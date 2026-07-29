"""
generate_qwen.py -- generation for Qwen2.5-VL-7B-Instruct, three modes.
Same interface and outputs as generate_llava.py.

  --mode capture --method {vcd,sid}   per-step top-30 record (expert / amateur /
        contrastive), one line per image. The amateur branch is recomputed
        cache-less at every step (correctness over speed: a hand-rolled mRoPE
        second-branch cache diverged from the exact computation), so capture on
        Qwen is slow. This record feeds both analyses.
  --mode proxy --stats-source {vcd,sid} --seed N   full-vocabulary Gaussian noise
        on the expert logits, no amateur branch. Caption only.
  --mode greedy   plain expert argmax baseline. Caption only.

Weights: <repo>/models/Qwen2.5-VL-7B-Instruct. Images: <repo>/data/coco/val2017.
Resumable and per-image seeded.
"""
import argparse, json, math, os, sys, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import torch
from PIL import Image
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          LogitsProcessor, LogitsProcessorList)
from qwen_vl_utils import process_vision_info

MODEL_PATH = str(REPO / "models" / "Qwen2.5-VL-7B-Instruct")
IMAGE_DIR = REPO / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(REPO / "image_ids_500.json"))
STATS_PATH = REPO / "proxy_stats_qwen.json"

IMAGE_TOKEN_ID = 151655
AGG_LAYER = 2
SID_KEEP_FRAC = 72.0 / 576.0
MAX_NEW_TOKENS = 256
CD_ALPHA, CD_BETA = 1.0, 0.2
LOG_BETA = math.log(CD_BETA)
TOPK = 30
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_diffusion_noise(x, t=500):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, 0)
    return torch.sqrt(ap[t]) * x + torch.sqrt(1 - ap[t]) * torch.randn_like(x)


class SIDState:
    def __init__(self):
        self.active = False
        self.masked_cols = None


def install_sid_hooks(model, state):
    minv = torch.finfo(model.dtype).min

    def mk(idx):
        def hook(mod, args, kwargs):
            if not state.active or idx < AGG_LAYER or state.masked_cols is None:
                return None
            hs = args[0] if args else kwargs.get("hidden_states")
            L = hs.shape[1]
            m4 = torch.zeros(1, 1, L, L, dtype=model.dtype, device=hs.device)
            m4[0, 0][torch.triu(torch.ones(L, L, dtype=torch.bool, device=hs.device), 1)] = minv
            m4[0, 0][:, state.masked_cols] = minv
            kwargs["attention_mask"] = m4
            return args, kwargs
        return hook
    for i, layer in enumerate(model.model.language_model.layers):
        layer.register_forward_pre_hook(mk(i), with_kwargs=True)


class CaptureProcessor(LogitsProcessor):
    """capture mode: compute amateur cache-less, record top-30 of E/A/C, return post-APC CD."""
    def __init__(self, model, method, pv_cd, thw, sid_state, sink):
        self.model, self.method, self.pv_cd, self.thw, self.sid, self.sink = \
            model, method, pv_cd, thw, sid_state, sink

    def __call__(self, input_ids, scores):
        E = scores[0].float()
        if self.method == "sid":
            self.sid.active = True
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             pixel_values=self.pv_cd, image_grid_thw=self.thw, use_cache=False)
        if self.method == "sid":
            self.sid.active = False
        A = out.logits[0, -1].float()
        cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
        cutoff = LOG_BETA + E.max().item()
        mask = E < cutoff
        cd_post = cd_pre.clone(); cd_post[mask] = NEG_INF
        ev, ei = torch.topk(E, TOPK); av, ai = torch.topk(A, TOPK); cv, ci = torch.topk(cd_pre, TOPK)
        self.sink.append({
            "step": len(self.sink), "chosen_id": int(cd_post.argmax().item()),
            "apc_cutoff": round(cutoff, 4),
            "expert_top_ids": ei.tolist(), "expert_top_logits": [round(v, 4) for v in ev.tolist()],
            "amateur_top_ids": ai.tolist(), "amateur_top_logits": [round(v, 4) for v in av.tolist()],
            "cd_top_ids": ci.tolist(), "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
            "cd_top_survives_apc": (~mask[ci]).tolist(),
        })
        return cd_post.unsqueeze(0).to(scores.dtype)


class ProxyProcessor(LogitsProcessor):
    def __init__(self, mu, sigma, gen, device):
        self.mu, self.sigma, self.gen, self.device = mu, sigma, gen, device

    def __call__(self, input_ids, scores):
        E = scores[0].float()
        noise = torch.randn(E.shape[-1], generator=self.gen, device=self.device) * self.sigma + self.mu
        scored = E + noise
        scored[E < (LOG_BETA + E.max().item())] = NEG_INF
        return scored.unsqueeze(0).to(scores.dtype)


def build(proc, img, device):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(m)
    return proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capture", "proxy", "greedy"], required=True)
    ap.add_argument("--method", choices=["vcd", "sid"], help="required for --mode capture")
    ap.add_argument("--stats-source", choices=["vcd", "sid"], default="vcd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "capture" and not args.method:
        ap.error("--mode capture requires --method {vcd,sid}")

    set_seed(args.seed)
    print(f"[qwen] mode={args.mode} method={args.method} seed={args.seed} n={args.n_images}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)
    device = model.device
    sid_state = SIDState()
    if args.mode == "capture" and args.method == "sid":
        install_sid_hooks(model, sid_state)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed)
    if args.mode == "proxy":
        stats = json.load(open(STATS_PATH))[args.stats_source]
        mu, sigma = float(stats["pooled_mean"]), float(stats["pooled_std"])
        print(f"[qwen] proxy noise N(mu={mu:.4f}, sigma={sigma:.4f}) over full vocab", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["image_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")

    import time; t0 = time.time(); n = 0
    for image_id in IMAGE_IDS[: args.n_images]:
        if image_id in done:
            continue
        iseed = (args.seed * 1_000_003 + image_id) % (2**31 - 1)
        torch.manual_seed(iseed); noise_gen.manual_seed(iseed)
        img = Image.open(IMAGE_DIR / f"{image_id:012d}.jpg").convert("RGB")
        inp = build(processor, img, device)
        rec = {"image_id": image_id}

        if args.mode == "greedy":
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        else:
            if args.mode == "capture":
                if args.method == "vcd":
                    pv_cd = add_diffusion_noise(inp["pixel_values"].float(), 500).to(inp["pixel_values"].dtype)
                else:
                    pv_cd = inp["pixel_values"]
                    pos = (inp.input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                    keep = max(1, int(round(SID_KEEP_FRAC * pos.numel())))
                    perm = torch.randperm(pos.numel(), device=device)
                    sid_state.masked_cols = pos[perm[keep:]]
                sink = []
                proc = CaptureProcessor(model, args.method, pv_cd, inp["image_grid_thw"], sid_state, sink)
            else:  # proxy
                proc = ProxyProcessor(mu, sigma, noise_gen, device)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     logits_processor=LogitsProcessorList([proc]))
            if args.mode == "capture":
                rec["steps"] = sink

        rec["caption"] = processor.tokenizer.decode(out[0, inp.input_ids.shape[1]:],
                                                    skip_special_tokens=True).strip()
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[qwen] DONE {args.mode} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
