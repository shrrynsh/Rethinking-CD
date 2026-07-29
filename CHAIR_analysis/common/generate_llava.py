"""
generate_llava.py -- generation for LLaVA-1.5-7B, three modes.

  --mode capture --method {vcd,sid}
        Real contrastive decoding. At every step it records the top-30 tokens
        and logits of the expert branch, the amateur branch, and the contrastive
        score C = 2E - A (pre-APC), plus the emitted token. This per-step record
        is the input for BOTH analyses (agreement/overlap and the contrastive
        adjustment d = E - A). One line per image.

  --mode proxy --stats-source {vcd,sid} --seed N
        No amateur branch. Adds independent Gaussian noise N(mu, sigma^2) to
        EVERY vocabulary logit of the expert, matched to the measured mean/std
        of d = E - A for the chosen method, then the same plausibility
        constraint. Caption only.

  --mode greedy
        Plain expert argmax baseline. Caption only.

Paths are relative to this repository. Weights live in <repo>/models,
images in <repo>/data/coco/val2017. Outputs are written to the path given by
--out. Runs are resumable and per-image seeded (reproducible on restart).
"""
import argparse, json, math, os, sys, types, random
from pathlib import Path

HERE = Path(__file__).resolve().parent          # common/
REPO = HERE.parent                              # CHAIR_analysis/
sys.path.insert(0, str(HERE))
import torch
from PIL import Image

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "llava" in CONFIG_MAPPING._mapping:
    del CONFIG_MAPPING._mapping["llava"]
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

MODEL_PATH = str(REPO / "models" / "llava-v1.5-7b")
IMAGE_DIR = REPO / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(REPO / "image_ids_500.json"))
STATS_PATH = REPO / "proxy_stats_llava.json"

MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.2
LOG_BETA = math.log(CD_BETA)
NOISE_STEP = 500
TOPK = 30
PROMPT = "Describe this image in detail."


def add_diffusion_noise(image_tensor, noise_step):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, dim=0).to(image_tensor.device)
    a, am1 = ap[noise_step].sqrt(), (1 - ap[noise_step]).sqrt()
    return a * image_tensor.clone() + am1 * torch.randn_like(image_tensor)


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print(f"[llava] mode={args.mode} method={args.method} seed={args.seed} n={args.n_images}", flush=True)

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    eos_id = tokenizer.eos_token_id or 2
    noise_gen = torch.Generator(device=device).manual_seed(args.seed)

    if args.mode == "proxy":
        stats = json.load(open(STATS_PATH))[args.stats_source]
        mu, sigma = float(stats["pooled_mean"]), float(stats["pooled_std"])
        print(f"[llava] proxy noise N(mu={mu:.4f}, sigma={sigma:.4f}) over full vocab", flush=True)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_ids_base = tokenizer_image_token(conv.get_prompt(), tokenizer,
                                            IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["image_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")
    two_branch = args.mode == "capture"

    import time; t0 = time.time(); n = 0
    for image_id in IMAGE_IDS[: args.n_images]:
        if image_id in done:
            continue
        iseed = (args.seed * 1_000_003 + image_id) % (2**31 - 1)
        torch.manual_seed(iseed); noise_gen.manual_seed(iseed)

        img = Image.open(IMAGE_DIR / f"{image_id:012d}.jpg").convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)
        prompt_ids = prompt_ids_base.to(device)
        noisy = None
        if two_branch and args.method == "vcd":
            noisy = add_diffusion_noise(img_tensor.squeeze(0).float(), NOISE_STEP).unsqueeze(0).to(dtype).to(device)

        expert_pkv = amateur_pkv = attn_e = attn_a = None
        chosen_ids, steps_out = [], []

        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                if two_branch:
                    if args.method == "vcd":
                        amateur_inp = dict(input_ids=prompt_ids, images=noisy, use_cache=True, return_dict=True)
                    else:
                        amateur_inp = dict(input_ids=prompt_ids, images=img_tensor, use_sid=True,
                                           use_cache=True, return_dict=True)
            else:
                new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_e,
                                  use_cache=True, return_dict=True)
                if two_branch:
                    amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_a,
                                       use_cache=True, return_dict=True)
                    if args.method == "sid":
                        amateur_inp["use_sid"] = True

            with torch.no_grad():
                expert_out = model(**expert_inp)
                E = expert_out.logits[0, -1].float()
                if two_branch:
                    amateur_out = model(**amateur_inp)
                    A = amateur_out.logits[0, -1].float()

            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            if args.mode == "greedy":
                scored = E.clone()
            elif two_branch:
                scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
                scored = scored.clone(); scored[mask] = float("-inf")
            else:  # proxy
                noise = torch.randn(E.shape[-1], generator=noise_gen, device=device) * sigma + mu
                scored = E + noise.float(); scored[mask] = float("-inf")
            chosen_id = int(scored.argmax().item())

            if two_branch:
                ev, ei = torch.topk(E, TOPK); av, ai = torch.topk(A, TOPK)
                cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
                cv, ci = torch.topk(cd_pre, TOPK)
                steps_out.append({
                    "step": step, "chosen_id": chosen_id, "apc_cutoff": round(cutoff, 4),
                    "expert_top_ids": ei.tolist(), "expert_top_logits": [round(v, 4) for v in ev.tolist()],
                    "amateur_top_ids": ai.tolist(), "amateur_top_logits": [round(v, 4) for v in av.tolist()],
                    "cd_top_ids": ci.tolist(), "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
                    "cd_top_survives_apc": (~mask[ci]).tolist(),
                })

            expert_pkv = expert_out.past_key_values
            if two_branch:
                amateur_pkv = amateur_out.past_key_values
            if step == 0:
                attn_e = torch.ones(1, expert_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
                if two_branch:
                    attn_a = torch.ones(1, amateur_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
            else:
                attn_e = torch.cat([attn_e, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                if two_branch:
                    attn_a = torch.cat([attn_a, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

        rec = {"image_id": image_id,
               "caption": tokenizer.decode(chosen_ids, skip_special_tokens=True).strip()}
        if two_branch:
            rec["steps"] = steps_out
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[llava] DONE {args.mode} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
