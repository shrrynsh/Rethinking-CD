"""
contrastive_analysis.py -- "contrastive adjustment" experiments for one model.

Reads the per-step capture files (outputs/captures/<model>_{vcd,sid}.jsonl) and,
if present, the proxy / greedy caption files (outputs/captions/), and reports:

  (1) The contrastive adjustment d = E - A on emitted object words and on the
      expert's top-10 / top-30 object candidates: mean d and the percentage with
      d > 0, split by real / hallucinated object. d > 0 means contrastive
      decoding amplifies the token; a working suppressor would give hallucinated
      objects clearly negative d.

  (2) The bucket view: among the top-5 object candidates at each object step, the
      normalized probability mass on real vs hallucinated objects, per branch
      (expert / amateur / contrastive), with 95% image-level bootstrap CIs.

  (3) CHAIR for the noise proxy vs greedy, if those caption files exist, so the
      proxy can be compared against the undefended baseline.

Usage:
  python contrastive_analysis.py --model llava
  python contrastive_analysis.py --model qwen --B 10000
"""
import argparse, json, math, sys, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
COCO = REPO / "data" / "coco" / "annotations"
CAPTURES = REPO / "outputs" / "captures"
CAPTIONS = REPO / "outputs" / "captions"
MODEL_DIR = {"llava": "llava-v1.5-7b", "qwen": "Qwen2.5-VL-7B-Instruct"}


def softmax(logits):
    m = max(logits); e = [math.exp(x - m) for x in logits]; z = sum(e)
    return [x / z for x in e]


# ---------- (1) d = E - A ----------
def d_stats(model, method, tok, chair_mod, singularize):
    recs = [json.loads(l) for l in open(CAPTURES / f"{model}_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    pops = {p: {"real": [], "hall": []} for p in ("emitted", "top10", "top30")}
    for r in recs:
        steps = r["steps"]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        chosen = [s["chosen_id"] for s in steps]
        emap = {s["step"]: dict(zip(s["expert_top_ids"], s["expert_top_logits"])) for s in steps}
        amap = {s["step"]: dict(zip(s["amateur_top_ids"], s["amateur_top_logits"])) for s in steps}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si >= len(steps):
                continue
            tid = steps[si]["chosen_id"]
            E, A = emap[si].get(tid), amap[si].get(tid)
            if E is not None and A is not None:
                pops["emitted"]["hall" if m["hallucinated"] else "real"].append(E - A)
        for s in steps:
            am = amap[s["step"]]
            for K, key in ((10, "top10"), (30, "top30")):
                for tid, E in zip(s["expert_top_ids"][:K], s["expert_top_logits"][:K]):
                    txt = tok.decode([tid])
                    if not (txt.startswith(" ") or txt.startswith("▁")):
                        continue
                    w = "".join(c for c in txt.lower() if c.isalpha())
                    if not w:
                        continue
                    w = singularize(w)
                    if w not in chair.mscoco_objects:
                        continue
                    A = am.get(tid)
                    if A is None:
                        continue
                    lab = "hall" if chair.inverse_synonym_dict[w] not in gt else "real"
                    pops[key][lab].append(E - A)
    def summ(v):
        if not v:
            return "n=0"
        a = np.array(v)
        return f"mean {a.mean():+.3f}  %d>0 {100*(a>0).mean():.1f}  (n={len(v)})"
    print(f"\n--- {method.upper()}: d = E - A ---")
    for p in ("emitted", "top10", "top30"):
        print(f"  {p:8} real: {summ(pops[p]['real'])}")
        print(f"  {p:8} hall: {summ(pops[p]['hall'])}")


# ---------- (2) bucket mass with CI ----------
BRANCHES = {"Expert": ("expert_top_ids", "expert_top_logits"),
            "Amateur": ("amateur_top_ids", "amateur_top_logits"),
            "Contrastive": ("cd_top_ids", "cd_top_pre_apc_logits")}


def bucket(model, method, tok, chair_mod, singularize, B):
    recs = [json.loads(l) for l in open(CAPTURES / f"{model}_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    rows = {b: [] for b in BRANCHES}
    for r in recs:
        steps = r["steps"]; chosen = [s["chosen_id"] for s in steps]
        gt = chair.imid_to_objects.get(r["image_id"], set())
        lab = {}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] == "object":
                si = m["step_indices"][0]
                if si < len(steps):
                    lab.setdefault(si, "hall" if m["hallucinated"] else "real")
        acc = {b: [0.0, 0] for b in BRANCHES}
        for si, ml in lab.items():
            s = steps[si]
            for b, (ik, lk) in BRANCHES.items():
                ids, probs = s[ik][:5], softmax(s[lk])[:5]
                rp = hp = 0.0
                for tid, p in zip(ids, probs):
                    if tid == s["chosen_id"]:
                        c = ml
                    else:
                        txt = tok.decode([tid])
                        if not (txt.startswith(" ") or txt.startswith("▁")):
                            continue
                        w = "".join(ch for ch in txt.lower() if ch.isalpha())
                        if not w:
                            continue
                        w = singularize(w)
                        if w not in chair.mscoco_objects:
                            continue
                        c = "real" if chair.inverse_synonym_dict[w] in gt else "hall"
                    if c == "real": rp += p
                    elif c == "hall": hp += p
                if rp + hp > 0:
                    acc[b][0] += rp / (rp + hp); acc[b][1] += 1
        for b in BRANCHES:
            rows[b].append(acc[b])
    print(f"\n--- {method.upper()}: normalized object-mass buckets (REAL share) [95% CI] ---")
    for b in BRANCHES:
        arr = np.array(rows[b], float)
        pt = arr[:, 0].sum() / arr[:, 1].sum()
        rng = np.random.default_rng(0); n = len(arr); d = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n); d[i] = arr[idx, 0].sum() / arr[idx, 1].sum()
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {b:12} REAL {pt:.5f} [{lo:.5f}, {hi:.5f}]   HALL {1-pt:.5f} [{1-hi:.5f}, {1-lo:.5f}]")


# ---------- (3) CHAIR for proxy vs greedy ----------
def chair_scores(model):
    from eval.chair import CHAIR
    files = sorted(glob.glob(str(CAPTIONS / f"{model}_greedy*.jsonl"))
                   + glob.glob(str(CAPTIONS / f"{model}_proxy*.jsonl")))
    if not files:
        print("\n--- CHAIR (proxy vs greedy): no caption files found in outputs/captions/, skipping ---")
        return
    print("\n--- CHAIR of proxy / greedy caption files (lower is better) ---")
    for f in files:
        caps = [json.loads(l) for l in open(f)]
        ids = {c["image_id"] for c in caps}
        ch = CHAIR(ids, str(COCO)); ch.get_annotations()
        res = ch.compute_chair(caps)
        print(f"  {Path(f).stem:32} CHAIR-S {res['CHAIR_s']*100:.1f}  CHAIR-I {res['CHAIR_i']*100:.1f}  (n={res['num_captions']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llava", "qwen"], required=True)
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import object_mentions as chair_mod
    from eval.chair import singularize
    slow = args.model == "llava"
    tok = AutoTokenizer.from_pretrained(str(REPO / "models" / MODEL_DIR[args.model]), use_fast=not slow)

    print(f"\n===== contrastive adjustment : {args.model} =====")
    for method in ["vcd", "sid"]:
        if not (CAPTURES / f"{args.model}_{method}.jsonl").exists():
            print(f"[skip] capture for {method} not found (run generate_{args.model}.py --mode capture --method {method})")
            continue
        d_stats(args.model, method, tok, chair_mod, singularize)
        bucket(args.model, method, tok, chair_mod, singularize, args.B)
    chair_scores(args.model)


if __name__ == "__main__":
    main()
