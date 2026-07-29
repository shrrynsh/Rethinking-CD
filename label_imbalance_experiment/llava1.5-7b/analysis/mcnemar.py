"""
Paired McNemar test of each method against the Baseline, per
(regime, dataset, split), on the built label-imbalance regimes.

Both methods answer the same questions, so their per-question correctness
vectors are paired. McNemar looks only at the disagreements:

  b = questions the baseline gets right and the method gets wrong
  c = questions the baseline gets wrong and the method gets right

Under the null (the two are equally accurate) each discordant pair is a fair
coin, so the p-value is the exact two-sided binomial tail of min(b, c) out of
b + c. This tests whether a method's accuracy differs from the baseline's on
the same items, which is stronger than comparing two separate accuracy CIs.

Reads the output of build_regimes.py; writes nothing (prints a table).
"""
import argparse
import json
import os
from math import comb

VARIANTS = ["random", "popular", "adversarial"]
METHODS = ["vcd", "sid", "icd", "pba", "olm"]  # each compared vs baseline
DATASETS = ["gqa", "coco", "aokvqa"]
REGIMES = ["25_75", "75_25"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def correct_map(ref_path, res_path):
    ref = load_jsonl(ref_path)
    res = {r["question_id"]: r for r in load_jsonl(res_path)}
    out = {}
    for r in ref:
        gt_yes = r["label"].strip().lower() == "yes"
        pred_yes = "yes" in res[r["question_id"]]["text"].strip().lower()
        out[r["question_id"]] = int(gt_yes == pred_yes)
    return out


def exact_two_sided_binom(b, c):
    """Two-sided exact binomial p for min(b, c) successes out of n=b+c, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="llava-7b")
    ap.add_argument("--regime-dir", default="./regimes")
    args = ap.parse_args()

    def res_path(base, method, dataset, variant):
        return os.path.join(base, "outputs", method,
                            f"{args.model_tag}-{dataset}-{variant}-greedy.jsonl")

    for regime in REGIMES:
        for dataset in DATASETS:
            base = os.path.join(args.regime_dir, regime, dataset)
            if not os.path.isdir(base):
                continue
            print(f"\n=== {regime}  {dataset}  (method vs baseline) ===")
            print(f"  {'method':9s} {'split':12s} {'b':>5s} {'c':>5s} {'p':>10s}")
            for variant in VARIANTS:
                ref = os.path.join(base, "data", f"{dataset}_pope_{variant}.json")
                base_res = res_path(base, "baseline", dataset, variant)
                if not (os.path.exists(ref) and os.path.exists(base_res)):
                    continue
                base_c = correct_map(ref, base_res)
                for method in METHODS:
                    mp = res_path(base, method, dataset, variant)
                    if not os.path.exists(mp):
                        continue
                    meth_c = correct_map(ref, mp)
                    b = sum(1 for q in base_c if base_c[q] == 1 and meth_c[q] == 0)
                    c = sum(1 for q in base_c if base_c[q] == 0 and meth_c[q] == 1)
                    p = exact_two_sided_binom(b, c)
                    print(f"  {method:9s} {variant:12s} {b:5d} {c:5d} {p:10.4g}")


if __name__ == "__main__":
    main()
