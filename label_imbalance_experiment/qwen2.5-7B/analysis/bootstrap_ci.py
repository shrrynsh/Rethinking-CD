"""
Attach a 95% bootstrap confidence interval to the accuracy of every
(regime, dataset, method, split) cell of the built label-imbalance regimes.

For each cell we form the per-question correctness vector (1 if the answer
matches the label, else 0), report point accuracy = mean * 100, and resample
the questions with replacement B times to get a percentile CI. The CI is on
accuracy only; it quantifies how much of a cell's accuracy is sampling noise
given the number of questions in that regime.

Reads the output of build_regimes.py; writes nothing (prints a table).
"""
import argparse
import json
import os
import numpy as np

VARIANTS = ["random", "popular", "adversarial"]
METHODS = ["baseline", "vcd", "sid", "icd", "pba", "olm"]
DATASETS = ["gqa", "coco", "aokvqa"]
REGIMES = ["25_75", "75_25"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def correctness(ref_path, res_path):
    ref = load_jsonl(ref_path)
    res = {r["question_id"]: r for r in load_jsonl(res_path)}
    out = []
    for r in ref:
        gt_yes = r["label"].strip().lower() == "yes"
        pred_yes = "yes" in res[r["question_id"]]["text"].strip().lower()
        out.append(1 if gt_yes == pred_yes else 0)
    return np.asarray(out, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="llava-7b")
    ap.add_argument("--regime-dir", default="./regimes")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for regime in REGIMES:
        for dataset in DATASETS:
            base = os.path.join(args.regime_dir, regime, dataset)
            if not os.path.isdir(base):
                continue
            print(f"\n=== {regime}  {dataset} ===")
            print(f"  {'method':9s} {'split':12s} {'acc':>7s}  {'95% CI':>16s}")
            for method in METHODS:
                for variant in VARIANTS:
                    ref = os.path.join(base, "data", f"{dataset}_pope_{variant}.json")
                    res = os.path.join(base, "outputs", method,
                                       f"{args.model_tag}-{dataset}-{variant}-greedy.jsonl")
                    if not (os.path.exists(ref) and os.path.exists(res)):
                        continue
                    c = correctness(ref, res)
                    n = len(c)
                    acc = c.mean() * 100
                    draws = c[rng.integers(0, n, size=(args.boot, n))].mean(axis=1) * 100
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    print(f"  {method:9s} {variant:12s} {acc:7.2f}  [{lo:6.2f}, {hi:6.2f}]")


if __name__ == "__main__":
    main()
