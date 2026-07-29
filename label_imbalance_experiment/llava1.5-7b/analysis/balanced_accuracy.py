"""
Score every (regime, dataset, method, split) cell of the built label-imbalance
regimes and report standard accuracy, balanced accuracy, and the predicted
yes-rate.

Standard accuracy on POPE is class-balance sensitive: the same answers score
differently under 25/75 and 75/25. Balanced accuracy is not:

  balanced accuracy = (TPR + TNR) / 2

which equals the AUC of the two-point ROC implied by a single hard threshold.
A pure class-balance shift with identical answers leaves it unchanged; only a
change in the model's yes/no threshold moves it. Comparing a method's balanced
accuracy across the two regimes therefore separates a genuine change in
discrimination from an artifact of the label ratio.

Reads the output of build_regimes.py; writes nothing (prints a table).
"""
import argparse
import json
import os

VARIANTS = ["random", "popular", "adversarial"]
METHODS = ["baseline", "vcd", "sid", "icd", "pba", "olm"]
DATASETS = ["gqa", "coco", "aokvqa"]
REGIMES = ["25_75", "75_25"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def counts(ref_path, res_path):
    ref = load_jsonl(ref_path)
    res = {r["question_id"]: r for r in load_jsonl(res_path)}
    tp = tn = fp = fn = 0
    for r in ref:
        gt_yes = r["label"].strip().lower() == "yes"
        pred_yes = "yes" in res[r["question_id"]]["text"].strip().lower()
        if gt_yes and pred_yes:
            tp += 1
        elif gt_yes:
            fn += 1
        elif pred_yes:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    std = (tp + tn) / total * 100
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal = (tpr + tnr) / 2 * 100
    yes = (tp + fp) / total * 100
    return std, bal, yes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="llava-7b")
    ap.add_argument("--regime-dir", default="./regimes")
    args = ap.parse_args()

    for regime in REGIMES:
        for dataset in DATASETS:
            base = os.path.join(args.regime_dir, regime, dataset)
            if not os.path.isdir(base):
                continue
            print(f"\n=== {regime}  {dataset} ===")
            print(f"  {'method':9s} {'split':12s} {'stdAcc':>8s} {'balAcc':>8s} {'yes%':>7s}")
            for method in METHODS:
                for variant in VARIANTS:
                    ref = os.path.join(base, "data", f"{dataset}_pope_{variant}.json")
                    res = os.path.join(base, "outputs", method,
                                       f"{args.model_tag}-{dataset}-{variant}-greedy.jsonl")
                    if not (os.path.exists(ref) and os.path.exists(res)):
                        continue
                    std, bal, yes = metrics(*counts(ref, res))
                    print(f"  {method:9s} {variant:12s} {std:8.2f} {bal:8.2f} {yes:7.1f}")


if __name__ == "__main__":
    main()
