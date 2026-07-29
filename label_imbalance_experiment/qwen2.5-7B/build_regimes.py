"""
Build the two label-imbalance regimes from a finished POPE run.

Standard POPE gives every image six questions: three whose answer is YES and
three whose answer is NO (a 50/50 balance). This script re-balances that ground
truth by deleting questions, while leaving every model answer untouched:

  25/75 regime : delete 2 of the 3 YES questions per image
                 -> 1 YES + 3 NO  = 25% YES / 75% NO
  75/25 regime : delete 2 of the 3 NO  questions per image
                 -> 3 YES + 1 NO  = 75% YES / 25% NO

The deletion mask is drawn once (fixed seed) from the RANDOM split, then applied
identically to every split and every decoding method, so all methods are always
scored on exactly the same set of questions and only the label balance changes.

Inputs (never modified):
  --anno-dir   POPE annotation files: <anno-dir>/<dataset>/<dataset>_pope_<split>.json
  --pred-dir   model answers:         <pred-dir>/<method>/<tag>-<dataset>-<split>-greedy.jsonl

Output:
  --out-dir/<regime>/<dataset>/data/...      subsampled annotations
  --out-dir/<regime>/<dataset>/outputs/...   subsampled answers (aligned 1:1)

No results are shipped; run this to (re)create them.
"""
import argparse
import json
import os
import random
from collections import defaultdict, Counter

VARIANTS = ["random", "popular", "adversarial"]
METHODS = ["baseline", "vcd", "sid", "icd", "pba", "olm"]
DATASETS = ["gqa", "coco", "aokvqa"]
# regime -> label whose questions are deleted (keep 1 of 3, keep all of the other)
DELETE_LABEL = {"25_75": "yes", "75_25": "no"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_keep_ids(anno_dir, dataset, regime, seed):
    """Return {variant: set(question_id to keep)} for one dataset and regime."""
    del_label = DELETE_LABEL[regime]

    def anno_path(variant):
        return os.path.join(anno_dir, dataset, f"{dataset}_pope_{variant}.json")

    # 1) canonical image order + deletion mask from the RANDOM split
    random_data = load_jsonl(anno_path("random"))
    image_order, by_image = [], defaultdict(list)
    for q in random_data:
        if q["image"] not in by_image:
            image_order.append(q["image"])
        by_image[q["image"]].append(q)

    rng = random.Random(seed)
    delete_ordinals = {}  # image -> {ordinals of del_label questions to remove}
    for img in image_order:
        target = [q for q in by_image[img] if q["label"] == del_label]
        assert len(target) == 3, (
            f"{dataset}/{img}: expected 3 '{del_label}' questions, got {len(target)}"
        )
        delete_ordinals[img] = set(rng.sample([0, 1, 2], 2))

    combos = Counter(tuple(sorted(v)) for v in delete_ordinals.values())
    print(f"    mask ({regime}, delete '{del_label}'): combos removed = {dict(combos)}")

    # 2) apply the same mask to every split
    keep_ids = {}
    for variant in VARIANTS:
        data = load_jsonl(anno_path(variant))
        per_image = defaultdict(list)
        for q in data:
            per_image[q["image"]].append(q)

        keep = set()
        for img in image_order:
            target = [q for q in per_image[img] if q["label"] == del_label]
            other = [q for q in per_image[img] if q["label"] != del_label]
            for ordinal, q in enumerate(target):
                if ordinal not in delete_ordinals[img]:
                    keep.add(q["question_id"])
            for q in other:
                keep.add(q["question_id"])
        keep_ids[variant] = keep

        yes = sum(1 for q in data if q["question_id"] in keep and q["label"] == "yes")
        no = sum(1 for q in data if q["question_id"] in keep and q["label"] == "no")
        print(f"    {variant:11s}: keep {len(keep):5d}  YES={yes:5d}  NO={no:5d}  "
              f"YES%={yes / (yes + no) * 100:.1f}")
    return keep_ids


def apply_and_write(anno_dir, pred_dir, out_dir, dataset, regime, tag, keep_ids):
    for variant in VARIANTS:
        keep = keep_ids[variant]

        # annotations
        src_anno = os.path.join(anno_dir, dataset, f"{dataset}_pope_{variant}.json")
        anno = [q for q in load_jsonl(src_anno) if q["question_id"] in keep]
        anno_order = [q["question_id"] for q in anno]
        write_jsonl(anno, os.path.join(
            out_dir, regime, dataset, "data", f"{dataset}_pope_{variant}.json"))

        # each method's answers, filtered to the same ids and checked for alignment
        for method in METHODS:
            src_pred = os.path.join(
                pred_dir, method, f"{tag}-{dataset}-{variant}-greedy.jsonl")
            if not os.path.exists(src_pred):
                print(f"    [skip] missing predictions: {src_pred}")
                continue
            pred = [q for q in load_jsonl(src_pred) if q["question_id"] in keep]
            pred_order = [q["question_id"] for q in pred]
            if pred_order != anno_order:
                raise ValueError(
                    f"question_id order mismatch: {method}/{dataset}/{variant}")
            write_jsonl(pred, os.path.join(
                out_dir, regime, dataset, "outputs", method,
                f"{tag}-{dataset}-{variant}-greedy.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag", default="llava-7b",
                    help="answer-file prefix, e.g. 'llava-7b' or 'Qwen2.5-7b'")
    ap.add_argument("--anno-dir", default="./data",
                    help="POPE annotations root (<dataset>/<dataset>_pope_<split>.json)")
    ap.add_argument("--pred-dir", required=True,
                    help="finished POPE answers root (<method>/<tag>-<dataset>-<split>-greedy.jsonl)")
    ap.add_argument("--out-dir", default="./regimes",
                    help="where the subsampled regimes are written")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--regimes", nargs="+", default=list(DELETE_LABEL),
                    choices=list(DELETE_LABEL))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for regime in args.regimes:
        print(f"\n=== regime {regime} ===")
        for dataset in args.datasets:
            print(f"  {dataset}:")
            keep_ids = build_keep_ids(args.anno_dir, dataset, regime, args.seed)
            apply_and_write(args.anno_dir, args.pred_dir, args.out_dir,
                            dataset, regime, args.model_tag, keep_ids)
    print("\nDone. Subsampled regimes written under", args.out_dir)


if __name__ == "__main__":
    main()
