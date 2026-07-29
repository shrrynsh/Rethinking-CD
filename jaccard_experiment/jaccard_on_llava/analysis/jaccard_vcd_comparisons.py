"""
Compute BPE-token Jaccard similarity between VCD and:
  1. Greedy search
  2. APC with beta=0.1 (VCD's default APC setting)
  3. Direct sampling
"""

import os
import json
import numpy as np
from transformers import AutoTokenizer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.dirname(SCRIPT_DIR)
OUTPUTS     = os.path.join(EXPERIMENT, "outputs", "llava_bench")
MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.5-7b")

VCD_FILE    = os.path.join(OUTPUTS, "vcd",    "llava-7b-llava_bench-vcd.jsonl")
GREEDY_FILE = os.path.join(OUTPUTS, "greedy", "llava-7b-llava_bench-greedy.jsonl")
SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "llava-7b-llava_bench-sample.jsonl")
APC01_FILE  = os.path.join(OUTPUTS, "apc", "beta_0.100", "llava-7b-llava_bench-apc-beta0.100.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def bpe_token_set(tokenizer, text):
    return set(tokenizer.encode(text, add_special_tokens=False))


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare(tokenizer, base, other, base_name, other_name):
    common_ids = sorted(set(base) & set(other))
    scores = [
        jaccard(bpe_token_set(tokenizer, base[qid]),
                bpe_token_set(tokenizer, other[qid]))
        for qid in common_ids
    ]
    mean_j = np.mean(scores)
    print(f"  {base_name:12s} vs {other_name:20s}  Jaccard = {mean_j:.4f}  (n={len(scores)})")
    return mean_j


def main():
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading output files …")
    vcd    = load_jsonl(VCD_FILE)
    greedy = load_jsonl(GREEDY_FILE)
    sample = load_jsonl(SAMPLE_FILE)
    apc01  = load_jsonl(APC01_FILE)

    print("\n── VCD comparisons ──────────────────────────────────")
    r1 = compare(tokenizer, vcd, greedy, "VCD", "Greedy")
    r2 = compare(tokenizer, vcd, apc01,  "VCD", "APC (β=0.1)")
    r3 = compare(tokenizer, vcd, sample, "VCD", "Direct Sampling")

    # save CSV
    out_dir  = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_vcd_comparisons.csv")
    with open(csv_path, "w") as f:
        f.write("comparison,mean_jaccard\n")
        f.write(f"VCD vs Greedy,{r1:.6f}\n")
        f.write(f"VCD vs APC_beta0.1,{r2:.6f}\n")
        f.write(f"VCD vs Direct Sampling,{r3:.6f}\n")

    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
