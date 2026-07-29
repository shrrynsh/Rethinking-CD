"""
Compute BPE-token Jaccard similarity between each APC beta condition and direct sampling.
"""

import sys
import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.dirname(SCRIPT_DIR)
OUTPUTS     = os.path.join(EXPERIMENT, "outputs", "llava_bench")
MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.5-7b")

SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "llava-7b-llava_bench-sample.jsonl")
APC_GLOB    = os.path.join(OUTPUTS, "apc", "beta_*", "*.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def bpe_token_set(tokenizer, text):
    return set(tokenizer.encode(text, add_special_tokens=False))


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading direct sampling outputs …")
    sample = load_jsonl(SAMPLE_FILE)

    apc_files = sorted(glob.glob(APC_GLOB))
    if not apc_files:
        print(f"No APC files found at {APC_GLOB}")
        sys.exit(1)

    results = []

    for path in apc_files:
        beta_str = os.path.basename(os.path.dirname(path)).replace("beta_", "")
        beta = float(beta_str)

        apc = load_jsonl(path)
        common_ids = sorted(set(sample) & set(apc))

        scores = []
        for qid in common_ids:
            s_tokens = bpe_token_set(tokenizer, sample[qid])
            a_tokens = bpe_token_set(tokenizer, apc[qid])
            scores.append(jaccard(s_tokens, a_tokens))

        mean_j = np.mean(scores)
        results.append((beta, mean_j))
        print(f"  beta={beta:.3f}  Jaccard={mean_j:.4f}  (n={len(scores)})")

    out_dir = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_apc_vs_sample.csv")

    with open(csv_path, "w") as f:
        f.write("beta,mean_jaccard\n")
        for beta, mean_j in results:
            f.write(f"{beta:.3f},{mean_j:.6f}\n")

    print(f"\nSaved → {csv_path}")

    print("\n── Summary ──────────────────────────────────")
    print(f"{'beta':>8}  {'Jaccard (mean)':>14}")
    print("─" * 28)
    for beta, mean_j in results:
        print(f"{beta:>8.3f}  {mean_j:>14.4f}")


if __name__ == "__main__":
    main()
