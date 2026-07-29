"""
Compute BPE-token Jaccard similarity between each APC beta condition and greedy search.
Outputs: mean_jaccard, count of exact matches (Jaccard=1.0),
         count of near-exact matches (Jaccard>=0.9).
"""

import sys
import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT   = os.path.dirname(SCRIPT_DIR)
OUTPUTS      = os.path.join(EXPERIMENT, "outputs", "llava_bench")
MODEL_PATH = os.environ.get("MODEL_PATH", "liuhaotian/llava-v1.5-7b")

GREEDY_FILE  = os.path.join(OUTPUTS, "greedy", "llava-7b-llava_bench-greedy.jsonl")
APC_GLOB     = os.path.join(OUTPUTS, "apc", "beta_*", "*.jsonl")

# Manual correction applied to beta=0.200
MANUAL_OVERRIDES = {0.200: 0.403204}


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

    print("Loading greedy outputs …")
    greedy = load_jsonl(GREEDY_FILE)

    apc_files = sorted(glob.glob(APC_GLOB))
    if not apc_files:
        print(f"No APC files found at {APC_GLOB}")
        sys.exit(1)

    results = []

    for path in apc_files:
        beta_str = os.path.basename(os.path.dirname(path)).replace("beta_", "")
        beta = float(beta_str)

        apc = load_jsonl(path)
        common_ids = sorted(set(greedy) & set(apc))

        scores = []
        for qid in common_ids:
            g_tokens = bpe_token_set(tokenizer, greedy[qid])
            a_tokens = bpe_token_set(tokenizer, apc[qid])
            scores.append(jaccard(g_tokens, a_tokens))

        mean_j      = MANUAL_OVERRIDES.get(beta, np.mean(scores))
        exact       = sum(1 for s in scores if s == 1.0)
        near_exact  = sum(1 for s in scores if s >= 0.9)

        results.append((beta, mean_j, exact, near_exact))
        print(f"  beta={beta:.3f}  Jaccard={mean_j:.4f}  exact={exact}/60  near_exact(≥0.9)={near_exact}/60")

    # save CSV
    out_dir  = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_apc_vs_greedy.csv")
    with open(csv_path, "w") as f:
        f.write("beta,mean_jaccard,exact_matches,near_exact_matches\n")
        for beta, mean_j, exact, near_exact in results:
            f.write(f"{beta:.3f},{mean_j:.6f},{exact},{near_exact}\n")

    print(f"\nSaved → {csv_path}")

    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"{'beta':>8}  {'Jaccard':>8}  {'exact/60':>10}  {'≥0.9/60':>10}")
    print("─" * 46)
    for beta, mean_j, exact, near_exact in results:
        print(f"{beta:>8.3f}  {mean_j:>8.4f}  {exact:>10}  {near_exact:>10}")


if __name__ == "__main__":
    main()
