"""
Table 1: BPE-token Jaccard similarity between each APC beta and greedy search.

Tokenizer: Qwen2TokenizerFast (tiktoken BPE, vocab=151936)
Normalization: strip markdown artifacts before tokenizing (see text_utils.py)

Columns: beta | mean_jaccard | exact_matches (=1.0) | near_exact_matches (>=0.9) | n_questions
"""

import sys
import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_utils import normalize

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.dirname(SCRIPT_DIR)
OUTPUTS     = os.path.join(EXPERIMENT, "outputs", "llava_bench")
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

GREEDY_FILE = os.path.join(OUTPUTS, "greedy", "qwen2.5-7b-llava_bench-greedy.jsonl")
APC_GLOB    = os.path.join(OUTPUTS, "apc", "beta_*", "*.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def token_set(tokenizer, text):
    return set(tokenizer.encode(normalize(text), add_special_tokens=False))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    print("Loading Qwen tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  vocab size: {tokenizer.vocab_size}")

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
            g_tok = token_set(tokenizer, greedy[qid])
            a_tok = token_set(tokenizer, apc[qid])
            scores.append(jaccard(g_tok, a_tok))

        mean_j     = float(np.mean(scores))
        exact      = sum(1 for s in scores if s == 1.0)
        near_exact = sum(1 for s in scores if s >= 0.9)
        n          = len(scores)

        results.append((beta, mean_j, exact, near_exact, n))
        print(f"  beta={beta:.3f}  Jaccard={mean_j:.4f}  exact={exact}/{n}  >=0.9={near_exact}/{n}")

    out_dir  = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_apc_vs_greedy.csv")
    with open(csv_path, "w") as f:
        f.write("beta,mean_jaccard,exact_matches,near_exact_matches,n_questions\n")
        for beta, mean_j, exact, near_exact, n in results:
            f.write(f"{beta:.3f},{mean_j:.6f},{exact},{near_exact},{n}\n")

    print(f"\nSaved → {csv_path}")

    print("\n── APC vs Greedy ────────────────────────────────────────────────")
    print(f"{'beta':>8}  {'Jaccard':>8}  {'exact/n':>10}  {'>=0.9/n':>10}")
    print("─" * 48)
    for beta, mean_j, exact, near_exact, n in results:
        print(f"{beta:>8.3f}  {mean_j:>8.4f}  {exact:>4}/{n:<4}  {near_exact:>4}/{n:<4}")


if __name__ == "__main__":
    main()
