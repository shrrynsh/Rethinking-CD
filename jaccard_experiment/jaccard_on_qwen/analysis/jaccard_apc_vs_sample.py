"""
Table 2: BPE-token Jaccard similarity between each APC beta and direct sampling.

Tokenizer: Qwen2TokenizerFast (tiktoken BPE, vocab=151936)
Normalization: strip markdown artifacts before tokenizing (see text_utils.py)

Columns: beta | mean_jaccard | n_questions
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

SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "qwen2.5-7b-llava_bench-sample.jsonl")
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
            s_tok = token_set(tokenizer, sample[qid])
            a_tok = token_set(tokenizer, apc[qid])
            scores.append(jaccard(s_tok, a_tok))

        mean_j = float(np.mean(scores))
        n      = len(scores)

        results.append((beta, mean_j, n))
        print(f"  beta={beta:.3f}  Jaccard={mean_j:.4f}  (n={n})")

    out_dir  = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_apc_vs_sample.csv")
    with open(csv_path, "w") as f:
        f.write("beta,mean_jaccard,n_questions\n")
        for beta, mean_j, n in results:
            f.write(f"{beta:.3f},{mean_j:.6f},{n}\n")

    print(f"\nSaved → {csv_path}")

    print("\n── APC vs Direct Sampling ────────────────────────")
    print(f"{'beta':>8}  {'Jaccard':>8}  {'n':>6}")
    print("─" * 30)
    for beta, mean_j, n in results:
        print(f"{beta:>8.3f}  {mean_j:>8.4f}  {n:>6}")


if __name__ == "__main__":
    main()
