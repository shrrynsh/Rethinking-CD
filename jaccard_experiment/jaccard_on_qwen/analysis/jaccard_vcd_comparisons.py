"""
Table 3: BPE-token Jaccard similarity between VCD and three reference conditions.

Comparisons:
  VCD vs Greedy
  VCD vs APC (beta=0.1)   -- VCD's own APC setting (cd_beta=0.1)
  VCD vs Direct Sampling

Note: VCD output has 42/60 questions (questions 0-41; run was interrupted).
All comparisons are restricted to the common question IDs.

Tokenizer: Qwen2TokenizerFast (tiktoken BPE, vocab=151936)
Normalization: strip markdown artifacts before tokenizing (see text_utils.py)
"""

import sys
import os
import json
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_utils import normalize

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.dirname(SCRIPT_DIR)
OUTPUTS     = os.path.join(EXPERIMENT, "outputs", "llava_bench")
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

VCD_FILE    = os.path.join(OUTPUTS, "vcd",    "qwen2.5-7b-llava_bench-vcd.jsonl")
GREEDY_FILE = os.path.join(OUTPUTS, "greedy", "qwen2.5-7b-llava_bench-greedy.jsonl")
SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "qwen2.5-7b-llava_bench-sample.jsonl")
APC01_FILE  = os.path.join(OUTPUTS, "apc", "beta_0.100", "qwen2.5-7b-llava_bench-apc-beta0.100.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def token_set(tokenizer, text):
    return set(tokenizer.encode(normalize(text), add_special_tokens=False))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare(tokenizer, base, other, base_name, other_name):
    common_ids = sorted(set(base) & set(other))
    scores = [
        jaccard(token_set(tokenizer, base[qid]),
                token_set(tokenizer, other[qid]))
        for qid in common_ids
    ]
    mean_j = float(np.mean(scores))
    print(f"  {base_name:6s} vs {other_name:20s}  Jaccard={mean_j:.4f}  (n={len(scores)})")
    return mean_j, len(scores)


def main():
    print("Loading Qwen tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  vocab size: {tokenizer.vocab_size}")

    print("Loading output files …")
    vcd    = load_jsonl(VCD_FILE)
    greedy = load_jsonl(GREEDY_FILE)
    sample = load_jsonl(SAMPLE_FILE)
    apc01  = load_jsonl(APC01_FILE)

    print(f"\n  VCD questions: {len(vcd)} (out of 60 total)")

    print("\n── VCD comparisons ──────────────────────────────────────────")
    r1, n1 = compare(tokenizer, vcd, greedy, "VCD", "Greedy")
    r2, n2 = compare(tokenizer, vcd, apc01,  "VCD", "APC (beta=0.1)")
    r3, n3 = compare(tokenizer, vcd, sample, "VCD", "Direct Sampling")

    out_dir  = os.path.join(EXPERIMENT, "analysis")
    csv_path = os.path.join(out_dir, "jaccard_vcd_comparisons.csv")
    with open(csv_path, "w") as f:
        f.write("comparison,mean_jaccard,n_questions\n")
        f.write(f"VCD vs Greedy,{r1:.6f},{n1}\n")
        f.write(f"VCD vs APC_beta0.1,{r2:.6f},{n2}\n")
        f.write(f"VCD vs Direct Sampling,{r3:.6f},{n3}\n")

    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
