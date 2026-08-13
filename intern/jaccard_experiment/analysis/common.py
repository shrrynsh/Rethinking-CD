"""Shared paths and Jaccard helpers for the InternVL3 analysis scripts."""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = os.path.dirname(SCRIPT_DIR)
OUTPUTS    = os.path.join(EXPERIMENT, "outputs", "llava_bench")

# Tokenizer source. The analysis only needs the tokenizer, not the weights, so
# this falls back to the Hub id if no local copy is configured.
MODEL_PATH = os.environ.get("MODEL_PATH", "OpenGVLab/InternVL3-8B-hf")

MODEL_ID = "internvl3-8b"

GREEDY_FILE = os.path.join(OUTPUTS, "greedy", f"{MODEL_ID}-llava_bench-greedy.jsonl")
SAMPLE_FILE = os.path.join(OUTPUTS, "sample", f"{MODEL_ID}-llava_bench-sample.jsonl")
VCD_FILE    = os.path.join(OUTPUTS, "vcd",    f"{MODEL_ID}-llava_bench-vcd.jsonl")
APC_GLOB    = os.path.join(OUTPUTS, "apc", "beta_*", "*.jsonl")
APC01_FILE  = os.path.join(OUTPUTS, "apc", "beta_0.100",
                           f"{MODEL_ID}-llava_bench-apc-beta0.100.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
