#!/usr/bin/env bash
# VCD: temperature=1.0, noise_step=900, cd_alpha=1.0, cd_beta=0.1
# Note: VCD runs a full contrastive forward pass at every decode step (no KV
# cache reuse on the noisy branch), so it is ~2x slower than greedy/sample.

set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/vcd/qwen2.5-7b-llava_bench-vcd.jsonl

python ./inference/qwen_bench_infer_vcd.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --temperature   1.0              \
    --top_p         1.0              \
    --noise-step    900              \
    --cd-alpha      1.0              \
    --cd-beta       0.1              \
    --min-pixels    $((256 * 28 * 28))  \
    --max-pixels    $((1280 * 28 * 28))

echo "VCD done. Output: $ANSWERS_FILE"
