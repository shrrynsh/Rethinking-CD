#!/usr/bin/env bash
# APC sweep: 14 beta values
#   beta=0.000  →  no masking    →  equivalent to direct sampling
#   beta=1.000  →  only argmax   →  equivalent to greedy search

set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images
MIN_PIXELS=$((256 * 28 * 28))
MAX_PIXELS=$((1280 * 28 * 28))

BETAS=(0.000 0.025 0.050 0.075 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)

for beta in "${BETAS[@]}"; do
    ANSWERS_FILE="./outputs/llava_bench/apc/beta_${beta}/qwen2.5-7b-llava_bench-apc-beta${beta}.jsonl"
    mkdir -p "$(dirname "$ANSWERS_FILE")"

    echo "Running APC beta=${beta} ..."

    python ./inference/qwen_bench_infer_apc.py \
        --model-path    "$MODEL_PATH"    \
        --question-file "$QUESTION_FILE" \
        --image-folder  "$IMAGE_FOLDER"  \
        --answers-file  "$ANSWERS_FILE"  \
        --temperature   1.0              \
        --top_p         1.0              \
        --beta          "$beta"          \
        --min-pixels    "$MIN_PIXELS"    \
        --max-pixels    "$MAX_PIXELS"

    echo "Done: $ANSWERS_FILE"
done

echo "All 14 APC runs complete."
