#!/usr/bin/env bash
# VCD (Visual Contrastive Decoding) with temperature=1.0 sampling

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/llava-v1.5-7b}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/vcd/llava-7b-llava_bench-vcd.jsonl

python ./inference/llava_bench_infer_vcd.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode vicuna_v1 \
    --temperature 1.0 \
    --top_p 1.0 \
    --noise-step 900 \
    --cd-alpha 1.0 \
    --cd-beta 0.1

echo "VCD inference done. Output: $ANSWERS_FILE"
