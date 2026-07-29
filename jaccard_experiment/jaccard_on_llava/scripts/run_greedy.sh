#!/usr/bin/env bash
# Greedy search (temperature=0, do_sample=False)

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/llava-v1.5-7b}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/greedy/llava-7b-llava_bench-greedy.jsonl

python ./inference/llava_bench_infer_greedy.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode vicuna_v1

echo "Greedy inference done. Output: $ANSWERS_FILE"
