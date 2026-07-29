#!/usr/bin/env bash
# Direct sampling (temperature=1.0, do_sample=True, no APC)

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/llava-v1.5-7b}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images
ANSWERS_FILE=./outputs/llava_bench/sample/llava-7b-llava_bench-sample.jsonl

python ./inference/llava_bench_infer_sample.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode vicuna_v1 \
    --temperature 1.0 \
    --top_p 1.0

echo "Direct sampling inference done. Output: $ANSWERS_FILE"
