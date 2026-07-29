#!/usr/bin/env bash
# Direct sampling + APC for all beta values
# beta=0   → equivalent to direct sampling (no masking)
# beta=1   → equivalent to greedy search (only argmax survives)

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/llava-v1.5-7b}"
QUESTION_FILE=./data/llava_bench/questions.jsonl
IMAGE_FOLDER=./data/llava_bench/images

BETAS=(0.000 0.025 0.050 0.075 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)

for beta in "${BETAS[@]}"; do
    ANSWERS_FILE="./outputs/llava_bench/apc/beta_${beta}/llava-7b-llava_bench-apc-beta${beta}.jsonl"

    echo "Running APC with beta=${beta} ..."

    python ./inference/llava_bench_infer_apc.py \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWERS_FILE" \
        --conv-mode vicuna_v1 \
        --temperature 1.0 \
        --top_p 1.0 \
        --beta "$beta" \
        --cd-alpha 1.0

    echo "Done: $ANSWERS_FILE"
done

echo "All APC runs complete."
