#!/usr/bin/env bash
# Greedy search (do_sample=False, num_beams=1)
# Image resolution: InternVL dynamic tiling, up to 12 tiles + thumbnail

set -e
cd "$(dirname "$0")/.."
source ./scripts/common.sh

ANSWERS_FILE="$OUT_ROOT/greedy/${MODEL_ID}-llava_bench-greedy.jsonl"

python ./inference/internvl_bench_infer_greedy.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --min-patches   "$MIN_PATCHES"   \
    --max-patches   "$MAX_PATCHES"

echo "Greedy done. Output: $ANSWERS_FILE"
