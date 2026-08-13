#!/usr/bin/env bash
# Direct sampling (do_sample=True, temperature=1.0, top_p=1.0)

set -e
cd "$(dirname "$0")/.."
source ./scripts/common.sh

ANSWERS_FILE="$OUT_ROOT/sample/${MODEL_ID}-llava_bench-sample.jsonl"

python ./inference/internvl_bench_infer_sample.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --temperature   1.0              \
    --top_p         1.0              \
    --min-patches   "$MIN_PATCHES"   \
    --max-patches   "$MAX_PATCHES"

echo "Sample done. Output: $ANSWERS_FILE"
