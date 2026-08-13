#!/usr/bin/env bash
# APC sweep: 14 beta values
#   beta=0.000  →  no masking    →  equivalent to direct sampling
#   beta=1.000  →  only argmax   →  equivalent to greedy search
# Same 14 values as jaccard_on_llava / jaccard_on_qwen, so the curves overlay.

set -e
cd "$(dirname "$0")/.."
source ./scripts/common.sh

BETAS=(0.000 0.025 0.050 0.075 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)

for beta in "${BETAS[@]}"; do
    ANSWERS_FILE="$OUT_ROOT/apc/beta_${beta}/${MODEL_ID}-llava_bench-apc-beta${beta}.jsonl"
    mkdir -p "$(dirname "$ANSWERS_FILE")"

    echo "Running APC beta=${beta} ..."

    python ./inference/internvl_bench_infer_apc.py \
        --model-path    "$MODEL_PATH"    \
        --question-file "$QUESTION_FILE" \
        --image-folder  "$IMAGE_FOLDER"  \
        --answers-file  "$ANSWERS_FILE"  \
        --temperature   1.0              \
        --top_p         1.0              \
        --beta          "$beta"          \
        --min-patches   "$MIN_PATCHES"   \
        --max-patches   "$MAX_PATCHES"

    echo "Done: $ANSWERS_FILE"
done

echo "All 14 APC runs complete."
