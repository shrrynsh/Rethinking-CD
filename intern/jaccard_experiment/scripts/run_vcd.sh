#!/usr/bin/env bash
# VCD: temperature=1.0, noise_step=900, cd_alpha=1.0, cd_beta=0.1
# Note: VCD runs a full contrastive forward pass at every decode step (no KV
# cache reuse on the noisy branch). On InternVL that means re-encoding up to 13
# vision tiles per step, so this is by far the slowest condition.

set -e
cd "$(dirname "$0")/.."
source ./scripts/common.sh

ANSWERS_FILE="$OUT_ROOT/vcd/${MODEL_ID}-llava_bench-vcd.jsonl"

python ./inference/internvl_bench_infer_vcd.py \
    --model-path    "$MODEL_PATH"    \
    --question-file "$QUESTION_FILE" \
    --image-folder  "$IMAGE_FOLDER"  \
    --answers-file  "$ANSWERS_FILE"  \
    --temperature   1.0              \
    --top_p         1.0              \
    --noise-step    900              \
    --cd-alpha      1.0              \
    --cd-beta       0.1              \
    --min-patches   "$MIN_PATCHES"   \
    --max-patches   "$MAX_PATCHES"

echo "VCD done. Output: $ANSWERS_FILE"
