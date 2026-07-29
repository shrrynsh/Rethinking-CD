#!/usr/bin/env bash
# MME contrastive-decoding inference: VCD / ICD / SID (greedy + sampling).
#
# Each method patches both the greedy-search and sampling code paths, so it
# runs correctly under --temperature 0 (greedy) and --temperature 1 (sampling).
# Exactly one contrastive flag may be set per invocation (mme_infer_cd.py
# rejects conflicting flags).
#
# Override the model with: MODEL_PATH=/path/to/llava bash scripts/mme_infer_cd.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/studios/this_studio/models/llava-v1.5-7b}
CONV_MODE=${CONV_MODE:-vicuna_v1}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_ROOT=${OUT_ROOT:-./outputs/mme}

# method name -> inference flag
declare -A CD_FLAGS=(
    [vcd]=--use-vcd
    [icd]=--use-icd
    [sid]=--use-sid
)

for method in vcd icd sid; do
    flag=${CD_FLAGS[$method]}
    out_dir="${OUT_ROOT}/${method}"
    mkdir -p "${out_dir}"

    ## greedy (temperature 0)
    python ./inference/mme_infer_cd.py \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${out_dir}/llava-7b-mme-greedy.jsonl" \
        --temperature 0 \
        --conv-mode "${CONV_MODE}" \
        "${flag}"

    ## sampling (temperature 1)
    python ./inference/mme_infer_cd.py \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${out_dir}/llava-7b-mme-sample.jsonl" \
        --temperature 1 \
        --conv-mode "${CONV_MODE}" \
        "${flag}"
done
