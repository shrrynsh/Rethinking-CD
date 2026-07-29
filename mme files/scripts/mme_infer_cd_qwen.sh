#!/usr/bin/env bash
# MME contrastive-decoding inference for Qwen2.5-VL-7B-Instruct: VCD / ICD / SID
# (greedy + sampling). Qwen counterpart of scripts/mme_infer_cd.sh.
#
# Each method runs a self-contained two-stream contrastive decode, so it works
# under --temperature 0 (greedy) and --temperature 1 (sampling). Exactly one
# contrastive flag is set per invocation (mme_infer_cd_qwen.py rejects conflicts).
#
# Override the model with: MODEL_PATH=/path/to/qwen bash scripts/mme_infer_cd_qwen.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/lightning_storage/model/Qwen2.5_7b}
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
    python ./inference/mme_infer_cd_qwen.py \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${out_dir}/qwen25-7b-mme-greedy.jsonl" \
        --temperature 0 \
        "${flag}"

    ## sampling (temperature 1)
    python ./inference/mme_infer_cd_qwen.py \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${out_dir}/qwen25-7b-mme-sample.jsonl" \
        --temperature 1 \
        "${flag}"
done
