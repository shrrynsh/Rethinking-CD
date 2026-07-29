#!/usr/bin/env bash
# MME spurious-mitigation inference for Qwen2.5-VL-7B-Instruct: PBA / OLM / APC.
# Qwen counterpart of scripts/mme_infer_spurious.sh.
#
#   - PBA: stock generation, greedy (temperature 0), with the PBA prompt suffix.
#   - OLM: greedy yes/no logits-processor; run greedy (temperature 0).
#   - APC: sampling plausibility-cutoff logits-processor; run sampling (temperature 1).
#
# Override the model with: MODEL_PATH=/path/to/qwen bash scripts/mme_infer_spurious_qwen.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/lightning_storage/model/Qwen2.5_7b}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_ROOT=${OUT_ROOT:-./outputs/mme}

## pba (greedy)
mkdir -p "${OUT_ROOT}/pba"
python ./inference/mme_infer_pba_qwen.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/pba/qwen25-7b-mme-greedy.jsonl" \
    --temperature 0

## olm (greedy yes/no logits-processor)
mkdir -p "${OUT_ROOT}/olm"
python ./inference/mme_infer_olm_qwen.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/olm/qwen25-7b-mme-greedy.jsonl" \
    --temperature 0 \
    --use-olm

## apc (sampling plausibility-cutoff logits-processor)
mkdir -p "${OUT_ROOT}/apc"
python ./inference/mme_infer_apc_qwen.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/apc/qwen25-7b-mme-sample.jsonl" \
    --temperature 1 \
    --use-apc
