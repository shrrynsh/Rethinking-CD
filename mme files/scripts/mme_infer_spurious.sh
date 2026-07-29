#!/usr/bin/env bash
# MME spurious-mitigation inference: PBA / OLM / APC.
#
# Decoding follows each method's patched code path (mirroring the POPE scripts):
#   - PBA: no patches; greedy (temperature 0), with the PBA prompt suffix.
#   - OLM: greedy-only patch; run greedy (temperature 0).
#   - APC: sampling-only patch; run sampling (temperature 1).
#
# Override the model with: MODEL_PATH=/path/to/llava bash scripts/mme_infer_spurious.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/studios/this_studio/models/llava-v1.5-7b}
CONV_MODE=${CONV_MODE:-vicuna_v1}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_ROOT=${OUT_ROOT:-./outputs/mme}

## pba (greedy)
mkdir -p "${OUT_ROOT}/pba"
python ./inference/mme_infer_pba.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/pba/llava-7b-mme-greedy.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

## olm (greedy-only patch)
mkdir -p "${OUT_ROOT}/olm"
python ./inference/mme_infer_olm.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/olm/llava-7b-mme-greedy.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}" \
    --use-olm

## apc (sampling-only patch)
mkdir -p "${OUT_ROOT}/apc"
python ./inference/mme_infer_apc.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_ROOT}/apc/llava-7b-mme-sample.jsonl" \
    --temperature 1 \
    --conv-mode "${CONV_MODE}" \
    --use-apc
