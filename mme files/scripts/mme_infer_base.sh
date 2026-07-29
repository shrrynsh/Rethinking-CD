#!/usr/bin/env bash
# MME base inference (greedy + sampling).
#
# Runs LLaVA generation over the combined MME question file and writes an
# answer file in the shared JSONL schema consumed by ./eval/mme_eval.py.
#
# Override the model with: MODEL_PATH=/path/to/llava bash scripts/mme_infer_base.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/studios/this_studio/models/llava-v1.5-7b}
CONV_MODE=${CONV_MODE:-vicuna_v1}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_DIR=${OUT_DIR:-./outputs/mme/baseline}

mkdir -p "${OUT_DIR}"

## greedy (temperature 0)
python ./inference/mme_infer_base.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/llava-7b-mme-greedy.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

## sampling (temperature 1)
python ./inference/mme_infer_base.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/llava-7b-mme-sample.jsonl" \
    --temperature 1 \
    --conv-mode "${CONV_MODE}"
