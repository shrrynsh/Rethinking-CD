#!/usr/bin/env bash
# MME base inference for Qwen2.5-VL-7B-Instruct (greedy + sampling).
#
# Qwen counterpart of scripts/mme_infer_base.sh. Writes answer files with the
# qwen25-7b-mme- filename prefix (vs the LLaVA llava-7b-mme- prefix) so the two
# result sets coexist in the same per-method directory and eval/mme_eval.py
# consumes both unchanged.
#
# Override the model with: MODEL_PATH=/path/to/qwen bash scripts/mme_infer_base_qwen.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/teamspace/lightning_storage/model/Qwen2.5_7b}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_DIR=${OUT_DIR:-./outputs/mme/baseline}

mkdir -p "${OUT_DIR}"

## greedy (temperature 0)
python ./inference/mme_infer_base_qwen.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/qwen25-7b-mme-greedy.jsonl" \
    --temperature 0

## sampling (temperature 1)
python ./inference/mme_infer_base_qwen.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/qwen25-7b-mme-sample.jsonl" \
    --temperature 1
