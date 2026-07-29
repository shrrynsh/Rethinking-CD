#!/usr/bin/env bash
# Download LLaVA-1.5-7B weights (liuhaotian/llava-v1.5-7b, the ORIGINAL LLaVA format
# that THIS repo's llava.model.builder.load_pretrained_model expects -- NOT the
# llava-hf transformers-native checkpoint).
#
# Usage:
#   bash mech_interp/download_model.sh [MODELS_DIR]
#
# Weights land in $MODELS_DIR/llava-v1.5-7b and persist across Lightning restarts
# as long as MODELS_DIR is under /teamspace/studios/this_studio/.
set -euo pipefail

MODELS_DIR="${1:-/teamspace/studios/this_studio/cd_pope_mech_interp/models}"
REPO_ID="liuhaotian/llava-v1.5-7b"
TARGET="${MODELS_DIR}/llava-v1.5-7b"

mkdir -p "${MODELS_DIR}"

# Point HF caches at persistent storage too, so nothing lands on the ephemeral root disk.
export HF_HOME="${MODELS_DIR}/hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

echo "[download_model] Installing huggingface_hub[cli] + hf_transfer (fast download) ..."
pip install -q -U "huggingface_hub[cli]" hf_transfer

echo "[download_model] Downloading ${REPO_ID} -> ${TARGET}"
# --local-dir gives a plain directory of files that from_pretrained(model_path=...) reads directly.
huggingface-cli download "${REPO_ID}" \
    --local-dir "${TARGET}" \
    --local-dir-use-symlinks False

echo "[download_model] Done. Sanity listing:"
ls -lh "${TARGET}"
echo
echo "[download_model] Pass --model-path ${TARGET} to extract_activations.py"
