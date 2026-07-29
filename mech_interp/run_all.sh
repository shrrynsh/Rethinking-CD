#!/usr/bin/env bash
# End-to-end driver for the POPE-COCO VCD mechanistic-interpretability experiment.
# Run from the repo root on a Lightning Studio (L4). Idempotent: re-running skips
# already-downloaded weights/images and overwrites results.
set -euo pipefail

# --- persistent-storage root (survives studio restarts) ---
ROOT="${ROOT:-/teamspace/studios/this_studio/cd_pope_mech_interp}"
MODELS_DIR="${ROOT}/models"
DATA_DIR="${ROOT}/data"
ACT_DIR="${ROOT}/activations"
RESULTS_DIR="${ROOT}/results"
MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/llava-v1.5-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # set e.g. 40 for a fast smoke test

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${MODELS_DIR}" "${DATA_DIR}" "${ACT_DIR}" "${RESULTS_DIR}"

echo "=========================================================="
echo " repo        : ${REPO_ROOT}"
echo " model path  : ${MODEL_PATH}"
echo " data dir    : ${DATA_DIR}"
echo " activations : ${ACT_DIR}"
echo " results     : ${RESULTS_DIR}"
echo " max samples : ${MAX_SAMPLES} (0 = all ~9000)"
echo "=========================================================="
nvidia-smi || true

echo "[1/4] Model weights"
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    bash mech_interp/download_model.sh "${MODELS_DIR}"
else
    echo "  already present: ${MODEL_PATH}"
fi

echo "[2/4] POPE-COCO question files + images"
python mech_interp/download_pope_data.py --data-dir "${DATA_DIR}"

echo "[3/4] Extract activations (clean + noisy, all splits)"
python mech_interp/extract_activations.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${ACT_DIR}" \
    --conv-mode vicuna_v1 \
    --noise-step 900 \
    --shard-size 500 \
    --max-samples "${MAX_SAMPLES}" \
    --sanity-check 8

echo "[4/4] Train per-layer probes + build curves/plot/summary"
python mech_interp/train_probes.py \
    --act-dir "${ACT_DIR}" \
    --results-dir "${RESULTS_DIR}"

echo "DONE. See ${RESULTS_DIR}/summary.txt, per_layer.csv, layerwise_curves.png"
