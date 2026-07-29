#!/usr/bin/env bash
# Build both label-imbalance regimes for LLaVA-1.5-7B and score them.
#
# Prerequisite: a finished POPE run whose answer files are laid out as
#   $PRED_DIR/<method>/llava-7b-<dataset>-<split>-greedy.jsonl
# (this is exactly what the companion POPE-table reproduction writes to its
#  outputs/pope/ folder). Point PRED_DIR at that folder.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PRED_DIR:?set PRED_DIR to the finished POPE answers root (…/outputs/pope)}"
TAG=llava-7b

echo "### building 25/75 and 75/25 regimes ###"
python build_regimes.py --model-tag "$TAG" --anno-dir ./data --pred-dir "$PRED_DIR" --out-dir ./regimes

echo "### standard vs balanced accuracy ###"
python analysis/balanced_accuracy.py --model-tag "$TAG" --regime-dir ./regimes

echo "### 95% bootstrap CI on accuracy ###"
python analysis/bootstrap_ci.py --model-tag "$TAG" --regime-dir ./regimes

echo "### McNemar vs baseline ###"
python analysis/mcnemar.py --model-tag "$TAG" --regime-dir ./regimes
