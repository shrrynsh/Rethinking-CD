#!/usr/bin/env bash
# MME data preparation: convert the raw MME_Benchmark release tree into the
# evaluator-ready reference + questions JSONL files and a reconciled images/
# tree under ./data/mme.
#
# Run this once before the inference scripts (mme_infer_base.sh /
# mme_infer_cd.sh / mme_infer_spurious.sh) and the results table (mme_eval.sh).
#
# Auto-detects FLAT vs SPLIT subtask layouts and writes mme_reference.jsonl,
# mme_questions.jsonl, per-subtask reference/{subtask}.jsonl files, and the
# images/{subtask}/ tree.
set -euo pipefail

RAW_ROOT=${RAW_ROOT:-./data/mme/MME_Benchmark_release_version/MME_Benchmark}
OUT_ROOT=${OUT_ROOT:-./data/mme}

python ./eval/mme_prepare.py \
    --raw-root "${RAW_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --per-subtask
