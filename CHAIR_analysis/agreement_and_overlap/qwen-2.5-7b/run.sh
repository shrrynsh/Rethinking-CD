#!/usr/bin/env bash
# Reproduce the agreement-and-overlap analysis for Qwen2.5-VL-7B-Instruct.
# NOTE: the VCD/SID capture recomputes the amateur branch cache-less and is slow.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
[ -f models/Qwen2.5-VL-7B-Instruct/config.json ] || bash download_assets.sh
$PY common/generate_qwen.py --mode capture --method vcd --out outputs/captures/qwen_vcd.jsonl
$PY common/generate_qwen.py --mode capture --method sid --out outputs/captures/qwen_sid.jsonl
$PY common/agreement_analysis.py --model qwen --B 10000
