#!/usr/bin/env bash
# Reproduce the agreement-and-overlap analysis for LLaVA-1.5-7B.
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> CHAIR_analysis/
PY=${PY:-python}
[ -f models/llava-v1.5-7b/config.json ] || bash download_assets.sh
# per-step top-30 capture (shared with the contrastive analysis; resumable)
$PY common/generate_llava.py --mode capture --method vcd --out outputs/captures/llava_vcd.jsonl
$PY common/generate_llava.py --mode capture --method sid --out outputs/captures/llava_sid.jsonl
# analysis: intervention rate + top-10 overlap, with 95% bootstrap CIs
$PY common/agreement_analysis.py --model llava --B 10000
