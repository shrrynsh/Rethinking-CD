#!/usr/bin/env bash
# Reproduce the contrastive-adjustment analysis for Qwen2.5-VL-7B-Instruct.
# NOTE: the VCD/SID capture is slow (cache-less amateur); the proxy is fast.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
[ -f models/Qwen2.5-VL-7B-Instruct/config.json ] || bash download_assets.sh
$PY common/generate_qwen.py --mode capture --method vcd --out outputs/captures/qwen_vcd.jsonl
$PY common/generate_qwen.py --mode capture --method sid --out outputs/captures/qwen_sid.jsonl
for s in 0 1 2; do
  $PY common/generate_qwen.py --mode proxy --stats-source vcd --seed $s \
      --out outputs/captions/qwen_proxy_vcd_seed$s.jsonl
done
$PY common/generate_qwen.py --mode greedy --out outputs/captions/qwen_greedy.jsonl
$PY common/contrastive_analysis.py --model qwen --B 10000
