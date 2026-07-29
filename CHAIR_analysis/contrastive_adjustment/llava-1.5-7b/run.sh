#!/usr/bin/env bash
# Reproduce the contrastive-adjustment analysis for LLaVA-1.5-7B:
# d = E - A, the normalized object-mass buckets, and the noise proxy vs greedy.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
[ -f models/llava-v1.5-7b/config.json ] || bash download_assets.sh
# per-step capture (shared with the agreement analysis; skipped if already done)
$PY common/generate_llava.py --mode capture --method vcd --out outputs/captures/llava_vcd.jsonl
$PY common/generate_llava.py --mode capture --method sid --out outputs/captures/llava_sid.jsonl
# full-vocabulary Gaussian noise proxy (3 seeds) + greedy baseline
for s in 0 1 2; do
  $PY common/generate_llava.py --mode proxy --stats-source vcd --seed $s \
      --out outputs/captions/llava_proxy_vcd_seed$s.jsonl
done
$PY common/generate_llava.py --mode greedy --out outputs/captions/llava_greedy.jsonl
# analysis: d-stats + buckets (with CIs) + CHAIR of proxy/greedy
$PY common/contrastive_analysis.py --model llava --B 10000
