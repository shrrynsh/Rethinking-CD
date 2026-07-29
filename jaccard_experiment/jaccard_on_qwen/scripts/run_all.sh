#!/usr/bin/env bash
# Run all 17 inference conditions in sequence:
#   1  × greedy
#   1  × direct sampling
#  14  × APC (beta sweep: 0.000 → 1.000)
#   1  × VCD
#
# Estimated time on L4 (24GB VRAM), max_pixels=1280*28*28:
#   greedy / sample / each APC:  ~5–10 min each   →  greedy+sample+14xAPC ≈ 90–150 min
#   VCD:                          ~45–90 min        (2× forward passes per step)
#   Total:                        ~2.5–4 hours

set -e
SCRIPT_DIR="$(dirname "$0")"

echo "=== [1/4] Greedy ==="
bash "$SCRIPT_DIR/run_greedy.sh"

echo "=== [2/4] Direct Sampling ==="
bash "$SCRIPT_DIR/run_sample.sh"

echo "=== [3/4] APC (14 beta values) ==="
bash "$SCRIPT_DIR/run_apc.sh"

echo "=== [4/4] VCD ==="
bash "$SCRIPT_DIR/run_vcd.sh"

echo ""
echo "All 17 conditions complete."
echo "Outputs: jaccard_on_qwen/outputs/llava_bench/"
