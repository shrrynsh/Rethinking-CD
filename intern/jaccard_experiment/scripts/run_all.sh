#!/usr/bin/env bash
# Run all 17 inference conditions in sequence:
#   1  × greedy
#   1  × direct sampling
#  14  × APC (beta sweep: 0.000 → 1.000)
#   1  × VCD
#
# InternVL3-8B is ~8B params (bf16 ≈ 16 GB) and its dynamic tiling pushes up to
# ~3.3k visual tokens per image, so a 24 GB card is tight but workable. If you
# hit OOM, lower MAX_PATCHES (e.g. MAX_PATCHES=6) before reaching for anything
# else — it is the single biggest lever on memory here.
#
# MEASURED on L40S (46GB), MAX_PATCHES=12, 3-question smoke test:
#   greedy:  3.14 s/question   ->  ~3 min per condition
#   VCD:     4.91 s/question   ->  ~5 min          (only 1.6x greedy, thanks to
#                                                   the cached amateur branch)
#
#   16 standard conditions ≈ 50-80 min
#   VCD                    ≈ 5-8 min
#   Total                  ≈ 1-1.5 hours
#
# The smoke sample skews short (image 001's answers run 20-120 words); longer
# detail/complex answers elsewhere in the set will push the real figure toward
# the upper end. An earlier L4 estimate put this at ~7 h — the L40S has 2.9x
# the memory bandwidth (864 vs 300 GB/s), and decoding is bandwidth-bound.
#
# VCD keeps its own KV cache on the amateur branch (see
# inference/vcd_logits_processor.py); equivalence to the stateless re-forward
# is verified by inference/test_vcd_cache_equivalence.py. Without it, VCD alone
# would dominate the whole experiment.
#
# Launch via scripts/launch_tmux.sh so a dropped connection cannot kill the run.

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
echo "Outputs: intern/jaccard_experiment/outputs/llava_bench/"
