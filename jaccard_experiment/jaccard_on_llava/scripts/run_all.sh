#!/usr/bin/env bash
# Run all 17 inference conditions for the Jaccard experiment
# Order: greedy → direct sampling → VCD → APC (14 beta values: 0.000–1.000)

SCRIPT_DIR="$(dirname "$0")"

echo "===== [1/17] Greedy Search ====="
bash "$SCRIPT_DIR/run_greedy.sh"

echo ""
echo "===== [2/17] Direct Sampling (temp=1.0) ====="
bash "$SCRIPT_DIR/run_sample.sh"

echo ""
echo "===== [3/17] VCD (temp=1.0, noise_step=900) ====="
bash "$SCRIPT_DIR/run_vcd.sh"

echo ""
echo "===== [4-17/17] APC (temp=1.0, beta sweep: 0.000 to 1.000) ====="
bash "$SCRIPT_DIR/run_apc.sh"

echo ""
echo "All 17 inference runs complete."
