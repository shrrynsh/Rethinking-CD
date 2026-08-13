#!/usr/bin/env bash
# Entrypoint for running the InternVL3-8B Jaccard sweep as a Lightning Job.
#
# Everything is absolute-pathed so it does not depend on the job's HOME or cwd.
#
# Point a job at it with:
#   bash /teamspace/studios/this_studio/Rethinking-CD/intern/jaccard_experiment/scripts/job_entrypoint.sh
#
# Safe to re-run: every condition resumes from whatever is already on disk, so a
# preempted or restarted job picks up where it stopped rather than starting over.

set -euo pipefail

EXP=/teamspace/studios/this_studio/Rethinking-CD/intern/jaccard_experiment
MODEL=/teamspace/studios/this_studio/models/InternVL3-8B-hf
QWEN_DATA=/teamspace/studios/this_studio/Rethinking-CD/jaccard_experiment/jaccard_on_qwen/data/llava_bench

cd "$EXP"
echo "=== workdir: $PWD"

# Outputs go to the artifacts dir so they survive the job ending; a job writes
# to its own snapshot copy of the filesystem, which is discarded otherwise.
export OUT_ROOT="${LIGHTNING_ARTIFACTS_DIR:-$EXP}/outputs/llava_bench"
export MODEL_PATH="$MODEL"
mkdir -p "$OUT_ROOT"
echo "=== OUT_ROOT: $OUT_ROOT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: no GPU visible in this job." >&2; exit 1; }

# The snapshot normally carries the conda env, but reinstall cheaply if not.
python3 -c "import transformers, accelerate, shortuuid" 2>/dev/null || {
    echo "=== installing deps ==="
    pip install -q "transformers>=4.52" accelerate shortuuid
}

# data/llava_bench is a symlink into the Qwen experiment; recreate if the
# snapshot dropped it.
if [ ! -f "$EXP/data/llava_bench/questions.jsonl" ]; then
    echo "=== relinking data ==="
    mkdir -p "$EXP/data"
    ln -sfn "$QWEN_DATA" "$EXP/data/llava_bench"
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo "=== weights absent; downloading (~16 GB) ==="
    STAGING="$MODEL" bash "$EXP/scripts/download_internvl3_wget.sh"
fi

echo "=== starting sweep ==="
bash "$EXP/scripts/run_all.sh"

echo "=== sweep complete ==="
find "$OUT_ROOT" -name '*.jsonl' | sort | while read -r f; do
    printf '%s  %s lines\n' "$f" "$(wc -l < "$f")"
done
