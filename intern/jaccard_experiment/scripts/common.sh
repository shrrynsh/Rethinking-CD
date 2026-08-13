#!/usr/bin/env bash
# Shared config for the InternVL3-8B Jaccard runs. Sourced by every run_*.sh.
#
# MODEL_PATH points at the weights. The Teamspace Drive (/teamspace/uploads)
# was the intended home, but it is read-only from a Studio AND both write APIs
# (`lightning cp`, SDK Teamspace.upload_file) are rejected server-side here, so
# the weights live on the Studio disk instead:
#     export MODEL_PATH=/teamspace/studios/this_studio/models/InternVL3-8B-hf

MODEL_PATH="${MODEL_PATH:-/teamspace/studios/this_studio/models/InternVL3-8B-hf}"

# LLaVA-Bench (In-the-Wild): 24 images, 60 questions.
# Either populate ./data via scripts/download_llava_bench.py, or symlink the
# copy already in the repo:
#     ln -s ../../../jaccard_experiment/jaccard_on_qwen/data/llava_bench data/llava_bench
DATA_DIR="${DATA_DIR:-./data/llava_bench}"
QUESTION_FILE="${QUESTION_FILE:-$DATA_DIR/questions.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$DATA_DIR/images}"

OUT_ROOT="${OUT_ROOT:-./outputs/llava_bench}"
MODEL_ID=internvl3-8b

# InternVL dynamic tiling: 448x448 tiles, 256 visual tokens each.
# 1/12 is the OpenGVLab single-image default.
MIN_PATCHES="${MIN_PATCHES:-1}"
MAX_PATCHES="${MAX_PATCHES:-12}"

if [ ! -f "$QUESTION_FILE" ]; then
    echo "ERROR: questions file not found: $QUESTION_FILE" >&2
    echo "Run scripts/download_llava_bench.py or set DATA_DIR." >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "See scripts/download_internvl3.py for how to place the weights on the Drive." >&2
    exit 1
fi
