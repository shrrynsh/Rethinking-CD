#!/usr/bin/env bash
# Download InternVL3-8B-hf weights with wget, then push them to the Teamspace
# Drive at /teamspace/uploads.
#
# WHY THE TWO STEPS:
#   /teamspace/uploads is mounted read-only inside a Studio:
#       ...on /teamspace/uploads type fuse.litfs (ro,relatime,...)
#   so wget (and curl, cp, hf download — anything) cannot write there directly.
#   The Drive is written through the Lightning control plane instead:
#       lightning cp -r <local> lit://<owner>/<teamspace>/uploads/<path>
#   So we stage on the Studio disk, upload, then delete the staging copy.
#
# Usage:
#   bash download_internvl3_wget.sh              # download + upload + clean up
#   KEEP_STAGING=1 bash download_internvl3_wget.sh   # keep the local copy
#   SKIP_UPLOAD=1  bash download_internvl3_wget.sh   # download only
#
# wget uses -c (resume), so re-running after an interruption picks up where it
# stopped rather than restarting the ~16 GB.

set -euo pipefail

REPO_ID="${REPO_ID:-OpenGVLab/InternVL3-8B-hf}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

STAGING="${STAGING:-/teamspace/studios/this_studio/models/InternVL3-8B-hf}"

# Drive destination. Resolved from the SDK if not overridden.
LIT_OWNER="${LIT_OWNER:-dsg-mssjw-org}"
LIT_TEAMSPACE="${LIT_TEAMSPACE:-cd-rethinking}"
DRIVE_SUBPATH="${DRIVE_SUBPATH:-models/InternVL3-8B-hf}"
LIT_DEST="lit://${LIT_OWNER}/${LIT_TEAMSPACE}/uploads/${DRIVE_SUBPATH}"

# Everything in the repo except docs/git metadata.
FILES=(
    config.json
    generation_config.json
    model.safetensors.index.json
    model-00001-of-00004.safetensors
    model-00002-of-00004.safetensors
    model-00003-of-00004.safetensors
    model-00004-of-00004.safetensors
    preprocessor_config.json
    processor_config.json
    chat_template.jinja
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    vocab.json
    merges.txt
)

mkdir -p "$STAGING"
echo "Staging dir : $STAGING"
echo "Drive dest  : $LIT_DEST"
echo "Files       : ${#FILES[@]}"
echo

for f in "${FILES[@]}"; do
    echo "── $f"
    wget -c -q --show-progress \
         --tries=5 --timeout=60 --waitretry=10 \
         -O "$STAGING/$f" \
         "$BASE_URL/$f"
done

echo
echo "Download complete:"
du -sh "$STAGING"
ls -la "$STAGING"

# Verify the shard set matches the index before uploading 16 GB.
python3 - "$STAGING" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
want = sorted(set(json.load(open(idx))["weight_map"].values()))
missing = [s for s in want if not os.path.exists(os.path.join(d, s))]
if missing:
    sys.exit("ERROR: shards missing from staging dir: " + ", ".join(missing))
print(f"OK: all {len(want)} shards present and match the index.")
PY

# Drive upload is OFF by default: every write path into /teamspace/uploads is
# rejected from a Studio (read-only mount; `lightning cp` -> 400; SDK
# Teamspace.upload_file -> 404). Set TRY_UPLOAD=1 to attempt it anyway, e.g.
# from a machine where the Drive APIs do work.
if [ "${TRY_UPLOAD:-0}" != "1" ]; then
    echo
    echo "Weights ready at: $STAGING"
    echo "export MODEL_PATH=$STAGING"
    echo
    echo "(Drive upload skipped — /teamspace/uploads is not writable from a"
    echo " Studio. Set TRY_UPLOAD=1 to attempt it anyway.)"
    exit 0
fi

echo
echo "Attempting Drive upload — expect a 400 from inside a Studio …"
lightning cp -r "$STAGING" "$LIT_DEST"

echo
echo "Upload done. Verify it landed:"
echo "  ls /teamspace/uploads/${DRIVE_SUBPATH}"

if [ "${KEEP_STAGING:-0}" = "1" ]; then
    echo "KEEP_STAGING=1 — local copy left at $STAGING"
else
    echo "Removing staging copy …"
    rm -rf "$STAGING"
fi

echo
echo "Done. Run:"
echo "  export MODEL_PATH=/teamspace/uploads/${DRIVE_SUBPATH}"
