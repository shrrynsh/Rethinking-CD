#!/usr/bin/env bash
# Download the model weights and the 500 COCO images that are not stored in git.
# Everything lands in the paths the scripts expect. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

echo "== model weights =="
mkdir -p models
[ -f models/llava-v1.5-7b/config.json ] || \
  huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir models/llava-v1.5-7b
[ -f models/Qwen2.5-VL-7B-Instruct/config.json ] || \
  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct

echo "== COCO val2017 annotations =="
mkdir -p data/coco/annotations
if [ ! -f data/coco/annotations/instances_val2017.json ]; then
  tmp=$(mktemp -d)
  wget -q --show-progress -O "$tmp/a.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  python - "$tmp/a.zip" <<'PY'
import sys, zipfile, shutil, os
z = zipfile.ZipFile(sys.argv[1])
for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
    with z.open(name) as s, open(os.path.join("data/coco/annotations", os.path.basename(name)), "wb") as d:
        shutil.copyfileobj(s, d)
PY
  rm -rf "$tmp"
fi

echo "== the 500 COCO val2017 images =="
mkdir -p data/coco/val2017
python - <<'PY'
import json, os, urllib.request
ids = json.load(open("image_ids_500.json")); base = "http://images.cocodataset.org/val2017/"
got = 0
for i in ids:
    dst = f"data/coco/val2017/{i:012d}.jpg"
    if os.path.exists(dst):
        continue
    urllib.request.urlretrieve(base + f"{i:012d}.jpg", dst); got += 1
    if got % 100 == 0:
        print(f"  {got} images...", flush=True)
print(f"images ready ({len(ids)} total)")
PY
echo "assets ready."
