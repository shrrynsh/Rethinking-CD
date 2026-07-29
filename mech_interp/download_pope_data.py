#!/usr/bin/env python3
"""Download POPE-COCO question files and ONLY the referenced COCO val2014 images.

POPE's json files already carry the ground-truth ``label`` per question, so we do
NOT need instances_val2014.json or the full val2014 archive. We collect the unique
set of image filenames referenced across the three COCO splits (random / popular /
adversarial) and pull just those from the COCO image server.

Layout produced (matches the repo's own expectations, images flattened one level):

    <data_dir>/pope/coco/coco_pope_random.json
    <data_dir>/pope/coco/coco_pope_popular.json
    <data_dir>/pope/coco/coco_pope_adversarial.json
    <data_dir>/pope/coco/images/COCO_val2014_000000XXXXXX.jpg

Usage:
    python mech_interp/download_pope_data.py \
        --data-dir /teamspace/studios/this_studio/cd_pope_mech_interp/data
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

SPLITS = ("random", "popular", "adversarial")

# POPE question files. Try a few URL shapes in case the repo layout/branch moved.
POPE_URL_TEMPLATES = (
    "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{split}.json",
    "https://raw.githubusercontent.com/RUCAIBox/POPE/master/output/coco/coco_pope_{split}.json",
)

COCO_IMG_URL = "http://images.cocodataset.org/val2014/{filename}"


def fetch_split_file(split: str, dst_path: str) -> None:
    """Download one POPE split json, trying each candidate URL until one works."""
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"[pope] {split}: already present -> {dst_path}")
        return
    last_err = None
    for tmpl in POPE_URL_TEMPLATES:
        url = tmpl.format(split=split)
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.text.strip():
                with open(dst_path, "w") as f:
                    f.write(resp.text)
                print(f"[pope] {split}: downloaded from {url}")
                return
            last_err = f"HTTP {resp.status_code} at {url}"
        except requests.RequestException as e:  # noqa: PERF203
            last_err = f"{type(e).__name__} at {url}: {e}"
    raise RuntimeError(f"[pope] Could not fetch split '{split}'. Last error: {last_err}")


def collect_image_names(pope_dir: str) -> set:
    """Union of the ``image`` field across all three downloaded splits."""
    names = set()
    for split in SPLITS:
        path = os.path.join(pope_dir, f"coco_pope_{split}.json")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                names.add(json.loads(line)["image"])
    return names


def download_one_image(filename: str, images_dir: str, retries: int = 3) -> tuple:
    """Return (filename, ok, msg). Skips files already on disk."""
    dst = os.path.join(images_dir, filename)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return filename, True, "cached"
    url = COCO_IMG_URL.format(filename=filename)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.content:
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    f.write(resp.content)
                os.replace(tmp, dst)
                return filename, True, "ok"
            time.sleep(1.0 + attempt)
        except requests.RequestException:
            time.sleep(1.0 + attempt)
    return filename, False, "failed after retries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="/teamspace/studios/this_studio/cd_pope_mech_interp/data",
        help="Root data dir; POPE lands under <data-dir>/pope/coco/",
    )
    ap.add_argument("--workers", type=int, default=16, help="Parallel image downloads.")
    args = ap.parse_args()

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 1) question files
    for split in SPLITS:
        fetch_split_file(split, os.path.join(pope_dir, f"coco_pope_{split}.json"))

    # 2) unique image set
    names = collect_image_names(pope_dir)
    print(f"[pope] Unique COCO images referenced across all 3 splits: {len(names)}")

    # 3) download only those
    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one_image, n, images_dir): n for n in names}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="images"):
            _, success, _ = fut.result()
            if success:
                ok += 1
            else:
                fail.append(futs[fut])

    print(f"[pope] Images ready: {ok}/{len(names)}  failed: {len(fail)}")
    if fail:
        print("[pope] First few failures:", fail[:10])
        print("[pope] Re-run this script to retry only the missing ones (cached files are skipped).")
    print(f"[pope] Question files: {pope_dir}")
    print(f"[pope] Images dir:     {images_dir}")


if __name__ == "__main__":
    main()
