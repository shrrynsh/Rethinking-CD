"""
Downloads LLaVA-Bench (In-the-Wild) from HuggingFace.

Dataset: liuhaotian/llava-bench-in-the-wild
  - 60 image-question pairs
  - Categories: conv (30), detail (15), complex (15)

Output layout:
  data/llava_bench/
      questions.jsonl   <- one JSON line per question
      images/           <- 60 images
"""

import os
import json
import shutil
from pathlib import Path

REPO_ID = "liuhaotian/llava-bench-in-the-wild"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "llava_bench"
IMG_DIR = OUT_DIR / "images"
QUESTIONS_OUT = OUT_DIR / "questions.jsonl"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    print(f"Downloading {REPO_ID} ...")
    cache_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(OUT_DIR / "_hf_cache"),
    )
    print(f"Downloaded to: {cache_dir}")

    cache_path = Path(cache_dir)

    # Locate and copy images
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    src_img_dir = cache_path / "images"
    if not src_img_dir.exists():
        # Some versions store directly at root
        src_img_dir = cache_path
    copied = 0
    for img_path in src_img_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            shutil.copy2(img_path, IMG_DIR / img_path.name)
            copied += 1
    print(f"Copied {copied} images to {IMG_DIR}")

    # Locate questions file (may be questions.jsonl or similar)
    q_src = None
    for candidate in ["questions.jsonl", "llava-bench-in-the-wild.jsonl", "question.jsonl"]:
        p = cache_path / candidate
        if p.exists():
            q_src = p
            break
    if q_src is None:
        # Try to build from HuggingFace datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset(REPO_ID, split="train")
            with open(QUESTIONS_OUT, "w") as f:
                for item in ds:
                    f.write(json.dumps(item) + "\n")
            print(f"Wrote {len(ds)} questions to {QUESTIONS_OUT}")
        except Exception as e:
            raise SystemExit(
                f"Could not find questions file in cache and datasets library failed: {e}\n"
                "Please manually place questions.jsonl in data/llava_bench/"
            )
    else:
        shutil.copy2(q_src, QUESTIONS_OUT)
        with open(QUESTIONS_OUT) as f:
            n = sum(1 for _ in f)
        print(f"Copied {n} questions to {QUESTIONS_OUT}")

    # Cleanup cache
    shutil.rmtree(OUT_DIR / "_hf_cache", ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
