"""
Download InternVL3-8B weights to the Teamspace Drive (NOT into the studio).

Model: OpenGVLab/InternVL3-8B-hf  (~16 GB, bf16 safetensors)

  The `-hf` suffix matters. It is the native-transformers port
  (`InternVLForConditionalGeneration`). The plain `OpenGVLab/InternVL3-8B`
  repo needs trust_remote_code and exposes only a `.chat()` helper that cannot
  take a logits_processor, which the VCD condition requires. See
  inference/internvl_utils.py.

IMPORTANT — the Teamspace Drive is NOT usable as a weights location here.

  1. /teamspace/uploads is mounted read-only inside a Studio:
         ...on /teamspace/uploads type fuse.litfs (ro,relatime,...)
     so wget/curl/cp/huggingface_hub all fail with `Read-only file system`.

  2. Both documented write-through APIs are also rejected server-side for this
     teamspace (tested 2026-08-13):
         lightning cp -r <dir> lit://<owner>/<ts>/uploads/<path>
             -> 400 "this endpoint only accepts logs/ and metrics/ paths"
         lightning_sdk Teamspace.upload_file(...)
             -> 404 "Failed to upload file. Status code: 404"

  So the weights live on the Studio disk (332 GB free):
         export MODEL_PATH=/teamspace/studios/this_studio/models/InternVL3-8B-hf

  If you want them on the Drive, upload via the Drive tab in the Lightning web
  UI, which uses a different (browser) upload path than either API above.

Usage:
    python download_internvl3.py                    # default dest, refuses if read-only
    python download_internvl3.py --dest /some/path  # explicit destination
    python download_internvl3.py --check            # just report where weights are
"""

import argparse
import os
import sys

REPO_ID = "OpenGVLab/InternVL3-8B-hf"
DEFAULT_DEST = os.environ.get(
    "MODEL_PATH", "/teamspace/studios/this_studio/models/InternVL3-8B-hf"
)

# Weights only — skip the duplicate .bin/.pth copies HF repos often carry.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
]


def is_writable(path):
    """Walk up to the nearest existing ancestor and test it for write access."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def looks_complete(path):
    if not os.path.isdir(path):
        return False
    names = os.listdir(path)
    return ("config.json" in names
            and any(n.endswith(".safetensors") for n in names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Where to write the weights (default: {DEFAULT_DEST})")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--check", action="store_true",
                        help="Report whether weights are already present, then exit.")
    args = parser.parse_args()

    if looks_complete(args.dest):
        print(f"Weights already present at {args.dest}")
        print(f"Run:  export MODEL_PATH={args.dest}")
        return
    if args.check:
        print(f"No weights at {args.dest}")
        sys.exit(1)

    if not is_writable(args.dest):
        print(f"ERROR: {args.dest} is not writable from this environment.", file=sys.stderr)
        if args.dest.startswith("/teamspace/uploads"):
            print(
                "\n/teamspace/uploads is a read-only mount inside a Studio. Upload the\n"
                "weights to the Drive instead (see the module docstring for the two\n"
                "options), or pass --dest to download somewhere writable.",
                file=sys.stderr,
            )
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    print(f"Downloading {args.repo_id} → {args.dest}  (~16 GB)")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.dest,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=8,
    )
    print(f"\nDone. Run:  export MODEL_PATH={args.dest}")


if __name__ == "__main__":
    main()
