# InternVL3-8B extension

Ports the six experiments in this repo from LLaVA-1.5-7B / Qwen2.5-VL-7B to a
third model, **InternVL3-8B**, to test whether the paper's conclusions about
contrastive decoding hold on an architecture neither original study covered.

| Subfolder | Mirrors | Status |
|---|---|---|
| `jaccard_experiment/` | `../jaccard_experiment/` | code written |
| `pope_repro/` | `../pope repro files/` | not started |
| `mme/` | `../mme files/` | not started |
| `chair_analysis/` | `../CHAIR_analysis/` | not started |
| `label_imbalance/` | `../label_imbalance_experiment/` | not started |
| `mech_interp/` | `../mech_interp/` | not started |

## Shared conventions

**Checkpoint.** `OpenGVLab/InternVL3-8B-hf` — the native-transformers port, not
the `trust_remote_code` original. Every experiment here needs a plain
`forward()` and a `generate()` that accepts `logits_processor`; the original
release exposes only a `.chat()` helper that provides neither. Requires
`transformers>=4.52`.

**Weights live on the Studio disk**, at
`/teamspace/studios/this_studio/models/InternVL3-8B-hf` (15.9 GB, 332 GB free).

The Teamspace Drive was the intended home but is not usable from a Studio.
Three independent write paths were tried on 2026-08-13 and all failed:

| Method | Result |
|---|---|
| `wget` / `curl` / `cp` / `huggingface_hub` → `/teamspace/uploads` | `Read-only file system` (`fuse.litfs (ro,...)`) |
| `lightning cp -r <dir> lit://<owner>/<ts>/uploads/<path>` | `400` — "this endpoint only accepts logs/ and metrics/ paths" |
| `lightning_sdk` `Teamspace.upload_file(...)` | `404` — upload rejected |

Note `lightning file/folder upload` is *not* an alternative — those target a
Studio's home directory, not the Drive. The only remaining route is the Drive
tab in the Lightning web UI, which uses a browser upload path.

`jaccard_experiment/scripts/download_internvl3_wget.sh` fetches the weights
(wget, resumable, verifies shards against the index). Every run script reads:

```
export MODEL_PATH=/teamspace/studios/this_studio/models/InternVL3-8B-hf
```

**Data is shared, not duplicated.** Each experiment expects a `data/` symlink
into the corresponding LLaVA/Qwen folder, or a fresh download via its own
`scripts/download_*.py`.

**Hyperparameters are frozen across models.** APC beta grid, VCD `noise_step`,
`cd_alpha`, `cd_beta`, prompts, and text normalization are all kept identical to
the LLaVA and Qwen runs. The only things allowed to differ are genuinely
architecture-specific: how images are tiled/resized, and what metadata the
forward pass needs.

## InternVL-specific differences that matter

InternVL uses **dynamic tiling** rather than a resize-to-token-budget: an image
becomes up to `max_patches` tiles of 448×448 plus a global thumbnail, at 256
visual tokens per tile. So `pixel_values` is `(num_tiles, 3, 448, 448)` and the
tile count varies per image with aspect ratio. Two consequences:

1. There is no `image_grid_thw`-style tensor to thread through forwards — the
   tile count is just the leading dim.
2. Any contrastive method that re-runs the model on a perturbed image (VCD, SID)
   pays to re-encode the vision tower over *all* tiles, which is much more
   expensive than on LLaVA's single 336² crop.

`max_patches` is also the main memory lever: bf16 weights (~16 GB) plus ~3.3k
visual tokens is tight on a 24 GB card. Because it changes the visual token
budget, changing it invalidates cross-condition comparisons — re-run everything.
