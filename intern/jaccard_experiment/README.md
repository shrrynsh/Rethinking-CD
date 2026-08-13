# Jaccard experiment — InternVL3-8B

Third-model extension of `jaccard_experiment/`, which already covers
LLaVA-1.5-7B and Qwen2.5-VL-7B. Same benchmark, same 17 conditions, same
metric, so the three sets of numbers are directly comparable.

InternVL3-8B is run on the 60 questions of LLaVA-Bench (In-the-Wild) under
greedy search, direct sampling, the Adaptive Plausibility Constraint swept from
0.0 to 1.0, and VCD.

```
inference/   code that runs the model and writes its answers
scripts/     shell scripts that run the full decoding sweep
analysis/    code that reads the answers and computes the Jaccard scores
data/        the LLaVA-Bench questions and images
```

## Which checkpoint

`OpenGVLab/InternVL3-8B-hf` — the **native-transformers port**, not the plain
`OpenGVLab/InternVL3-8B`.

This is a load-bearing choice, not a convenience. The original OpenGVLab
release ships custom modeling code behind `trust_remote_code=True` and exposes
generation only through a bespoke `model.chat(...)` helper, which does not
forward a `logits_processor` and whose `forward()` takes an extra `image_flags`
tensor. Both APC and VCD here are `LogitsProcessor`s, and VCD additionally
re-runs `model(...)` directly on a noised image every decode step. The `-hf`
port (`InternVLForConditionalGeneration`) supports both, and lets the code
track `jaccard_on_qwen` almost line for line.

## Setup

### 1. Weights

**Already downloaded** to `/teamspace/studios/this_studio/models/InternVL3-8B-hf`
(15.9 GB, all 4 shards verified against `model.safetensors.index.json`).

```
export MODEL_PATH=/teamspace/studios/this_studio/models/InternVL3-8B-hf
python scripts/download_internvl3.py --check    # verifies they're readable
```

To re-fetch: `bash scripts/download_internvl3_wget.sh` pulls the 16 files
(4 safetensors shards + tokenizer/processor config) with wget. It runs `-c`, so
an interrupted run resumes rather than restarting the 16 GB, and it verifies
the shard set against the index before finishing. `SKIP_UPLOAD=1` downloads
only; `STAGING=<dir>` changes the destination.

The Teamspace Drive (`/teamspace/uploads`) is **not** a usable location — see
[../README.md](../README.md) for the three write methods tried and how each
failed. `scripts/download_internvl3.py` is the `huggingface_hub` equivalent,
useful if you are running from a machine that can write to the Drive directly.

### 2. Dependencies

```
pip install "transformers>=4.52" accelerate torch torchvision \
            pillow shortuuid tqdm numpy pandas matplotlib sentencepiece
```

`transformers>=4.52` is the floor — that is the release that added
`InternVLForConditionalGeneration`. No `qwen-vl-utils` equivalent is needed;
InternVL's tiling lives inside its own image processor.

Note this conflicts with `CHAIR_analysis` (`transformers==5.12.1`) and
`mech_interp` (`transformers==4.31.0`), matching the existing per-experiment
environment split in this repo.

### 3. Data

LLaVA-Bench (In-the-Wild): 24 images, 60 questions. Either symlink the copy
already in the repo (no re-download):

```
ln -s ../../../jaccard_experiment/jaccard_on_qwen/data/llava_bench data/llava_bench
```

or fetch it fresh:

```
python scripts/download_llava_bench.py
```

## Run

**Always launch through tmux.** The full sweep takes ~7 hours on an L4 — longer
than an SSH session, a browser tab, or an IDE reload will survive, and a plain
`bash scripts/run_all.sh` dies with its parent shell.

```
bash scripts/launch_tmux.sh              # full sweep, detached
bash scripts/launch_tmux.sh run_vcd.sh   # a single condition
```

```
tmux attach -t internvl        # watch
Ctrl-b d                       # detach, run keeps going
tail -f logs/internvl-*.log    # follow without attaching
tmux kill-session -t internvl  # stop
```

Output is tee'd to `logs/` as well as the tmux pane, so you can diagnose a
crash after the fact. The launcher refuses to start if a session of the same
name already exists (set `SESSION=<name>` to run two side by side).

The individual scripts — `run_greedy.sh`, `run_sample.sh`, `run_apc.sh`,
`run_vcd.sh` — can also be run directly for short tests. All read `MODEL_PATH`,
`DATA_DIR`, `MAX_PATCHES` etc. from `scripts/common.sh`, which validates that
the weights and questions exist before spending GPU time.

**Smoke-test first.** Before committing 7 hours, run 3 questions to confirm the
tiling printout and generation look right:

```
python inference/internvl_bench_infer_greedy.py \
    --model-path "$MODEL_PATH" \
    --question-file data/llava_bench/questions.jsonl \
    --image-folder  data/llava_bench/images \
    --answers-file  /tmp/smoke.jsonl \
    --num-chunks 20 --chunk-idx 0
```

It should print `[internvl] tiling: crop_to_patches=True ... <= 3328 visual tokens`.
If `crop_to_patches` is False or the token bound is 256, tiling is off and the
run would not be comparable to the LLaVA/Qwen numbers.

### Expected runtime (L4, 24 GB)

| Conditions | Time |
|---|---|
| greedy + sample + 14× APC | ~6.3 h (~24 min each) |
| VCD | ~0.8 h |
| **Total** | **~7 h** |

Calculated from the L4's 300 GB/s bandwidth and ~121 TFLOPS bf16 against a
~275-token average answer — arithmetic, not a measurement, so ±40%. Decoding is
bandwidth-bound (15.9 GB of weights / 300 GB/s ≈ 12 tok/s), which is also why
lowering `MAX_PATCHES` saves little time: it shrinks prefill, already only ~5%
of per-question cost.

Then:

```
cd analysis
python3 jaccard_apc_vs_greedy.py
python3 jaccard_apc_vs_sample.py
python3 jaccard_vcd_comparisons.py
python3 plot_jaccard_apc_vs_greedy.py
```

The analysis only loads the tokenizer, not the weights, so it runs fine on CPU.

## What changed vs. the Qwen implementation

The two are deliberately near-identical. Real differences:

| | Qwen2.5-VL-7B | InternVL3-8B |
|---|---|---|
| Resolution knob | `min_pixels`/`max_pixels` (256–1280 visual tokens, continuous) | `min_patches`/`max_patches` (1–12 tiles of 448², 256 tokens each, + thumbnail) |
| Grid metadata | `image_grid_thw` must be threaded into every forward | none — tile count is the leading dim of `pixel_values` |
| `pixel_values` shape | flat patchified sequence | `(num_tiles, 3, 448, 448)` |
| VCD amateur branch | stateless full re-forward every step | **KV-cached** — prefill once, then 1 token/step |

`add_diffusion_noise` and `APCLogitsProcessor` are unchanged from the Qwen
version — the first is elementwise and therefore shape-agnostic, the second
operates only on the logit vector.

**The VCD amateur branch keeps its own KV cache here.** jaccard_on_qwen re-runs
an uncached forward over the whole sequence every decode step, which is
harmless on POPE (1–3 token answers) but is ~981k token-forwards per question
on LLaVA-Bench's ~275-token answers — it would put VCD at ~6.5 h, half the
experiment. Caching makes it ~0.8 h.

This is a pure optimisation, not a methodological change: attention at position
*k* depends only on the K/V at positions ≤ *k*, and those are identical whether
computed now or 200 steps ago, so the logits match the stateless version up to
float nondeterminism. Worth one sentence in the writeup all the same, since it
diverges from the reference VCD implementation in form if not in output.

The APC beta grid (14 values) and VCD hyperparameters (`noise_step=900`,
`cd_alpha=1.0`, `cd_beta=0.1`) are identical across all three models.

## Notes

- **Memory.** bf16 weights are ~16 GB and tiling adds up to ~3.3k visual
  tokens, so a 24 GB card is tight. `MAX_PATCHES=6` is the first knob to turn
  on OOM; it changes the visual token budget, so re-run *all* conditions if you
  change it, not just the one that crashed.
- **VCD runtime** is the bottleneck — the no-KV-cache amateur branch re-encodes
  the vision tower every step. If the run is interrupted,
  `jaccard_vcd_comparisons.py` intersects question IDs and reports `n`, so a
  partial file still gives valid (if lower-powered) numbers.
- The plot's y-limits are computed from the data rather than hard-coded to the
  Qwen window `(0.30, 0.85)`, since InternVL3's sampling-vs-greedy spread isn't
  known in advance.
