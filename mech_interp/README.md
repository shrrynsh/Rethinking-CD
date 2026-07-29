# POPE-COCO VCD mechanistic-interpretability experiment

Does VCD's perturbation-induced logit shift actually modify the residual stream at the layer where LLaVA encodes "does this object exist", or only at unrelated layers?

Single-decision binary task (T=1, no generation loop), so this is the cheap version of the mech experiment.

## What this adds

Everything lives in `mech_interp/` and imports the repo's own LLaVA + VCD code (`llava.model.builder.load_pretrained_model`, `cd_utils.vcd_utils.add_diffusion_noise`, `conv_mode=vicuna_v1`, the exact `pope_infer_base.py` prompt). **No existing repo file is modified** and the LLaVA/CD config is untouched.

| File | Role |
|---|---|
| `download_model.sh` | Pull `liuhaotian/llava-v1.5-7b` (original LLaVA format this repo loads) into persistent storage |
| `download_pope_data.py` | Fetch the 3 POPE-COCO json splits + only the unique COCO val2014 images they reference |
| `extract_activations.py` | Two forward passes/sample (clean=expert, noisy=amateur); saves last-token hidden states for all 33 layers + per-layer `CD_residual_change` + gt label + baseline greedy pred, sharded |
| `train_probes.py` | Probe A (truth), Probe B (hallucination on gt-No), CD curve; emits csv + plot + summary |
| `run_all.sh` | One-shot driver |

## The three quantities

- **Probe A (truth)** `hidden_l -> object-present?` over all samples -> layer `l*_truth` where the true answer is most decodable.
- **Probe B (hallucination)** restricted to ground-truth-No samples, `hidden_l -> did baseline emit a false-positive "Yes"?` -> layer `l*_halluc`, the hallucination-specific representation.
- **CD residual change** `mean_l ||h_clean[l] - h_noisy[l]||_2` -> layer `l*_CD` where VCD's perturbation moves the residual stream most.

If `l*_CD` sits at a different depth from `l*_truth` / `l*_halluc`, that is direct evidence VCD's correction does not operate on the representation encoding object presence.

---

## Running on Lightning (L4)

You are **not** on Lightning yet - do these in order once you open the Studio.

### 0. Open a Studio, attach an L4, clone this repo
```bash
cd /teamspace/studios/this_studio
git clone <your-fork-url> cd-rethinking      # or however this repo is on the Studio
cd cd-rethinking
nvidia-smi        # confirm the L4 is attached
```

### 1. Environment
This repo pins old deps (`transformers==4.31.0`, `torch==2.0.1`) that the custom LLaVA needs. Install the repo itself, then the two extras the mech scripts add:
```bash
pip install -e .                                   # repo deps (llava)
pip install matplotlib pandas hf_transfer          # extras: plotting + fast HF download
```
`scikit-learn`, `numpy`, `tqdm`, `requests`, `Pillow` already come from the repo's `pyproject.toml`.

> If `pip install -e .` fights the pre-installed torch on the Studio image, install torch first per the repo pin, then `pip install -e . --no-deps` and `pip install` any missing runtime deps. The mech scripts don't add exotic requirements - no flash-attn needed.

### 2. Smoke test first (2-3 min, ~40 samples/split)
Always validate the pipeline end-to-end on a tiny slice before the full run:
```bash
MAX_SAMPLES=40 bash mech_interp/run_all.sh
```
Check in the log:
- `num_hidden_layers=32` (so 33 hidden states),
- the `[sanity]` block shows `generate()` and logit-argmax agree (should be 8/8),
- `[baseline] ... greedy acc` is sane,
- `results/layerwise_curves.png` + `per_layer.csv` + `summary.txt` were written.

### 3. Full run
```bash
bash mech_interp/run_all.sh
```
- Downloads weights (~14 GB, once) and the unique COCO images (POPE reuses a small image pool across splits, so this is a few hundred to ~2k images, not all of val2014).
- Extraction: ~9000 questions x 2 short forward passes. On an L4 expect roughly **30-70 min**. Activations are ~2-3 GB of sharded `.npz` under `activations/`.
- Probing + plotting: a few minutes.

Override paths/behaviour via env vars:
```bash
ROOT=/teamspace/studios/this_studio/cd_pope_mech_interp \
MODEL_PATH=/teamspace/.../llava-v1.5-7b \
MAX_SAMPLES=0 \
bash mech_interp/run_all.sh
```

### Run stages individually (if you prefer)
```bash
bash   mech_interp/download_model.sh   /teamspace/studios/this_studio/cd_pope_mech_interp/models
python mech_interp/download_pope_data.py --data-dir /teamspace/studios/this_studio/cd_pope_mech_interp/data
python mech_interp/extract_activations.py --model-path <weights> --sanity-check 8
python mech_interp/train_probes.py
```

---

## Outputs (`results/`)

- `per_layer.csv` - layer, `probe_truth_acc/auc/bal_acc`, `probe_halluc_acc/auc/bal_acc`, `cd_change_mean`, `cd_change_norm`.
- `layerwise_curves.png` - three curves (truth AUC, hallucination AUC, normalized CD change) with `l*` markers.
- `summary.txt` / `summary.json` - `l*_truth`, `l*_halluc`, `l*_CD`, class balances, hallucination rate, and the coincide-vs-different-depth verdict.
- `activations/extract_summary.json` - per-split baseline greedy accuracy (sanity vs known POPE-COCO numbers).

## Sanity checks (Step 6 of the brief) - baked in

- **generate vs logit** - `--sanity-check` runs `model.generate(max_new_tokens=5)` on a handful of samples and prints agreement with the forward-pass argmax used for `pred`.
- **Baseline accuracy** - printed per split; COCO-random should land ~80-87% for LLaVA-1.5-7B, lower on adversarial. If it's near chance, the prompt/preprocessing is wrong - stop and fix before trusting probes.
- **Class balance / hallucination rate** - printed before probe numbers. If the gt-No hallucination rate is <10% or >90%, trust `bal_acc`/`auc` over raw accuracy (Probe B reports all three; the plot/peaks use AUC).

## Design notes / deviations from the brief

- **Uses this repo's LLaVA, not `llava-hf` transformers.** The brief's `LlavaForConditionalGeneration`/`AutoProcessor` snippet is the HF-native path; this repo ships its own `LlavaLlamaForCausalLM` + `tokenizer_image_token` + `add_diffusion_noise`, and `conv_mode=vicuna_v1` with the `" Answer the question using a single word or phrase."` suffix. Matching the repo guarantees the extracted representations are the ones the real VCD pipeline sees.
- **Noise** is the repo's `add_diffusion_noise(image_tensor, 900)` verbatim, seeded per-sample (`--noise-seed`) for reproducible/resumable runs.
- **Storage** keeps full clean hidden states (needed by the probes) plus only the per-layer L2 norm of the clean-vs-noisy difference (the CD curve only needs the norm), which halves disk vs storing both hidden tensors.
- **Peak selection uses AUC**, not raw accuracy, because POPE gt-No hallucination can be class-imbalanced.
