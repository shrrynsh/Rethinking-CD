# CHAIR analysis: does contrastive decoding reduce object hallucination?

This repository reproduces a mechanistic study of two contrastive decoding
methods for multimodal language models, VCD and SID, on the CHAIR object
hallucination benchmark. It runs on 500 MSCOCO images for two models,
LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct, and asks whether the "amateur" branch
that these methods subtract actually changes which objects the model names.

The code is organized into two independent analysis families:

1. **agreement_and_overlap** measures how often contrastive decoding changes the
   emitted token relative to plain greedy decoding, and how much the expert,
   amateur, and contrastive branches share the same top candidates.

2. **contrastive_adjustment** measures the per-token adjustment d = E - A that
   contrastive decoding adds to the expert logits, and tests whether replacing
   the amateur branch with matched Gaussian noise reproduces its effect on CHAIR.

## Repository layout

```
CHAIR_analysis/
  README.md                 this guide
  requirements.txt          package versions
  download_assets.sh        fetches model weights and the 500 COCO images
  image_ids_500.json        the fixed 500 MSCOCO val2017 image ids (same order everywhere)
  proxy_stats_llava.json    measured mean/std of d = E - A, used by the noise proxy
  proxy_stats_qwen.json
  common/                   shared code (all experiments use it)
    generate_llava.py       generation for LLaVA: capture / proxy / greedy
    generate_qwen.py        generation for Qwen
    agreement_analysis.py   intervention rate + top-10 overlap (one model)
    contrastive_analysis.py d = E - A, object-mass buckets, proxy vs greedy CHAIR
    object_mentions.py      caption-level object attribution (CHAIR mapping)
    eval/chair.py           CHAIR metric
    llava/                  the LLaVA model package (needed to run LLaVA)
  agreement_and_overlap/
    README.md
    llava-1.5-7b/run.sh
    qwen-2.5-7b/run.sh
  contrastive_adjustment/
    README.md
    llava-1.5-7b/run.sh
    qwen-2.5-7b/run.sh
```

Results are not included. Each `run.sh` regenerates everything from the models
and images. Generated files are written under `outputs/` and are ignored by git.

## Setup

A single machine with one 24 GB GPU is enough. In a Python environment with a
CUDA-capable PyTorch, install the dependencies:

```
pip install -r requirements.txt
```

Keep `transformers==5.12.1`; the bundled LLaVA package is aligned to it.

Then download the weights and images (about 29 GB, one time):

```
bash download_assets.sh
```

This fetches `liuhaotian/llava-v1.5-7b`, `Qwen/Qwen2.5-VL-7B-Instruct`, the COCO
val2017 annotations, and the 500 images listed in `image_ids_500.json`. It is
idempotent and skips anything already present. Each `run.sh` also calls it
automatically if the assets are missing, so a single `bash run.sh` works from a
fresh clone.

## How to reproduce

Each leaf folder has one `run.sh` that produces the data it needs and prints the
results for that model and experiment family. From the repository root:

```
bash agreement_and_overlap/llava-1.5-7b/run.sh
bash agreement_and_overlap/qwen-2.5-7b/run.sh
bash contrastive_adjustment/llava-1.5-7b/run.sh
bash contrastive_adjustment/qwen-2.5-7b/run.sh
```

If your environment uses a specific interpreter, pass it once:
`PY=/path/to/python bash agreement_and_overlap/llava-1.5-7b/run.sh`.

The per-experiment READMEs explain what each script prints and what it means.

## Practical notes

- **Shared capture.** Both analysis families read the same per-step capture
  files (`outputs/captures/<model>_{vcd,sid}.jsonl`). Whichever `run.sh` you run
  first produces them; later runs skip images that are already done, so the
  capture is not repeated.
- **Cost.** The proxy and greedy runs are fast (one forward per token). The
  VCD/SID capture on Qwen is slow, because the amateur branch is recomputed
  without a key-value cache at every step for numerical correctness (a cached
  second branch under Qwen's multimodal rotary embeddings did not match the
  exact computation). Expect hours for the Qwen capture on 500 images.
- **Resumable.** Every generation run appends per image and skips finished
  images, so it can be stopped and restarted.
- **Reproducibility.** Decoding is greedy with `max_new_tokens=256`, alpha = 1,
  beta = 0.2, and the prompt "Describe this image in detail." The proxy and the
  stochastic parts of VCD/SID are seeded per image, so a run is reproducible and
  a restart does not change the result.
- **Determinism note.** VCD's noise draw and SID's kept-token subset are random
  by construction. They are seeded here, but exact values across different
  hardware or library builds may differ slightly; aggregate results over 500
  images are stable.
