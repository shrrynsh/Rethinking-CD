# MME & POPE Results — Contrastive Decoding Reproducibility Study

This folder contains the code, experiment scripts, and results for reproducing and
analyzing **Contrastive Decoding (CD)** and related decoding strategies for
hallucination mitigation in vision-language models (LLaVA), evaluated on the
**MME** and **POPE** benchmarks.

> This folder is self-contained: all paths in the commands below are relative to
> this directory. The raw benchmark data (`data/`) and the bundled `llava/` model
> source are **not** included here — install the LLaVA stack into your environment
> and download the MME/POPE releases separately (see Prerequisites and Prepare the data).

## Directory overview

| Path | Description |
| --- | --- |
| `inference/` | Inference scripts, one per decoding method, for MME and POPE (baseline, VCD, ICD, SID, PBA, OLM, APC). |
| `eval/` | Evaluators that consume answer files and render the per-subtask results table. |
| `scripts/` | Shell wrappers to prepare data, run inference, and build the tables end to end. |
| `outputs/` | Generated answer files (`.jsonl`) and result tables (`.csv`) per method. |
| `plots/` | Figures produced from the results. |
| `R_score.py`, `R_score_eda.py` | Compute and analyze the semantic flip ratio (R) between decoding methods. |
| `transitions.py` | EDA on answer transitions across MME results. |
| `pyproject.toml` | Project configuration and dependencies. |

## Contents

- MME evaluation table (below): what gets produced, repo layout, prerequisites,
  preparing data, running inference, and building the table.
- POPE evaluation: see the [POPE](#pope-evaluation) section at the end.

---

## MME Evaluation Table

This guide explains how to use this codebase to run LLaVA inference on the
**MME benchmark** and produce the per-subtask results table (Yes% | Accuracy |
Score | F1) with Perception and Cognition group totals.

The MME workflow mirrors the existing POPE workflow one-to-one: a set of
`mme_infer_*.py` inference scripts (one per decoding method) write answer files,
and a single `eval/mme_eval.py` evaluator consumes them and renders the table.

---

## 1. What gets produced

For each subtask the evaluator reports four metrics, plus two group totals:

| Metric        | Definition |
|---------------|------------|
| **Yes%**      | answers classified `"yes"` / total questions × 100 |
| **Accuracy**  | correctly answered questions / total questions |
| **Score**     | `(Accuracy + Accuracy_Plus) × 100` (the MME score) |
| **F1**        | harmonic mean of precision/recall, `"yes"` as the positive class |

where **Accuracy_Plus** is the per-image correctness rate that credits an image
only when *both* of its two questions are answered correctly. Group totals are
the sum of `Score` over the subtasks in each group:

- **Perception (10):** existence, count, position, color, posters, celebrity, scene, landmark, artwork, OCR
- **Cognition (4):** commonsense_reasoning, numerical_calculation, text_translation, code_reasoning

An answer is classified `yes`/`no`/`unknown` by case-insensitive substring match
(`yes` takes precedence over `no`); `unknown` always counts as incorrect.

---

## 2. Repository layout (MME-relevant)

```
cd_rethink/
  data/mme/
    questions/{subtask}.jsonl     # inference input: {question_id, image, text}
    reference/{subtask}.jsonl     # ground truth:    {question_id, image, text, label, category}
    images/{subtask}/*.jpg        # images referenced by the `image` field
    mme_questions.jsonl           # combined inference input (all subtasks)
    mme_reference.jsonl           # combined ground truth (all subtasks)
  inference/
    mme_infer_common.py           # shared helpers (chunking, do_sample, answer record, flag guard)
    mme_infer_base.py             # base generation
    mme_infer_cd.py               # VCD / ICD / SID contrastive decoding
    mme_infer_apc.py              # APC spurious-mitigation
    mme_infer_olm.py              # OLM spurious-mitigation
    mme_infer_pba.py              # PBA spurious-mitigation
  eval/
    mme_constants.py              # the 14 subtask names + Perception/Cognition split
    mme_eval.py                   # evaluator + results table
  scripts/
    mme_infer_base.sh             # base inference (greedy + sample)
    mme_infer_cd.sh               # VCD/ICD/SID inference (greedy + sample)
    mme_infer_spurious.sh         # PBA / OLM / APC inference
    mme_eval.sh                   # render the table for every produced answer file
```

---

## 3. Prerequisites

- **Python** with the LLaVA stack installed (the same environment used for POPE):
  `torch`, the `llava` package, `shortuuid`, `tqdm`, `Pillow`.
- A **LLaVA checkpoint** (e.g. `llava-v1.5-7b`) and a GPU for inference.
- For the evaluator and the test suite only: `pytest` and `hypothesis`
  (no GPU/model required).

> Run every command from the repository root (`cd_rethink/`). All default paths
> in the scripts are relative to the repo root.

---

## 4. Prepare the data

The committed raw MME release ships under
`data/mme/MME_Benchmark_release_version/MME_Benchmark/{subtask}/`. The converter
`eval/mme_prepare.py` turns that raw tree into the exact JSONL + image layout the
inference and evaluation pipeline consumes. Run it once from the repo root:

```bash
python eval/mme_prepare.py \
    --raw-root ./data/mme/MME_Benchmark_release_version/MME_Benchmark \
    --out-root ./data/mme \
    --per-subtask
# or: bash scripts/mme_prepare.sh
```

### Outputs

The converter writes four artifacts under `--out-root` (`./data/mme`):

- **`mme_reference.jsonl`** — the combined ground truth, one
  `{question_id, image, text, label, category}` object per line, ordered by
  subtask (following `SUBTASKS`) then by stem.
- **`mme_questions.jsonl`** — the combined inference input, derived from the
  reference by projecting each record to `{question_id, image, text}`. The
  converter now produces this for you (no manual projection step needed).
- **`reference/{subtask}.jsonl`** — per-subtask reference files (written only
  when `--per-subtask` is passed).
- **`images/{subtask}/`** — the uniform reconciled image view that the `image`
  field resolves against under `--image-folder ./data/mme/images`.

### FLAT vs SPLIT auto-detection

Each subtask folder is classified by its on-disk shape, with no manual
configuration:

- **SPLIT** — selected when the subtask folder contains *both* an `images/`
  sub-folder and a `questions_answers_YN/` sub-folder. Images are read from
  `images/` and annotations from `questions_answers_YN/`.
- **FLAT** — selected otherwise. Image files (`.jpg`/`.jpeg`/`.png`) and their
  `{stem}.txt` annotation files sit as siblings directly inside the subtask
  folder.

The observable property is simply the presence of both `images/` and
`questions_answers_YN/` sub-folders: present → SPLIT, otherwise → FLAT. A FLAT
subtask and a SPLIT subtask carrying the same logical content produce identical
reference records.

### Image reconciliation and `--copy`

Regardless of the raw layout, the converter materializes every image at the
uniform path `data/mme/images/{subtask}/{filename}` so a single
`--image-folder ./data/mme/images` resolves all records. By default each
reconciled image is a **relative symlink** back to its source in the raw tree,
which avoids duplicating the dataset on disk. Pass **`--copy`** to materialize
byte-for-byte physical copies instead — use this on symlink-averse filesystems
(for example some network shares or Windows mounts) where symlinks are
unsupported. If symlink mode is selected on a filesystem that cannot create
symlinks, the converter aborts with an actionable error directing you to re-run
with `--copy`.

---

## 5. Run inference

Each script defaults to `MODEL_PATH=/code/pretrained_models/llava-v1.5-7b` and
`CONV_MODE=vicuna_v1`. Override via environment variables as needed.

```bash
# Base generation (greedy @ temperature 0, and sampling @ temperature 1)
bash scripts/mme_infer_base.sh

# Contrastive decoding: VCD, ICD, SID (greedy + sampling for each)
bash scripts/mme_infer_cd.sh

# Spurious-mitigation: PBA (greedy), OLM (greedy), APC (sampling)
bash scripts/mme_infer_spurious.sh
```

Override the model or data, for example:

```bash
MODEL_PATH=/path/to/llava-v1.5-13b CONV_MODE=vicuna_v1 \
QUESTION_FILE=./data/mme/mme_questions.jsonl \
IMAGE_FOLDER=./data/mme/images \
bash scripts/mme_infer_base.sh
```

Answer files are written to `./outputs/mme/{method}/llava-7b-mme-{greedy,sample}.jsonl`
in the shared JSONL schema `{question_id, prompt, text, answer_id, model_id, metadata}`.

### Decoding notes (carried over from POPE)

- **base / PBA**: stock `model.generate`; `do_sample = temperature > 0`.
- **VCD / ICD / SID** (`mme_infer_cd.py`): patch *both* greedy and sampling paths,
  so they work under `--temperature 0` and `--temperature 1`. Select exactly one
  of `--use-vcd` / `--use-icd` / `--use-sid` per run (the script rejects conflicts).
- **OLM**: greedy-only patch → run greedy (`--temperature 0`).
- **APC**: sampling-only patch → run sampling (`--temperature 1`).

---

## 6. Build the table

After running any subset of the inference scripts:

```bash
bash scripts/mme_eval.sh
```

This evaluates every answer file that exists under `./outputs/mme/` against the
combined reference and prints a labeled table per method, e.g.:

```
========================================================================
Method: baseline (greedy)
Answer file: ./outputs/mme/baseline/llava-7b-mme-greedy.jsonl
========================================================================
Subtask                 Yes%     Accuracy Score    F1
existence               50.00    1.0000   200.00   1.0000
commonsense_reasoning   50.00    1.0000   200.00   1.0000
------------------------------------------------------------
Perception Total                          200.00
Cognition Total                           200.00
```

Override the reference or output root if needed:

```bash
REF_FILE=./data/mme/mme_reference.jsonl OUT_ROOT=./outputs/mme bash scripts/mme_eval.sh
```

### Evaluate a single answer file directly

```bash
python ./eval/mme_eval.py \
    --ref-files ./data/mme/mme_reference.jsonl \
    --res-files ./outputs/mme/baseline/llava-7b-mme-greedy.jsonl
```

The evaluator raises a descriptive error if the answer count or any
`question_id` does not match the reference, or if any image is not associated
with exactly two questions.

---

## 7. Run a single method manually (without the scripts)

```bash
# Example: VCD, greedy
python ./inference/mme_infer_cd.py \
    --model-path /code/pretrained_models/llava-v1.5-7b \
    --question-file ./data/mme/mme_questions.jsonl \
    --image-folder ./data/mme/images \
    --answers-file ./outputs/mme/vcd/llava-7b-mme-greedy.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-vcd
```

Shared inference arguments (all scripts): `--model-path`, `--model-base`,
`--image-folder`, `--question-file`, `--answers-file`, `--conv-mode`,
`--num-chunks`, `--chunk-idx`, `--temperature`, `--top_p`, `--num_beams`.
Method-specific flags: `--use-vcd` / `--use-icd` / `--use-sid` / `--noise-step`
(cd), `--use-apc` (apc), `--use-olm` (olm).

---

## 8. Tests

The pure logic (evaluator metrics, classification, validation, inference
helpers) is covered by property-based tests (Hypothesis) plus example and
mocked smoke tests — no GPU or model required:

```bash
# Converter (parsing, layout detection, reconciliation, self-validation)
python -m pytest eval/test_mme_prepare.py -q

# Evaluator (classification, metrics, grouping, table, alignment)
python -m pytest eval/test_mme_eval.py -q

# Inference helpers + mocked per-method smoke tests
python -m pytest \
    inference/test_mme_infer_common.py \
    inference/test_mme_infer_base.py \
    inference/test_mme_infer_cd.py \
    inference/test_mme_infer_apc.py \
    inference/test_mme_infer_olm.py \
    inference/test_mme_infer_pba.py -q
```

---

## 9. End-to-end summary

```bash
# 1. Prepare data/mme/ from the committed raw MME tree
python eval/mme_prepare.py \
    --raw-root ./data/mme/MME_Benchmark_release_version/MME_Benchmark \
    --out-root ./data/mme \
    --per-subtask
# 2. Inference
bash scripts/mme_infer_base.sh
bash scripts/mme_infer_cd.sh
bash scripts/mme_infer_spurious.sh
# 3. Table
bash scripts/mme_eval.sh
```

---

## POPE Evaluation

The POPE (Polling-based Object Probing Evaluation) workflow mirrors the MME setup:
one `inference/pope_infer_*.py` script per decoding method writes answer files, and
a POPE evaluator scores them for hallucination (accuracy, precision, recall, F1, and
yes-ratio) over the `random`, `popular`, and `adversarial` splits.

### Run POPE inference

```bash
# Baseline greedy generation
bash scripts/pope_infer_base.sh

# Contrastive decoding (VCD / ICD / SID)
bash scripts/pope_infer_cd.sh

# Spurious-mitigation variants (PBA / OLM / APC)
bash scripts/pope_infer_spurious.sh
```

The POPE inference scripts accept the same shared arguments as the MME scripts
(`--model-path`, `--question-file`, `--image-folder`, `--answers-file`, `--conv-mode`,
and the method-specific flags such as `--use-vcd` / `--use-icd` / `--use-sid` /
`--use-apc` / `--use-olm`). Override the model or data via environment variables
exactly as documented for MME above.

### Score POPE

```bash
bash scripts/pope_eval_base.sh        # score a single method
bash scripts/pope_eval_transfer.sh    # cross-method / transfer analysis
```

Answer files are written under `outputs/pope/{method}/` following the same JSON
schema as MME, and the evaluator prints a per-split table of hallucination metrics.
