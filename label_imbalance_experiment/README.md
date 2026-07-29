# Label-imbalance experiment

Code to build and score the two label-imbalance stress tests of POPE for two
vision-language models:

- `llava1.5-7b/` : LLaVA-1.5-7B
- `qwen2.5-7B/`  : Qwen2.5-VL-7B-Instruct

Standard POPE is balanced: every image carries three YES questions and three NO
questions, so the benchmark is 50% YES / 50% NO. This experiment re-balances the
ground truth by deleting questions, while leaving every model answer untouched,
and asks what happens to each decoding method's accuracy:

- **25/75 regime**: delete 2 of the 3 YES questions per image, keep all 3 NO.
  The result is 25% YES / 75% NO (negatives dominate).
- **75/25 regime**: delete 2 of the 3 NO questions per image, keep all 3 YES.
  The result is 75% YES / 25% NO (positives dominate).

The deletion mask is drawn once, with a fixed seed, from the RANDOM split, and
then applied identically to every split and every decoding method. All methods
are therefore always scored on exactly the same questions; only the label
balance of the benchmark changes between the two regimes, which isolates the
effect of the label ratio from the behaviour of the model.

No re-inference is done. The regimes are subsamples of a finished POPE run, so
the whole experiment is a fast, deterministic transform of answer files that
already exist.

## Layout

Both model folders share the same layout:

```
data/            source POPE annotations (gqa / coco / aokvqa, 3 splits each)
build_regimes.py builds the 25/75 and 75/25 subsamples from a finished POPE run
eval/            per-file scorer (accuracy, precision/recall/F1, yes-rate)
analysis/
  balanced_accuracy.py   standard vs balanced accuracy + yes-rate, every cell
  bootstrap_ci.py        95% bootstrap CI on accuracy, every cell
  mcnemar.py             paired McNemar test of each method vs the baseline
scripts/run_all.sh       build both regimes, then run all three analyses
```

The subsampled regimes and every metric are generated on the fly; nothing
derived is shipped.

## Prerequisite: a finished POPE run

`build_regimes.py` consumes the answer files of an ordinary POPE run, laid out as

```
<PRED_DIR>/<method>/<tag>-<dataset>-<split>-greedy.jsonl
```

with `method` in `{baseline, vcd, sid, icd, pba, olm}`, `dataset` in
`{gqa, coco, aokvqa}`, `split` in `{random, popular, adversarial}`, and `tag`
being `llava-7b` or `Qwen2.5-7b`. This is exactly the layout written by the
companion POPE-table reproduction package to its `outputs/pope/` folder, so the
usual workflow is to run that first and point `PRED_DIR` at its output.

The POPE annotations are already included under `data/`; only the model answers
need to be supplied.

## How to run

From inside a model folder:

```bash
cd llava1.5-7b        # or  cd qwen2.5-7B
export PRED_DIR=/path/to/finished/pope/outputs/pope
bash scripts/run_all.sh
```

`run_all.sh` runs, in order:

1. `build_regimes.py`  writes the subsampled regimes to `./regimes/<regime>/<dataset>/`
2. `analysis/balanced_accuracy.py`  prints standard vs balanced accuracy per cell
3. `analysis/bootstrap_ci.py`       prints a 95% bootstrap CI on accuracy per cell
4. `analysis/mcnemar.py`            prints McNemar p-values (method vs baseline)

Each step can also be run on its own; every script takes `--model-tag` and a
directory argument and has sensible defaults (see `--help`).

## What the analyses show

- **Standard vs balanced accuracy.** Standard accuracy on POPE moves with the
  label ratio even when the answers do not; balanced accuracy, the mean of the
  two per-class recall rates, does not. Comparing a method's balanced accuracy
  across the two regimes separates a real change in discrimination from an
  artifact of the ratio.
- **Bootstrap CI.** Resamples the questions of each cell to bound how much of
  its accuracy is sampling noise given the regime's question count.
- **McNemar.** Uses the paired per-question correctness of a method and the
  baseline on identical questions, which is a stronger comparison than two
  separate accuracy intervals.

## Reproducibility

- The deletion mask uses a fixed seed (`--seed`, default 42) and is built from
  the RANDOM split, so the same questions are removed for every method and split.
- `build_regimes.py` checks that each method's filtered answers line up with the
  filtered annotations `question_id` for `question_id`, and refuses to write a
  cell whose order does not match.
- The scorers treat any answer containing "yes" as positive and any answer
  containing "no" as negative, matching the POPE protocol.
