# Agreement and overlap

These two experiments test whether contrastive decoding actually intervenes, and
whether it has any new candidates to intervene with.

## What is measured

**Intervention rate.** At every decoding step we compare the token contrastive
decoding emits with the token the expert branch alone would have emitted (plain
greedy on the clean image). The reported number is the percentage of steps where
they are the same, that is, where contrastive decoding changed nothing. We report
this over all steps, and separately over steps that emit a real object and steps
that emit a hallucinated object. Object steps are identified with the standard
CHAIR mapping applied to the generated caption.

**Top-10 candidate overlap.** At every step we take the top-10 tokens of the
expert branch, the amateur branch, and the contrastive score, and count how many
tokens the expert shares with the amateur (E&A) and with the contrastive score
(E&C). If the three branches consider the same shortlist, contrastive decoding
can only reorder existing candidates, not introduce better ones. We again report
this over all steps and split by real and hallucinated object steps.

All numbers come with 95 percent confidence intervals from an image-level
bootstrap (10,000 resamples). Images are the resampling unit because steps within
one caption share the same picture and are correlated.

## How to run

From the repository root:

```
bash agreement_and_overlap/llava-1.5-7b/run.sh     # LLaVA-1.5-7B
bash agreement_and_overlap/qwen-2.5-7b/run.sh      # Qwen2.5-VL-7B-Instruct
```

Each script downloads the assets if needed, produces the per-step capture for
VCD and SID, and prints the intervention rate and the top-10 overlap with
confidence intervals. The Qwen capture is slow (see the top-level README).

## How to read the output

The script prints, per method, the intervention rate for all / real / hall
steps, and the E&A and E&C overlaps for all / real / hall steps, each with a
95 percent interval. A high intervention rate means contrastive decoding rarely
changes the token. A high overlap means the branches deliberate over the same
words, so the contrastive step can only re-rank a shared shortlist.
