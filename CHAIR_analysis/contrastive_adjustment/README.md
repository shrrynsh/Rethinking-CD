# Contrastive adjustment

Contrastive decoding forms its score as C = (1 + alpha) E - alpha A, which at
alpha = 1 is C = 2E - A = E + d, where d = E - A is the amount it adds to the
expert logit of every token. These experiments look directly at d, and then test
whether d can be replaced by random noise of the same size.

## What is measured

**The adjustment d = E - A.** For each object token we read its expert logit E
and amateur logit A and form d. A positive d means contrastive decoding amplifies
the token; a negative d means it suppresses it. A method that removes
hallucinations should give hallucinated objects clearly negative d. We report the
mean of d and the percentage of tokens with d > 0, split by real and hallucinated
object, over three populations: the object words the model actually emits, and
the expert's top-10 and top-30 object candidates.

**Object-mass buckets.** Among the top-5 object candidates at each object step,
we compute the normalized probability mass placed on real versus hallucinated
objects, for the expert, amateur, and contrastive branches. This shows whether
the contrastive step moves probability toward real objects or away from them. The
real share is reported with a 95 percent image-level bootstrap interval.

**Noise proxy versus greedy.** We discard the amateur branch and instead add
independent Gaussian noise to every expert logit, with mean and standard
deviation matched to the measured distribution of d (from `proxy_stats_*.json`).
The proxy is run for three seeds. We score its captions and the greedy baseline
with CHAIR. If matched random noise reproduces the CHAIR level of the real
methods, the amateur branch is not contributing anything a generic perturbation
could not.

## How to run

From the repository root:

```
bash contrastive_adjustment/llava-1.5-7b/run.sh    # LLaVA-1.5-7B
bash contrastive_adjustment/qwen-2.5-7b/run.sh     # Qwen2.5-VL-7B-Instruct
```

Each script downloads the assets if needed, reuses the per-step capture (shared
with the agreement analysis, so it is skipped if already produced), runs the
noise proxy for three seeds plus a greedy baseline, and prints the d statistics,
the object-mass buckets with confidence intervals, and the CHAIR scores of the
proxy and greedy caption files.

## How to read the output

For d, compare the hallucinated row against the real row: a suppressor would make
the hallucinated mean clearly more negative, which does not happen. For the
buckets, compare the contrastive branch against the expert branch. For the proxy,
compare its CHAIR to greedy: proxy near greedy means the noise does not reduce
hallucination, which is the point.
