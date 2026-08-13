"""Verify the KV-cached amateur branch matches the stateless re-forward.

The cached VCDLogitsProcessor is only a valid optimisation if it returns the
same amateur logits as recomputing the whole sequence every step. A subtly
misaligned cache would still produce fluent text — just conditioned on the
wrong prefix — so this is checked numerically rather than by eyeballing output.

Run:
    python inference/test_vcd_cache_equivalence.py --model-path $MODEL_PATH
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internvl_utils import build_inputs, load_model_and_processor
from vcd_logits_processor import VCDLogitsProcessor, add_diffusion_noise


def stateless_amateur_logits(model, input_ids, pixel_values_cd, keep_kwarg):
    """The original jaccard_on_qwen behaviour: full uncached forward."""
    extra = {keep_kwarg: 1} if keep_kwarg else {}
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values_cd,
            cache_position=torch.arange(input_ids.shape[1], device=input_ids.device),
            use_cache=False,
            **extra,
        )
    return out.logits[:, -1, :].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image", default="data/llava_bench/images/001.jpg")
    ap.add_argument("--question", default="Describe this photo in detail.")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--noise-step", type=int, default=900)
    args = ap.parse_args()

    model, processor = load_model_and_processor(args.model_path)
    inputs = build_inputs(processor, model, args.image, args.question)
    pixel_values_cd = add_diffusion_noise(
        inputs["pixel_values"].float(), args.noise_step
    ).to(device=model.device, dtype=model.dtype)

    proc = VCDLogitsProcessor(model, pixel_values_cd, cd_alpha=1.0, cd_beta=0.1)

    # Walk a growing prefix the way generate() would, feeding the cached
    # processor incrementally while recomputing the stateless answer each step.
    ids = inputs["input_ids"]
    vocab = model.config.text_config.vocab_size
    dummy_scores = torch.zeros((1, vocab), device=ids.device, dtype=torch.float32)

    max_abs, max_rel_rank = 0.0, 0
    print(f"\ncomparing {args.steps} steps (prompt = {ids.shape[1]} tokens)\n")
    for step in range(args.steps):
        # cached path — mirrors what generate() drives
        proc(ids, dummy_scores)
        cached = proc._last_cd_logits.float()

        # stateless path — recompute from scratch
        ref = stateless_amateur_logits(
            model, ids, pixel_values_cd, proc._keep_kwarg
        )

        diff = (cached - ref).abs().max().item()
        # rank agreement matters more than raw float distance
        same_argmax = (cached.argmax(-1) == ref.argmax(-1)).all().item()
        top5_c = cached.topk(5, dim=-1).indices.tolist()[0]
        top5_r = ref.topk(5, dim=-1).indices.tolist()[0]

        max_abs = max(max_abs, diff)
        max_rel_rank += 0 if top5_c == top5_r else 1
        print(f"  step {step:2d}  seq={ids.shape[1]:5d}  max|Δlogit|={diff:.4e}  "
              f"argmax_match={same_argmax}  top5_match={top5_c == top5_r}")

        # append the amateur's own argmax as the next token, just to advance
        ids = torch.cat([ids, ref.argmax(-1, keepdim=True)], dim=1)

    print(f"\nmax abs logit difference over {args.steps} steps: {max_abs:.4e}")
    print(f"steps where top-5 ordering differed: {max_rel_rank}/{args.steps}")
    # bf16 forward passes are not bitwise reproducible; what must hold is that
    # the ranking the sampler sees is unchanged.
    if max_rel_rank == 0:
        print("\nPASS — cached and stateless amateur branches agree.")
    else:
        print("\nFAIL — cache is not equivalent to the stateless forward.")
        sys.exit(1)


if __name__ == "__main__":
    main()
