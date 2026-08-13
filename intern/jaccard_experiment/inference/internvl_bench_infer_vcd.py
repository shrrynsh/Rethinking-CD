"""VCD (Visual Contrastive Decoding) inference on LLaVA-Bench using InternVL3-8B.

Per-step logic:
  1. Main forward:        clean image        → logits_clean
  2. Contrastive forward: noisy pixel_values → logits_cd
  3. cd_logits = (1+alpha)*logits_clean - alpha*logits_cd
  4. APC mask at cd_beta=0.1 applied to logits_clean
  5. Sample from masked cd_logits

Uses VCDLogitsProcessor. The contrastive forward runs without KV-cache (full
re-encode each step), so this is the slowest condition by a wide margin —
InternVL re-encodes up to 13 vision tiles on every single decode step.

Image resolution: InternVL dynamic tiling (see inference/internvl_utils.py).
"""
import argparse
import json
import os
import sys

import shortuuid
import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internvl_utils import (MODEL_ID, add_common_args, build_inputs,
                            decode_new_tokens, load_model_and_processor)
from io_utils import load_questions, prepare_answers_file
from vcd_logits_processor import VCDLogitsProcessor, add_diffusion_noise


def eval_model(args):
    ans_handle, done_ids = prepare_answers_file(args.answers_file, resume=not args.no_resume)
    questions = load_questions(
        args.question_file, args.num_chunks, args.chunk_idx, done_ids
    )
    if not questions:
        print("[resume] nothing left to do for this file.")
        ans_handle.close()
        return

    model, processor = load_model_and_processor(
        args.model_path, args.min_patches, args.max_patches
    )

    with ans_handle as ans_file:
        for line in tqdm(questions):
            idx        = line["question_id"]
            image_file = line["image"]
            qs         = line["text"]

            inputs = build_inputs(
                processor, model,
                os.path.join(args.image_folder, image_file),
                qs,
            )

            # Noisy pixel_values for the VCD amateur branch. Same tile count and
            # shape as the clean tensor, so the image placeholder tokens already
            # in input_ids line up on both branches.
            pixel_values_cd = add_diffusion_noise(
                inputs["pixel_values"].float(), args.noise_step
            ).to(device=model.device, dtype=model.dtype)

            logits_processor = LogitsProcessorList([
                VCDLogitsProcessor(
                    model,
                    pixel_values_cd,
                    args.cd_alpha,
                    args.cd_beta,
                )
            ])

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    logits_processor=logits_processor,
                )

            input_len   = inputs["input_ids"].shape[1]
            output_text = decode_new_tokens(processor, output_ids, input_len)

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt":      qs,
                "text":        output_text,
                "answer_id":   shortuuid.uuid(),
                "model_id":    MODEL_ID,
                "metadata":    {
                    "noise_step": args.noise_step,
                    "cd_alpha":   args.cd_alpha,
                    "cd_beta":    args.cd_beta,
                },
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=1.0)
    parser.add_argument("--noise-step",  type=int,   default=900)
    parser.add_argument("--cd-alpha",    type=float, default=1.0)
    parser.add_argument("--cd-beta",     type=float, default=0.1)
    args = parser.parse_args()
    eval_model(args)
