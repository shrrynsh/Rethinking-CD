"""Greedy decoding inference on LLaVA-Bench (In-the-Wild) using InternVL3-8B.

Image resolution: InternVL dynamic tiling, 448x448 tiles, min_patches=1 /
max_patches=12 (the OpenGVLab single-image default) plus a global thumbnail.
See inference/internvl_utils.py for why this is the analogue of Qwen's
min_pixels/max_pixels knob.
"""
import argparse
import json
import os
import sys

import shortuuid
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internvl_utils import (MODEL_ID, add_common_args, build_inputs,
                            decode_new_tokens, load_model_and_processor)
from io_utils import load_questions, prepare_answers_file


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

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            input_len   = inputs["input_ids"].shape[1]
            output_text = decode_new_tokens(processor, output_ids, input_len)

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt":      qs,
                "text":        output_text,
                "answer_id":   shortuuid.uuid(),
                "model_id":    MODEL_ID,
                "metadata":    {},
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    eval_model(args)
