"""APC (Adaptive Plausibility Constraint) inference on LLaVA-Bench using Qwen2.5-VL-7B.

APC masks tokens whose logit falls below (log(beta) + max_logit):
  beta=0.000  →  cutoff=-inf  →  no masking  →  equivalent to direct sampling
  beta=1.000  →  cutoff=max   →  only argmax  →  equivalent to greedy search

Implemented as a LogitsProcessor — no monkey-patching, no second forward pass.
This is functionally identical to the LLaVA apc_utils.py (cd_alpha=1.0).

Image resolution: set at processor level per Qwen2.5-VL official docs.
"""
import argparse
import json
import math
import os

import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from apc_logits_processor import APCLogitsProcessor


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def eval_model(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    logits_processor = LogitsProcessorList([APCLogitsProcessor(args.beta)])

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx        = line["question_id"]
            image_file = line["image"]
            qs         = line["text"]

            full_image_path = os.path.join(args.image_folder, image_file)
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Always respond in English only, in plain text without any markdown formatting (no bold, no headers, no bullet symbols)."},
                {"role": "user", "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "text",  "text": qs},
                ]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                    logits_processor=logits_processor,
                )

            input_len   = inputs.input_ids.shape[1]
            output_text = processor.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt":      qs,
                "text":        output_text,
                "answer_id":   shortuuid.uuid(),
                "model_id":    "qwen2.5-vl-7b",
                "metadata":    {"beta": args.beta},
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",    type=str, required=True)
    parser.add_argument("--image-folder",  type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file",  type=str, required=True)
    parser.add_argument("--num-chunks",    type=int, default=1)
    parser.add_argument("--chunk-idx",     type=int, default=0)
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--top_p",         type=float, default=1.0)
    parser.add_argument("--beta",          type=float, required=True,
                        help="APC beta: 0=pure sampling, 1=greedy.")
    parser.add_argument("--min-pixels",    type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels",    type=int, default=1280 * 28 * 28)
    args = parser.parse_args()
    eval_model(args)
