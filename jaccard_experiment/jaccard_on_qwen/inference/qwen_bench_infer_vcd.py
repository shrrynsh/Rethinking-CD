"""VCD (Visual Contrastive Decoding) inference on LLaVA-Bench using Qwen2.5-VL-7B.

Per-step logic:
  1. Main forward:        clean image     → logits_clean
  2. Contrastive forward: noisy pixel_values → logits_cd
  3. cd_logits = (1+alpha)*logits_clean - alpha*logits_cd
  4. APC mask at cd_beta=0.1 applied to logits_clean
  5. Sample from masked cd_logits

Uses VCDLogitsProcessor (same as qwen2.5-7B/inference/pope_infer_cd.py).
The contrastive forward runs without KV-cache (full re-encode each step).

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

from vcd_logits_processor import VCDLogitsProcessor, add_diffusion_noise


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

            # Noisy pixel_values for VCD contrastive branch (same shape as clean)
            pixel_values_cd = add_diffusion_noise(
                inputs["pixel_values"].float(), args.noise_step
            ).to(device=model.device, dtype=model.dtype)

            logits_processor = LogitsProcessorList([
                VCDLogitsProcessor(
                    model,
                    pixel_values_cd,
                    inputs["image_grid_thw"],
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
                "metadata":    {
                    "noise_step": args.noise_step,
                    "cd_alpha":   args.cd_alpha,
                    "cd_beta":    args.cd_beta,
                },
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
    parser.add_argument("--noise-step",    type=int,   default=900)
    parser.add_argument("--cd-alpha",      type=float, default=1.0)
    parser.add_argument("--cd-beta",       type=float, default=0.1)
    parser.add_argument("--min-pixels",    type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels",    type=int, default=1280 * 28 * 28)
    args = parser.parse_args()
    eval_model(args)
