import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# Bring in standard Transformers and Qwen utilities
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LogitsProcessorList
from qwen_vl_utils import process_vision_info
from llava.utils import disable_torch_init

from PIL import Image
import math

# Contrastive decoding hooks. These are LogitsProcessors (a stable, public
# generation extension point), not monkey-patches of GenerationMixin private
# methods -- see the comments in each cd_utils module for why.
from cd_utils.vcd_utils import add_diffusion_noise, VCDLogitsProcessor
from cd_utils.icd_utils import get_random_icd_prompt, ICDLogitsProcessor
from cd_utils.sid_utils import SIDLogitsProcessor, SID_MAX_PIXELS


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = os.path.basename(model_path.rstrip("/"))

    # Load native Qwen2.5-VL model and processor. sdpa is used instead of
    # flash_attention_2 since flash-attn isn't installed/built in this env.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        qs = qs + " Answer the question using a single word or phrase."
        full_image_path = os.path.join(args.image_folder, image_file)

        # 1. Prepare Standard (Clean) Inputs using Qwen ChatML Schema
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "text", "text": qs}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        logits_processor = LogitsProcessorList()

        # 2. Visual Contrastive Decoding (VCD): same prompt, diffusion-noised image
        if args.use_vcd:
            pixel_values_cd = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
            pixel_values_cd = pixel_values_cd.to(device=model.device, dtype=model.dtype)
            logits_processor.append(
                VCDLogitsProcessor(model, pixel_values_cd, inputs["image_grid_thw"], args.cd_alpha, args.cd_beta)
            )

        # 3. Instruction Contrastive Decoding (ICD): different system prompt, same image
        if args.use_icd:
            icd_prompt = get_random_icd_prompt()
            messages_cd = [
                {"role": "system", "content": icd_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_image_path},
                        {"type": "text", "text": qs}
                    ]
                }
            ]
            text_cd = processor.apply_chat_template(messages_cd, tokenize=False, add_generation_prompt=True)
            inputs_cd = processor(
                text=[text_cd], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(model.device)
            logits_processor.append(
                ICDLogitsProcessor(
                    model, inputs.input_ids.shape[1], inputs_cd.input_ids,
                    inputs["pixel_values"], inputs["image_grid_thw"], args.cd_alpha, args.cd_beta
                )
            )

        # 4. Self-Introspective Decoding (SID): same prompt, heavily downsampled "myopic" image
        if args.use_sid:
            messages_sid = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_image_path, "min_pixels": SID_MAX_PIXELS, "max_pixels": SID_MAX_PIXELS},
                        {"type": "text", "text": qs}
                    ]
                }
            ]
            text_sid = processor.apply_chat_template(messages_sid, tokenize=False, add_generation_prompt=True)
            image_inputs_sid, _ = process_vision_info(messages_sid)
            inputs_sid = processor(
                text=[text_sid], images=image_inputs_sid, padding=True, return_tensors="pt"
            ).to(model.device)
            logits_processor.append(
                SIDLogitsProcessor(
                    model, inputs.input_ids.shape[1], inputs_sid.input_ids,
                    inputs_sid["pixel_values"], inputs_sid["image_grid_thw"], args.cd_alpha, args.cd_beta
                )
            )

        # 5. Generate. Greedy vs sampling is controlled by do_sample/temperature as usual;
        # the contrastive logic above runs identically either way since it's a LogitsProcessor.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=20,  # POPE answers are a single word/short phrase; bounds worst-case latency
                use_cache=True,
                logits_processor=logits_processor,
            )

        input_token_len = inputs.input_ids.shape[1]
        outputs = processor.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to Qwen2.5-VL checkpoint")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_chat")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)  # Defaulting to 0.0 for Greedy Baseline robustness
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    # CD parameters
    parser.add_argument("--use-icd", action='store_true', default=False)
    parser.add_argument("--use-vcd", action='store_true', default=False)
    parser.add_argument("--use-sid", action='store_true', default=False)
    parser.add_argument("--noise-step", type=int, default=900)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)
