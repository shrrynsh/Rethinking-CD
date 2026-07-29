import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# Bring in standard Transformers and Qwen utilities
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from llava.utils import disable_torch_init

from PIL import Image
import math


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
    
    # Extract model name for logging/metadata
    model_name = os.path.basename(model_path.rstrip("/"))

    # 1. Load native Qwen2.5-VL model and processor.
    # sdpa instead of flash_attention_2: flash-attn isn't installed/built in
    # this environment (no CUDA toolchain available to compile it).
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

        # Append execution suffix standard to the original POPE setup
        qs = qs + " Answer the question using a single word or phrase."
        full_image_path = os.path.join(args.image_folder, image_file)

        # 2. Convert standard conversational items into Qwen ChatML schema
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "text", "text": qs}
                ]
            }
        ]

        # 3. Generate formatting strings and query vision characteristics
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # 4. Synthesize input structures together to extract tensors (pixel_values + grid_thw)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. Generate with dynamic configuration checks
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=20,  # POPE answers are a single word/short phrase; bounds worst-case latency
                use_cache=True
            )

        # 6. Extract only generated sequence parts out of context windows
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
    parser.add_argument("--conv-mode", type=str, default="qwen_chat")  # Kept parameter interface unchanged
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)      # Defaulting to 0.0 for Greedy Baseline reproducibility
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)