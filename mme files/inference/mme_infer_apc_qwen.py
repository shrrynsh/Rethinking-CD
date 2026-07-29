"""MME APC inference for Qwen2.5-VL-7B-Instruct — memory-optimised.

Key changes vs original:
  1. max_new_tokens: 1024 → 16   (MME answers are single words)
  2. Images capped at MAX_IMAGE_SIZE before processor sees them
  3. torch.cuda.empty_cache() + del inputs/outputs each iteration
  4. attn_implementation="flash_attention_2" + bfloat16 (saves ~1.5 GB)
  5. Processor loaded with min_pixels / max_pixels caps
  6. Optional --load-in-4bit path for very tight VRAM budgets
"""

import argparse
import os
import json
import gc

import torch
import shortuuid
from tqdm import tqdm
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers import LogitsProcessor, LogitsProcessorList

from mme_infer_common import (
    split_list,
    get_chunk,
    derive_do_sample,
    build_answer_record,
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
DEFAULT_QWEN_PATH = "/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct"

# Qwen2.5-VL uses dynamic resolution. Capping pixels avoids surprise OOMs on
# large MME images. 512×512 is ample for binary yes/no perception tasks.
MAX_IMAGE_SIZE = (512, 512)

# Processor pixel caps (Qwen2.5-VL specific). These control how many vision
# tokens the model allocates before any forward pass.
MIN_PIXELS = 256 * 28 * 28   # lower bound
MAX_PIXELS = 512 * 28 * 28   # ~400K pixels → far fewer vision tokens than default


class APCPlausibilityProcessor(LogitsProcessor):
    """Adaptive Plausibility Constraint cutoff (cd_beta=0.2).

    Masks every token whose logit falls below log(cd_beta) + max(logit) to
    -inf before the sampling warpers run, identical to the LLaVA APC sample
    path once the redundant identity contrastive stream is removed.
    """

    def __init__(self, cd_beta: float = 0.2):
        self.cd_beta = cd_beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cutoff = torch.log(
            torch.tensor(self.cd_beta, device=scores.device, dtype=scores.dtype)
        ) + scores.max(dim=-1, keepdim=True).values
        return scores.masked_fill(scores < cutoff, float("-inf"))


def load_qwen(args):
    model_path = os.path.expanduser(args.model_path)

    # --- dtype & attention backend ---
    # bfloat16 is numerically more stable than float16 and preferred on Ampere+.
    # flash_attention_2 is used when available; falls back to sdpa (PyTorch
    # scaled-dot-product attention), which is still more memory-efficient than
    # the default eager implementation.
    dtype = torch.bfloat16

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        # sdpa requires torch >= 2.1.1; fall back to eager for older envs
        import torch as _torch
        attn_impl = "sdpa" if tuple(int(x) for x in _torch.__version__.split(".")[:2]) >= (2, 1) else "eager"

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation=attn_impl,
    )

    if getattr(args, "load_in_4bit", False):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # device_map must be "auto" for 4-bit
        load_kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.eval()

    # Processor pixel caps limit vision tokens without touching the text side.
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model_name = os.path.basename(model_path.rstrip("/"))
    return model, processor, model_name


def resize_image(image_path: str) -> Image.Image:
    """Open and cap image size before the processor converts it to tensors."""
    img = Image.open(image_path).convert("RGB")
    if img.width > MAX_IMAGE_SIZE[0] or img.height > MAX_IMAGE_SIZE[1]:
        img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return img


def build_inputs(model, processor, image: Image.Image, question_text: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question_text},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def decode_trimmed(processor, inputs, generated_ids: torch.Tensor) -> str:
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def eval_model(args):
    model, processor, model_name = load_qwen(args)

    logits_processor = None
    if args.use_apc:
        logits_processor = LogitsProcessorList([APCPlausibilityProcessor(cd_beta=0.2)])

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    do_sample = derive_do_sample(args.temperature)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            image_path = os.path.join(args.image_folder, line["image"])
            qs = line["text"] + INSTRUCTION_SUFFIX

            # --- load & resize before tensor allocation ---
            image = resize_image(image_path)

            inputs = build_inputs(model, processor, image, qs)

            gen_kwargs = dict(
                max_new_tokens=16,     # was 1024 — single words don't need more
                use_cache=True,
                do_sample=do_sample,
                num_beams=args.num_beams,
            )
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            output_text = decode_trimmed(processor, inputs, generated_ids)
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(build_answer_record(line, output_text, ans_id, model_name)) + "\n"
            )
            ans_file.flush()

            # --- explicit memory cleanup each iteration ---
            del inputs, generated_ids, image
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-apc", action="store_true", default=False)
    # Extra: 4-bit quantisation for very constrained GPUs (requires bitsandbytes)
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                        help="Load model in NF4 4-bit (saves ~4 GB, needs bitsandbytes)")
    args = parser.parse_args()

    eval_model(args)