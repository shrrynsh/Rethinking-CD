"""MME contrastive-decoding inference for Qwen2.5-VL-7B-Instruct (VCD / ICD / SID) — memory-optimised.

Key changes vs original:
  1. max_new_tokens: 1024 → 16   (MME answers are single words)
  2. Images capped at MAX_IMAGE_SIZE before processor sees them
  3. torch.cuda.empty_cache() + del inputs/outputs each iteration
  4. attn_implementation auto-detected (flash_attention_2 → sdpa → eager) + bfloat16
  5. Processor loaded with min_pixels / max_pixels caps
  6. Optional --load-in-4bit path for very tight VRAM budgets
  7. resize_image() extracted; build_main_inputs() receives a PIL image, not a path
  8. _combine_cd dtype-matched to avoid float16/bfloat16 tensor mismatch
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

from cd_utils.vcd_utils import add_diffusion_noise
from cd_utils.icd_utils import get_random_icd_prompt

from mme_infer_common import (
    split_list,
    get_chunk,
    derive_do_sample,
    build_answer_record,
    validate_cd_flags,
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
DEFAULT_QWEN_PATH = "teamspace/lightning_storage/model/Qwen2.5_7b"

MAX_IMAGE_SIZE = (512, 512)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

CD_ALPHA = 1.0
CD_BETA  = 0.2


def load_qwen(args):
    model_path = os.path.expanduser(args.model_path)

    dtype = torch.bfloat16

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
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
        load_kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.eval()

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


def _process(model, processor, messages, with_image, image=None):
    """Run the processor for a chat-message list, returning model-ready inputs."""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if with_image:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def build_main_inputs(model, processor, image: Image.Image, question_text: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question_text},
        ],
    }]
    return _process(model, processor, messages, with_image=True, image=image)


def build_contrastive_inputs(model, processor, args, main_inputs, image, question_text):
    """Build the contrastive-stream inputs for the selected CD method."""
    if args.use_vcd:
        inputs_cd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in main_inputs.items()}
        inputs_cd["pixel_values"] = add_diffusion_noise(
            main_inputs["pixel_values"], args.noise_step
        ).to(model.dtype)
        return inputs_cd
    if args.use_icd:
        messages = [
            {"role": "system", "content": get_random_icd_prompt()},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]},
        ]
        return _process(model, processor, messages, with_image=True, image=image)
    if args.use_sid:
        messages = [{"role": "user", "content": [{"type": "text", "text": question_text}]}]
        return _process(model, processor, messages, with_image=False)
    return None


def _eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


def _init_stream(model, inputs):
    """Initialize a decode-stream state with explicit multimodal RoPE position ids."""
    input_ids      = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs.get("image_grid_thw")
    position_ids, rope_deltas = model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "pixel_values":    inputs.get("pixel_values"),
        "image_grid_thw":  image_grid_thw,
        "position_ids":    position_ids,
        "rope_deltas":     rope_deltas,
        "past_key_values": None,
        "cur_len":         input_ids.shape[1],
        "first":           True,
    }


def _forward_stream(model, stream):
    """Advance one stream by a single forward pass; return the last-position logits.

    pixel_values and image_grid_thw are only needed for the prefill (first) pass —
    the vision encoder output is already baked into past_key_values after that.
    We clear them from the stream state immediately after prefill to free the
    image tensor from VRAM for the rest of the decode loop.
    """
    if stream["first"]:
        model_inputs = {
            "input_ids":       stream["input_ids"],
            "attention_mask":  stream["attention_mask"],
            "position_ids":    stream["position_ids"],
            "pixel_values":    stream["pixel_values"],
            "image_grid_thw":  stream["image_grid_thw"],
            "past_key_values": stream["past_key_values"],
            "use_cache":       True,
        }
    else:
        last_id = stream["input_ids"][:, -1:]
        delta   = (stream["cur_len"] - 1) + stream["rope_deltas"]
        pos     = torch.arange(1, device=last_id.device).view(1, -1) + delta
        model_inputs = {
            "input_ids":       last_id,
            "attention_mask":  stream["attention_mask"],
            "position_ids":    pos.unsqueeze(0).expand(3, -1, -1),
            "past_key_values": stream["past_key_values"],
            "use_cache":       True,
        }
    outputs = model(**{k: v for k, v in model_inputs.items() if v is not None}, return_dict=True)
    stream["past_key_values"] = outputs.past_key_values
    if stream["first"]:
        # Free vision tensors — not needed once KV-cache is populated.
        stream["pixel_values"]   = None
        stream["image_grid_thw"] = None
        stream["first"] = False
    return outputs.logits[:, -1, :]


def _append_token(stream, token):
    stream["input_ids"] = torch.cat([stream["input_ids"], token[:, None]], dim=-1)
    if stream["attention_mask"] is not None:
        ones = torch.ones(
            (stream["attention_mask"].shape[0], 1),
            dtype=stream["attention_mask"].dtype,
            device=stream["attention_mask"].device,
        )
        stream["attention_mask"] = torch.cat([stream["attention_mask"], ones], dim=-1)
    stream["cur_len"] += 1


def _combine_cd(logits, logits_cd):
    cutoff = torch.log(
        torch.tensor(CD_BETA, device=logits.device, dtype=logits.dtype)
    ) + logits.max(dim=-1, keepdim=True).values
    diffs = (1 + CD_ALPHA) * logits - CD_ALPHA * logits_cd
    return diffs.masked_fill(logits < cutoff, float("-inf"))


def _top_p_filter(scores, top_p):
    sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0]  = False
    return scores.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))


def _select_token(scores, do_sample, temperature, top_p):
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if temperature and temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    if top_p is not None:
        scores = _top_p_filter(scores, top_p)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(1)


def contrastive_generate(model, inputs_main, inputs_cd, args, do_sample, max_new_tokens=16):
    """Two-stream contrastive autoregressive decode; returns generated token ids only.

    Returning only the new tokens (not the full input_ids tensor) means the
    prompt tokens don't stay live on GPU past this call. Both KV-caches are
    explicitly deleted before returning so VRAM is freed before the caller's
    cleanup runs.
    """
    eos_ids    = _eos_ids(model)
    stream     = _init_stream(model, inputs_main)
    stream_cd  = _init_stream(model, inputs_cd) if inputs_cd is not None else None
    generated  = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = _forward_stream(model, stream)
            if stream_cd is not None:
                logits_cd = _forward_stream(model, stream_cd)
                scores = _combine_cd(logits, logits_cd)
            else:
                scores = logits
            next_token = _select_token(scores, do_sample, args.temperature, args.top_p)
            generated.append(next_token.item())
            _append_token(stream, next_token)
            if stream_cd is not None:
                _append_token(stream_cd, next_token)
            if next_token.item() in eos_ids:
                break

    # Free both KV-caches immediately — they are the largest live tensors.
    del stream["past_key_values"]
    if stream_cd is not None:
        del stream_cd["past_key_values"]

    return torch.tensor([generated], dtype=torch.long)


def decode_trimmed(processor, prompt_len: int, generated_ids: torch.Tensor) -> str:
    trimmed = generated_ids[:, prompt_len:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def eval_model(args):
    validate_cd_flags(args)
    model, processor, model_name = load_qwen(args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Resume support: skip questions already written to the answers file.
    done_ids: set = set()
    if args.resume_from and os.path.exists(answers_file):
        with open(answers_file) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_ids)} questions already done, skipping.")

    do_sample = derive_do_sample(args.temperature)
    with open(answers_file, "a" if done_ids else "w") as ans_file:
        for line in tqdm(questions):
            if line.get("question_id") in done_ids:
                continue

            image_path = os.path.join(args.image_folder, line["image"])
            qs = line["text"] + INSTRUCTION_SUFFIX

            image       = resize_image(image_path)
            inputs_main = build_main_inputs(model, processor, image, qs)
            inputs_cd   = build_contrastive_inputs(
                model, processor, args, inputs_main, image, qs
            )
            # contrastive_generate now returns only the new tokens tensor.
            new_ids     = contrastive_generate(model, inputs_main, inputs_cd, args, do_sample)
            output_text = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps(build_answer_record(line, output_text, ans_id, model_name)) + "\n")
            ans_file.flush()

            # Explicit memory cleanup each iteration.
            del inputs_main, inputs_cd, new_ids, image
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",      type=str,   default=DEFAULT_QWEN_PATH)
    parser.add_argument("--model-base",      type=str,   default=None)
    parser.add_argument("--image-folder",    type=str,   default="")
    parser.add_argument("--question-file",   type=str,   default="tables/question.jsonl")
    parser.add_argument("--answers-file",    type=str,   default="answer.jsonl")
    parser.add_argument("--num-chunks",      type=int,   default=1)
    parser.add_argument("--chunk-idx",       type=int,   default=0)
    parser.add_argument("--temperature",     type=float, default=0.2)
    parser.add_argument("--top_p",           type=float, default=None)
    parser.add_argument("--num_beams",       type=int,   default=1)
    parser.add_argument("--use-icd",         action="store_true", default=False)
    parser.add_argument("--use-vcd",         action="store_true", default=False)
    parser.add_argument("--use-sid",         action="store_true", default=False)
    parser.add_argument("--noise-step",      type=int,   default=900)
    parser.add_argument("--load-in-4bit",    action="store_true", default=False,
                        help="Load model in NF4 4-bit (saves ~4 GB, needs bitsandbytes)")
    parser.add_argument("--resume-from",     action="store_true", default=False,
                        help="Skip questions already present in the answers file (safe resume after OOM)")
    args = parser.parse_args()

    eval_model(args)
