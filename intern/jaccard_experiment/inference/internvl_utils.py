"""Shared InternVL3-8B loading / prompting helpers for the Jaccard experiment.

Model: OpenGVLab/InternVL3-8B-hf  (the *native transformers* port).

Why the `-hf` port and not `OpenGVLab/InternVL3-8B`:
    The original OpenGVLab release ships custom modeling code behind
    `trust_remote_code=True` and exposes generation only through a bespoke
    `model.chat(tokenizer, pixel_values, question, gen_config)` helper. That
    helper does not forward a `logits_processor`, and its `forward()` takes an
    extra `image_flags` tensor. VCD here is implemented as a `LogitsProcessor`
    that re-runs `model(...)` on a noised image every decode step (same design
    as jaccard_on_qwen), so we need a plain, stable `forward()` and a
    `generate()` that accepts `logits_processor`. The `-hf` port
    (`InternVLForConditionalGeneration`) gives exactly that, and matches the
    Qwen2.5-VL code path almost line for line.

Image tiling — the InternVL analogue of Qwen's min_pixels/max_pixels:
    InternVL does not resize to a token budget. It splits the image into
    448x448 tiles ("dynamic high resolution") and appends a global thumbnail.
    Each tile costs 256 visual tokens (448/14 = 32 patches, 32^2 = 1024,
    then pixel-shuffle x0.25 -> 256). `max_patches=12` is the OpenGVLab default
    for single-image inference, giving up to 12+1 tiles = up to 3328 visual
    tokens. `pixel_values` therefore has shape (num_tiles, 3, 448, 448) and
    num_tiles varies per image with its aspect ratio.
"""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Same system prompt as jaccard_on_qwen, so the three models are compared under
# identical instructions. InternVL3 (Qwen2.5-7B LLM backbone) has the same
# habit of emitting markdown regardless; analysis/text_utils.py strips it.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Always respond in English only, in plain text "
    "without any markdown formatting (no bold, no headers, no bullet symbols)."
)

MODEL_ID = "internvl3-8b"


def load_model_and_processor(model_path, min_patches=1, max_patches=12):
    """Load InternVL3-8B and its processor with tiling limits pinned.

    IMPORTANT: this checkpoint ships `"crop_to_patches": false` in
    preprocessor_config.json. Left at that default the image processor skips
    dynamic tiling entirely and squashes the whole image into ONE 448x448 tile
    (256 visual tokens instead of up to 3328). It does not warn — you just get
    quietly degraded inputs and a ruined comparison against the LLaVA/Qwen runs.

    So the setting is forced on twice: once through from_pretrained kwargs, and
    again directly on the image processor in case kwarg routing through
    ProcessorMixin changes between transformers versions. Then it is verified,
    because a silent wrong default is exactly the failure this experiment
    cannot afford.
    """
    # transformers 5.x renamed `torch_dtype` to `dtype`; 4.x only knows the old
    # name. Try the new one first and fall back, so the same file works under
    # both (this repo already spans 4.31 -> 5.x across experiments).
    common = dict(attn_implementation="sdpa", device_map="auto")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, **common
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **common
        )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        crop_to_patches=True,
        min_patches=min_patches,
        max_patches=max_patches,
    )

    ip = processor.image_processor
    ip.crop_to_patches = True
    ip.min_patches = min_patches
    ip.max_patches = max_patches

    if not getattr(ip, "crop_to_patches", False):
        raise RuntimeError(
            "crop_to_patches is still disabled after being set explicitly — "
            "InternVL would run on a single squashed tile. Check the "
            "transformers version against the image processor API."
        )
    print(f"[internvl] tiling: crop_to_patches={ip.crop_to_patches} "
          f"min_patches={ip.min_patches} max_patches={ip.max_patches} "
          f"(<= {(max_patches + 1) * 256} visual tokens)")

    return model, processor


def build_inputs(processor, model, image_path, question):
    """Render the chat template and tokenize one (image, question) pair.

    This checkpoint's chat_template.jinja renders a `{"type": "image"}` content
    block as a single `<IMG_CONTEXT>` marker (not `<image>` — verified against
    the shipped template). `processor.__call__` then expands that one marker
    into `<img>` + N*256 `<IMG_CONTEXT>` tokens + `</img>`, where N is the tile
    count chosen for this particular image. That expansion is why input_ids and
    pixel_values must always be built together, and why the VCD amateur branch
    can reuse the same input_ids with a different pixel_values of equal tile
    count.

    The template emits `<|im_start|>{role}` for any role, so the system prompt
    is passed as a normal message.
    """
    image = Image.open(os.path.expanduser(image_path)).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=image, text=[text], return_tensors="pt")
    inputs = inputs.to(model.device)
    # The vision tower runs in the model's compute dtype; the image processor
    # always returns float32.
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def decode_new_tokens(processor, output_ids, input_len):
    return processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def add_common_args(parser):
    """Arguments shared by all four decoding scripts."""
    parser.add_argument("--model-path",    type=str, required=True)
    parser.add_argument("--image-folder",  type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file",  type=str, required=True)
    parser.add_argument("--num-chunks",    type=int, default=1)
    parser.add_argument("--chunk-idx",     type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-patches",   type=int, default=1,
                        help="Min 448x448 tiles per image (OpenGVLab default: 1)")
    parser.add_argument("--max-patches",   type=int, default=12,
                        help="Max 448x448 tiles per image (OpenGVLab default: 12)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Truncate the answers file instead of resuming it.")
    return parser
