#!/usr/bin/env python3
"""Extract last-token hidden states (expert=clean, amateur=noisy) + VCD residual
change per layer for POPE-COCO, using THIS repo's own LLaVA + VCD machinery.

For every POPE question we run two single forward passes (T=1, no generation loop):

  * clean  : model(images=clean_image_tensor,  output_hidden_states=True)
  * noisy  : model(images=add_diffusion_noise(image_tensor, noise_step), ...)

At the final prompt-token position (index -1, the token whose logits predict the
Yes/No answer) we record, for each of the 33 hidden layers (embeddings + 32 blocks):

  * clean_hidden[l]                       -> features for the linear probes
  * cd_change[l] = ||h_clean[l] - h_noisy[l]||_2   -> VCD's residual-stream effect
  * ground-truth label (from POPE 'label')
  * baseline greedy prediction (argmax of clean logits, decoded)

Prompt construction, conv_mode, image preprocessing, and the noise function are
copied verbatim from the repo's pope_infer_base.py / vcd_utils.add_diffusion_noise
so the extracted representations match what the real VCD pipeline sees.

Output: sharded .npz (default 500 samples/shard) + a meta.jsonl per shard, under
<out-dir>/. One split at a time (--split), or all three.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# --- make the repo importable no matter where this script is invoked from ---
# REPO_ROOT for `llava`; REPO_ROOT/inference for `cd_utils` (that's where the repo
# keeps it, and pope_infer_cd.py imports it as `cd_utils.vcd_utils`).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

from cd_utils.vcd_utils import add_diffusion_noise  # noqa: E402  (inference/cd_utils on sys.path via inference/)

SPLITS = ("random", "popular", "adversarial")
ANSWER_SUFFIX = " Answer the question using a single word or phrase."


def label_to_int(label: str) -> int:
    """POPE label 'yes'/'no' -> 1 (object present) / 0 (absent)."""
    return 1 if str(label).strip().lower() == "yes" else 0


def build_input_ids(tokenizer, model, question_text: str, conv_mode: str):
    """Replicate pope_infer_base.py prompt construction exactly."""
    qs = question_text
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    qs = qs + ANSWER_SUFFIX

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids


def yes_no_token_ids(tokenizer):
    """Token ids that begin a 'Yes' / 'No' answer, for logging a scalar margin."""
    yes_ids, no_ids = set(), set()
    for word, bucket in (("Yes", yes_ids), ("yes", yes_ids), ("No", no_ids), ("no", no_ids)):
        for variant in (word, " " + word):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                bucket.add(ids[0])
    return sorted(yes_ids), sorted(no_ids)


@torch.inference_mode()
def run_pass(model, input_ids, image_tensor):
    """One forward pass; return (last_token_hidden [33,4096] fp32 cpu, last_logits [V] fp32 cpu)."""
    out = model(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # hidden_states: tuple length num_layers+1, each [1, seq, 4096]
    hs = torch.stack([h[0, -1, :].float() for h in out.hidden_states], dim=0)  # [33, 4096]
    last_logits = out.logits[0, -1, :].float().cpu()
    return hs.cpu(), last_logits


def load_questions(pope_dir: str, split: str):
    path = os.path.join(pope_dir, f"coco_pope_{split}.json")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_shard(out_dir, split, shard_idx, buf):
    """buf: dict of lists. Writes <split>_shard{idx}.npz + .meta.jsonl."""
    npz_path = os.path.join(out_dir, f"{split}_shard{shard_idx:04d}.npz")
    np.savez_compressed(
        npz_path,
        clean_hidden=np.stack(buf["clean_hidden"]).astype(np.float16),  # [n,33,4096]
        cd_change=np.stack(buf["cd_change"]).astype(np.float32),        # [n,33]
        gt=np.asarray(buf["gt"], dtype=np.int8),                         # [n]
        pred=np.asarray(buf["pred"], dtype=np.int8),                     # [n]
    )
    meta_path = os.path.join(out_dir, f"{split}_shard{shard_idx:04d}.meta.jsonl")
    with open(meta_path, "w") as f:
        for m in buf["meta"]:
            f.write(json.dumps(m) + "\n")


def extract_split(args, tokenizer, model, image_processor, split):
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    questions = load_questions(os.path.join(args.data_dir, "pope", "coco"), split)
    if args.max_samples > 0:
        questions = questions[: args.max_samples]
    images_dir = os.path.join(args.data_dir, "pope", "coco", "images")

    os.makedirs(args.out_dir, exist_ok=True)
    buf = {k: [] for k in ("clean_hidden", "cd_change", "gt", "pred", "meta")}
    shard_idx, n_correct, n_total = 0, 0, 0

    for i, line in enumerate(tqdm(questions, desc=f"extract:{split}")):
        image_file = line["image"]
        try:
            image = Image.open(os.path.join(images_dir, image_file)).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] {image_file}: {e}")
            continue

        input_ids = build_input_ids(tokenizer, model, line["text"], args.conv_mode)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # deterministic per-sample noise so re-runs / resume reproduce exactly
        torch.manual_seed(args.noise_seed + i)
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)

        hs_clean, logits_clean = run_pass(model, input_ids, image_tensor)
        hs_noisy, _ = run_pass(model, input_ids, image_tensor_cd)

        cd_change = torch.linalg.vector_norm(hs_clean - hs_noisy, dim=1).numpy()  # [33]

        # baseline greedy prediction from clean logits (== generate() greedy first token)
        pred_id = int(torch.argmax(logits_clean).item())
        pred_str = tokenizer.decode([pred_id]).strip()
        pred_yes = 1 if pred_str.lower().startswith("yes") else 0
        gt = label_to_int(line["label"])
        n_correct += int(pred_yes == gt)
        n_total += 1

        buf["clean_hidden"].append(hs_clean.numpy())
        buf["cd_change"].append(cd_change)
        buf["gt"].append(gt)
        buf["pred"].append(pred_yes)
        buf["meta"].append(
            {
                "split": split,
                "question_id": line.get("question_id"),
                "image": image_file,
                "text": line["text"],
                "gt": gt,
                "pred_yes": pred_yes,
                "pred_token": pred_str,
                "yes_logit": float(logits_clean[yes_ids].max()) if yes_ids else None,
                "no_logit": float(logits_clean[no_ids].max()) if no_ids else None,
            }
        )

        if len(buf["gt"]) >= args.shard_size:
            save_shard(args.out_dir, split, shard_idx, buf)
            shard_idx += 1
            buf = {k: [] for k in buf}

    if buf["gt"]:
        save_shard(args.out_dir, split, shard_idx, buf)

    acc = n_correct / max(n_total, 1)
    print(f"[extract:{split}] samples={n_total}  baseline greedy accuracy={acc:.4f}")
    return {"split": split, "n": n_total, "baseline_accuracy": acc}


@torch.inference_mode()
def sanity_generate_vs_logit(args, tokenizer, model, image_processor, split, n=8):
    """Confirm argmax(clean logits) == generate(max_new_tokens=5) first token."""
    from llava.conversation import SeparatorStyle
    from llava.mm_utils import KeywordsStoppingCriteria

    questions = load_questions(os.path.join(args.data_dir, "pope", "coco"), split)[:n]
    images_dir = os.path.join(args.data_dir, "pope", "coco", "images")
    print(f"\n[sanity] generate() vs logit-argmax on {n} '{split}' samples:")
    match = 0
    for line in questions:
        image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
        input_ids = build_input_ids(tokenizer, model, line["text"], args.conv_mode)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        _, logits_clean = run_pass(model, input_ids, image_tensor)
        logit_tok = tokenizer.decode([int(torch.argmax(logits_clean))]).strip()

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        crit = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        out_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            max_new_tokens=5,
            use_cache=True,
            stopping_criteria=[crit],
        )
        gen = tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        ok = gen.lower().startswith(logit_tok.lower()) and logit_tok != ""
        match += int(ok)
        print(f"  gt={line['label']:>3}  logit='{logit_tok}'  generate='{gen}'  match={ok}")
    print(f"[sanity] agreement: {match}/{n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to llava-v1.5-7b weights dir")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--data-dir", default="/teamspace/studios/this_studio/cd_pope_mech_interp/data")
    ap.add_argument("--out-dir", default="/teamspace/studios/this_studio/cd_pope_mech_interp/activations")
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--noise-step", type=int, default=900)  # VCD default in this repo
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all; else cap per split (debug)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--sanity-check", type=int, default=8, help="0 to skip generate-vs-logit check")
    args = ap.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    n_layers = model.config.num_hidden_layers
    print(f"[model] {model_name}  num_hidden_layers={n_layers} (expect 32 -> 33 hidden states)")

    if args.sanity_check > 0:
        sanity_generate_vs_logit(args, tokenizer, model, image_processor, args.splits[0], args.sanity_check)

    summary = [extract_split(args, tokenizer, model, image_processor, s) for s in args.splits]
    with open(os.path.join(args.out_dir, "extract_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] per-split baseline accuracy:")
    for s in summary:
        print(f"  {s['split']:>12}: n={s['n']:>5}  acc={s['baseline_accuracy']:.4f}")


if __name__ == "__main__":
    main()
