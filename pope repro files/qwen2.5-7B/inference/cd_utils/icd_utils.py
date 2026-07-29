import random

import torch
from transformers import LogitsProcessor


def get_random_icd_prompt():
    icd_prompts = [
        "You are an object detector to recognize every different objects.",
        "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
        "I want you avoid any specific identification or categorization of the objects depicted.",
        "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
        "You are an object detector to provide a general overview or impression of the image."
    ]
    return random.choice(icd_prompts)


class ICDLogitsProcessor(LogitsProcessor):
    """Instruction Contrastive Decoding for Qwen2.5-VL.

    Same rationale as VCDLogitsProcessor (see vcd_utils.py): implemented via
    `LogitsProcessor` instead of monkey-patching `GenerationMixin.greedy_search`
    /`sample`, which don't exist anymore in current transformers.

    ICD's contrastive branch uses a *different* prompt (an alternate system
    instruction) but the *same* image and the *same* generated continuation.
    Since the two branches' prompts have different lengths, the continuation
    generated so far has to be sliced out of the main branch's growing
    `input_ids` using `main_prompt_len` and re-attached to the cd branch's own
    prompt before each forward pass.
    """

    def __init__(self, model, main_prompt_len, cd_prompt_ids, pixel_values, image_grid_thw, cd_alpha=1.0, cd_beta=0.2):
        self.model = model
        self.main_prompt_len = main_prompt_len
        self.cd_prompt_ids = cd_prompt_ids
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.main_prompt_len:]
        cd_input_ids = torch.cat([self.cd_prompt_ids, generated], dim=1)
        attention_mask = torch.ones_like(cd_input_ids)
        cache_position = torch.arange(cd_input_ids.shape[1], device=cd_input_ids.device)

        with torch.no_grad():
            cd_out = self.model(
                input_ids=cd_input_ids,
                attention_mask=attention_mask,
                pixel_values=self.pixel_values,
                image_grid_thw=self.image_grid_thw,
                cache_position=cache_position,
                use_cache=False,
            )
        cd_logits = cd_out.logits[:, -1, :].to(scores.dtype)

        ## cd_comments: Adaptive Plausibility Constraint, same formula as the original repo
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * scores - self.cd_alpha * cd_logits
        return diffs.masked_fill(scores < cutoff, -float("inf"))
