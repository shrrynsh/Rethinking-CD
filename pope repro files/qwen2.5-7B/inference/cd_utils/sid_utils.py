import torch
from transformers import LogitsProcessor

# How small the "myopic" image gets for the SID contrastive branch, in raw
# pixels handed to Qwen2.5-VL's image processor (qwen_vl_utils reads
# min_pixels/max_pixels straight off each message's image entry). 28*28*4 is
# one merged visual token after the 2x2 spatial merge; this gives roughly a
# dozen tokens, deliberately small but not degenerate.
SID_MAX_PIXELS = 28 * 28 * 4 * 16


class SIDLogitsProcessor(LogitsProcessor):
    """Approximate Self-Introspective Decoding (SID) for Qwen2.5-VL.

    The original repo's SID branch (see `llava/model/language_model/
    custom_modeling_llama.py`, `use_sid` path) only works for LLaVA: it
    hooks LLaMA's decoder layers directly and, from layer 2 onward, masks
    all but a random ~12.5% of the image's *fixed* 576 CLIP tokens out of
    attention. Qwen2.5-VL has no such fixed-length image span (the number of
    image tokens is dynamic, driven by `image_grid_thw`), and its decoder
    layers aren't monkey-patchable the same way without forking the model
    class.

    This reimplements the same underlying idea — contrast the model's full
    view of the image against a deliberately impoverished ("myopic") view of
    it — at the input level instead of the attention level: the contrastive
    branch gets the *same* prompt and the *same* generated continuation, but
    a heavily downsampled version of the image (built via qwen_vl_utils'
    min_pixels/max_pixels), so it only ever sees a coarse, low-detail view of
    the scene. Combined via the same Adaptive Plausibility Constraint VCD/ICD
    use. This is an adaptation, not a byte-for-byte port — treat its outputs
    as approximate and validate with a smoke test before trusting numbers.
    """

    def __init__(self, model, main_prompt_len, cd_prompt_ids, pixel_values_cd, image_grid_thw_cd, cd_alpha=1.0, cd_beta=0.2):
        self.model = model
        self.main_prompt_len = main_prompt_len
        self.cd_prompt_ids = cd_prompt_ids
        self.pixel_values_cd = pixel_values_cd
        self.image_grid_thw_cd = image_grid_thw_cd
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
                pixel_values=self.pixel_values_cd,
                image_grid_thw=self.image_grid_thw_cd,
                cache_position=cache_position,
                use_cache=False,
            )
        cd_logits = cd_out.logits[:, -1, :].to(scores.dtype)

        ## cd_comments: Adaptive Plausibility Constraint, same formula as the original repo
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * scores - self.cd_alpha * cd_logits
        return diffs.masked_fill(scores < cutoff, -float("inf"))
