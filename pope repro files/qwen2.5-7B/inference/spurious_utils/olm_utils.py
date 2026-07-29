import torch
from transformers import LogitsProcessor


def get_yes_no_token_ids(tokenizer):
    """Find every single-token spelling of "yes"/"no" in the given tokenizer's
    vocabulary (covers case and leading-space BPE variants), instead of the
    original repo's hardcoded Vicuna token ids (YES_INDEX=3869, NO_INDEX=1939)
    which are meaningless for Qwen2.5's tokenizer.
    """
    yes_variants = ["Yes", " Yes", "yes", " yes"]
    no_variants = ["No", " No", "no", " no"]
    yes_ids = sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in yes_variants})
    no_ids = sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in no_variants})
    return yes_ids, no_ids


class OLMLogitsProcessor(LogitsProcessor):
    """Output Layer Modification (OLM) for Qwen2.5-VL.

    Reimplemented as a LogitsProcessor instead of monkey-patching
    `GenerationMixin.greedy_search`/`_greedy_search`, which don't exist in
    current transformers (see cd_utils/vcd_utils.py for the same rationale).

    Unlike VCD/SID/ICD, OLM never needs a second forward pass -- it only
    inspects the single forward pass's own logits, so this is a plain,
    single-branch LogitsProcessor.

    Same rule as the original repo: when the model is meaningfully torn
    between "yes" and "no" (their combined probability mass is large but
    neither dominates), force the answer to "yes".
    """

    def __init__(self, yes_ids, no_ids, force_id, sum_threshold=0.2, diff_threshold=0.5):
        self.yes_ids = yes_ids
        self.no_ids = no_ids
        self.force_id = force_id
        self.sum_threshold = sum_threshold
        self.diff_threshold = diff_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        yes_prob = probs[:, self.yes_ids].sum(dim=-1)
        no_prob = probs[:, self.no_ids].sum(dim=-1)
        sum_prob = yes_prob + no_prob

        if sum_prob.item() > self.sum_threshold and (yes_prob - no_prob).abs().item() < self.diff_threshold:
            forced = torch.full_like(scores, -float("inf"))
            forced[:, self.force_id] = 0.0
            return forced

        return scores
