import math
import torch
from transformers import LogitsProcessor


class APCLogitsProcessor(LogitsProcessor):
    """Adaptive Plausibility Constraint (APC) as a LogitsProcessor.

    Mechanism (same as VCD paper):
        cutoff = log(beta) + max_logit
        tokens where logit < cutoff are masked to -inf

    beta=0   → log(0)=-inf → cutoff=-inf → no masking → pure sampling
    beta=1   → log(1)=0    → cutoff=max  → only argmax survives → greedy

    Model-agnostic: operates purely on the logit vector, so this is byte-for-byte
    the same processor used in jaccard_on_qwen, and equivalent to the LLaVA
    apc_utils.py monkey-patch with cd_alpha=1.0 (the standard setting) but
    without a second forward pass.
    """

    def __init__(self, beta: float):
        self.log_beta = math.log(beta) if beta > 0 else float('-inf')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cutoff = self.log_beta + scores.max(dim=-1, keepdim=True).values
        return scores.masked_fill(scores < cutoff, float('-inf'))
