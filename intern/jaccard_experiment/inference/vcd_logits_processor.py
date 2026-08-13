import torch
from transformers import DynamicCache, LogitsProcessor


def add_diffusion_noise(image_tensor, noise_step):
    """Unchanged from the original LLaVA/VCD implementation: forward-diffusion
    Gaussian noise on a float image tensor. Fully elementwise and therefore
    shape-agnostic, so it applies unmodified to InternVL's tiled
    `pixel_values` of shape (num_tiles, 3, 448, 448) just as it did to
    LLaVA/CLIP's (3, 336, 336) and Qwen2.5-VL's patchified layout.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    noise_delta = int(noise_step)  # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    return image_tensor_cd


class VCDLogitsProcessor(LogitsProcessor):
    """Visual Contrastive Decoding for InternVL3-8B.

    The original VCD repo implemented this by monkey-patching
    `GenerationMixin.greedy_search`/`sample`, which no longer exist in current
    transformers. This hooks in through `LogitsProcessor` instead — a stable
    public extension point that `generate()` calls once per step under both
    greedy and sampling decoding.

    Each step it evaluates the *same* token prefix against a diffusion-noised
    image and combines the two logit streams with the Adaptive Plausibility
    Constraint from the VCD paper.

    KV cache on the amateur branch
    ------------------------------
    jaccard_on_qwen re-runs the amateur branch statelessly: a full uncached
    forward over the entire sequence at every decode step. That is fine on POPE,
    where answers are 1-3 tokens, but LLaVA-Bench answers average ~275 tokens,
    which turns it into ~981k token-forwards per question (vs ~3.7k) and puts
    VCD at ~6.5 min/question on an L4 — half the whole experiment's runtime for
    one of 17 conditions.

    So this keeps the amateur branch's own `past_key_values` as state:

        first call  : noisy image + full prompt -> build cache, return logits
        later calls : feed ONLY the newest token -> attend against cache

    This is a pure optimisation, not a change of method. Attention at position k
    depends only on the K/V at positions <= k, and those tensors are identical
    whether computed now or 200 steps ago, so the logits match the stateless
    version up to float nondeterminism. Vision re-encoding also disappears: the
    image's contribution is already baked into the cached K/V of the prompt
    positions, so `pixel_values` is passed on the prefill call only.

    Validity rests on both branches sharing one token prefix — true here because
    `generate()` feeds every processor the same running `input_ids`, and this
    experiment uses num_beams=1 (no beam reordering) with one sequence at a
    time. `_reset()` guards the assumption: if the incoming prefix ever fails to
    extend what the cache holds, the cache is rebuilt from scratch rather than
    silently returning logits for the wrong prefix.

    InternVL vs Qwen difference: Qwen2.5-VL needs `image_grid_thw` alongside
    `pixel_values` to reconstruct its variable-resolution patch grid. InternVL
    tiles to a fixed 448x448 and infers tile count from the leading dim of
    `pixel_values`, so there is no grid tensor to thread through.
    """

    def __init__(self, model, pixel_values_cd, cd_alpha=1.0, cd_beta=0.2):
        self.model = model
        self.pixel_values_cd = pixel_values_cd
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta

        self._reset()

        # Only the final position's logits are ever used, but a bare forward()
        # runs lm_head over the whole sequence: at ~3.9k tokens and a 151674
        # vocab that is a 1.2 GB bf16 tensor which transformers then upcasts to
        # fp32 (2.4 GB). Asking for one position drops that to ~0.3 MB. This
        # still matters with the cache on, for the prefill call.
        # transformers renamed the arg (num_logits_to_keep -> logits_to_keep
        # around 4.49), so probe for both rather than pinning a version.
        self._keep_kwarg = None
        try:
            import inspect
            params = inspect.signature(model.forward).parameters
            for name in ("logits_to_keep", "num_logits_to_keep"):
                if name in params:
                    self._keep_kwarg = name
                    break
        except (TypeError, ValueError):
            pass

    def _reset(self):
        self.past_key_values = None
        self.cached_len = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]

        # The cache holds positions [0, cached_len). A valid continuation adds
        # at least one token to exactly that prefix. Anything else (a new
        # question reusing this object, a retry, a reordered beam) means the
        # cache describes a different sequence — drop it and re-prefill.
        if self.past_key_values is not None and seq_len <= self.cached_len:
            self._reset()

        with torch.no_grad():
            if self.past_key_values is None:
                # Prefill: whole prompt + the noised image.
                new_ids = input_ids
                model_kwargs = {"pixel_values": self.pixel_values_cd}
            else:
                # Incremental: only the tokens generated since the last call.
                # Normally exactly one; a slice keeps it correct regardless.
                new_ids = input_ids[:, self.cached_len:]
                model_kwargs = {}

            cache_position = torch.arange(
                seq_len - new_ids.shape[1], seq_len, device=input_ids.device
            )
            extra = {self._keep_kwarg: 1} if self._keep_kwarg else {}

            cd_out = self.model(
                input_ids=new_ids,
                attention_mask=torch.ones(
                    (input_ids.shape[0], seq_len),
                    dtype=torch.long, device=input_ids.device,
                ),
                cache_position=cache_position,
                past_key_values=self.past_key_values if self.past_key_values is not None else DynamicCache(),
                use_cache=True,
                **model_kwargs,
                **extra,
            )

        self.past_key_values = cd_out.past_key_values
        self.cached_len = seq_len

        cd_logits = cd_out.logits[:, -1, :].to(scores.dtype)
        # Exposed for inference/test_vcd_cache_equivalence.py, which checks the
        # cached branch against a stateless re-forward.
        self._last_cd_logits = cd_logits

        ## cd_comments: Adaptive Plausibility Constraint, same formula as the original repo
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * scores - self.cd_alpha * cd_logits
        return diffs.masked_fill(scores < cutoff, -float("inf"))
