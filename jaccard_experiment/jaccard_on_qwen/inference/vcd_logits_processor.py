import torch
from transformers import LogitsProcessor


def add_diffusion_noise(image_tensor, noise_step):
    """Unchanged from the original LLaVA/VCD implementation: this just adds
    forward-diffusion Gaussian noise to a float image tensor. It is fully
    shape-agnostic (works elementwise), so it applies just as well to
    Qwen2.5-VL's patchified `pixel_values` as it did to LLaVA/CLIP's CHW
    tensors.
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
    """Visual Contrastive Decoding for Qwen2.5-VL.

    The original repo implemented VCD by monkey-patching
    `GenerationMixin.greedy_search`/`sample`. Those methods no longer exist in
    current transformers (everything routes through a single, frequently
    reshaped `_sample`), so this instead hooks in through `LogitsProcessor` —
    a stable, public extension point that `generate()` already calls once per
    step for both greedy and sampling decoding.

    Each step it reruns the *same* prompt with a diffusion-noised image
    (stateless, no KV cache reuse — simplest and most robust given how short
    POPE's single-word/phrase answers are) and combines the two logit
    streams with the Adaptive Plausibility Constraint from the VCD paper.
    """

    def __init__(self, model, pixel_values_cd, image_grid_thw, cd_alpha=1.0, cd_beta=0.2):
        self.model = model
        self.pixel_values_cd = pixel_values_cd
        self.image_grid_thw = image_grid_thw
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        attention_mask = torch.ones_like(input_ids)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        with torch.no_grad():
            cd_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=self.pixel_values_cd,
                image_grid_thw=self.image_grid_thw,
                cache_position=cache_position,
                use_cache=False,
            )
        cd_logits = cd_out.logits[:, -1, :].to(scores.dtype)

        ## cd_comments: Adaptive Plausibility Constraint, same formula as the original repo
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + scores.max(dim=-1, keepdim=True).values
        diffs = (1 + self.cd_alpha) * scores - self.cd_alpha * cd_logits
        return diffs.masked_fill(scores < cutoff, -float("inf"))
