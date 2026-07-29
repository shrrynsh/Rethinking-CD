try:
    # This pulls in llava.model.language_model.llava_mpt, which imports
    # legacy MPT modeling code that reaches into private transformers
    # internals (e.g. transformers.models.bloom.modeling_bloom._expand_mask)
    # removed in newer transformers releases. Nothing in the inference path
    # needs this package-root re-export (only llava/model/apply_delta.py
    # does), so don't let a merely-importing-the-package script (e.g. one
    # that only wants llava.utils) crash because of it.
    from .model import LlavaLlamaForCausalLM
except ImportError:
    pass
