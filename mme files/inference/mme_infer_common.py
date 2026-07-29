"""Shared pure helper functions for the MME inference scripts.

These helpers are factored out of the per-method ``mme_infer_*.py`` scripts so
that the input-driven logic can be imported and property-tested in isolation,
without requiring a live LLaVA model. The chunking helpers are copied verbatim
from the POPE inference skeleton to keep the chunking behavior identical.
"""

import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def derive_do_sample(temperature):
    """Return the ``do_sample`` flag implied by ``temperature``.

    Sampling is enabled exactly when the temperature is greater than 0; a
    temperature of 0 selects greedy decoding (``do_sample = False``).
    """
    return temperature > 0


def build_answer_record(question, output_text, answer_id, model_id):
    """Build an answer-file record for a single question.

    The record uses the input question's ``question_id`` (preserved verbatim)
    and ``text`` (as the prompt), the generated ``output_text``, and the passed
    ``answer_id`` and ``model_id``. ``metadata`` is always an empty dict. The
    returned dict has exactly the keys
    ``{question_id, prompt, text, answer_id, model_id, metadata}``.
    """
    return {
        "question_id": question["question_id"],
        "prompt": question["text"],
        "text": output_text,
        "answer_id": answer_id,
        "model_id": model_id,
        "metadata": {},
    }


def validate_cd_flags(args):
    """Reject invocations that select more than one contrastive-decoding option.

    Inspects the boolean attributes ``use_vcd``, ``use_icd`` and ``use_sid`` on
    ``args``. If two or more are set, raises a descriptive ``ValueError`` naming
    the selected flags. Zero or one selected flag never raises.
    """
    selected = [
        name
        for name, on in (
            ("--use-vcd", args.use_vcd),
            ("--use-icd", args.use_icd),
            ("--use-sid", args.use_sid),
        )
        if on
    ]
    if len(selected) > 1:
        raise ValueError(
            f"Conflicting contrastive-decoding options selected: "
            f"{', '.join(selected)}. Choose exactly one."
        )
