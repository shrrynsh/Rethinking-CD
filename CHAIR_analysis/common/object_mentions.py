"""
object_mentions.py -- maps CHAIR object mentions (real and hallucinated),
plus number-words and colour-words, in a generated caption back to the exact
decoding step(s) that produced them.

Used by generate_report.py to pick which decoding steps to tabulate logits
for, instead of a generic "did CD flip the choice" heuristic.
"""

import re
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: E402
if "llava" in CONFIG_MAPPING._mapping:
    del CONFIG_MAPPING._mapping["llava"]
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

from transformers import AutoTokenizer  # noqa: E402
from eval.chair import CHAIR  # noqa: E402

_WORD_RE = re.compile(r"[a-z]+")

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "silver", "gold", "golden", "tan",
    "beige", "maroon", "navy", "teal", "violet", "cream", "khaki",
}
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "several", "few", "many", "couple",
    "multiple", "dozen", "single", "pair", "numerous", "group",
}


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


def load_chair(image_ids, coco_path):
    chair = CHAIR(imids=image_ids, coco_path=coco_path)
    chair.get_annotations()
    return chair


def step_char_offsets(tokenizer, chosen_ids):
    """Per-step (char_start, char_end) into the fully-decoded caption, from
    incrementally decoding ids[:i+1] and diffing string length -- robust to
    tokenizer normalisation quirks (leading spaces, merged subwords, etc.)."""
    offsets, cum_text, ids_so_far = [], "", []
    for tid in chosen_ids:
        ids_so_far.append(tid)
        new_text = tokenizer.decode(ids_so_far, skip_special_tokens=True)
        offsets.append((len(cum_text), len(new_text)))
        cum_text = new_text
    return offsets, cum_text


def _steps_for_span(offsets, span):
    return [i for i, (s, e) in enumerate(offsets) if e > span[0] and s < span[1]]


def _object_mentions(chair, image_id, cum_text, offsets):
    words, node_words, idxs, _raw = chair.caption_to_words(cum_text)
    gt_objects = chair.imid_to_objects.get(image_id, set())
    word_spans = [m.span() for m in _WORD_RE.finditer(cum_text.lower())]

    mentions = []
    for w, nw, idx in zip(words, node_words, idxs):
        n = len(w.split())
        if idx + n - 1 >= len(word_spans):
            continue
        span = (word_spans[idx][0], word_spans[idx + n - 1][1])
        step_indices = _steps_for_span(offsets, span)
        if not step_indices:
            continue
        mentions.append({
            "word": w, "node_word": nw, "category": "object",
            "hallucinated": nw not in gt_objects,
            "char_span": span, "step_indices": step_indices,
        })
    return mentions


def _attribute_mentions(cum_text, offsets, vocab, category):
    mentions = []
    for m in _WORD_RE.finditer(cum_text.lower()):
        w = m.group()
        if w not in vocab:
            continue
        step_indices = _steps_for_span(offsets, m.span())
        if not step_indices:
            continue
        mentions.append({
            "word": w, "node_word": w, "category": category,
            "hallucinated": None,
            "char_span": m.span(), "step_indices": step_indices,
        })
    return mentions


def all_mentions(chair, tokenizer, image_id, chosen_ids):
    """Returns every object/number/colour mention in one caption:
        {word, node_word, category, hallucinated, char_span, step_indices}
    ``category`` is "object" (hallucinated is bool), "number", or "color"
    (hallucinated is None -- no ground truth to check these against).
    ``step_indices`` are the decoding steps whose generated characters
    overlap the mention's character span in the caption text.
    """
    offsets, cum_text = step_char_offsets(tokenizer, chosen_ids)
    mentions = _object_mentions(chair, image_id, cum_text, offsets)
    mentions += _attribute_mentions(cum_text, offsets, NUMBER_WORDS, "number")
    mentions += _attribute_mentions(cum_text, offsets, COLOR_WORDS, "color")
    mentions.sort(key=lambda m: m["char_span"][0])
    return mentions


# Backwards-compatible alias (older calls expected only CHAIR object mentions)
def object_mentions(chair, tokenizer, image_id, chosen_ids):
    offsets, cum_text = step_char_offsets(tokenizer, chosen_ids)
    return _object_mentions(chair, image_id, cum_text, offsets)
