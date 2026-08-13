"""
Shared text normalization for InternVL3 Jaccard analysis.

InternVL3-8B is built on a Qwen2.5-7B language backbone and inherits the same
habit of partially ignoring the no-markdown system prompt: it still produces
**bold**, # headers, - bullets, and double newlines stochastically. Stripping
these before tokenizing ensures Jaccard measures content/semantic overlap, not
incidental formatting choices.

Kept identical to jaccard_on_qwen/analysis/text_utils.py so the numbers are
comparable across models.
"""

import re


def normalize(text: str) -> str:
    # Remove **bold** and *italic* markers, keep inner text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*',     r'\1', text, flags=re.DOTALL)
    # Remove markdown header markers (## Title → Title)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bullet/dash list markers at line start
    text = re.sub(r'^\s*[-]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()
