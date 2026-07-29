import os
"""
Shared text normalization for Qwen Jaccard analysis.

Qwen2.5-VL partially ignores the no-markdown system prompt and still produces
**bold**, # headers, - bullets, and double newlines stochastically.
Stripping these before tokenizing ensures Jaccard measures content/semantic
overlap, not incidental formatting choices.
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
