# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cut long page text into parts: at paragraphs first, then sentences, then anywhere."""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pack(units: list[str], sep: str, limit: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(sep) + len(unit) > limit:
            parts.append(current)
            current = unit
        else:
            current = f"{current}{sep}{unit}" if current else unit
    if current:
        parts.append(current)
    return parts


def _hard_cut(text: str, limit: int) -> list[str]:
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def _split_paragraph(paragraph: str, limit: int) -> list[str]:
    """Pieces of one over-long paragraph, each within ``limit``."""
    units: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE.split(paragraph)):
        if sentence:
            units += _hard_cut(sentence, limit) if len(sentence) > limit else [sentence]
    return _pack(units, " ", limit)


def chunk_text(text: str, limit: int) -> list[str]:
    """Parts of at most ``limit`` characters; empty for blank text."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    units: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(text)):
        if paragraph:
            units += _split_paragraph(paragraph, limit) if len(paragraph) > limit else [paragraph]
    return _pack(units, "\n\n", limit)
