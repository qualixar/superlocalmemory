#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Text processing utilities for synthetic bootstrap.

Simple keyword extraction and text processing functions
with no external NLP dependencies.
"""

import re
from collections import Counter
from typing import List

from .constants import STOPWORDS, MIN_KEYWORD_LENGTH


def extract_keywords(content: str, top_n: int = 3) -> List[str]:
    """
    Extract meaningful keywords from memory content.

    Simple frequency-based extraction:
    1. Tokenize (alphanumeric words)
    2. Remove stopwords and short words
    3. Return top N by frequency

    No external NLP dependencies — just regex + counter.

    Args:
        content: Text content to extract keywords from.
        top_n: Number of top keywords to return.

    Returns:
        List of top N keywords by frequency.
    """
    if not content:
        return []

    # Tokenize: extract alphanumeric words
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9_.-]*[a-zA-Z0-9]|[a-zA-Z]', content.lower())

    # Filter stopwords and short words
    meaningful = [
        w for w in words
        if w not in STOPWORDS and len(w) >= MIN_KEYWORD_LENGTH
    ]

    if not meaningful:
        return []

    # Count and return top N
    counter = Counter(meaningful)
    return [word for word, _count in counter.most_common(top_n)]


def clean_fts_query(query: str) -> str:
    """
    Clean and prepare query for FTS5 search.

    Extracts word tokens and joins them with OR for FTS5 MATCH syntax.

    Args:
        query: Raw query string.

    Returns:
        FTS5-compatible query string, or empty string if no valid tokens.
    """
    fts_tokens = re.findall(r'\w+', query)
    if not fts_tokens:
        return ''
    return ' OR '.join(fts_tokens)
