# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22

"""Deterministic scoring for the LoCoMo reproducibility harness.

Two metrics:
  - ``exact_match(pred, gold)``  — string equality after whitespace /
    case / punctuation normalisation. LoCoMo gold answers are short,
    so exact-match is the standard reference.
  - ``mrr_at_k(ranked_ids, gold_ids, k)`` — mean reciprocal rank on
    the first position where a gold fact_id appears in the ranked
    retrieval list. MRR is the LoCoMo paper's headline metric.
"""

from __future__ import annotations

import re
from typing import Iterable

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    return _NORM_RE.sub(" ", s.lower()).strip()


def exact_match(pred: str, gold: str) -> bool:
    """Normalised exact-match. Short LoCoMo answers are safe here."""
    return normalize(pred) == normalize(gold)


def mrr_at_k(
    ranked_ids: Iterable[str],
    gold_ids: Iterable[str],
    k: int = 10,
) -> float:
    """Mean reciprocal rank at K over a single query.

    Returns ``1 / (rank+1)`` for the first match, or 0 if none in top-k.
    """
    gold_set = {str(g) for g in gold_ids}
    for i, rid in enumerate(ranked_ids):
        if i >= k:
            return 0.0
        if str(rid) in gold_set:
            return 1.0 / (i + 1)
    return 0.0


__all__ = ("normalize", "exact_match", "mrr_at_k")
