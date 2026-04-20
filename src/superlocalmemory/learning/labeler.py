# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §4.7

"""Integer-label mapping for LightGBM ``lambdarank`` training.

LLD reference: ``.backup/active-brain/lld/LLD-02-signal-pipeline-and-lightgbm.md``
Section 4.7 — single source of truth for outcome-reward / position → int label.

Rules:
    - Labels are integers in ``[0, 4]`` (5 relevance tiers).
    - ``label_gain`` length MUST be ``>= max(label) + 1``; we ship
      ``label_gain=[0, 1, 3, 7, 15]`` (length 5).
    - Prefer ``outcome_reward`` (v3.4.22) if present; else position proxy.
    - ``NaN`` reward is treated as missing and falls through to position.
"""

from __future__ import annotations

import math
from typing import Any

# Exactly five tiers — do not widen without updating ``label_gain`` callers.
_LABEL_GAIN: tuple[int, ...] = (0, 1, 3, 7, 15)


def label_gain() -> list[int]:
    """Return the canonical ``label_gain`` list for LightGBM.

    Length is ``max(label) + 1 = 5``. Must be passed verbatim to
    ``lgb.train(params=..., label_gain=...)``.
    """
    return list(_LABEL_GAIN)


def _coerce_reward(raw: Any) -> float | None:
    """Coerce a reward-ish input to float, rejecting None / NaN."""
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def label_for_row(row: dict) -> int:
    """Map a training row to integer relevance in ``[0, 4]``.

    Args:
        row: Dict with optional ``outcome_reward`` (float in [0, 1]) and
             ``position`` (int, 0-based rank at recall time).

    Returns:
        An integer label in ``[0, 4]``. Higher = more relevant.
    """
    reward = _coerce_reward(row.get("outcome_reward"))
    if reward is not None:
        if reward >= 0.90:
            return 4
        if reward >= 0.60:
            return 3
        if reward >= 0.30:
            return 2
        if reward > 0.00:
            return 1
        return 0

    # 3.4.22 proxy: position (0 = best, higher = worse).
    try:
        pos = int(row.get("position", 99))
    except (TypeError, ValueError):
        return 0
    if pos == 0:
        return 4
    if pos <= 2:
        return 3
    if pos <= 4:
        return 2
    if pos <= 9:
        return 1
    return 0


__all__ = ("label_for_row", "label_gain")
