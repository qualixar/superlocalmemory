# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Logged adjacent swaps in explicit recall: how much attention a position gets (D-46).

An agent acts more on what is shown first partly because it is shown first.
To tell the two apart, in a share ``SWAP_SHARE`` of explicit recalls two
neighbouring results among the top ``SWAP_DEPTH`` trade places: positions k and
k + 1, k drawn uniformly. Nothing is withheld or added. Both keyed draws and
the result are logged on the pull receipt, so the attention ratio
p(k + 1) / p(k) can be estimated without bias (``learning.position_bias``) and
the ranker's offline gate can weight each use by 1 / attention (D-45).

On by default, with withholding, now that the D-28 disclosure and off switch
ship (D-34's rule, D-46; ``outcomes.switch`` turns both off). A person reading (the
dashboard, a view run from it) is never swapped: only agents' recalls are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from superlocalmemory.outcomes import randomiser

logger = logging.getLogger(__name__)

#: Design constant (D-46, Varun 18 Nov): share of explicit recalls swapped.
SWAP_SHARE = 0.05
#: Design constant: swaps stay within the first five results, where agents act.
SWAP_DEPTH = 5


@dataclass(frozen=True)
class Swap:
    delivery_id: str
    #: The keyed draw deciding whether to swap; swapped when below ``share``.
    draw: float
    #: k when positions k and k + 1 (1-based) traded places, else None.
    at: int | None
    share: float


def pairs(n_results: int) -> int:
    """Neighbouring pairs a swap may choose from."""
    return max(min(SWAP_DEPTH, n_results) - 1, 0)


def decide(slm_home: Path | None, delivery_id: str, n_results: int, *,
           flags=None) -> Swap | None:
    """This delivery's swap (or its logged decision not to); None when off."""
    try:
        if slm_home is None:
            return None
        if flags is None:
            from superlocalmemory.core.feature_flags import FeatureFlags

            flags = FeatureFlags(slm_home)
        if not (flags.enabled("outcomes.swaps") and flags.observing("outcomes.receipts")):
            return None
        from superlocalmemory.core.receipt_key import load_or_create

        key = load_or_create(slm_home)
        u = randomiser.draw(key, delivery_id, "swap")
        a = randomiser.draw(key, delivery_id, "swap_at")
        n = pairs(n_results)
        at = 1 + int(a * n) if u < SWAP_SHARE and n else None
        return Swap(delivery_id, u, at, SWAP_SHARE)
    except Exception as exc:  # a swap is never worth a failed recall
        logger.debug("swap skipped: %s", exc)
        return None


def apply(results: list, swap: Swap | None) -> list:
    """The results in the order shown: a new list, the input never changed."""
    out = list(results)
    if swap is not None and swap.at is not None and len(out) > swap.at:
        i = swap.at - 1
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


__all__ = ["SWAP_DEPTH", "SWAP_SHARE", "Swap", "apply", "decide", "pairs"]
