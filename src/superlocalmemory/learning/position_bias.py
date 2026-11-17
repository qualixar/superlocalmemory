# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""How much attention each result position gets, measured from logged swaps (D-46).

Examination model: P(an agent acts on a result) = relevance x attention(position).
When positions k and k + 1 trade places at random (``outcomes.swaps``), the
same kind of item is seen at both positions, so the ratio of use rates is the
ratio of attention, r_k = p(k + 1) / p(k), with nothing else confounded:

* A = use rate of items whose place was k, shown at k + 1 (swapped at k)
      / their use rate shown at k (control deliveries: drawn, not swapped);
* B = use rate of items whose place was k + 1, shown at k + 1 (control)
      / their use rate shown at k (swapped at k);

both estimate r_k; the estimate is their geometric mean, with a log-scale
interval from the binomial variances of the four rates (delta method). The
interval level is ``MODEL_ALPHA`` split over the pairs (Bonferroni), so the
whole curve holds jointly. Attention never rises going down the list, so each
ratio and its bounds are capped at 1. p(1) = 1 and p(k) is the product of the
ratios above it, for positions 1 .. ``SWAP_DEPTH``.

A pair with an empty cell (no swaps yet, or no use in one of the four) is not
measured, and a curve with any unmeasured pair cannot be used (the offline gate
then certifies nothing: it would be guessing). Read-only on memory.db.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import NormalDist

from superlocalmemory.outcomes import effects, usage
from superlocalmemory.outcomes.swaps import SWAP_DEPTH

#: Policy: half of the overall 0.05 goes to the attention curve, half to the gate.
MODEL_ALPHA = effects.ALPHA / 2.0

_ROWS = """
SELECT d.delivery_id, d.swap_at, r.position,
       EXISTS (SELECT 1 FROM injection_usage u
                WHERE u.delivery_id = r.delivery_id AND u.fact_id = r.fact_id
                  AND u.signal IN ({signals})
                  AND julianday(u.observed_at) <= julianday(d.delivered_at) + ? / 24.0),
       (SELECT COUNT(*) FROM injection_receipts c
         WHERE c.delivery_id = d.delivery_id AND c.delivered = 1)
  FROM injection_deliveries d
  JOIN injection_receipts r ON r.delivery_id = d.delivery_id AND r.delivered = 1
 WHERE d.profile_id = ? AND d.swap_draw IS NOT NULL AND r.position <= ?
   AND julianday(d.delivered_at) + ? / 24.0 <= julianday(?)
"""


@dataclass(frozen=True)
class Ratio:
    pair: int                     # k: attention(k + 1) / attention(k)
    estimate: float | None
    low: float | None
    high: float | None
    cells: tuple[tuple[int, int], ...]   # (uses, shown) for a_s, a_u, b_u, b_s

    @property
    def measured(self) -> bool:
        return self.estimate is not None


@dataclass(frozen=True)
class PositionModel:
    ratios: tuple[Ratio, ...] = field(default_factory=tuple)

    @property
    def measured(self) -> bool:
        return bool(self.ratios) and all(r.measured for r in self.ratios)

    def attention(self, which: str = "estimate") -> list[float]:
        """p(1) .. p(SWAP_DEPTH) under the estimate, or the ``low`` (steepest)
        or ``high`` (flattest) end of every ratio's interval."""
        if not self.measured:
            raise ValueError("position attention is not measured yet")
        p = [1.0]
        for r in self.ratios:
            p.append(p[-1] * getattr(r, which))
        return p


def _cells(rows) -> dict[int, list[list[int]]]:
    """pair k -> [[uses, shown] for a_s, a_u, b_u, b_s]."""
    cells = {k: [[0, 0] for _ in range(4)] for k in range(1, SWAP_DEPTH)}
    for _did, swap_at, shown, used, n in rows:
        for k in cells:
            if swap_at is None and n >= k + 1:          # control with this pair present
                slot = 1 if shown == k else 2 if shown == k + 1 else None
            elif swap_at == k:                          # swapped at this pair
                slot = 0 if shown == k + 1 else 3 if shown == k else None
            else:
                slot = None
            if slot is not None:
                cells[k][slot][0] += int(bool(used))
                cells[k][slot][1] += 1
    return cells


def _ratio(k: int, c: list[list[int]], z: float) -> Ratio:
    cells = tuple((u, n) for u, n in c)
    if any(u == 0 or n == 0 for u, n in c):
        return Ratio(k, None, None, None, cells)
    rate = [u / n for u, n in c]
    log_r = 0.5 * (math.log(rate[0] / rate[1]) + math.log(rate[2] / rate[3]))
    var = 0.25 * sum((1.0 - p) / (n * p) for p, (_u, n) in zip(rate, c, strict=True))
    se = math.sqrt(var)
    cap = lambda x: min(x, 1.0)  # noqa: E731 - attention never rises going down
    return Ratio(k, cap(math.exp(log_r)), cap(math.exp(log_r - z * se)),
                 cap(math.exp(log_r + z * se)), cells)


def estimate(conn: sqlite3.Connection, profile_id: str, *,
             as_of: datetime | None = None) -> PositionModel:
    """The attention curve for one profile from its logged swaps (window closed)."""
    sql = _ROWS.format(signals=", ".join("?" * len(effects.USED_SIGNALS)))
    when = usage.stamp(as_of or datetime.now(UTC))
    try:
        rows = conn.execute(sql, (*effects.USED_SIGNALS, effects.OUTCOME_WINDOW_H, profile_id,
                                  SWAP_DEPTH, effects.OUTCOME_WINDOW_H, when)).fetchall()
    except sqlite3.OperationalError:
        return PositionModel(tuple(Ratio(k, None, None, None, ()) for k in range(1, SWAP_DEPTH)))
    pairs = SWAP_DEPTH - 1
    z = NormalDist().inv_cdf(1.0 - MODEL_ALPHA / (2.0 * pairs))
    cells = _cells(rows)
    return PositionModel(tuple(_ratio(k, cells[k], z) for k in sorted(cells)))


__all__ = ["MODEL_ALPHA", "PositionModel", "Ratio", "estimate"]
