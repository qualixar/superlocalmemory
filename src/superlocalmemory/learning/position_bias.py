# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""How much attention each result position gets, measured from logged swaps (D-46).

**Model and Assumptions.** Examination model: P(an agent acts on a result) =
relevance x attention(position). The estimate is consistent for the attention
ratios under these assumptions, stated rather than hidden (it is not "unbiased"
in any model-free sense):

1. *Examination:* use = relevance x attention(position), attention depends on the
   position only, not on which item sits there or on its list;
2. *Random swaps:* in a share ``SWAP_SHARE`` of agent recalls positions k and k + 1
   (k uniform over the list's ``pairs(n)``) trade places, drawn by a keyed hash
   independent of the results, so a swapped delivery and a control delivery of the
   same list length hold items of the same kind;
3. *No interference:* a swap changes where two items sit and nothing else about the
   agent's behaviour (no other result's use depends on the swap);
4. *Use is observed alike* at every position and arm (detection recall does not depend
   on the position).

**The four cells.** Where positions k and k + 1 trade places, the same kind of item
is seen at both positions, so the ratio of use rates is the ratio of attention,
r_k = p(k + 1) / p(k):

* A = use rate of items whose place was k, shown at k + 1 (swapped at k)
      / their use rate shown at k (control deliveries: drawn, not swapped);
* B = use rate of items whose place was k + 1, shown at k + 1 (control)
      / their use rate shown at k (swapped at k);

both estimate r_k; the estimate is their geometric mean, with a log-scale
interval from the variances of the four rates (delta method). The interval level
is ``MODEL_ALPHA`` split over the pairs (Bonferroni), so the whole curve holds
jointly. Attention never rises going down the list, so each ratio and its bounds
are capped at 1. p(1) = 1 and p(k) is the product of the ratios above it, for
positions 1 .. ``SWAP_DEPTH``.

**List length (A-20).** A swap at pair k has probability ``SWAP_SHARE / pairs(n)``
for a list of n results, so swapped cells over-represent short lists relative to
controls. If the use rate depends on the list length, an unweighted ratio is
biased. Each swapped delivery therefore carries weight ``pairs(n)`` (the inverse of
its share of the swap probability; controls weigh 1), so both arms stand for the
same mix of list lengths, and rates are weighted means with weighted variances.

**Empty cells (A-20).** A pair was dropped when any of its four cells had no use,
which keeps exactly the pairs where uses happened to occur: a selection on the
outcome. Now a pair is measured when every cell was *shown* at least
``MIN_CELL_SHOWN`` times (a count fixed by the design, not by what agents did), and
a rate is the weighted mean with one added pseudo-observation (half a use, half a
non-use) so a zero count gives a finite rate and a wide interval rather than a
dropped pair. The smoothing pulls a sparse cell's rate toward a half, so a small
cell's interval is approximate (delta method on a smoothed rate): the limitation
is real and is why ``MIN_CELL_SHOWN`` exists.

A pair with a thin cell is not measured, and a curve with any unmeasured pair
cannot be used (the offline gate then certifies nothing: it would be guessing).
Read-only on memory.db.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import NormalDist

from superlocalmemory.outcomes import effects, usage
from superlocalmemory.outcomes.swaps import SWAP_DEPTH, pairs

#: Policy: half of the overall 0.05 goes to the attention curve, half to the gate.
MODEL_ALPHA = effects.ALPHA / 2.0
#: A pair is measured only when each of its four cells holds at least this many shown
#: items. A design count, never an outcome count (A-20). Routing for the normal
#: approximation of the smoothed rates: the interval is wide below it anyway.
MIN_CELL_SHOWN = 30

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
class Cell:
    """Weighted use counts of one of a pair's four groups."""

    uses: int = 0
    shown: int = 0
    w: float = 0.0          # sum of weights
    wy: float = 0.0         # sum of weight x use
    w2: float = 0.0         # sum of squared weights
    w2y: float = 0.0        # sum of squared weight x use

    @classmethod
    def of(cls, uses: int, shown: int, weight: float = 1.0) -> Cell:
        """A cell of ``shown`` items, ``uses`` of them used, all with one weight."""
        return cls(uses, shown, weight * shown, weight * uses, weight ** 2 * shown,
                   weight ** 2 * uses)

    def add(self, used: bool, weight: float) -> Cell:
        y = 1.0 if used else 0.0
        return Cell(self.uses + int(used), self.shown + 1, self.w + weight, self.wy + weight * y,
                    self.w2 + weight ** 2, self.w2y + weight ** 2 * y)

    def rate(self) -> tuple[float, float]:
        """(smoothed weighted use rate, its variance), one pseudo-observation added
        (half a use, half a non-use, at the mean weight): defined for every cell with
        something shown, with no selection on the outcome."""
        mean_w = self.w / self.shown
        total = self.w + mean_w
        r = (self.wy + 0.5 * mean_w) / total
        pseudo = (0.5 * mean_w) ** 2            # each pseudo-observation has weight mean_w / 2
        var = (self.w2y * (1 - r) ** 2 + (self.w2 - self.w2y) * r ** 2
               + pseudo * ((1 - r) ** 2 + r ** 2)) / total ** 2
        return r, var


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


def _cells(rows) -> dict[int, list[Cell]]:
    """pair k -> [a_s, a_u, b_u, b_s] cells, weighted by list length (A-20)."""
    cells = {k: [Cell() for _ in range(4)] for k in range(1, SWAP_DEPTH)}
    for _did, swap_at, shown, used, n in rows:
        for k in cells:
            if swap_at is None and n >= k + 1:          # control with this pair present
                slot, weight = (1 if shown == k else 2 if shown == k + 1 else None), 1.0
            elif swap_at == k:                          # swapped at this pair
                slot, weight = (0 if shown == k + 1 else 3 if shown == k else None), \
                    float(max(pairs(n), 1))
            else:
                slot, weight = None, 1.0
            if slot is not None:
                cells[k][slot] = cells[k][slot].add(bool(used), weight)
    return cells


def _ratio(k: int, c: list[Cell], z: float) -> Ratio:
    shown = tuple((x.uses, x.shown) for x in c)
    if any(x.shown < MIN_CELL_SHOWN for x in c):
        return Ratio(k, None, None, None, shown)
    rate, var = zip(*(x.rate() for x in c), strict=True)
    log_r = 0.5 * (math.log(rate[0] / rate[1]) + math.log(rate[2] / rate[3]))
    se = 0.5 * math.sqrt(sum(v / (r * r) for r, v in zip(rate, var, strict=True)))
    cap = lambda x: min(x, 1.0)  # noqa: E731 - attention never rises going down
    return Ratio(k, cap(math.exp(log_r)), cap(math.exp(log_r - z * se)),
                 cap(math.exp(log_r + z * se)), shown)


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


__all__ = ["MIN_CELL_SHOWN", "MODEL_ALPHA", "Cell", "PositionModel", "Ratio", "estimate"]
