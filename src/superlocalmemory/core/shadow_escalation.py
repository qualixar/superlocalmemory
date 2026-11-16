# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""How often a stronger judge, asked in shadow, is sure of something else (M-06, R-30).

The ladder asks a pair a tier was sure about of the next tier up with chance ε
(``judge_ladder._shadow_escalate``) and receipts the answer under
``<population>:shadow``. For each (lower tier, model) → (upper tier, model) this
counts the shadow pairs, how many the upper tier was sure of a *different*
verdict on (its confidence at or above its routing threshold), and the exact
one-sided bounds on that rate at ``risk_policy.DELTA``.

It is a warning, not an error rate: the upper judge can be the wrong one. People
settle it through the blind audit, which draws shadow receipts like any other.
Counts and ids only, never memory text.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass

from superlocalmemory.core.audit_sampler import NOT_ANSWERS
from superlocalmemory.core.judge_calibration import clopper_pearson_lower, clopper_pearson_upper
from superlocalmemory.core.judge_ladder import SHADOW_SUFFIX
from superlocalmemory.core.risk_policy import DELTA

_PAIRS = """
SELECT s.profile_id, l.tier, l.model, s.tier, s.model, l.inputs_hash, l.verdict, s.verdict, s.confidence,
       s.threshold
  FROM judge_receipts s
  JOIN judge_receipts l
    ON l.profile_id = s.profile_id AND l.question_id = s.question_id
   AND l.question_version = s.question_version AND l.inputs_hash = s.inputs_hash
   AND l.population || ? = s.population AND l.tier_index = s.tier_index - 1
   AND l.accepted = 1
 WHERE (? IS NULL OR s.profile_id = ?) AND s.population LIKE ? AND s.failure_kind IS NULL
 ORDER BY s.decided_at
"""


@dataclass(frozen=True)
class Disagreement:
    lower_tier: str
    lower_model: str
    upper_tier: str
    upper_model: str
    pairs: int
    disagreed: int
    rate_lower: float
    rate_upper: float

    def as_dict(self) -> dict:
        return asdict(self)


def disagreements(conn: sqlite3.Connection, profile_id: str | None) -> list[Disagreement]:
    """Per tier step, over distinct pairs the upper tier really answered in shadow
    (one profile, or every profile when ``None``)."""
    seen: dict[tuple, dict[str, bool]] = {}
    try:
        rows = conn.execute(_PAIRS, (SHADOW_SUFFIX, profile_id, profile_id,
                                     "%" + SHADOW_SUFFIX)).fetchall()
    except sqlite3.OperationalError:  # no receipts table yet
        return []
    for profile, lt, lm, ut, um, pair, lower, upper, conf, threshold in rows:
        if upper in NOT_ANSWERS:
            continue
        sure = conf is not None and threshold is not None and conf >= threshold
        step = seen.setdefault((lt, lm, ut, um), {})
        step[(profile, pair)] = step.get((profile, pair), False) or (sure and upper != lower)
    out = []
    for (lt, lm, ut, um), pairs in sorted(seen.items()):
        n, k = len(pairs), sum(pairs.values())
        out.append(Disagreement(lt, lm, ut, um, n, k, clopper_pearson_lower(k, n, DELTA),
                                clopper_pearson_upper(k, n, DELTA)))
    return out


__all__ = ["Disagreement", "disagreements"]
