# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which audit question to ask a person first (W2, innovation 5: value of information).

A person's time is the scarce input, so the audit queue asks first where an
answer is worth most:

1. **It moves a certificate.** The acting gate reads each cell's drawn receipts
   in decision order and stops at the first one without a label (W0-03), so a
   label on any later receipt changes nothing until the earlier ones have one.
   The one item per cell that moves it is that cell's oldest unlabelled drawn
   receipt (its *head*), and only in a regime that can certify (W0-04). Heads
   come first, the cell with the fewest labels still needed first: closer to a
   certificate is closer to a judge that may act.
2. **It guards against the costliest mistake.** Everything else is ordered by
   harm x doubt x use: a retirement (``replaces``) is weighted 3 against 1 for an
   answer that leaves both memories true; doubt is 1 - the judge's confidence
   (0.5 for a rule, which has none); use is 1 + ln(1 + access_count) of the
   most-used memory involved.

Ties keep the oldest first. Nothing here reads memory text or a label's value;
it only orders. Without receipts it returns the input order.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence

from superlocalmemory.core import risk_policy
from superlocalmemory.core.judge_calibration import CERTIFIABLE_REGIMES, clean_decisions_needed

#: The weight of a wrong retirement against a wrong "keep both" (harm).
HARM = {"replaces": 3.0}

_HEADS = """
SELECT inputs_hash, question_id, question_version, tier, model, verdict, population,
       batching_regime
  FROM (SELECT r.*, ROW_NUMBER() OVER (
               PARTITION BY r.question_id, r.question_version, r.tier, r.model, r.verdict,
                            r.population, r.batching_regime
               ORDER BY r.decided_at, r.receipt_id) AS rn
          FROM judge_receipts r
         WHERE r.profile_id = ? AND r.audit_selected = 1 AND r.inputs_hash IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM judge_labels l WHERE l.receipt_id = r.receipt_id
                           AND l.source = 'audit'))
 WHERE rn = 1
"""

_LABELLED = """
SELECT COUNT(*) FROM judge_receipts r
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ? AND r.tier = ?
   AND r.model = ? AND r.verdict = ? AND r.population = ? AND r.batching_regime = ?
   AND r.audit_selected = 1
   AND EXISTS (SELECT 1 FROM judge_labels l WHERE l.receipt_id = r.receipt_id
               AND l.source = 'audit')
"""


def _heads(conn: sqlite3.Connection, profile_id: str) -> dict[str, int]:
    """inputs_hash -> labels its cell still needs, for each certifiable cell's head."""
    out: dict[str, int] = {}
    for h, qid, ver, tier, model, verdict, pop, regime in conn.execute(_HEADS, (profile_id,)):
        if regime not in CERTIFIABLE_REGIMES:
            continue
        have = conn.execute(_LABELLED, (profile_id, qid, ver, tier, model, verdict, pop,
                                        regime)).fetchone()[0]
        need = clean_decisions_needed(risk_policy.alpha_for(verdict), risk_policy.DELTA)
        remaining = max(0, need - int(have))
        out[h] = min(out.get(h, remaining), remaining)
    return out


def _risk(conn: sqlite3.Connection, profile_id: str, inputs_hash: str,
          fact_ids: Sequence[str]) -> float:
    rows = conn.execute(
        "SELECT verdict, confidence FROM judge_receipts WHERE profile_id = ? AND inputs_hash = ? "
        "AND verdict NOT IN ('no_answer', 'tier_failed', 'abstain')",
        (profile_id, inputs_hash)).fetchall()
    if not rows:
        return 0.0
    harm = max(HARM.get(v, 1.0) for v, _ in rows)
    doubt = max(0.5 if c is None else 1.0 - float(c) for _, c in rows)
    use = 1.0
    if fact_ids:
        marks = ",".join("?" * len(fact_ids))
        try:
            top = conn.execute(f"SELECT MAX(access_count) FROM atomic_facts "  # noqa: S608
                               f"WHERE fact_id IN ({marks})", tuple(fact_ids)).fetchone()[0]
        except sqlite3.OperationalError:
            top = None
        use = 1.0 + math.log1p(max(0, int(top or 0)))
    return harm * max(doubt, 0.0) * use


def order(conn: sqlite3.Connection, profile_id: str, rows: Sequence[tuple]) -> list[tuple]:
    """``rows`` are (inputs_hash, question_id, fact_ids_json, decided_at), oldest first."""
    try:
        heads = _heads(conn, profile_id)
    except sqlite3.OperationalError:
        return list(rows)

    def key(indexed: tuple[int, tuple]) -> tuple:
        i, (h, _qid, fids, _when) = indexed
        risk = _risk(conn, profile_id, h, json.loads(fids or "[]"))
        if h in heads:
            return (0, heads[h], -risk, i)
        return (1, 0, -risk, i)

    return [row for _, row in sorted(enumerate(rows), key=key)]


__all__ = ["HARM", "order"]
