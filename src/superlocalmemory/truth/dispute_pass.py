# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open slot disputes, settled by the judge ladder in the background.

Each open dispute is one ``SLOT_RELATION`` question: what does the newer memory
do to the older one? The answer, once a tier is sure enough for its label:

* ``replaces``  -> the newer value wins; the older memory is retired through
  the correction ledger (only with ``truth.slots.auto_apply``; otherwise the
  dispute stays open with its receipt, for review);
* ``same``      -> the newer memory is a duplicate; nothing is retired;
* ``both_true`` / ``separate_events`` -> both stay current;
* ``unrelated`` -> the slot key was wrong; both stay current.

``judge.ladder`` in shadow asks and records receipts but changes nothing; on,
it acts -- but only where the store's own labels support it
(``core/judge_calibration.ActingGate``: a threshold chosen by fixed-sequence
testing so the error rate among acted-on decisions is <= alpha with
probability >= 1 - delta). Tier 0 rules never retire anything and act on their
non-retiring answers without a gate. Measured 15 Nov 2026 on the maintainer's
backlog, Jev's "replaces" had 20-40% precision and no threshold passed, so
retirement by a model stays off until labels say otherwise.

A dispute the ladder already asked about under this question version is not
asked again: an abstention is final until a person looks, so a store
never pays twice for the same unanswerable pair.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from superlocalmemory.core import recall_gate
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import Item, JudgeLadder, write_receipts
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.truth import slot_store
from superlocalmemory.truth.slot_resolver import retire_superseded

logger = logging.getLogger(__name__)

SUBJECT_KIND = "slot_dispute"
_RESOLUTION = {"replaces": "b", "same": "same", "both_true": "both",
               "separate_events": "both", "unrelated": "unrelated"}

_OPEN = """
SELECT d.dispute_id, d.profile_id, fa.fact_id, fa.content, fb.fact_id, fb.content, fb.scope
  FROM slot_disputes d
  JOIN slot_revisions ra ON ra.revision_id = d.revision_a
  JOIN slot_revisions rb ON rb.revision_id = d.revision_b
  JOIN atomic_facts fa ON fa.fact_id = ra.fact_id
  JOIN atomic_facts fb ON fb.fact_id = rb.fact_id
 WHERE d.state = 'open'
   AND NOT EXISTS (SELECT 1 FROM judge_receipts j WHERE j.subject_kind = ?
                   AND j.subject_id = d.dispute_id AND j.question_id = ?
                   AND j.question_version = ?)
 ORDER BY d.opened_at
 LIMIT ?
"""


class DisputePass:
    def __init__(self, db: Any, flags: FeatureFlags, ladder: JudgeLadder, *,
                 limit: int = 32) -> None:
        self._db = db
        self._flags = flags
        self._ladder = ladder
        self._limit = limit
        self.counts: Counter[str] = Counter()

    def run(self) -> dict[str, int]:
        if not self._flags.observing("judge.ladder"):
            return {}
        with self._db.raw_connection() as conn:  # 1. read, then let go of the connection
            rows = conn.execute(_OPEN, (SUBJECT_KIND, SLOT_RELATION.question_id,
                                        SLOT_RELATION.version, self._limit)).fetchall()
        if not rows:
            return {}
        items = [Item(SUBJECT_KIND, r[0], r[1], {"older": r[3], "newer": r[5]}) for r in rows]
        with recall_gate.background_work():  # 2. judge with no lock held: saves never wait
            decisions, receipts = self._ladder.decide(SLOT_RELATION, items)
        with self._db.raw_connection() as conn:  # 3. one short write
            write_receipts(conn, receipts)
            self._apply(conn, rows, decisions)
        logger.info("Dispute pass: %s", dict(self.counts))
        return dict(self.counts)

    def _apply(self, conn: Any, rows: list, decisions: list) -> None:
        acting = self._flags.enabled("judge.ladder")
        auto_apply = (self._flags.enabled("truth.slots.auto_apply")
                      and self._flags.enabled("truth.slots"))
        gates: dict[str, ActingGate] = {}
        for row, decision in zip(rows, decisions):
            self.counts[f"{decision.tier}:{decision.verdict}"] += 1
            if not acting or decision.abstained:
                continue
            gate = gates.setdefault(row[1], ActingGate(
                conn, profile_id=row[1], question_id=SLOT_RELATION.question_id,
                version=SLOT_RELATION.version))
            if decision.tier != "rules" and not gate.allows(
                    decision.tier, decision.model, decision.verdict, decision.confidence):
                self.counts["held_uncalibrated"] += 1
                continue
            self._act(conn, row, decision.verdict, auto_apply)

    def _act(self, conn: Any, row: tuple, verdict: str, auto_apply: bool) -> None:
        dispute_id, profile_id, older, _a, newer, _b, scope = row
        if verdict == "replaces" and not auto_apply:
            self.counts["held_for_review"] += 1
            return
        conn.execute("SAVEPOINT dispute")
        try:
            if verdict == "replaces":
                ledger, _case = retire_superseded(conn, profile_id=profile_id, scope=scope,
                                                  predecessor=older, successor=newer)
                if ledger != "applied":
                    conn.execute("ROLLBACK TO dispute")
                    self.counts[f"ledger_{ledger}"] += 1
                    return
            slot_store.resolve_dispute(conn, dispute_id, _RESOLUTION[verdict],
                                       resolved_by="judge_ladder")
            self.counts[f"resolved_{verdict}"] += 1
        except slot_store.StaleDisputeError:
            self.counts["stale_dismissed"] += 1  # keep the dismissal it wrote
        except Exception as exc:  # noqa: BLE001 - one dispute must not stall the pass
            conn.execute("ROLLBACK TO dispute")
            self.counts["error"] += 1
            logger.debug("dispute %s not settled: %s", dispute_id, exc)
        finally:
            conn.execute("RELEASE dispute")


__all__ = ["DisputePass", "SUBJECT_KIND"]
