# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The correction backlog, judged by the ladder in the background (D-13, W2).

Modes B and C file a ``proposed`` correction case whenever the save path only
guesses that a newer memory updates an older one; nobody reviews them, and a
real store holds thousands. Each pending machine case is one ``SLOT_RELATION``
question (older = the case's predecessor, newer = its successor), judged under
population ``backlog``:

* a closing verdict (``same``, ``both_true``, ``separate_events``,
  ``unrelated``: the older memory stays true) closes the case, but only with
  ``judge.ladder`` on and a certificate for that cell
  (``judge_calibration.ActingGate``: a rule on an audit of its own decisions,
  D-35; a model also under the base-rate condition, M-02). The case moves whole
  to ``correction_cases_overtaken`` and ``slm corrections restore-overtaken``
  puts it back;
* ``replaces`` never acts here: the case stays in the review queue for a person,
  whatever any certificate says (D-13);
* no sure answer: nothing changes.

In shadow, or without certificates (every store at first), the pass only
receipts, which is what lets the audit draw (M-04) give people a fair sample to
label, and so lets a judge earn a certificate. A pair is never judged twice
(``truth/judge_reuse``). Ids and counts only in logs.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from superlocalmemory.core import overtaken_cases as ot
from superlocalmemory.core import recall_gate, risk_policy
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import (
    Context,
    Decision,
    Item,
    JudgeLadder,
    inputs_hash,
    write_receipts,
)
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.truth import judge_reuse

logger = logging.getLogger(__name__)

POPULATION = "backlog"
SUBJECT_KIND = "correction_case"
CLOSING = frozenset({"same", "both_true", "separate_events", "unrelated"})
USER_ACTION = "judge_closed"
ACTOR = "judge_ladder"

_PENDING = """
SELECT c.case_id, c.profile_id, c.predecessor_fact_id, p.content, c.successor_fact_id,
       s.content, c.reason_code
  FROM correction_cases c
  JOIN atomic_facts p ON p.fact_id = c.predecessor_fact_id
  JOIN atomic_facts s ON s.fact_id = c.successor_fact_id
 WHERE c.status = 'proposed' AND c.proposed_by_actor_kind = ?
   AND c.reason_code IN ({reasons})
 ORDER BY c.created_at, c.case_id
"""


def _item(row: tuple) -> Item:
    case_id, profile_id, older_id, older, newer_id, newer, reason = row
    return Item(SUBJECT_KIND, case_id, profile_id, {"older": older or "", "newer": newer or ""},
                fact_ids=(older_id, newer_id), basis=reason or "")


class BacklogPass:
    def __init__(self, db: Any, flags: FeatureFlags, ladder: JudgeLadder, *,
                 limit: int = 32, hash_key: bytes | None = None) -> None:
        self._db = db
        self._flags = flags
        self._ladder = ladder
        self._limit = limit
        self._key = hash_key
        self.counts: Counter[str] = Counter()

    def run(self) -> dict[str, int]:
        if not self._flags.observing("judge.ladder"):
            return {}
        reasons = sorted(ot.MACHINE_REASONS)
        with self._db.raw_connection() as conn:  # 1. read, then let go of the connection
            try:
                rows = conn.execute(_PENDING.format(reasons=",".join("?" * len(reasons))),
                                    (ot.MACHINE_ACTOR_KIND, *reasons)).fetchall()
            except Exception as exc:  # noqa: BLE001 - a store without the ledger has no backlog
                if "no such" in str(exc).lower():
                    return {}
                raise
            items = [_item(r) for r in rows]
            keys = [(it.profile_id, inputs_hash(SLOT_RELATION, it, self._key)) for it in items]
            seen = judge_reuse.earlier(conn, SLOT_RELATION.version, keys)
        available = self._ladder.available_model_tiers(SLOT_RELATION)
        limit = self._limit if available else len(items)
        reused, ask = judge_reuse.plan(keys, seen, available, limit)
        if not reused and not ask:
            return {}
        ctx = Context(population=POPULATION, flag_state=self._flags.state("judge.ladder").value,
                      hash_key=self._key, audit_probability=risk_policy.AUDIT_PROBABILITY,
                      escalation_probability=risk_policy.SHADOW_ESCALATION)
        fresh: list[Decision] = []
        receipts: list = []
        if ask:
            with recall_gate.background_work():  # 2. judge with no lock held: saves never wait
                fresh, receipts = self._ladder.decide(SLOT_RELATION, [items[i] for i in ask],
                                                      ctx)
        decided = {**reused, **dict(zip(ask, fresh))}
        with self._db.raw_connection() as conn:  # 3. one short write
            write_receipts(conn, receipts)
            self._apply(conn, [(items[i], decided[i]) for i in sorted(decided)])
        logger.info("Backlog pass: %s", dict(self.counts))
        return dict(self.counts)

    def _apply(self, conn: Any, pairs: list[tuple[Item, Decision]]) -> None:
        acting = self._flags.enabled("judge.ladder")
        gates: dict[str, ActingGate] = {}
        for item, decision in pairs:
            self.counts[f"{decision.tier}:{decision.verdict}"] += 1
            if decision.abstained:
                continue
            if decision.verdict not in CLOSING:  # replaces: a person's to review (D-13)
                self.counts["left_for_review"] += 1
                continue
            if not acting:
                continue
            gate = gates.setdefault(item.profile_id, ActingGate(
                conn, profile_id=item.profile_id, question_id=SLOT_RELATION.question_id,
                version=SLOT_RELATION.version))
            if not gate.allows(decision.tier, decision.model, decision.verdict,
                               decision.confidence, population=POPULATION,
                               regime=decision.regime,
                               routing_threshold=decision.routing_threshold):
                self.counts["held_uncalibrated"] += 1
                continue
            self._close(conn, item, decision)

    def _close(self, conn: Any, item: Item, decision: Decision) -> None:
        conn.execute("SAVEPOINT backlog_case")
        try:
            cur = conn.execute("SELECT * FROM correction_cases WHERE case_id = ?",
                               (item.subject_id,))
            cols = [c[0] for c in cur.description]
            cases = [dict(zip(cols, r)) for r in cur.fetchall()]  # read again, in this write
            closed = ot.overtake(conn, cases, user_action=USER_ACTION, actor_id=ACTOR,
                                 operation_id=decision.receipts[-1] if decision.receipts else "",
                                 closed_reason=f"closed by a certified judge ({decision.tier}): "
                                               f"{decision.verdict}")
            if closed and decision.receipts:
                conn.execute("UPDATE judge_receipts SET action_taken = ? WHERE receipt_id = ?",
                             (f"closed_{decision.verdict}", decision.receipts[-1]))
            self.counts["closed" if closed else "already_reviewed"] += 1
        except ot.OvertakeRaced:
            conn.execute("ROLLBACK TO backlog_case")
            self.counts["already_reviewed"] += 1
        except Exception as exc:  # noqa: BLE001 - one case must not stall the pass
            conn.execute("ROLLBACK TO backlog_case")
            self.counts["error"] += 1
            logger.debug("backlog case %s not closed: %s", item.subject_id, exc)
        finally:
            conn.execute("RELEASE backlog_case")


__all__ = ["BacklogPass", "CLOSING", "POPULATION", "SUBJECT_KIND"]
