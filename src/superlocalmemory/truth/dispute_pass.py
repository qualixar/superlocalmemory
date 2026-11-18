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
* ``unrelated`` -> the slot key was wrong: the newer memory leaves the slot
  (``unbound``, a keying error on record) and both stay current.

``judge.ladder`` in shadow asks and records receipts but changes nothing; on,
it acts -- but only where the store's own labels support it
(``core/judge_calibration.ActingGate``: an e-process certificate from blind
audit labels, so the error rate among acted-on decisions is <= alpha with
probability >= 1 - delta). Tier 0 rules never retire anything, and even their
closing answers act only on an audit of the rules' own decisions (I-10, M-18):
closing a dispute is a memory change. Measured 15 Nov 2026 on the maintainer's
backlog, Jev's "replaces" had 20-40% precision and no threshold passed, so
retirement by a model stays off until labels say otherwise.

A dispute is not asked again once every model tier that could answer it here
has abstained on it under this question version (keyed pair hash): it waits
for a person, so a store never pays twice for the same unanswerable pair. It
is asked again when a tier failed (a failed ask judged nothing, R-35) or when a
tier is newly available, e.g. the ladder went from shadow (rules and Laya) to
on (with Jev, D-27). With no model tier at all only the free rules run, over
every open dispute, so nothing waits behind a page limit.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from superlocalmemory.core import recall_gate, risk_policy
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import Context, Decision, Item, JudgeLadder, inputs_hash, write_receipts
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.truth import judge_reuse, slot_store
from superlocalmemory.truth.slot_resolver import (
    RETIRED_FOR_ANOTHER,
    auto_apply_allowed,
    retire_superseded,
)

logger = logging.getLogger(__name__)

SUBJECT_KIND = "slot_dispute"
_RESOLUTION = {"replaces": "b", "same": "same", "both_true": "both",
               "separate_events": "both", "unrelated": "unrelated"}


def _item(r: tuple) -> Item:
    """The pair in time order: a value older than the head is the older text (R-10)."""
    head, challenger = (r[3], r[2]), (r[5], r[4])
    older, newer = (challenger, head) if r[7] in slot_store.OLDER_REASONS else (head, challenger)
    return Item(SUBJECT_KIND, r[0], r[1], {"older": older[0], "newer": newer[0]},
                fact_ids=(older[1], newer[1]), basis=r[7] or "")


def _resolution(verdict: str, basis: str) -> str:
    """``replaces`` means the newer text wins: the head, when the challenger is older."""
    if verdict == "replaces" and basis in slot_store.OLDER_REASONS:
        return "a"
    return _RESOLUTION[verdict]

_OPEN = """
SELECT d.dispute_id, d.profile_id, fa.fact_id, fa.content, fb.fact_id, fb.content, fb.scope,
       d.reason_code
  FROM slot_disputes d
  JOIN slot_revisions ra ON ra.revision_id = d.revision_a
  JOIN slot_revisions rb ON rb.revision_id = d.revision_b
  JOIN atomic_facts fa ON fa.fact_id = ra.fact_id
  JOIN atomic_facts fb ON fb.fact_id = rb.fact_id
 WHERE d.state = 'open' AND d.reason_code NOT IN ('rolled_back', 'origin_below_head')
 ORDER BY d.opened_at
"""
#: A dispute the ledger reopened by rolling a retirement back is a person's to
#: settle (R-09): no judge answers it, so the same judge cannot act on it again.
#: The same holds for one the origin lattice refused (``origin_below_head``, M-15).

class DisputePass:
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
        with self._db.raw_connection() as conn:  # 1. read, then let go of the connection
            rows = conn.execute(_OPEN).fetchall()
            items = [_item(r) for r in rows]
            hashes = [(it.profile_id, inputs_hash(SLOT_RELATION, it, self._key)) for it in items]
            seen = judge_reuse.earlier(conn, SLOT_RELATION.version, hashes)
        # Which model tiers could answer here now (nothing is asked to find out).
        # Without one, only the free deterministic rules run, so no limit applies.
        available = self._ladder.available_model_tiers(SLOT_RELATION)
        limit = self._limit if available else len(items)
        reused, ask = judge_reuse.plan(hashes, seen, available, limit)
        if not reused and not ask:
            return {}
        ctx = Context(population=SUBJECT_KIND, flag_state=self._flags.state("judge.ladder").value,
                      hash_key=self._key, audit_probability=risk_policy.AUDIT_PROBABILITY,
                      escalation_probability=risk_policy.SHADOW_ESCALATION)
        with recall_gate.background_work():  # 2. judge with no lock held: saves never wait
            fresh, receipts = self._ladder.decide(SLOT_RELATION, [items[i] for i in ask], ctx)
        decided = {**reused, **dict(zip(ask, fresh))}
        with self._db.raw_connection() as conn:  # 3. one short write
            write_receipts(conn, receipts)
            self._apply(conn, [(rows[i], decided[i]) for i in sorted(decided)])
        logger.info("Dispute pass: %s", dict(self.counts))
        return dict(self.counts)

    def _apply(self, conn: Any, pairs: list) -> None:
        acting = self._flags.enabled("judge.ladder")
        auto_apply = auto_apply_allowed(self._flags)
        gates: dict[str, ActingGate] = {}
        for row, decision in pairs:
            self.counts[f"{decision.tier}:{decision.verdict}"] += 1
            if not acting or decision.abstained:
                continue
            gate = gates.setdefault(row[1], ActingGate(
                conn, profile_id=row[1], question_id=SLOT_RELATION.question_id,
                version=SLOT_RELATION.version))
            # Rules too: closing a dispute is a memory change, and a rule acts only
            # on an audit of its own decisions (I-10, M-18), like any judge.
            if not gate.allows(decision.tier, decision.model, decision.verdict,
                               decision.confidence, population=SUBJECT_KIND,
                               regime=decision.regime,
                               routing_threshold=decision.routing_threshold):
                self.counts["held_uncalibrated"] += 1
                continue
            if (decision.verdict == "replaces" and self._flags.enabled("trust.taint")
                    and self._origin_held(conn, row)):
                continue
            self._act(conn, row, decision, auto_apply)

    def _origin_held(self, conn: Any, row: tuple) -> bool:
        """M-15: a judge may retire a memory only for one whose origin is at least its own.

        ``replaces`` retires the older text: the newer when the challenger is the
        newer value, the challenger when it is the older one (the head stays). If the
        winner's origin does not reach the loser's, or either is ``unknown``, nothing
        is retired and the dispute waits for a person (relabelled so no judge asks again;
        an older-value dispute keeps its reason, which its settling needs).
        """
        from superlocalmemory.trust import supersession

        dispute_id, _p, head, _a, challenger, _b, _scope, basis = row
        older = (basis or "") in slot_store.OLDER_REASONS
        winner, loser = (head, challenger) if older else (challenger, head)
        if supersession.allows(conn, winner, loser):
            return False
        self.counts["origin_below_head"] += 1
        if not older:
            conn.execute("UPDATE slot_disputes SET reason_code = ? WHERE dispute_id = ? "
                         "AND state = 'open'", (supersession.ORIGIN_BELOW_HEAD, dispute_id))
        return True

    def _act(self, conn: Any, row: tuple, decision: Decision, auto_apply: bool) -> None:
        dispute_id, profile_id, _older, _a, _newer, _b, scope, basis = row
        verdict = decision.verdict
        if verdict == "replaces" and not auto_apply:
            self.counts["held_for_review"] += 1
            return
        conn.execute("SAVEPOINT dispute")
        try:
            retire = slot_store.resolve_dispute(conn, dispute_id, _resolution(verdict, basis or ""),
                                                resolved_by="judge_ladder")
            for loser in retire:  # exactly what the resolution names, read in this transaction
                successor = slot_store.successor_fact(conn, dispute_id, loser)
                ledger, case = retire_superseded(conn, profile_id=profile_id, scope=scope,
                                                 predecessor=loser, successor=successor)
                if ledger != "applied":
                    raise _NotApplied(ledger)
            self._mark(conn, decision, f"resolved_{verdict}", dispute_id)
            self.counts[f"resolved_{verdict}"] += 1
        except slot_store.StaleDisputeError:
            conn.execute("ROLLBACK TO dispute")  # undo anything done before the check
            slot_store.dismiss_dispute(conn, dispute_id, by="judge_ladder:stale")
            self.counts["stale_dismissed"] += 1
        except _NotApplied as exc:
            conn.execute("ROLLBACK TO dispute")
            self.counts[f"ledger_{exc}"] += 1
            if str(exc) == RETIRED_FOR_ANOTHER:  # the head's fact is already gone (R-22)
                slot_store.dismiss_dispute(conn, dispute_id, by="judge_ladder:retired_for_another")
        except Exception as exc:  # noqa: BLE001 - one dispute must not stall the pass
            conn.execute("ROLLBACK TO dispute")
            self.counts["error"] += 1
            logger.debug("dispute %s not settled: %s", dispute_id, exc)
        finally:
            conn.execute("RELEASE dispute")

    @staticmethod
    def _mark(conn: Any, decision: Decision, action: str, dispute_id: str | None = None) -> None:
        if decision.receipts:
            conn.execute("UPDATE judge_receipts SET action_taken = ? WHERE receipt_id = ?",
                         (action, decision.receipts[-1]))
            if dispute_id is not None:  # R-09: the revision the decision changed names it
                conn.execute("UPDATE slot_revisions SET receipt_id = ? WHERE revision_id = "
                             "(SELECT revision_b FROM slot_disputes WHERE dispute_id = ?)",
                             (decision.receipts[-1], dispute_id))


class _NotApplied(Exception):
    pass


__all__ = ["DisputePass", "SUBJECT_KIND"]
