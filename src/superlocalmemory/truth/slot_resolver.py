# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""From a stored fact to a slot decision, and from a decision to the ledger.

``resolve_fact`` keys one fact (Tier 0 rules), binds it to its slot, and, when
the decision is *supersede* and ``truth.slots.auto_apply`` is on, retires the
older fact through the correction ledger, so the change is recorded, visible
in ``slm corrections`` and reversible with the exact prior state.

When the older fact already has an open proposal (the backlog of consolidator
proposals), the slot decision settles it instead of adding a second case: the
same successor applies that proposal; a different successor leaves it for
review, because two machine opinions disagreeing is exactly what a person
should see.

A supersede decided while retirement was not allowed (``truth.slots`` in
shadow, or ``auto_apply`` off) is not lost (R-07). The revision itself records
it (state ``superseded``, no ``case_id``), and ``reconcile_ledger`` -- run by the
slot pass only while ``truth.slots`` is on, since shadow never writes the
ledger -- turns it into a ledger case: proposed (the review queue) while
``auto_apply`` is off, applied once it is on; proposals it made earlier are
applied then too. A predecessor that already has a proposal with a different
successor is linked to that proposal and left for a person.

Callers own the transaction. Nothing here commits, and nothing reads text
beyond the one fact being keyed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from superlocalmemory.storage.correction_cases import (
    CorrectionActor,
    CorrectionCaseError,
    open_case_id,
    propose_on_connection,
    transition_on_connection,
)
from superlocalmemory.truth import slot_keys, slot_store

REASON = "slot_supersede"
#: The truth layer's own ledger identity: machine-decided, never a person.
ACTOR = CorrectionActor(actor_id="slm-truth-layer", actor_kind="system",
                        trust_tier="canonical_writer")

APPLIED, PENDING, DEFERRED_TO_REVIEW = "applied", "pending", "deferred_to_review"
PROPOSED = "proposed"
#: The predecessor is already retired in favour of a different successor: the
#: ledger is left as it is and the slot waits for a person (R-22).
RETIRED_FOR_ANOTHER = "retired_for_another"

#: W0-18: ``truth.slots.auto_apply`` let the resolver retire memories with no
#: certificate and no approval. Since D-36 nothing retires here: an explicit
#: key's new value opens a dispute, and the dispute pass retires only what the
#: ``ActingGate`` allows on a certificate (rules ``explicit-v1`` included), so
#: the flag now has a lawful basis (I-10). It stays off by default.
AUTO_APPLY_BASIS_READY = True


def auto_apply_allowed(flags) -> bool:
    """The effective ``truth.slots.auto_apply`` for a background pass."""
    return (AUTO_APPLY_BASIS_READY and flags.enabled("truth.slots.auto_apply")
            and flags.enabled("truth.slots"))


@dataclass(frozen=True)
class Resolution:
    fact_id: str
    keyed: bool
    outcome: str | None = None
    slot_id: str | None = None
    ledger: str | None = None
    case_id: str | None = None
    #: ``held`` when ``trust.taint`` is on and the lattice refused the newer value
    #: (the dispute is relabelled ``origin_below_head``); ``would_hold`` in shadow.
    origin: str | None = None


def _case_id(profile_id: str, predecessor: str, successor: str) -> str:
    raw = f"{REASON}\0{profile_id}\0{predecessor}\0{successor}".encode("utf-8")
    return "slot-" + hashlib.sha256(raw).hexdigest()[:40]


def _trusted(actor: CorrectionActor) -> bool:
    return actor == ACTOR


def _apply(conn: sqlite3.Connection, case_id: str, version: int, profile_id: str) -> None:
    transition_on_connection(
        conn, case_id=case_id, expected_version=version, actor=ACTOR,
        operation_id=f"{REASON}:apply:{case_id}", from_status="proposed",
        to_status="applied", mutate_temporal=True,
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=_trusted)


def retire_superseded(conn: sqlite3.Connection, *, profile_id: str, scope: str,
                      predecessor: str, successor: str,
                      apply: bool = True) -> tuple[str, str | None]:
    """Retire ``predecessor`` in favour of ``successor`` through the ledger.

    ``apply`` False only proposes (the review queue); nothing is retired.
    """
    open_id = open_case_id(conn, profile_id, predecessor)
    if open_id is not None:
        row = conn.execute("SELECT successor_fact_id, status, version FROM correction_cases "
                           "WHERE case_id = ?", (open_id,)).fetchone()
        if row[1] == "applied":
            return (APPLIED if row[0] == successor else RETIRED_FOR_ANOTHER), open_id
        if row[0] != successor:
            return DEFERRED_TO_REVIEW, open_id
        if not apply:
            return PROPOSED, open_id
        _apply(conn, open_id, int(row[2]), profile_id)
        return APPLIED, open_id
    case_id = _case_id(profile_id, predecessor, successor)
    propose_on_connection(
        conn, case_id=case_id, profile_id=profile_id, scope=scope,
        predecessor_fact_id=predecessor, successor_fact_id=successor, reason_code=REASON,
        actor=ACTOR, idempotency_key=f"{REASON}:{case_id}",
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=_trusted)
    if not apply:
        return PROPOSED, case_id
    _apply(conn, case_id, 0, profile_id)
    return APPLIED, case_id


#: Supersede decisions whose retirement is not settled. ``r`` is the superseded
#: revision, ``n`` the one that superseded it; the case lives on ``n`` (as in
#: ``resolve_fact``). Unsettled: no case yet, or its own proposal not yet applied,
#: and the predecessor fact still live. A link to a rival proposal (another
#: successor) waits for a person and is not read again.
_UNSETTLED = """
SELECT n.revision_id, r.profile_id, s.scope, r.fact_id, n.fact_id, n.case_id
  FROM slot_revisions r
  JOIN memory_slots s ON s.slot_id = r.slot_id
  JOIN slot_revisions n ON n.revision_id = r.superseded_by
  LEFT JOIN correction_cases c ON c.case_id = n.case_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = r.fact_id
 WHERE r.state = 'superseded' AND v.system_expired_at IS NULL
   AND (n.case_id IS NULL OR (c.status = 'proposed' AND c.successor_fact_id = n.fact_id))
 ORDER BY r.recorded_at, r.revision_id
 LIMIT ?
"""


def reconcile_ledger(conn: sqlite3.Connection, *, auto_apply: bool, limit: int = 200,
                     origin_gate: Callable[[sqlite3.Connection, str, str], bool] | None = None
                     ) -> Counter[str]:
    """Bring unsettled supersede decisions into the ledger (call only with slots on).

    Each revision in its own savepoint; a ledger refusal leaves it as it was.
    ``origin_gate(conn, winner, loser)`` (``trust.taint`` on, M-15) False means the
    winner's origin is below the loser's: the case is proposed for a person, never applied.
    """
    counts: Counter[str] = Counter()
    for rev, profile, scope, predecessor, successor, linked in conn.execute(
            _UNSETTLED, (limit,)).fetchall():
        conn.execute("SAVEPOINT slot_reconcile")
        try:
            held = origin_gate is not None and not origin_gate(conn, successor, predecessor)
            if held:
                counts["origin_below_head"] += 1
            status, case_id = retire_superseded(
                conn, profile_id=profile, scope=scope, predecessor=predecessor,
                successor=successor, apply=auto_apply and not held)
        except (CorrectionCaseError, ValueError):
            conn.execute("ROLLBACK TO slot_reconcile")
            conn.execute("RELEASE slot_reconcile")
            counts["ledger_refused"] += 1
            continue
        if case_id is not None and case_id != linked:
            slot_store.record_case(conn, rev, case_id)
        conn.execute("RELEASE slot_reconcile")
        counts[f"ledger_{status}"] += 1
    return counts


def resolve_fact(conn: sqlite3.Connection, *, fact_id: str, profile_id: str, scope: str,
                 content: str, valid_from: str | None,
                 resolve_subject: Callable[[str], str | None] | None = None,
                 explicit_key: str | None = None,
                 explicit_value: str | None = None,
                 taint: str = "off") -> Resolution:
    """Key and bind one fact. Never raises for a bad key, and never retires.

    What a binding decides is recorded on the slot; a retirement only ever
    comes from a settled dispute (``truth.dispute_pass``, D-36, I-10).

    ``taint`` is the ``trust.taint`` state (``off``, ``shadow``, ``on``, M-15). On,
    a dispute against a head the new fact's origin does not reach (or either
    side ``unknown``) is relabelled ``origin_below_head``: no judge answers it
    and only a recorded decision by a caller settles it (D-49). Shadow only reports what it would have done.
    Off, nothing changes.
    """
    key = slot_keys.extract(content, resolve_subject=resolve_subject, explicit=explicit_key,
                            explicit_value=explicit_value)
    if key is None:
        return Resolution(fact_id=fact_id, keyed=False)
    binding = slot_store.bind(conn, profile_id=profile_id, scope=scope, fact_id=fact_id,
                              key=key, valid_from=valid_from)
    origin = None
    disputed = (binding.outcome == slot_store.DISPUTE and binding.dispute_id
                and binding.other_fact_id)
    if disputed:
        # WS-9: a fact an import saved never displaces a head, whichever way trust.taint is set.
        from superlocalmemory.portability import guards

        origin = guards.hold(conn, fact_id=fact_id, dispute_id=binding.dispute_id,
                             older_reasons=slot_store.OLDER_REASONS)
    if origin is None and taint != "off" and disputed:
        origin = _origin_check(conn, binding, fact_id, enforce=taint == "on")
    return Resolution(fact_id, True, binding.outcome, binding.slot_id, origin=origin)


def _origin_check(conn: sqlite3.Connection, binding: slot_store.Binding, fact_id: str, *,
                  enforce: bool) -> str | None:
    """``held`` / ``would_hold`` when the new value's origin is below the head's.

    A value older than the head leaves the head in place (the judge may retire
    the newcomer), so it is checked when the judge answers, not here.
    """
    from superlocalmemory.trust import supersession

    reason = conn.execute("SELECT reason_code FROM slot_disputes WHERE dispute_id = ?",
                          (binding.dispute_id,)).fetchone()[0]
    if reason in slot_store.OLDER_REASONS:
        return None
    if supersession.allows(conn, fact_id, binding.other_fact_id):
        return None
    if not enforce:
        return "would_hold"
    conn.execute("UPDATE slot_disputes SET reason_code = ? WHERE dispute_id = ? "
                 "AND state = 'open'", (supersession.ORIGIN_BELOW_HEAD, binding.dispute_id))
    return "held"


def rollback_slot_cases(conn: sqlite3.Connection, *, profile_id: str | None = None
                        ) -> Counter[str]:
    """Roll back every applied case the truth layer filed (D-23).

    The kill switch stops 4.2.30 from acting and reading slots, but leaves
    retirements in place; this is the separate, explicit way to undo them. Only
    cases with this module's reason code are touched: never a person's or the
    consolidator's. Each case in its own savepoint, so a refusal skips only that
    case. Restores each predecessor exactly (the ledger's snapshot); the slots
    themselves are ignored while the kill switch is on and are re-derived later
    (R-09).
    """
    sql = ("SELECT case_id, version, profile_id FROM correction_cases "
           "WHERE status = 'applied' AND reason_code = ?")
    args: tuple = (REASON,)
    if profile_id is not None:
        sql, args = sql + " AND profile_id = ?", (*args, profile_id)
    counts: Counter[str] = Counter()
    for case_id, version, profile in conn.execute(sql, args).fetchall():
        conn.execute("SAVEPOINT slot_rollback")
        try:
            transition_on_connection(
                conn, case_id=case_id, expected_version=int(version), actor=ACTOR,
                operation_id=f"{REASON}:rollback:{case_id}", from_status="applied",
                to_status="rolled_back", mutate_temporal=True,
                is_profile_active=lambda candidate, p=profile: candidate == p,
                is_actor_trusted=_trusted)
        except (CorrectionCaseError, ValueError):
            conn.execute("ROLLBACK TO slot_rollback")
            counts["refused"] += 1
        else:
            counts["rolled_back"] += 1
        finally:
            conn.execute("RELEASE slot_rollback")
    return counts


__all__ = ["ACTOR", "APPLIED", "AUTO_APPLY_BASIS_READY", "DEFERRED_TO_REVIEW", "PENDING",
           "PROPOSED", "REASON", "RETIRED_FOR_ANOTHER", "auto_apply_allowed",
           "Resolution", "reconcile_ledger", "resolve_fact", "retire_superseded",
           "rollback_slot_cases"]
