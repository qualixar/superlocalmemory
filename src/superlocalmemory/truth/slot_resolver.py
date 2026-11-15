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

Callers own the transaction. Nothing here commits, and nothing reads text
beyond the one fact being keyed.
"""

from __future__ import annotations

import hashlib
import sqlite3
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


@dataclass(frozen=True)
class Resolution:
    fact_id: str
    keyed: bool
    outcome: str | None = None
    slot_id: str | None = None
    ledger: str | None = None
    case_id: str | None = None


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
                      predecessor: str, successor: str) -> tuple[str, str | None]:
    """Retire ``predecessor`` in favour of ``successor`` through the ledger."""
    open_id = open_case_id(conn, profile_id, predecessor)
    if open_id is not None:
        row = conn.execute("SELECT successor_fact_id, status, version FROM correction_cases "
                           "WHERE case_id = ?", (open_id,)).fetchone()
        if row[1] == "applied":
            return APPLIED, open_id
        if row[0] != successor:
            return DEFERRED_TO_REVIEW, open_id
        _apply(conn, open_id, int(row[2]), profile_id)
        return APPLIED, open_id
    case_id = _case_id(profile_id, predecessor, successor)
    propose_on_connection(
        conn, case_id=case_id, profile_id=profile_id, scope=scope,
        predecessor_fact_id=predecessor, successor_fact_id=successor, reason_code=REASON,
        actor=ACTOR, idempotency_key=f"{REASON}:{case_id}",
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=_trusted)
    _apply(conn, case_id, 0, profile_id)
    return APPLIED, case_id


def resolve_fact(conn: sqlite3.Connection, *, fact_id: str, profile_id: str, scope: str,
                 content: str, valid_from: str | None, auto_apply: bool,
                 resolve_subject: Callable[[str], str | None] | None = None,
                 explicit_key: str | None = None) -> Resolution:
    """Key, bind and (optionally) apply one fact. Never raises for a bad key."""
    key = slot_keys.extract(content, resolve_subject=resolve_subject, explicit=explicit_key)
    if key is None:
        return Resolution(fact_id=fact_id, keyed=False)
    binding = slot_store.bind(conn, profile_id=profile_id, scope=scope, fact_id=fact_id,
                              key=key, valid_from=valid_from)
    if binding.outcome != slot_store.SUPERSEDE or binding.other_fact_id is None:
        return Resolution(fact_id, True, binding.outcome, binding.slot_id)
    if not auto_apply:
        return Resolution(fact_id, True, binding.outcome, binding.slot_id, PENDING)
    try:
        ledger, case_id = retire_superseded(conn, profile_id=profile_id, scope=scope,
                                            predecessor=binding.other_fact_id,
                                            successor=fact_id)
    except (CorrectionCaseError, ValueError):
        # The ledger refused (scope moved, fact being deleted, ...): the slot
        # decision stands and the retirement waits for review.
        return Resolution(fact_id, True, binding.outcome, binding.slot_id, DEFERRED_TO_REVIEW)
    if ledger == APPLIED and case_id is not None:
        slot_store.record_case(conn, binding.revision_id, case_id)
    return Resolution(fact_id, True, binding.outcome, binding.slot_id, ledger, case_id)


__all__ = ["ACTOR", "APPLIED", "DEFERRED_TO_REVIEW", "PENDING", "REASON", "Resolution",
           "resolve_fact", "retire_superseded"]
