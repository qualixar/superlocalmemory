# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Bring back the memories 4.2.30 retired: the ledger command beside the kill switch (D-23, D-49).

The kill switch stops every 4.2.30 pass and makes reads ignore slots, leases, disputes and taint,
but it does not undo a retirement already applied. This is the separate, explicit command that
does: it rolls back every applied case the truth layer filed (``slot_resolver.rollback_slot_cases``
touches no other case: never a person's, never the consolidator's), each predecessor restored
exactly as the ledger snapshot says.

It is a decision like any other (D-49): any caller may run it (an agent, a script, a person on the
dashboard), it is recorded in the decision log with who and through which channel, ``dry_run``
says how many would come back before anything changes, and it can be undone. Undo applies the same
cases again, but only if the value of each has not moved since (the older memory is still live,
nothing else is deciding about it, the newer one is still live); if any one has moved it refuses in
words and changes nothing, all or nothing.

The caller owns the transaction. No memory text is read or logged: ids and counts only.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid

from superlocalmemory.core import decisions
from superlocalmemory.storage.correction_cases import (
    CorrectionCaseError,
    open_case_id,
    propose_on_connection,
)
from superlocalmemory.truth import slot_resolver as sr

ACTION_RUN, ACTION_UNDO = "run", "undo"
MOVED_WORDS = ("Some of those memories have changed since they came back, so they were not "
               "retired again.")


def _plural(n: int) -> str:
    return f"{n} memory" if n == 1 else f"{n} memories"


def _applied(conn: sqlite3.Connection, profile_id: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT case_id FROM correction_cases WHERE status = 'applied' AND reason_code = ? "
        "AND profile_id = ? ORDER BY case_id", (sr.REASON, profile_id)).fetchall()]


def would_come_back(conn: sqlite3.Connection, profile_id: str) -> int:
    """How many retirements the command would bring back now (read-only)."""
    return len(_applied(conn, profile_id))


def run(conn: sqlite3.Connection, *, profile_id: str, decider: decisions.Decider,
        dry_run: bool = False) -> dict:
    """Roll back every truth-layer retirement of the profile, or only count them (``dry_run``).

    Returns ``would_come_back``, ``came_back``, ``refused`` (a case the ledger would not roll back,
    left as it was) and ``decision_id`` (``""`` when nothing was recorded).
    """
    before = _applied(conn, profile_id)
    if dry_run:
        return {"dry_run": True, "would_come_back": len(before), "came_back": 0, "refused": 0,
                "decision_id": ""}
    counts = sr.rollback_slot_cases(conn, profile_id=profile_id)
    left = set(_applied(conn, profile_id))
    back = [c for c in before if c not in left]
    decision_id = ""
    if back:
        decision_id = decisions.log(
            conn, profile_id=profile_id, kind=decisions.ROLLBACK_RETIREMENTS,
            subject_id=f"retirements-{uuid.uuid4().hex[:12]}", action=ACTION_RUN, decider=decider,
            summary=f"Brought back {_plural(len(back))} that 4.2.30 had retired",
            detail={"case_ids": back})
    return {"dry_run": False, "would_come_back": len(before), "came_back": len(back),
            "refused": int(counts.get("refused", 0)), "decision_id": decision_id}


def _live(conn: sqlite3.Connection, fact_id: str) -> bool:
    row = conn.execute("SELECT valid_until, system_expired_at FROM fact_temporal_validity "
                       "WHERE fact_id = ?", (fact_id,)).fetchone()
    exists = conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,)).fetchone()
    return bool(exists) and (row is None or (row[0] is None and row[1] is None))


def _unmoved(conn: sqlite3.Connection, case: sqlite3.Row | tuple, profile_id: str) -> bool:
    _case_id, _scope, predecessor, successor, status = case
    return (status == "rolled_back" and _live(conn, predecessor) and _live(conn, successor)
            and open_case_id(conn, profile_id, predecessor) is None)


def _reapply_id(decision_id: str, case_id: str) -> str:
    raw = f"{sr.REASON}\0reapply\0{decision_id}\0{case_id}".encode()
    return "slot-" + hashlib.sha256(raw).hexdigest()[:40]


def undo(conn: sqlite3.Connection, *, profile_id: str, found: dict,
         decider: decisions.Decider) -> dict:
    """Apply the cases of a logged run again. ``found`` is its decision-log row.

    Raises ``ValueError`` (with the reason in words, nothing changed) if the value of any case
    has moved since the run.
    """
    case_ids = [str(c) for c in (found.get("detail") or {}).get("case_ids", [])]
    rows = []
    for case_id in case_ids:
        row = conn.execute(
            "SELECT case_id, scope, predecessor_fact_id, successor_fact_id, status "
            "FROM correction_cases WHERE case_id = ? AND profile_id = ?",
            (case_id, profile_id)).fetchone()
        if row is None or not _unmoved(conn, row, profile_id):
            raise ValueError(MOVED_WORDS)
        rows.append(tuple(row))
    conn.execute("SAVEPOINT retirement_undo")
    try:
        for case_id, scope, predecessor, successor, _status in rows:
            fresh = _reapply_id(found["decision_id"], case_id)
            propose_on_connection(
                conn, case_id=fresh, profile_id=profile_id, scope=scope,
                predecessor_fact_id=predecessor, successor_fact_id=successor,
                reason_code=sr.REASON, actor=sr.ACTOR, idempotency_key=f"{sr.REASON}:{fresh}",
                is_profile_active=lambda candidate: candidate == profile_id,
                is_actor_trusted=sr._trusted)
            sr._apply(conn, fresh, 0, profile_id)
    except (CorrectionCaseError, ValueError):
        conn.execute("ROLLBACK TO retirement_undo")
        conn.execute("RELEASE retirement_undo")
        raise ValueError(MOVED_WORDS) from None
    conn.execute("RELEASE retirement_undo")
    decisions.mark_undone(conn, found["decision_id"], decider)
    decisions.log(conn, profile_id=profile_id, kind=decisions.ROLLBACK_RETIREMENTS,
                  subject_id=found["subject_id"], action=ACTION_UNDO, decider=decider,
                  summary=f"Retired {_plural(len(rows))} again (took back the roll back)",
                  detail={"reapplied": len(rows)})
    return {"reapplied": len(rows)}


__all__ = ["MOVED_WORDS", "run", "undo", "would_come_back"]
