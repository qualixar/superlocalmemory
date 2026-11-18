# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What applying and rolling back a correction does to the predecessor's validity.

Split out of ``correction_cases.py`` (4.2.30, R-39: that module had reached its
size limit). Applying a case snapshots the predecessor's
``fact_temporal_validity`` row onto the case and supersedes it; rolling back
restores that snapshot exactly. The case store calls these inside its own
serialized transaction.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from superlocalmemory.storage.correction_errors import (
    CorrectionAuthorizationError,
    CorrectionCaseError,
    CorrectionNotFoundError,
)

if TYPE_CHECKING:
    from superlocalmemory.storage.correction_cases import CorrectionCase


def has_until_source(conn: sqlite3.Connection) -> bool:
    """Whether M073 has run: a store may run this code before its deferred migration."""
    return any(r[1] == "valid_until_source"
               for r in conn.execute("PRAGMA table_info(fact_temporal_validity)"))


def _snapshot_source(conn: sqlite3.Connection, case: CorrectionCase) -> None:
    # A side table (M073): M042 pins the ledger's exact columns.
    conn.execute(
        "INSERT OR REPLACE INTO correction_case_until_source (case_id, "
        "predecessor_valid_until_source) SELECT ?, (SELECT valid_until_source "
        "FROM fact_temporal_validity WHERE fact_id = ? AND profile_id = ?)",
        (case.case_id, case.predecessor_fact_id, case.profile_id))


def _check_facts(conn: sqlite3.Connection, case: CorrectionCase) -> None:
    rows = conn.execute(
        "SELECT fact_id, profile_id, scope FROM atomic_facts WHERE fact_id IN (?, ?)",
        (case.predecessor_fact_id, case.successor_fact_id),
    ).fetchall()
    facts = {str(row["fact_id"]): row for row in rows}
    if set(facts) != {case.predecessor_fact_id, case.successor_fact_id}:
        raise CorrectionNotFoundError("correction facts are no longer available")
    for row in facts.values():
        if row["profile_id"] != case.profile_id or row["scope"] != case.scope:
            raise CorrectionAuthorizationError("correction facts are outside the approved scope")


def _snapshot_prior(conn: sqlite3.Connection, case: CorrectionCase,
                    occurred_at: str) -> tuple | None:
    """Copy the predecessor's validity row onto the case (or note it had none)."""
    prior = conn.execute(
        "SELECT valid_from, valid_until, system_created_at, system_expired_at, "
        "invalidated_by, invalidation_reason FROM fact_temporal_validity "
        "WHERE fact_id=? AND profile_id=?",
        (case.predecessor_fact_id, case.profile_id),
    ).fetchone()
    if prior is None:
        conn.execute(
            "UPDATE correction_cases SET predecessor_temporal_existed=0, "
            "predecessor_valid_from=NULL, predecessor_valid_until=NULL, "
            "predecessor_system_created_at=NULL, predecessor_system_expired_at=NULL, "
            "predecessor_invalidated_by=NULL, predecessor_invalidation_reason=NULL "
            "WHERE case_id=?",
            (case.case_id,),
        )
        conn.execute(
            "INSERT INTO fact_temporal_validity "
            "(fact_id, profile_id, system_created_at) VALUES (?, ?, ?)",
            (case.predecessor_fact_id, case.profile_id, occurred_at),
        )
    else:
        conn.execute(
            "UPDATE correction_cases SET predecessor_temporal_existed=1, "
            "predecessor_valid_from=?, predecessor_valid_until=?, "
            "predecessor_system_created_at=?, predecessor_system_expired_at=?, "
            "predecessor_invalidated_by=?, predecessor_invalidation_reason=? "
            "WHERE case_id=?",
            (*tuple(prior), case.case_id),
        )
    return prior


def _retirement_end(conn: sqlite3.Connection, case: CorrectionCase, prior: tuple | None, *,
                    occurred_at: str, sourced: bool) -> tuple[str | None, str | None]:
    """4.2.30: a retired memory also stops being true in the world, not only in
    SLM's knowledge. A reviewer's time wins, then a close already recorded, then
    the best evidence of when the successor took over (truth/validity), each
    with the source it came from (R-11)."""
    from superlocalmemory.truth.validity import retire_until_sourced, successor_takeover_interval

    existing_source = None
    if sourced:
        _snapshot_source(conn, case)
        got = conn.execute("SELECT valid_until_source FROM fact_temporal_validity "
                           "WHERE fact_id = ? AND profile_id = ?",
                           (case.predecessor_fact_id, case.profile_id)).fetchone()
        existing_source = got[0] if got is not None else None
    start = prior[0] if prior is not None else None
    derived_until, derived_source, derived_upper = successor_takeover_interval(
        conn, case.successor_fact_id, predecessor_valid_from=start, occurred_at=occurred_at)
    try:
        return retire_until_sourced(
            reviewer=case.event_valid_until, existing=prior[1] if prior is not None else None,
            derived=derived_until, predecessor_valid_from=start,
            existing_source=existing_source, derived_source=derived_source,
            derived_upper=derived_upper)
    except ValueError as exc:
        raise CorrectionCaseError(str(exc)) from exc


def _apply_predecessor_temporal(
    conn: sqlite3.Connection, case: CorrectionCase, *, occurred_at: str
) -> None:
    """Snapshot and supersede exactly one profile-and-scope-owned predecessor."""
    _check_facts(conn, case)
    prior = _snapshot_prior(conn, case, occurred_at)
    sourced = has_until_source(conn)
    until, source = _retirement_end(conn, case, prior, occurred_at=occurred_at, sourced=sourced)
    cursor = conn.execute(
        "UPDATE fact_temporal_validity SET "
        "valid_until=?, system_expired_at=?, "
        "invalidated_by=?, invalidation_reason=? WHERE fact_id=? AND profile_id=?",
        (until, occurred_at, case.successor_fact_id, case.reason_code,
         case.predecessor_fact_id, case.profile_id),
    )
    if cursor.rowcount != 1:
        # Defence in depth, not a known live failure: the read above and the
        # insert-if-missing use this same (fact_id, profile_id), inside one
        # serialized transaction, so a zero-row update should be unreachable.
        # It is asserted because the alternative is review_correction returning
        # "applied" having retired nothing -- and supersession is the ONE
        # signal every read path now resolves its answer from.
        raise CorrectionNotFoundError("predecessor temporal record was not superseded")
    if sourced:
        from superlocalmemory.truth.validity import check_source

        conn.execute("UPDATE fact_temporal_validity SET valid_until_source = ? "
                     "WHERE fact_id = ? AND profile_id = ?",
                     (check_source(source) if until is not None else None, case.predecessor_fact_id,
                      case.profile_id))


def _restore_predecessor_temporal(conn: sqlite3.Connection, case: CorrectionCase) -> None:
    """Restore the snapshot captured at apply; history rows themselves remain."""
    if case.predecessor_temporal_existed is None:
        raise CorrectionCaseError("approved case has no predecessor temporal snapshot")
    if not case.predecessor_temporal_existed:
        conn.execute(
            "DELETE FROM fact_temporal_validity WHERE fact_id=? AND profile_id=?",
            (case.predecessor_fact_id, case.profile_id),
        )
        return
    cursor = conn.execute(
        "UPDATE fact_temporal_validity SET valid_from=?, valid_until=?, system_created_at=?, "
        "system_expired_at=?, invalidated_by=?, invalidation_reason=? "
        "WHERE fact_id=? AND profile_id=?",
        (
            case.predecessor_valid_from,
            case.predecessor_valid_until,
            case.predecessor_system_created_at,
            case.predecessor_system_expired_at,
            case.predecessor_invalidated_by,
            case.predecessor_invalidation_reason,
            case.predecessor_fact_id,
            case.profile_id,
        ),
    )
    if cursor.rowcount != 1:
        raise CorrectionNotFoundError("predecessor temporal record is no longer available")
    if has_until_source(conn):
        conn.execute(
            "UPDATE fact_temporal_validity SET valid_until_source = (SELECT "
            "predecessor_valid_until_source FROM correction_case_until_source WHERE case_id = ?) "
            "WHERE fact_id = ? AND profile_id = ?",
            (case.case_id, case.predecessor_fact_id, case.profile_id))


__all__ = ["_apply_predecessor_temporal", "_restore_predecessor_temporal", "has_until_source"]
