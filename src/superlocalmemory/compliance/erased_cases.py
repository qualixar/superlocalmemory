# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A right-to-erasure request is not blocked by correction history.

Varun's decision (2026-10-07): "Erasure wins: erase the fact anyway, close its
correction cases as 'erased on request' and keep only a text-free audit row
(ids, when, who). The right to erasure isn't blocked by internal history."

The correction ledger (M042) refers to both facts of every case
``ON DELETE RESTRICT`` and never removes a case, so an ordinary delete of a fact
a person edited, or whose correction was applied, is refused (and still is:
core/correction_protection.py). A GDPR erasure is not an ordinary delete. Every
case naming an erased fact, whatever its status and whichever profile it is
filed under (the foreign key does not care), is closed here: one audit row with
ids, prior status, when and who, then its events and the case itself go, in
the same transaction that deletes the facts. So either the facts and their
cases go together, or nothing does and the erasure can simply be run again.

The audit row holds no memory text, like the ledger itself: the columns are
checked against M042's forbidden raw-text columns below. A case closed this way
cannot be put back: both its facts are gone.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

ERASED_ON_REQUEST = "erased_on_request"

#: Created on first use (no migration: a migration copies memory.db first, and
#: this table only receives rows when a person asks to be erased).
_COLUMNS = ("case_id", "profile_id", "predecessor_fact_id", "successor_fact_id",
            "reason_code", "prior_status", "erasure_id", "actor_id", "closed_reason",
            "erased_at")
DDL = (
    """CREATE TABLE IF NOT EXISTS correction_cases_erased (
    erased_case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    predecessor_fact_id TEXT NOT NULL,
    successor_fact_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    prior_status TEXT NOT NULL,
    erasure_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    closed_reason TEXT NOT NULL,
    erased_at TEXT NOT NULL
)""",
    "CREATE INDEX IF NOT EXISTS idx_correction_cases_erased_profile "
    "ON correction_cases_erased (profile_id, erased_case_id)",
)


def _assert_text_free() -> None:
    from superlocalmemory.storage.migrations.M042_correction_case_ledger import (
        _FORBIDDEN_RAW_COLUMNS,
    )

    leaked = _FORBIDDEN_RAW_COLUMNS & set(_COLUMNS)
    if leaked:  # pragma: no cover - a guard against a future column
        raise RuntimeError(f"erased-case audit must not hold text columns: {sorted(leaked)}")


def ensure_table(target: Any) -> None:
    """Create the audit table if this store predates it. Idempotent, no data."""
    _assert_text_free()
    for statement in DDL:
        target.execute(statement)


def close_for_erasure(target: Any, fact_ids: Iterable[str], *, erasure_id: str,
                      actor_id: str) -> list[str]:
    """Close every case naming one of ``fact_ids``. Call inside the transaction
    that deletes the facts, before the delete. Returns the closed case ids."""
    from superlocalmemory.core.overtaken_cases import cases_naming

    cases = cases_naming(target, fact_ids)
    if not cases:
        return []
    ensure_table(target)
    now = datetime.now(timezone.utc).isoformat()
    for case in cases:
        values = (str(case["case_id"]), str(case["profile_id"]),
                  str(case["predecessor_fact_id"]), str(case["successor_fact_id"]),
                  str(case["reason_code"]), str(case["status"]), erasure_id,
                  actor_id or "unknown", ERASED_ON_REQUEST, now)
        target.execute(f"INSERT INTO correction_cases_erased ({', '.join(_COLUMNS)}) "
                       f"VALUES ({', '.join('?' * len(_COLUMNS))})", values)
        target.execute("DELETE FROM correction_events WHERE case_id = ?", (values[0],))
        target.execute("DELETE FROM correction_cases WHERE case_id = ?", (values[0],))
    closed = [str(c["case_id"]) for c in cases]
    logger.info("correction cases closed, erased on request: %s",
                ",".join(c[:12] for c in closed))
    return closed


def delete_erased_facts(db: Any, targets: list[tuple[str, str | None]], profile_id: str, *,
                        erasure_id: str, has_siblings: Callable[[str, str], bool],
                        counts: dict) -> None:
    """Close the cases, delete the facts (and the scenes only they made up) and
    their orphaned source memories: one transaction. On failure nothing is
    deleted and the erasure can be run again."""
    with db.transaction():
        closed = close_for_erasure(db, [fid for fid, _ in targets],
                                   erasure_id=erasure_id, actor_id="gdpr")
        for fid, mid in targets:
            db.remove_fact_from_scenes(fid, profile_id)  # an emptied scene keeps its theme text
            db.delete_fact(fid)
            if mid and not has_siblings(mid, profile_id):
                db.execute("DELETE FROM memories WHERE memory_id = ? AND profile_id = ?",
                           (mid, profile_id))
    counts["correction_cases_erased"] = len(closed)


def erase_profile_cases(db: Any, profile_id: str, *, erasure_id: str) -> int:
    """Before a whole profile is erased: close every case naming one of its
    facts (an audit row is kept only where the case is filed under another
    profile), drop the cases still filed under it, and its own audit rows.
    One transaction. Returns the number of cases removed.

    Without this the ledger's ``ON DELETE RESTRICT`` refused the delete of
    every fact and memory the profile's corrections named, and the erasure
    stopped partway with a foreign-key error.
    """
    # A custom DB wrapper may offer only ``execute``: the same statements, in
    # the same order, then run one by one (GDPRCompliance accepts such wrappers).
    begin = getattr(db, "transaction", None) or contextlib.nullcontext
    try:
        with begin():
            fact_ids = [str(dict(r)["fact_id"]) for r in db.execute(
                "SELECT fact_id FROM atomic_facts WHERE profile_id = ?", (profile_id,))]
            closed = close_for_erasure(db, fact_ids, erasure_id=erasure_id, actor_id="gdpr")
            own = [str(dict(r)["case_id"]) for r in db.execute(
                "SELECT case_id FROM correction_cases WHERE profile_id = ?", (profile_id,))]
            for case_id in own:
                db.execute("DELETE FROM correction_events WHERE case_id = ?", (case_id,))
                db.execute("DELETE FROM correction_cases WHERE case_id = ?", (case_id,))
            if closed:  # the table exists: close_for_erasure created it
                db.execute("DELETE FROM correction_cases_erased WHERE profile_id = ?",
                           (profile_id,))
            _erase_overtaken(db, profile_id)
    except Exception as exc:
        if "no such table: correction_cases" in str(exc):
            return 0  # a store from before the correction ledger
        raise
    return len(closed) + len(own)


def _erase_overtaken(db: Any, profile_id: str) -> None:
    """The profile's cases a user action or a bulk dismissal moved out of the
    ledger (core/overtaken_cases.py): they name its facts and outlive nothing."""
    try:
        db.execute("DELETE FROM correction_cases_overtaken WHERE profile_id = ?", (profile_id,))
    except Exception as exc:
        if "no such table: correction_cases_overtaken" not in str(exc):
            raise  # a store that never overtook a case has no such table


__all__ = ["ERASED_ON_REQUEST", "close_for_erasure", "delete_erased_facts",
           "ensure_table", "erase_profile_cases"]
