# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Dismiss the machine-proposed correction cases nobody reviewed, in bulk.

``slm corrections dismiss`` (issue #309). In Mode B/C the save path files a
``proposed`` case whenever it only guesses that a new memory updates an older
one. Nobody reviews them and they pile up (17,423 on one real store).

It never REJECTS them. A rejected case counts as decided, and a decided case
protects its facts from deletion forever (core/correction_protection.py), so
rejecting 17k cases would lock about 34k facts. A pending machine proposal
protects nothing and changes no recall result already, so the honest move is
the one a user's own delete or edit makes: close it as overtaken
(``overtaken_cases.overtake``). Each case moves whole, with its event log, to
``correction_cases_overtaken`` under ``user_action = 'dismissed'`` and can be
put back with ``slm corrections restore-overtaken``.

Only a case that is STILL a pending machine proposal is moved: it is read
again inside the write that moves it, and the move itself is compare-and-set
on the case version. A case someone reviewed meanwhile is skipped and counted.
The work runs in chunks of ``CHUNK_SIZE`` cases per commit, so a crash loses at
most one chunk and a re-run simply continues.

Ids and counts only; no memory text is read or returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from superlocalmemory.core import overtaken_cases as ot
from superlocalmemory.storage.correction_dismissals import DISMISSED_USER_ACTION

logger = logging.getLogger(__name__)

DISMISSED = DISMISSED_USER_ACTION
DISMISSED_REASON = "dismissed in bulk by the user"
CHUNK_SIZE = 500
_SAVEPOINT = "dismiss_one_case"


@dataclass(frozen=True)
class DismissFilter:
    """Which pending machine cases: one profile, optionally some reasons and
    only those created before ``before`` (an ISO timestamp, UTC)."""

    profile_id: str
    reasons: tuple[str, ...] = ()
    before: str | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.reasons) - ot.MACHINE_REASONS)
        if unknown:
            raise ValueError(
                f"unknown reason {', '.join(unknown)}: only machine-proposed cases can be "
                f"dismissed ({', '.join(sorted(ot.MACHINE_REASONS))})")

    @property
    def reason_set(self) -> tuple[str, ...]:
        return tuple(sorted(self.reasons or ot.MACHINE_REASONS))


def _utc(moment: datetime) -> datetime:
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc)


def cutoff_for(*, older_than_days: int | None, before: str | None,
               now: datetime | None = None) -> str | None:
    """The ISO instant a case must predate, from ``--older-than`` and/or
    ``--before`` (a case must satisfy both, so the earlier one wins)."""
    cutoffs: list[datetime] = []
    if older_than_days is not None:
        if int(older_than_days) < 0:
            raise ValueError("--older-than must be a number of days, 0 or more")
        start = _utc(now or datetime.now(timezone.utc))
        cutoffs.append(start - timedelta(days=int(older_than_days)))
    if before:
        try:
            parsed = (datetime.combine(date.fromisoformat(before), datetime.min.time())
                      if len(before) == 10 else datetime.fromisoformat(before))
        except ValueError:
            raise ValueError("--before must be a date such as 2026-11-01 "
                             "(or a full ISO date and time)") from None
        cutoffs.append(_utc(parsed))
    return min(cutoffs).isoformat() if cutoffs else None


def _where(flt: DismissFilter) -> tuple[str, tuple]:
    reasons = flt.reason_set
    sql = (f"profile_id = ? AND status = 'proposed' AND proposed_by_actor_kind = ? "
           f"AND reason_code IN ({','.join('?' * len(reasons))})")
    params: tuple = (flt.profile_id, ot.MACHINE_ACTOR_KIND, *reasons)
    if flt.before:
        sql, params = sql + " AND created_at < ?", (*params, flt.before)
    return sql, params


def _no_ledger(exc: Exception) -> bool:
    return "no such table" in str(exc).lower()


def preview(db: Any, flt: DismissFilter) -> dict[str, Any]:
    """What a dismissal would move: counts by reason and the oldest and newest
    creation dates. Reads only."""
    where, params = _where(flt)
    try:
        rows = db.execute(
            "SELECT reason_code, COUNT(*) AS n, MIN(created_at) AS oldest, "  # noqa: S608
            f"MAX(created_at) AS newest FROM correction_cases WHERE {where} "
            "GROUP BY reason_code", params)
    except Exception as exc:
        if not _no_ledger(exc):
            raise
        rows = []
    found = [dict(r) for r in rows]
    return {"eligible": sum(r["n"] for r in found),
            "by_reason": {r["reason_code"]: r["n"] for r in found},
            "oldest": min((r["oldest"] for r in found), default=None),
            "newest": max((r["newest"] for r in found), default=None)}


def _eligible_ids(db: Any, flt: DismissFilter) -> list[str]:
    where, params = _where(flt)
    try:
        rows = db.execute(f"SELECT case_id FROM correction_cases WHERE {where} "  # noqa: S608
                          "ORDER BY created_at, case_id", params)
    except Exception as exc:
        if not _no_ledger(exc):
            raise
        return []
    return [str(dict(r)["case_id"]) for r in rows]


def _current(db: Any, case_id: str, flt: DismissFilter) -> dict | None:
    """The case as it is NOW, or None if it is no longer a pending machine
    proposal of the filtered profile and reasons."""
    rows = db.execute("SELECT * FROM correction_cases WHERE case_id = ? AND profile_id = ?",
                      (case_id, flt.profile_id))
    case = dict(rows[0]) if rows else None
    if case is None or not ot.is_machine_pending(case) or case["reason_code"] not in flt.reason_set:
        return None
    return case


def _dismiss_chunk(db: Any, ids: list[str], flt: DismissFilter, *, actor_id: str,
                   operation_id: str) -> tuple[int, int, int]:
    """One transaction's worth. Returns (dismissed, raced, changed)."""
    # Open the transaction now: a savepoint taken first would be the outermost
    # one, and releasing it would commit that single case on its own.
    db.execute("UPDATE correction_cases SET case_id = case_id WHERE 0")
    ot.ensure_table(db)
    dismissed = raced = changed = 0
    for case_id in ids:
        case = _current(db, case_id, flt)
        if case is None:
            changed += 1
            continue
        db.execute(f"SAVEPOINT {_SAVEPOINT}")
        try:
            ot.overtake(db, [case], user_action=DISMISSED, actor_id=actor_id,
                        operation_id=operation_id, closed_reason=DISMISSED_REASON)
        except ot.OvertakeRaced:
            db.execute(f"ROLLBACK TO {_SAVEPOINT}")   # undo this case's partial writes only
            db.execute(f"RELEASE {_SAVEPOINT}")
            raced += 1
            continue
        db.execute(f"RELEASE {_SAVEPOINT}")
        dismissed += 1
    return dismissed, raced, changed


def dismiss(db: Any, flt: DismissFilter, *, actor_id: str, operation_id: str,
            chunk_size: int = CHUNK_SIZE) -> dict[str, int]:
    """Move every eligible case to the overtaken table, ``chunk_size`` per commit.

    Returns ``dismissed`` plus ``skipped_raced`` (changed during the move) and
    ``skipped_changed`` (no longer a pending machine proposal when reached).
    """
    ids = _eligible_ids(db, flt)
    total = [0, 0, 0]
    for start in range(0, len(ids), max(1, int(chunk_size))):
        with db.transaction():
            part = _dismiss_chunk(db, ids[start:start + chunk_size], flt,
                                  actor_id=actor_id, operation_id=operation_id)
        total = [a + b for a, b in zip(total, part)]
    logger.info("correction cases dismissed in bulk in %s: %d moved, %d raced, %d changed "
                "(operation %s)", flt.profile_id, *total, operation_id)
    return {"dismissed": total[0], "skipped_raced": total[1], "skipped_changed": total[2]}


__all__ = ["CHUNK_SIZE", "DISMISSED", "DISMISSED_REASON", "DismissFilter", "cutoff_for",
           "dismiss", "preview"]
