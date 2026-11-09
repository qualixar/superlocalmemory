# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The user's action wins over a correction only a machine proposed.

Varun's decision (2026-10-06): an explicit delete, replace or edit by the user
or their assistant goes through even when a correction case that SLM itself
proposed, and nobody reviewed, names that fact. That pending case is closed as
"overtaken by a user action": moved, whole and unchanged, to
``correction_cases_overtaken`` with who did what and when (the audit row), and
it can be put back (``restore``) while both of its facts still exist.

Before this, such a case made its facts undeletable forever: the ledger refers
to both facts ``ON DELETE RESTRICT`` and keeps every case. On a real 22k-fact
store 1,692 facts were held only by machine proposals (1,081 cases, all still
``proposed``, none reviewed), and a memory's own facts often proposed to correct
each other.

Who proposed a case is recorded on the case itself: the automatic detectors in
core/store_pipeline.py write ``proposed_by_actor_kind = 'host_attested'`` with
reason ``consolidation_update``, ``consolidation_supersede`` or
``temporal_contradiction``; a person's edit is ``host_authenticated`` with
``direct_content_correction``, and a caller's ``replaces`` is applied at once.
A case counts as machine-proposed only when BOTH say so. A case a person
proposed, or any case already applied (or rejected, or rolled back), protects
exactly as before.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MACHINE_REASONS = frozenset({"consolidation_update", "consolidation_supersede",
                             "temporal_contradiction"})
MACHINE_ACTOR_KIND = "host_attested"
OVERTAKEN = "overtaken by a user action"

#: Created with the rest of the schema (storage/schema.py create_all_tables)
#: and, as a no-op guard, by ``overtake`` itself. Not a migration on purpose:
#: a migration on memory.db makes every upgrade copy memory.db first (about
#: 8 s per GB before SLM answers), and this table only ever receives rows
#: when a user acts. Ids and lifecycle metadata only, like the ledger itself.
DDL = (
    """CREATE TABLE IF NOT EXISTS correction_cases_overtaken (
    overtake_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    predecessor_fact_id TEXT NOT NULL,
    successor_fact_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    case_json TEXT NOT NULL,
    events_json TEXT NOT NULL,
    user_action TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    closed_reason TEXT NOT NULL,
    overtaken_at TEXT NOT NULL,
    restored_at TEXT,
    restored_by TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_correction_cases_overtaken_case "
    "ON correction_cases_overtaken (case_id, restored_at)",
    "CREATE INDEX IF NOT EXISTS idx_correction_cases_overtaken_profile "
    "ON correction_cases_overtaken (profile_id, overtake_id)",
)


class RestoreRefused(ValueError):
    """The case cannot be put back as it was."""


class OvertakeRaced(RuntimeError):
    """A case changed between being read and being closed; the action aborts."""


def _run(target: Any, sql: str, params: tuple = ()) -> list[dict]:
    """Rows as dicts from a sqlite3 connection or a DatabaseManager."""
    result = target.execute(sql, params)
    if hasattr(result, "description"):
        cols = [d[0] for d in result.description] if result.description else []
        return [dict(zip(cols, row)) for row in result.fetchall()]
    return [dict(row) for row in result]


def is_machine_pending(case: dict) -> bool:
    return (str(case.get("status")) == "proposed"
            and str(case.get("reason_code")) in MACHINE_REASONS
            and str(case.get("proposed_by_actor_kind")) == MACHINE_ACTOR_KIND)


def cases_naming(target: Any, fact_ids: Iterable[str], *, predecessor_only: bool = False,
                 profile_id: str | None = None) -> list[dict]:
    """Every case whose foreign key refers to one of the facts: any profile
    (the foreign key does not care), or only ``profile_id``'s when given."""
    ids = list(dict.fromkeys(str(f) for f in fact_ids))
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    where = f"(predecessor_fact_id IN ({ph})" + (
        ")" if predecessor_only else f" OR successor_fact_id IN ({ph}))")
    params = tuple(ids) if predecessor_only else tuple(ids) * 2
    if profile_id is not None:
        where, params = where + " AND profile_id = ?", (*params, profile_id)
    try:
        return _run(target, f"SELECT * FROM correction_cases WHERE {where} "  # noqa: S608
                    "ORDER BY created_at, case_id", params)
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return []
        raise


def blocking(cases: Iterable[dict]) -> list[dict]:
    """The cases a user action does NOT overtake."""
    return [c for c in cases if not is_machine_pending(c)]


def ensure_table(target: Any) -> None:
    """Create the audit table if this store predates it. Idempotent, no data."""
    for statement in DDL:
        target.execute(statement)


def overtake(target: Any, cases: Iterable[dict], *, user_action: str, actor_id: str,
             operation_id: str, closed_reason: str = OVERTAKEN) -> list[str]:
    """Close machine-pending cases for a user action. Call inside the action's
    own write transaction so the case closes only if the action happens.

    Each case is removed only if it is STILL a pending machine proposal; if it
    changed since it was read (reviewed meanwhile), nothing is moved and
    ``OvertakeRaced`` aborts the caller's transaction, action included.

    ``closed_reason`` is the audit text; the default says a user action closed
    the case. The bulk dismissal (core/dismissed_cases.py) passes its own.
    """
    closed: list[str] = []
    todo = [c for c in cases if is_machine_pending(c)]
    if not todo:
        return closed
    ensure_table(target)
    now = datetime.now(timezone.utc).isoformat()
    for case in todo:
        case_id = str(case["case_id"])
        events = _run(target, "SELECT * FROM correction_events WHERE case_id = ? "
                      "ORDER BY system_occurred_at, event_id", (case_id,))
        target.execute(
            "INSERT INTO correction_cases_overtaken (case_id, profile_id, predecessor_fact_id, "
            "successor_fact_id, reason_code, case_json, events_json, user_action, actor_id, "
            "operation_id, closed_reason, overtaken_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (case_id, str(case["profile_id"]), str(case["predecessor_fact_id"]),
             str(case["successor_fact_id"]), str(case["reason_code"]),
             json.dumps(case, sort_keys=True), json.dumps(events, sort_keys=True),
             user_action, actor_id or "unknown", operation_id, closed_reason, now))
        target.execute("DELETE FROM correction_events WHERE case_id = ?", (case_id,))
        gone = _run(target, "DELETE FROM correction_cases WHERE case_id = ? AND status = ? "
                    "AND version = ? RETURNING case_id",
                    (case_id, "proposed", case.get("version")))
        if not gone:
            raise OvertakeRaced(f"correction case {case_id} changed while the {user_action} "
                                "ran; nothing was changed, try again")
        closed.append(case_id)
    # A bulk dismissal logs one summary line itself; one line per case would be thousands.
    log = logger.info if closed_reason == OVERTAKEN else logger.debug
    log("correction cases overtaken by a user %s: %s", user_action,
        ",".join(c[:12] for c in closed))
    return closed


def _insert(target: Any, table: str, row: dict) -> None:
    cols = list(row)
    target.execute(f"INSERT INTO {table} ({', '.join(cols)}) "  # noqa: S608
                   f"VALUES ({', '.join('?' * len(cols))})", tuple(row[c] for c in cols))


def _refusal(target: Any, case: dict) -> str | None:
    """Why ``case`` cannot be put back right now, or None if it can."""
    case_id = case["case_id"]
    for side in ("predecessor_fact_id", "successor_fact_id"):
        if not _run(target, "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (case[side],)):
            return (f"{case[side]} no longer exists (it was deleted), so case {case_id} "
                    "cannot come back")
    if _run(target, "SELECT 1 FROM correction_cases WHERE case_id = ? OR (profile_id = ? "
            "AND idempotency_key = ?)", (case_id, case["profile_id"], case["idempotency_key"])):
        return f"case {case_id} was proposed again since, so it is already back"
    if _run(target, "SELECT 1 FROM correction_cases WHERE profile_id = ? AND "
            "predecessor_fact_id = ? AND status IN ('proposed', 'applied')",
            (case["profile_id"], case["predecessor_fact_id"])):
        return (f"{case['predecessor_fact_id']} has another open correction now (the user's "
                "own); reject or roll that back first")
    return None


def _waiting(target: Any, case_id: str) -> dict | None:
    try:
        rows = _run(target, "SELECT * FROM correction_cases_overtaken WHERE case_id = ? AND "
                    "restored_at IS NULL ORDER BY overtake_id DESC LIMIT 1", (case_id,))
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return rows[0] if rows else None


def restore(target: Any, case_id: str, *, profile_id: str | None = None,
            actor_id: str = "") -> dict[str, Any]:
    """Put an overtaken case back, exactly as it was. Once per overtake.

    ``profile_id`` limits it to that profile's cases (another profile's case is
    "not found", like a missing one). Call inside one write transaction.
    """
    row = _waiting(target, case_id)
    if row is None or (profile_id is not None and row["profile_id"] != profile_id):
        raise RestoreRefused(f"no overtaken case {case_id} waiting to be restored")
    case, events = json.loads(row["case_json"]), json.loads(row["events_json"])
    why = _refusal(target, case)
    if why:
        raise RestoreRefused(why)
    _insert(target, "correction_cases", case)
    for event in events:
        _insert(target, "correction_events", event)
    target.execute("UPDATE correction_cases_overtaken SET restored_at = ?, restored_by = ? "
                   "WHERE overtake_id = ?", (datetime.now(timezone.utc).isoformat(),
                                             actor_id or "unknown", row["overtake_id"]))
    return {"restored": case_id, "events": len(events)}


_LISTED = ("overtake_id, case_id, profile_id, predecessor_fact_id, successor_fact_id, "
           "reason_code, user_action, actor_id, operation_id, closed_reason, overtaken_at, "
           "restored_at, restored_by, case_json")


def listing(target: Any, profile_id: str | None = None, limit: int = 100) -> list[dict]:
    """Overtaken cases, newest first: ids, who, when, and whether (and why not)
    each can be put back. No memory text: the ledger never holds any."""
    where, params = ("WHERE profile_id = ? ", (profile_id,)) if profile_id is not None else ("", ())
    try:
        rows = _run(target, f"SELECT {_LISTED} FROM correction_cases_overtaken {where}"  # noqa: S608
                    "ORDER BY overtake_id DESC LIMIT ?", (*params, int(limit)))
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return []
        raise
    out = []
    for row in rows:
        case = json.loads(row.pop("case_json"))
        why = "already restored" if row["restored_at"] else _refusal(target, case)
        out.append({**row, "restorable": why is None, "not_restorable_because": why})
    return out


__all__ = ["MACHINE_REASONS", "OVERTAKEN", "OvertakeRaced", "RestoreRefused", "blocking",
           "cases_naming", "ensure_table", "is_machine_pending", "listing", "overtake",
           "restore"]
