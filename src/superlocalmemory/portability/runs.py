# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The import run records (M085): create, progress, read, list. Ids, counts and plain words only.

Every function takes an open ``sqlite3`` connection and does one short statement or a few,
so no caller holds the writer longer than a statement. Memory text is never stored here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.model import SKIP_WORDS

QUEUED, RUNNING, DONE, CANCELLED, FAILED = "queued", "running", "done", "cancelled", "failed"
ROLLING_BACK, ROLLED_BACK = "rolling_back", "rolled_back"
#: States in which a run is still doing something.
ACTIVE = (QUEUED, RUNNING, ROLLING_BACK)
#: A run in one of these can be undone: it saved something and has not been rolled back.
UNDOABLE_STATES = (DONE, CANCELLED, FAILED, QUEUED, RUNNING)
MAX_LIST = 100
MAX_LABEL = 120

_COLS = ("run_id, profile_id, format, source_label, state, actor, channel, conversations, "
         "total_items, saved_items, skipped_json, first_at, last_at, dispute_cap, judge_cap, "
         "disputes_opened, disputes_unlisted, note, cancel_requested, decision_id, created_at, "
         "started_at, finished_at, rolled_back_at, rolled_back_by")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def has_tables(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                             "AND name='import_runs'").fetchone())


def clean_label(name: str) -> str:
    """A file name to show: no folders, no control characters, bounded."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(ch for ch in base if ch.isprintable())[:MAX_LABEL] or "file"


def create(conn: sqlite3.Connection, *, profile_id: str, fmt: str, source_label: str, actor: str,
           channel: str, conversations: int, total_items: int, skipped: dict[str, int],
           first_at: str, last_at: str, decision_id: str = "") -> str:
    run_id = "imp_" + uuid.uuid4().hex[:20]
    conn.execute(
        "INSERT INTO import_runs (run_id, profile_id, format, source_label, state, actor, channel, "
        "conversations, total_items, skipped_json, first_at, last_at, dispute_cap, judge_cap, "
        "decision_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, profile_id, fmt, clean_label(source_label), QUEUED, actor, channel, conversations,
         total_items, json.dumps(skipped, sort_keys=True), first_at or None, last_at or None,
         C.DISPUTES_PER_RUN, C.JUDGE_CALLS_PER_RUN, decision_id or None, now()))
    return run_id


def view(row: Any) -> dict:
    """A run as plain data, with the skipped counts also in the words a person reads."""
    values = dict(zip(_COLS.replace("\n", "").split(", "), tuple(row), strict=True))
    try:
        skipped = json.loads(values.pop("skipped_json") or "{}")
    except ValueError:
        skipped = {}
    values["skipped"] = skipped
    values["skipped_words"] = [{"reason": k, "count": v, "words": SKIP_WORDS.get(k, k)}
                               for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]) if v]
    values["cancel_requested"] = bool(values["cancel_requested"])
    values["active"] = values["state"] in ACTIVE
    values["undoable"] = values["state"] in UNDOABLE_STATES
    return values


def get(conn: sqlite3.Connection, profile_id: str, run_id: str) -> dict | None:
    if not has_tables(conn):
        return None
    row = conn.execute(f"SELECT {_COLS} FROM import_runs WHERE run_id = ? AND profile_id = ?",
                       (run_id, profile_id)).fetchone()
    return view(row) if row else None


def recent(conn: sqlite3.Connection, profile_id: str, limit: int = 20) -> list[dict]:
    if not has_tables(conn):
        return []
    rows = conn.execute(
        f"SELECT {_COLS} FROM import_runs WHERE profile_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (profile_id, max(1, min(int(limit), MAX_LIST)))).fetchall()
    return [view(r) for r in rows]


def active_for(conn: sqlite3.Connection, profile_id: str) -> dict | None:
    """The run of this profile that is doing something now, if any (one at a time)."""
    if not has_tables(conn):
        return None
    marks = ",".join("?" * len(ACTIVE))
    row = conn.execute(f"SELECT {_COLS} FROM import_runs WHERE profile_id = ? "
                       f"AND state IN ({marks}) ORDER BY created_at DESC LIMIT 1",
                       (profile_id, *ACTIVE)).fetchone()
    return view(row) if row else None


def set_state(conn: sqlite3.Connection, run_id: str, state: str, *, note: str | None = None,
              only_from: tuple[str, ...] | None = None) -> bool:
    sets, args = ["state = ?"], [state]
    if state == RUNNING:
        sets.append("started_at = COALESCE(started_at, ?)")
        args.append(now())
    if state in (DONE, CANCELLED, FAILED, ROLLED_BACK):
        sets.append("finished_at = ?")
        args.append(now())
    if note is not None:
        sets.append("note = ?")
        args.append(note[:300])
    where, tail = "run_id = ?", [run_id]
    if only_from:
        where += f" AND state IN ({','.join('?' * len(only_from))})"
        tail += list(only_from)
    cursor = conn.execute(f"UPDATE import_runs SET {', '.join(sets)} WHERE {where}", (*args, *tail))
    return cursor.rowcount == 1


def record_batch(conn: sqlite3.Connection, run_id: str, profile_id: str,
                 saved: list[tuple[str, str]], skipped_extra: dict[str, int]) -> None:
    """One short transaction: the memories a batch saved, and what it left out."""
    stamp = now()
    conn.executemany(
        "INSERT OR IGNORE INTO import_run_items (run_id, profile_id, memory_id, fingerprint, "
        "saved_at) VALUES (?, ?, ?, ?, ?)",
        [(run_id, profile_id, memory_id, fp, stamp) for memory_id, fp in saved])
    if saved:
        conn.execute("UPDATE import_runs SET saved_items = (SELECT COUNT(*) FROM import_run_items "
                     "WHERE run_id = ?) WHERE run_id = ?", (run_id, run_id))
    if skipped_extra:
        row = conn.execute("SELECT skipped_json FROM import_runs WHERE run_id = ?",
                           (run_id,)).fetchone()
        merged = json.loads(row[0] or "{}") if row else {}
        for key, n in skipped_extra.items():
            merged[key] = int(merged.get(key, 0)) + n
        conn.execute("UPDATE import_runs SET skipped_json = ? WHERE run_id = ?",
                     (json.dumps(merged, sort_keys=True), run_id))


def request_cancel(conn: sqlite3.Connection, run_id: str) -> bool:
    cursor = conn.execute("UPDATE import_runs SET cancel_requested = 1 WHERE run_id = ? "
                          "AND state IN ('queued', 'running')", (run_id,))
    return cursor.rowcount == 1


def cancel_requested(conn: sqlite3.Connection, run_id: str) -> bool:
    row = conn.execute("SELECT cancel_requested FROM import_runs WHERE run_id = ?",
                       (run_id,)).fetchone()
    return bool(row and row[0])


def memory_ids(conn: sqlite3.Connection, run_id: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT memory_id FROM import_run_items WHERE run_id = ? ORDER BY rowid", (run_id,))]


def known_fingerprints(conn: sqlite3.Connection, profile_id: str, fps: list[str]) -> set[str]:
    """Which of ``fps`` an earlier run of this profile (not rolled back) already saved."""
    found: set[str] = set()
    for i in range(0, len(fps), 400):
        part = fps[i:i + 400]
        marks = ",".join("?" * len(part))
        found.update(r[0] for r in conn.execute(
            f"SELECT fingerprint FROM import_run_items WHERE profile_id = ? "
            f"AND fingerprint IN ({marks})", (profile_id, *part)))
    return found


def bump(conn: sqlite3.Connection, run_id: str, column: str, by: int = 1) -> None:
    """Add to one of the dispute counters (``disputes_opened`` or ``disputes_unlisted``)."""
    if column not in ("disputes_opened", "disputes_unlisted"):
        raise ValueError(column)
    conn.execute(f"UPDATE import_runs SET {column} = {column} + ? WHERE run_id = ?",  # noqa: S608
                 (by, run_id))


__all__ = ["ACTIVE", "CANCELLED", "DONE", "FAILED", "QUEUED", "ROLLED_BACK", "ROLLING_BACK",
           "RUNNING", "UNDOABLE_STATES", "active_for", "bump", "cancel_requested", "clean_label",
           "create", "get", "has_tables", "known_fingerprints", "memory_ids", "now",
           "record_batch", "recent", "request_cancel", "set_state", "view"]
