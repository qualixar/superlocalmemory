# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M085: import runs, one row for each file a caller brought in from another app (4.2.30, WS-9, D-19).

An import is one unit. ``import_runs`` holds the run: which profile, which format, how many
items were found, saved and skipped (and why), who started it through which channel, the caps
it ran under and how many contradictions it opened, and where it stands (queued, running,
done, cancelled, failed, rolled back). ``import_run_items`` holds one row for each memory the
run saved: its memory id and a fingerprint of the item it came from. Rolling a run back removes
exactly those memories (and everything derived from them) and nothing else; the fingerprint is
how a second import of the same file finds what a first one already brought in.

The tables hold ids, counts and words for a person to read (a file name, a plain note), never
memory text. They carry ``profile_id`` so an erased profile takes its runs with it and a folded
profile moves them (I-11, ``storage/profile_fold``). Erasing a memory removes the run's record
of it through a trigger on ``memories``. CHECKs hold invariants only, never a list of formats
or states, which SQLite could not extend after release (W0-29).

Additive: a new pair of tables an older build never reads.
"""

from __future__ import annotations

import sqlite3

NAME = "M085_import_runs"
DB_TARGET = "memory"
#: Additive: new tables an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS import_runs (
    run_id           TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    format           TEXT NOT NULL CHECK (length(format) > 0),
    source_label     TEXT NOT NULL DEFAULT '',
    state            TEXT NOT NULL CHECK (length(state) > 0),
    actor            TEXT NOT NULL DEFAULT '',
    channel          TEXT NOT NULL DEFAULT '',
    conversations    INTEGER NOT NULL DEFAULT 0 CHECK (conversations >= 0),
    total_items      INTEGER NOT NULL DEFAULT 0 CHECK (total_items >= 0),
    saved_items      INTEGER NOT NULL DEFAULT 0 CHECK (saved_items >= 0),
    skipped_json     TEXT NOT NULL DEFAULT '{}',
    first_at         TEXT,
    last_at          TEXT,
    dispute_cap      INTEGER NOT NULL DEFAULT 0 CHECK (dispute_cap >= 0),
    judge_cap        INTEGER NOT NULL DEFAULT 0 CHECK (judge_cap >= 0),
    disputes_opened  INTEGER NOT NULL DEFAULT 0 CHECK (disputes_opened >= 0),
    disputes_unlisted INTEGER NOT NULL DEFAULT 0 CHECK (disputes_unlisted >= 0),
    note             TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
    decision_id      TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    rolled_back_at   TEXT,
    rolled_back_by   TEXT
);
CREATE INDEX IF NOT EXISTS idx_import_runs_recent ON import_runs (profile_id, created_at);
CREATE TABLE IF NOT EXISTS import_run_items (
    run_id      TEXT NOT NULL,
    profile_id  TEXT NOT NULL,
    memory_id   TEXT NOT NULL,
    fingerprint TEXT NOT NULL CHECK (length(fingerprint) > 0),
    saved_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_import_items_memory ON import_run_items (memory_id);
CREATE INDEX IF NOT EXISTS idx_import_items_fingerprint ON import_run_items (profile_id, fingerprint)
"""

#: I-11: erasing a memory erases the run's record of it.
_ERASE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_import_items_memory_erased
AFTER DELETE ON memories
BEGIN
    DELETE FROM import_run_items WHERE memory_id = OLD.memory_id;
END
"""


def _has_memories(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone())


def repair(conn: sqlite3.Connection) -> None:
    """Schema only: nothing here is data to rebuild."""
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    if _has_memories(conn):
        conn.execute(_ERASE_TRIGGER)


def apply(conn: sqlite3.Connection) -> None:
    repair(conn)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not {"import_runs", "import_run_items", "idx_import_runs_recent",
            "idx_import_items_memory", "idx_import_items_fingerprint"} <= names:
        return False
    return not _has_memories(conn) or "trg_import_items_memory_erased" in names
