# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M078: cues on prospective memories (4.2.30, W3-8, blueprint WS-4).

A cue says when a memory should be shown: when an agent edits a path that
matches a glob, uses a tool, hits an error message, works in a repository, or
reaches a date. One row binds one cue to one fact. ``origin`` is ``explicit``
(the agent or the user gave it) or ``extracted`` (a path found in the text of a
prospective memory). The matching is deterministic and lives in
``core/cue_triggers.py``; the hook that delivers is ``slm hook cue``.

CHECKs hold invariants only (non-empty), never a list of kinds or origins,
which SQLite could not extend after release (W0-29). The table holds the cue
pattern and ids, no memory text. Erasing a fact erases its cues (I-11).
"""

from __future__ import annotations

import sqlite3

NAME = "M078_memory_cues"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS memory_cues (
    cue_id     TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (length(kind) > 0),
    pattern    TEXT NOT NULL CHECK (length(pattern) > 0),
    created_at TEXT NOT NULL,
    origin     TEXT NOT NULL CHECK (length(origin) > 0)
);
CREATE INDEX IF NOT EXISTS idx_memory_cues_kind ON memory_cues (profile_id, kind);
CREATE INDEX IF NOT EXISTS idx_memory_cues_fact ON memory_cues (fact_id)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_memory_cues_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM memory_cues WHERE fact_id = OLD.fact_id;
END
"""


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    if _has_facts(conn):
        conn.execute(_FACT_TRIGGER)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not {"memory_cues", "idx_memory_cues_kind"} <= names:
        return False
    return not _has_facts(conn) or "trg_memory_cues_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
