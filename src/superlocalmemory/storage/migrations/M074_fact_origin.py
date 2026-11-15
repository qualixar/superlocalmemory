# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M074: where each fact came from, recorded at admission (4.2.30, D-21).

One row per fact, written by ``storage/fact_origin.record`` beside the fact
itself: the channel, the transform, and the raw signals (source type, actor,
agent attribution, speaker, role, parent ids). No text. Facts stored before
this migration have no row; nothing is backfilled, because origin cannot be
reconstructed truthfully after the fact. Erasing a fact erases its row.
"""

from __future__ import annotations

import sqlite3

NAME = "M074_fact_origin"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS fact_origin (
    fact_id          TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    channel          TEXT NOT NULL,
    transform        TEXT NOT NULL,
    source_type      TEXT,
    actor_id         TEXT,
    agent_id         TEXT,
    agent_id_source  TEXT,
    speaker          TEXT,
    role             TEXT,
    parent_memory_id TEXT,
    parent_fact_id   TEXT,
    recorded_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_origin_channel ON fact_origin (profile_id, channel)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_fact_origin_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM fact_origin WHERE fact_id = OLD.fact_id;
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
    if "fact_origin" not in names:
        return False
    return not _has_facts(conn) or "trg_fact_origin_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
