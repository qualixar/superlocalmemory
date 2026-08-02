# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3

NAME = "M036_vector_row_map"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS vector_row_map (
    fact_id    TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    vec_rowid  INTEGER NOT NULL,
    PRIMARY KEY (fact_id)
);
CREATE INDEX IF NOT EXISTS idx_vector_row_map_profile
    ON vector_row_map (profile_id);
"""

_REQUIRED_COLUMNS = frozenset({"fact_id", "profile_id", "vec_rowid"})

_REQUIRED_OBJECTS = frozenset({
    "vector_row_map",
    "idx_vector_row_map_profile",
})


def apply(conn: sqlite3.Connection) -> None:
    _reconcile(conn)


def repair(conn: sqlite3.Connection) -> None:
    _reconcile(conn)


def _reconcile(conn: sqlite3.Connection) -> None:
    _rebuild_if_malformed(conn)
    conn.executescript(DDL)
    _backfill(conn)


def _rebuild_if_malformed(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "vector_row_map"):
        return
    if _REQUIRED_COLUMNS <= _columns(conn, "vector_row_map"):
        return
    conn.execute("DROP TABLE vector_row_map")


def _backfill(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "embedding_metadata"):
        return
    if not {"fact_id", "profile_id", "vec_rowid"} <= _columns(conn, "embedding_metadata"):
        return
    conn.execute(
        "INSERT INTO vector_row_map (fact_id, profile_id, vec_rowid) "
        "SELECT em.fact_id, em.profile_id, em.vec_rowid FROM embedding_metadata em "
        "WHERE em.fact_id IS NOT NULL AND em.vec_rowid IS NOT NULL "
        "AND NOT EXISTS ("
        "SELECT 1 FROM vector_row_map vrm WHERE vrm.fact_id = em.fact_id"
        ")"
    )


def verify(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "vector_row_map"):
        return False
    if not _REQUIRED_COLUMNS <= _columns(conn, "vector_row_map"):
        return False
    present = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?)",
            tuple(sorted(_REQUIRED_OBJECTS)),
        ).fetchall()
    }
    if not _REQUIRED_OBJECTS <= present:
        return False
    return _backfill_complete(conn)


def _backfill_complete(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "embedding_metadata"):
        return True
    if not {"fact_id", "vec_rowid"} <= _columns(conn, "embedding_metadata"):
        return True
    uncovered = conn.execute(
        "SELECT 1 FROM embedding_metadata em "
        "WHERE em.fact_id IS NOT NULL AND em.vec_rowid IS NOT NULL "
        "AND NOT EXISTS ("
        "SELECT 1 FROM vector_row_map vrm WHERE vrm.fact_id = em.fact_id"
        ") LIMIT 1"
    ).fetchone()
    return uncovered is None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
