# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3

NAME = "M035_erasure_receipts"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS erasure_receipts (
    erasure_id          TEXT PRIMARY KEY,
    profile_id          TEXT NOT NULL,
    subject_type        TEXT NOT NULL CHECK (
                            subject_type IN ('fact', 'entity', 'profile')
                        ),
    subject_id          TEXT NOT NULL,
    requested_by        TEXT NOT NULL DEFAULT '',
    fact_count          INTEGER NOT NULL DEFAULT 0,
    state               TEXT NOT NULL CHECK (state IN ('COMPLETE', 'FAILED')),
    all_erased          INTEGER NOT NULL DEFAULT 0 CHECK (all_erased IN (0, 1)),
    owner_evidence_json TEXT NOT NULL DEFAULT '[]',
    audit_hash          TEXT NOT NULL,
    requested_at        REAL NOT NULL,
    completed_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_erasure_receipts_profile
    ON erasure_receipts (profile_id);
CREATE INDEX IF NOT EXISTS idx_erasure_receipts_state
    ON erasure_receipts (state);
CREATE TABLE IF NOT EXISTS projection_tombstones (
    profile_id  TEXT NOT NULL,
    fact_id     TEXT NOT NULL,
    erasure_id  TEXT NOT NULL,
    memory_id   TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (profile_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_projection_tombstones_fact
    ON projection_tombstones (fact_id);
"""

_REQUIRED_RECEIPT_COLUMNS = frozenset({
    "erasure_id",
    "profile_id",
    "subject_type",
    "subject_id",
    "requested_by",
    "fact_count",
    "state",
    "all_erased",
    "owner_evidence_json",
    "audit_hash",
    "requested_at",
    "completed_at",
})

_REQUIRED_TOMBSTONE_COLUMNS = frozenset({
    "profile_id",
    "fact_id",
    "erasure_id",
    "memory_id",
    "created_at",
})

_REQUIRED_OBJECTS = frozenset({
    "erasure_receipts",
    "projection_tombstones",
    "idx_erasure_receipts_profile",
    "idx_erasure_receipts_state",
    "idx_projection_tombstones_fact",
})


def apply(conn: sqlite3.Connection) -> None:
    if verify(conn):
        return
    conn.executescript(DDL)


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "erasure_receipts"):
        return False
    if not _table_exists(conn, "projection_tombstones"):
        return False
    if not _REQUIRED_RECEIPT_COLUMNS <= _columns(conn, "erasure_receipts"):
        return False
    if not _REQUIRED_TOMBSTONE_COLUMNS <= _columns(conn, "projection_tombstones"):
        return False
    present = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?, ?, ?)",
            tuple(sorted(_REQUIRED_OBJECTS)),
        ).fetchall()
    }
    return _REQUIRED_OBJECTS <= present


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
