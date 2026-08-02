# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3

NAME = "M033_projection_transactions"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS projection_obligations (
    obligation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id    TEXT NOT NULL,
    profile_id      TEXT NOT NULL,
    owner           TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('apply', 'erase')),
    subject_id      TEXT NOT NULL,
    revision        INTEGER NOT NULL DEFAULT 0,
    state           TEXT NOT NULL DEFAULT 'pending' CHECK (
                        state IN (
                            'pending', 'applied', 'verified',
                            'failed', 'compensated', 'erased'
                        )
                    ),
    checksum        TEXT,
    detail          TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE (operation_id, owner, kind)
);
CREATE INDEX IF NOT EXISTS idx_projection_obligations_operation
    ON projection_obligations (operation_id);
CREATE INDEX IF NOT EXISTS idx_projection_obligations_profile_state
    ON projection_obligations (profile_id, state);
CREATE INDEX IF NOT EXISTS idx_projection_obligations_state_updated
    ON projection_obligations (state, updated_at);
CREATE TABLE IF NOT EXISTS completion_manifests (
    operation_id        TEXT PRIMARY KEY,
    profile_id          TEXT NOT NULL,
    state               TEXT NOT NULL CHECK (
                            state IN ('COMPLETE', 'DEGRADED', 'FAILED')
                        ),
    all_met             INTEGER NOT NULL DEFAULT 0 CHECK (all_met IN (0, 1)),
    obligation_count    INTEGER NOT NULL DEFAULT 0,
    owner_evidence_json TEXT NOT NULL,
    manifest_hash       TEXT NOT NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_completion_manifests_profile_state
    ON completion_manifests (profile_id, state);
"""

_REQUIRED_OBLIGATION_COLUMNS = frozenset({
    "obligation_id",
    "operation_id",
    "profile_id",
    "owner",
    "kind",
    "subject_id",
    "revision",
    "state",
    "checksum",
    "detail",
    "attempts",
    "created_at",
    "updated_at",
})

_REQUIRED_MANIFEST_COLUMNS = frozenset({
    "operation_id",
    "profile_id",
    "state",
    "all_met",
    "obligation_count",
    "owner_evidence_json",
    "manifest_hash",
    "created_at",
    "updated_at",
})

_REQUIRED_OBJECTS = frozenset({
    "projection_obligations",
    "completion_manifests",
    "idx_projection_obligations_operation",
    "idx_projection_obligations_profile_state",
    "idx_projection_obligations_state_updated",
    "idx_completion_manifests_profile_state",
})


def apply(conn: sqlite3.Connection) -> None:
    if verify(conn):
        return
    conn.executescript(DDL)


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "projection_obligations"):
        return False
    if not _table_exists(conn, "completion_manifests"):
        return False
    if not _REQUIRED_OBLIGATION_COLUMNS <= _columns(conn, "projection_obligations"):
        return False
    if not _REQUIRED_MANIFEST_COLUMNS <= _columns(conn, "completion_manifests"):
        return False
    if not _has_unique_index(conn, "projection_obligations", ("operation_id", "owner", "kind")):
        return False
    present = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN "
            "(?, ?, ?, ?, ?, ?)",
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


def _has_unique_index(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...],
) -> bool:
    for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if not index[2]:
            continue
        index_columns = tuple(
            row[2] for row in conn.execute(f"PRAGMA index_info({index[1]})").fetchall()
        )
        if index_columns == columns:
            return True
    return False
