# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3

NAME = "M034_obligation_integrity"
DB_TARGET = "memory"

DDL = """
ALTER TABLE projection_obligations ADD COLUMN context_digest TEXT;
ALTER TABLE projection_obligations ADD COLUMN verify_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE projection_obligations ADD COLUMN last_verified_at REAL;
"""

_ADDED_COLUMNS = {
    "context_digest": "ALTER TABLE projection_obligations ADD COLUMN context_digest TEXT",
    "verify_attempts": (
        "ALTER TABLE projection_obligations "
        "ADD COLUMN verify_attempts INTEGER NOT NULL DEFAULT 0"
    ),
    "last_verified_at": (
        "ALTER TABLE projection_obligations ADD COLUMN last_verified_at REAL"
    ),
}


def apply(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn):
        return
    existing = _columns(conn)
    for column, statement in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(statement)


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn):
        return False
    return set(_ADDED_COLUMNS) <= _columns(conn)


def _table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projection_obligations'"
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {
        row[1]
        for row in conn.execute("PRAGMA table_info(projection_obligations)").fetchall()
    }
