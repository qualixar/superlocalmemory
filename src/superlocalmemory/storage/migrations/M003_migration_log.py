# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.3

"""M003 — bootstrap migration_log table.

Runs FIRST on both DBs. Purely idempotent ``CREATE TABLE IF NOT EXISTS``.
No transaction needed — a single DDL statement on SQLite is atomic.
"""

from __future__ import annotations

import sqlite3

NAME = "M003_migration_log"


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if migration_log already exists with expected columns."""
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(migration_log)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return {"name", "applied_at", "ddl_sha256",
            "rows_affected", "status"} <= cols
DB_TARGET = "learning"  # Also gets replicated to memory DB by the runner.

DDL = """
CREATE TABLE IF NOT EXISTS migration_log (
    name           TEXT PRIMARY KEY,
    applied_at     TEXT NOT NULL,
    ddl_sha256     TEXT NOT NULL,
    rows_affected  INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL
);
"""
