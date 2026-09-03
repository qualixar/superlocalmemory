# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""M031 — dead-letter queue for exhausted ingestion operations (Fix E, issue #77).

Additive migration: creates dead_letter_operations if not present.
Existing rows in ingestion_operations are unaffected.

When an M018 ingestion operation exhausts _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS
(10) it previously remained silently in FAILED state — invisible to operators and
unreachable by the materialiser.  This table gives operators a persistent,
inspectable record of every poisoned operation: original content, error, attempt
count, timestamps, and profile scope.

Schema design:
  - original_op_id references ingestion_operations.operation_id (soft ref — no FK
    so that dead-lettered rows survive if the source row is later cleaned up).
  - profile_id allows per-profile DLQ dashboards.
  - dead_lettered_at defaults to the current epoch for point-in-time auditing.
  - No TTL/expiry here — retention policy belongs to a future maintenance sweep.
"""

from __future__ import annotations

import sqlite3

NAME = "M031_dead_letter_operations"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS dead_letter_operations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    original_op_id      TEXT NOT NULL,
    operation_type      TEXT NOT NULL DEFAULT 'M018',
    content             TEXT,
    metadata_json       TEXT,
    error               TEXT,
    attempt_count       INTEGER,
    first_attempt_at    REAL,
    dead_lettered_at    REAL NOT NULL DEFAULT (unixepoch('now')),
    profile_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_dlq_profile
    ON dead_letter_operations (profile_id);
CREATE INDEX IF NOT EXISTS idx_dlq_op_id
    ON dead_letter_operations (original_op_id);
"""


def apply(conn: sqlite3.Connection) -> None:
    """Create the dead_letter_operations table and indexes idempotently."""
    conn.executescript(DDL)


def verify(conn: sqlite3.Connection) -> bool:
    """Return True only when the complete M031 contract is present."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='dead_letter_operations'"
    ).fetchone()
    if table is None:
        return False
    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(dead_letter_operations)"
        ).fetchall()
    }
    required = {
        "id",
        "original_op_id",
        "operation_type",
        "content",
        "error",
        "attempt_count",
        "dead_lettered_at",
        "profile_id",
    }
    return required <= columns


def repair(conn: sqlite3.Connection) -> None:
    """Re-run the idempotent apply as end-state repair (4.1.14 #133)."""
    apply(conn)
