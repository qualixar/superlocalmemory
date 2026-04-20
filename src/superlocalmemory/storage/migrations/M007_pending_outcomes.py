# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §1.2 + LLD-08

"""M007 — pending_outcomes table (memory.db).

LLD-00 §1.2 makes ``pending_outcomes`` the single source of truth for
in-flight recall outcomes awaiting reward settlement. An earlier draft
split pending state across a cache.db table which has been retired
(see LLD-00 §1.2 for the name of the predecessor). All pending rows
are now crash-safe, profile-scoped, and one-row-per-recall — signals
live in the ``signals_json`` blob, not separate rows.

Target DB: memory.db. Schema follows LLD-00 §1.2 verbatim. Idempotent:
``verify(conn)`` returns True once every required column is present on
the ``pending_outcomes`` table.
"""

from __future__ import annotations

import sqlite3

NAME = "M007_pending_outcomes"
DB_TARGET = "memory"

_REQUIRED_COLS = frozenset({
    "outcome_id", "profile_id", "session_id", "recall_query_id",
    "fact_ids_json", "query_text_hash", "created_at_ms", "expires_at_ms",
    "signals_json", "status",
})


def verify(conn: sqlite3.Connection) -> bool:
    try:
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(pending_outcomes)"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS.issubset(cols)


DDL = """
CREATE TABLE IF NOT EXISTS pending_outcomes (
    outcome_id       TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    recall_query_id  TEXT NOT NULL,
    fact_ids_json    TEXT NOT NULL,
    query_text_hash  TEXT NOT NULL,
    created_at_ms    INTEGER NOT NULL,
    expires_at_ms    INTEGER NOT NULL,
    signals_json     TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_pending_profile_expires
    ON pending_outcomes(profile_id, expires_at_ms);
CREATE INDEX IF NOT EXISTS idx_pending_status
    ON pending_outcomes(status, expires_at_ms);
"""
