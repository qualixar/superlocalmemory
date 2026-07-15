# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""M018 — durable canonical-ingestion operation records.

This is the EXPAND phase of the V3.7 ingestion migration.  It is additive:
legacy ``pending_memories`` and ``ingestion_log`` remain untouched until the
backfill and dual-write comparison prove the new contract.
"""

from __future__ import annotations

import sqlite3

NAME = "M018_ingestion_operations"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS ingestion_operations (
    operation_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    raw_metadata_json TEXT NOT NULL DEFAULT '{}',
    scope TEXT NOT NULL DEFAULT 'personal'
        CHECK (scope IN ('personal', 'project', 'shared', 'global')),
    shared_with_json TEXT NOT NULL DEFAULT '[]',
    trusted_actor_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    session_date TEXT NOT NULL DEFAULT '',
    speaker TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user',
    state TEXT NOT NULL DEFAULT 'raw'
        CHECK (state IN ('raw', 'queryable', 'enriching', 'complete', 'failed')),
    queryable_fact_ids_json TEXT NOT NULL DEFAULT '[]',
    final_fact_ids_json TEXT NOT NULL DEFAULT '[]',
    derivation_version TEXT NOT NULL DEFAULT '',
    derivation_state_json TEXT NOT NULL DEFAULT '{}',
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at REAL NOT NULL DEFAULT 0,
    next_retry_at REAL NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(profile_id, source_type, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_ingestion_operations_state
    ON ingestion_operations(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_operations_source_hash
    ON ingestion_operations(profile_id, source_hash);
"""


def apply(conn: sqlite3.Connection) -> None:
    """Create the additive operation table and indexes idempotently."""
    conn.executescript(DDL)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(ingestion_operations)").fetchall()
    }
    additive_columns = {
        "derivation_state_json": "TEXT NOT NULL DEFAULT '{}'",
        "session_date": "TEXT NOT NULL DEFAULT ''",
        "speaker": "TEXT NOT NULL DEFAULT ''",
        "role": "TEXT NOT NULL DEFAULT 'user'",
        "lease_owner": "TEXT NOT NULL DEFAULT ''",
        "lease_expires_at": "REAL NOT NULL DEFAULT 0",
        "next_retry_at": "REAL NOT NULL DEFAULT 0",
    }
    for name, declaration in additive_columns.items():
        if name not in columns:
            conn.execute(
                f"ALTER TABLE ingestion_operations ADD COLUMN {name} {declaration}"
            )


def verify(conn: sqlite3.Connection) -> bool:
    """Return true only when the complete M018 contract is present."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='ingestion_operations'"
    ).fetchone()
    if table is None:
        return False
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(ingestion_operations)").fetchall()
    }
    required = {
        "operation_id",
        "profile_id",
        "source_type",
        "idempotency_key",
        "source_hash",
        "raw_content",
        "state",
        "queryable_fact_ids_json",
        "final_fact_ids_json",
        "derivation_version",
        "derivation_state_json",
        "session_date",
        "speaker",
        "role",
        "lease_owner",
        "lease_expires_at",
        "next_retry_at",
        "attempt_count",
        "last_error",
    }
    if not required <= columns:
        return False
    indexes = {
        row[1]
        for row in conn.execute("PRAGMA index_list(ingestion_operations)").fetchall()
    }
    return "idx_ingestion_operations_state" in indexes
