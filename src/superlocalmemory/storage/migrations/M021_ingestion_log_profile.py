# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory — per-profile isolation (I-4)

"""M021 — per-profile ingestion_log dedup (memory.db, deferred).

``ingestion_log`` is the compatibility ledger for external-adapter ingests. Its
dedup UNIQUE constraint was global — ``UNIQUE(source_type, dedup_key)`` — so a
second profile ingesting the same source key could not record its own ledger
row (INSERT OR IGNORE silently no-op'd), losing that profile's fact_ids in the
ledger. (The authoritative dedup already moved to the M018 ingestion_operations
table, which is profile-scoped, so the memory itself is not starved — this
migration fixes the ledger so per-profile bookkeeping is complete.)

SQLite cannot alter a UNIQUE constraint in place, so the table is rebuilt:
rename → create with profile_id + UNIQUE(profile_id, source_type, dedup_key) →
copy rows backfilling profile_id='default' → drop old. Idempotent and tolerant
of a missing table (fresh installs create it correctly via schema_v343).

Deferred because ingestion_log is created at engine init (apply_v343_schema),
after ``apply_all`` runs — same reason as M016.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M021_ingestion_log_profile"
DB_TARGET = "memory"

# Documentation + drift hash. apply() below is the authoritative executor.
DDL = (
    "ALTER TABLE ingestion_log ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default';"
    "-- rebuild for UNIQUE(profile_id, source_type, dedup_key)"
)

_NEW_TABLE = """
CREATE TABLE ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    source_type TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    fact_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    status TEXT DEFAULT 'ingested',
    ingested_at TEXT NOT NULL,
    UNIQUE(profile_id, source_type, dedup_key)
)
"""


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def apply(conn: sqlite3.Connection) -> None:
    """Rebuild ingestion_log with a profile-scoped dedup constraint.

    No-op on a fresh install (table already has profile_id) or when the table
    doesn't exist yet. Existing rows backfill to the 'default' profile.
    """
    if not _table_exists(conn, "ingestion_log"):
        return
    if "profile_id" in _cols(conn, "ingestion_log"):
        return

    conn.execute("ALTER TABLE ingestion_log RENAME TO _ingestion_log_old")
    conn.executescript(_NEW_TABLE)
    # Copy legacy rows under the 'default' profile. Column set matches the
    # pre-migration schema (id, source_type, dedup_key, fact_ids, metadata,
    # status, ingested_at).
    old_cols = _cols(conn, "_ingestion_log_old")
    has_meta = "metadata" in old_cols
    if has_meta:
        conn.execute(
            "INSERT INTO ingestion_log "
            "(id, profile_id, source_type, dedup_key, fact_ids, metadata, "
            " status, ingested_at) "
            "SELECT id, 'default', source_type, dedup_key, fact_ids, metadata, "
            " status, ingested_at FROM _ingestion_log_old"
        )
    else:
        conn.execute(
            "INSERT INTO ingestion_log "
            "(id, profile_id, source_type, dedup_key, fact_ids, status, ingested_at) "
            "SELECT id, 'default', source_type, dedup_key, fact_ids, "
            " status, ingested_at FROM _ingestion_log_old"
        )
    conn.execute("DROP TABLE _ingestion_log_old")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_dedup "
        "ON ingestion_log(profile_id, source_type, dedup_key)"
    )


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once ingestion_log carries profile_id (or is absent on fresh DB)."""
    if not _table_exists(conn, "ingestion_log"):
        return True  # nothing to migrate; fresh install creates it correctly
    return "profile_id" in _cols(conn, "ingestion_log")
