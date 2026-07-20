# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M019 — durable, non-fabricated derivation lineage."""

from __future__ import annotations

import sqlite3

NAME = "M019_derivation_lineage"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS derivation_lineage (
    lineage_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    operation_id TEXT NOT NULL DEFAULT '',
    derivation_version TEXT NOT NULL DEFAULT '',
    source_status TEXT NOT NULL
        CHECK (source_status IN ('exact', 'derived_from_facts', 'unresolved', 'not_applicable')),
    source_start INTEGER,
    source_end INTEGER,
    source_text_sha256 TEXT NOT NULL DEFAULT '',
    source_fact_ids_json TEXT NOT NULL DEFAULT '[]',
    unresolved_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(profile_id, object_type, object_id, operation_id)
);
CREATE INDEX IF NOT EXISTS idx_derivation_lineage_profile
    ON derivation_lineage(profile_id, object_type, object_id);
CREATE INDEX IF NOT EXISTS idx_derivation_lineage_operation
    ON derivation_lineage(operation_id);
"""


def apply(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


def verify(conn: sqlite3.Connection) -> bool:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='derivation_lineage'"
    ).fetchone()
    if table is None:
        return False
    columns = {row[1] for row in conn.execute("PRAGMA table_info(derivation_lineage)")}
    return {
        "lineage_id", "profile_id", "object_type", "object_id",
        "operation_id", "derivation_version", "source_status",
        "source_start", "source_end", "source_text_sha256",
        "source_fact_ids_json", "unresolved_reason",
    } <= columns
