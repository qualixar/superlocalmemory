# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.2

"""M002 — rebuild learning_model_state without UNIQUE(profile_id).

Enables shadow testing (old + new model live side-by-side). SQLite cannot
drop a UNIQUE constraint directly, so we use the new-table-rename pattern
wrapped in a single transaction. Existing rows are copied forward and
marked ``is_active = 1``.

SEC-02-02: ``bytes_sha256`` integrity column added. The daemon will
compute and verify this on every load to block corrupted model BLOBs from
reaching the LightGBM deserialiser.
"""

from __future__ import annotations

import sqlite3

NAME = "M002_model_state_history"
DB_TARGET = "learning"

_REQUIRED_COLS = frozenset({
    "model_version", "bytes_sha256", "trained_on_count",
    "feature_names", "metrics_json", "is_active",
    "trained_at", "updated_at",
})


def verify(conn: sqlite3.Connection) -> bool:
    """Verify columns plus both indexes promised by this migration."""
    try:
        cols = {r[1]: r for r in conn.execute(
            "PRAGMA table_info(learning_model_state)"
        ).fetchall()}
        index_rows = conn.execute(
            "PRAGMA index_list(learning_model_state)"
        ).fetchall()
    except sqlite3.Error:
        return False
    if not _REQUIRED_COLS <= set(cols):
        return False

    indexes = {row[1]: row for row in index_rows}
    active = indexes.get("idx_model_active")
    profile_time = indexes.get("idx_model_profile_time")
    if active is None or profile_time is None:
        return False
    # idx_model_active must remain a UNIQUE partial index.
    if int(active[2]) != 1 or int(active[4]) != 1:
        return False
    active_cols = [row[2] for row in conn.execute(
        "PRAGMA index_info(idx_model_active)"
    ).fetchall()]
    time_cols = [row[2] for row in conn.execute(
        "PRAGMA index_info(idx_model_profile_time)"
    ).fetchall()]
    if active_cols != ["profile_id"]:
        return False
    if time_cols != ["profile_id", "trained_at"]:
        return False
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        ("idx_model_active",),
    ).fetchone()
    normalized = " ".join(str(sql_row[0] if sql_row else "").lower().split())
    return "where is_active = 1" in normalized


# IMPORTANT: this DDL shipped in V3.4.21.  Migration hashes are immutable
# upgrade contracts, so later improvements belong in a new forward migration
# (M020), never in this historical definition.
DDL = """
BEGIN IMMEDIATE;

CREATE TABLE learning_model_state_new (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id        TEXT NOT NULL,
    model_version     TEXT NOT NULL DEFAULT '3.4.21',
    state_bytes       BLOB NOT NULL,
    bytes_sha256      TEXT NOT NULL DEFAULT '',
    trained_on_count  INTEGER NOT NULL DEFAULT 0,
    feature_names     TEXT NOT NULL DEFAULT '[]',
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    is_active         INTEGER NOT NULL DEFAULT 0,
    trained_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

INSERT INTO learning_model_state_new
    (profile_id, state_bytes, is_active, trained_at, updated_at)
SELECT profile_id, state_bytes, 1, updated_at, updated_at
FROM learning_model_state;

DROP TABLE learning_model_state;
ALTER TABLE learning_model_state_new RENAME TO learning_model_state;

CREATE UNIQUE INDEX idx_model_active
    ON learning_model_state(profile_id)
    WHERE is_active = 1;

CREATE INDEX idx_model_profile_time
    ON learning_model_state(profile_id, trained_at);

COMMIT;
"""
