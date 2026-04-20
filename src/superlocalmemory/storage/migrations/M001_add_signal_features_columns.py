# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.1

"""M001 — add rich-signal columns to learning_signals + learning_features.

Additive only — uses ``ALTER TABLE ADD COLUMN``. No data loss, no type
changes. Every ADD is guarded by the runner's idempotency check against
``migration_log``; on rerun the migration is skipped outright so duplicate
ALTER errors never surface.

The runner wraps this DDL in ``BEGIN IMMEDIATE`` / ``COMMIT`` when needed.
SQLite DDL is transactional except for schema-version bumps, so the BEGIN
here is defense-in-depth against partial application on crash.
"""

from __future__ import annotations

import sqlite3

NAME = "M001_add_signal_features_columns"
DB_TARGET = "learning"

_REQUIRED_SIGNAL_COLS = frozenset({
    "query_id", "query_text_hash", "position", "channel_scores", "cross_encoder",
})
_REQUIRED_FEATURE_COLS = frozenset({"signal_id", "is_synthetic"})


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if the migration's end-state is already present."""
    try:
        sig_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(learning_signals)"
        ).fetchall()}
        feat_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(learning_features)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return (_REQUIRED_SIGNAL_COLS <= sig_cols
            and _REQUIRED_FEATURE_COLS <= feat_cols)

DDL = """
BEGIN IMMEDIATE;

ALTER TABLE learning_signals ADD COLUMN query_id        TEXT DEFAULT '';
ALTER TABLE learning_signals ADD COLUMN query_text_hash TEXT DEFAULT '';
ALTER TABLE learning_signals ADD COLUMN position        INTEGER DEFAULT 0;
ALTER TABLE learning_signals ADD COLUMN channel_scores  TEXT DEFAULT '{}';
ALTER TABLE learning_signals ADD COLUMN cross_encoder   REAL;

CREATE INDEX IF NOT EXISTS idx_signals_profile_time
    ON learning_signals(profile_id, created_at);
CREATE INDEX IF NOT EXISTS idx_signals_query_id
    ON learning_signals(query_id);

ALTER TABLE learning_features ADD COLUMN signal_id    INTEGER DEFAULT 0;
ALTER TABLE learning_features ADD COLUMN is_synthetic INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_features_signal
    ON learning_features(signal_id);
CREATE INDEX IF NOT EXISTS idx_features_synthetic
    ON learning_features(is_synthetic) WHERE is_synthetic = 0;

COMMIT;
"""
