# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §1.3 + LLD-10

"""M009 — learning_model_state lineage columns (learning.db).

LLD-10 online retrain + shadow test + auto-rollback needs to track
three models concurrently: the live active model, the previously
active (for rollback), and a candidate under A/B shadow validation.

This migration extends ``learning_model_state`` (created by M002) with
six additive columns and two partial unique indexes that enforce
single-active + single-candidate per profile. All additive — no
existing behaviour changes.
"""

from __future__ import annotations

import sqlite3

NAME = "M009_model_lineage"
DB_TARGET = "learning"

_REQUIRED_COLS = frozenset({
    "is_previous", "is_rollback", "is_candidate",
    "shadow_results_json", "promoted_at", "rollback_reason",
})

# Column definitions in application order: name → SQLite type fragment.
_COLUMN_ADDITIONS: tuple[tuple[str, str], ...] = (
    ("is_previous",         "INTEGER DEFAULT 0"),
    ("is_rollback",         "INTEGER DEFAULT 0"),
    ("is_candidate",        "INTEGER DEFAULT 0"),
    ("shadow_results_json", "TEXT"),
    ("promoted_at",         "TEXT"),
    ("rollback_reason",     "TEXT"),
)


def apply(conn: sqlite3.Connection) -> None:
    """Idempotent, column-by-column ALTER for crash-safe recovery.

    Each column is only added when absent, so any committed prefix from a
    prior interrupted run is handled transparently.  Indexes use the standard
    IF NOT EXISTS guard.  Callers must not call commit/rollback — the runner
    owns the transaction boundary.
    """
    existing_cols: set[str] = {
        r[1]
        for r in conn.execute(
            "PRAGMA table_info(learning_model_state)"
        ).fetchall()
    }
    for col_name, col_def in _COLUMN_ADDITIONS:
        if col_name not in existing_cols:
            conn.execute(
                f"ALTER TABLE learning_model_state ADD COLUMN {col_name} {col_def}"
            )
    # Partial-index creation: IF NOT EXISTS guards idempotency.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_active_one "
        "ON learning_model_state(profile_id) WHERE is_active=1"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_candidate_one "
        "ON learning_model_state(profile_id) WHERE is_candidate=1"
    )


def verify(conn: sqlite3.Connection) -> bool:
    try:
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(learning_model_state)"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS.issubset(cols)


DDL = """
ALTER TABLE learning_model_state ADD COLUMN is_previous INTEGER DEFAULT 0;
ALTER TABLE learning_model_state ADD COLUMN is_rollback INTEGER DEFAULT 0;
ALTER TABLE learning_model_state ADD COLUMN is_candidate INTEGER DEFAULT 0;
ALTER TABLE learning_model_state ADD COLUMN shadow_results_json TEXT;
ALTER TABLE learning_model_state ADD COLUMN promoted_at TEXT;
ALTER TABLE learning_model_state ADD COLUMN rollback_reason TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_active_one
    ON learning_model_state(profile_id) WHERE is_active=1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_candidate_one
    ON learning_model_state(profile_id) WHERE is_candidate=1;
"""
