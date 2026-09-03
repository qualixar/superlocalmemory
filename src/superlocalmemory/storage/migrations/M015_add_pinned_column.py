# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.65 — Context Injection v2

"""M015 — add `pinned` column to atomic_facts (v3.4.65, core-memory explicit pins).

Additive only — ALTER TABLE ADD COLUMN, default 0. No data loss, no type
changes. Idempotent via verify() + migration_log. Mirrors M001 pattern.

Backward-compat: old code that doesn't know about the column simply gets
DEFAULT 0 on SELECT *, which is the safe "not-pinned" state.
"""

from __future__ import annotations

import sqlite3

NAME = "M015_add_pinned_column"
DB_TARGET = "memory"

_REQUIRED_COLS = frozenset({"pinned"})


def verify(conn: sqlite3.Connection) -> bool:
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(atomic_facts)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS <= cols


DDL = """
BEGIN IMMEDIATE;
ALTER TABLE atomic_facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_facts_pinned
    ON atomic_facts(profile_id, pinned);
COMMIT;
"""


def apply(conn: sqlite3.Connection) -> None:
    """Apply M015 safely when a fresh schema already contains ``pinned``.

    New installations are created from the current base schema, whereas
    upgrades need the additive ALTER. SQLite has no ``ADD COLUMN IF NOT
    EXISTS``, so the old static migration failed on fresh databases.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
    if "pinned" not in cols:
        conn.execute(
            "ALTER TABLE atomic_facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_pinned "
        "ON atomic_facts(profile_id, pinned)"
    )


def repair(conn: sqlite3.Connection) -> None:
    """Re-run the idempotent apply as end-state repair (4.1.14 #133)."""
    apply(conn)
