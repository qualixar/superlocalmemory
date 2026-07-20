# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.6.18

"""M017 — scope column on ccq_consolidated_blocks.

The cognitive-consolidation queue table (``ccq_consolidated_blocks``) was
created before the M016 multi-scope migration and therefore has no ``scope``
column.  Without it, consolidation summaries are always stored as
``personal`` regardless of the source facts' scope — a silent data-loss edge
case when a user has opted into global or shared memory.

Fix: add a ``scope TEXT NOT NULL DEFAULT 'personal'`` column plus a
``(profile_id, scope)`` covering index for scope-filtered queries.

Deferred (like M006, M011, M013, M016): the CCQ table is created by the
engine, not by migration DDL, so we apply after engine init via
``apply_deferred``.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M017_ccq_scope_column"
DB_TARGET = "memory"

TABLE = "ccq_consolidated_blocks"

DDL = (
    f"ALTER TABLE {TABLE} ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal';"
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_scope ON {TABLE}(scope);"
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_profile_scope ON {TABLE}(profile_id, scope);"
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def apply(conn: sqlite3.Connection) -> None:
    """Idempotently add scope column + indexes to ccq_consolidated_blocks.

    Skips silently when the table doesn't exist yet (engine hasn't created it)
    or when the column is already present (re-apply or fresh install).
    """
    if not _table_exists(conn, TABLE):
        return
    cols = _column_names(conn, TABLE)
    if "scope" not in cols:
        conn.execute(
            f"ALTER TABLE {TABLE} ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal'"
        )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_scope ON {TABLE}(scope)")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_profile_scope ON {TABLE}(profile_id, scope)"
    )


def verify(conn: sqlite3.Connection) -> bool:
    """Applied when ccq_consolidated_blocks has the scope column and index."""
    if not _table_exists(conn, TABLE):
        return True  # table absent — apply() skips it; nothing to verify
    cols = _column_names(conn, TABLE)
    if "scope" not in cols:
        return False
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (f"idx_{TABLE}_scope",),
    ).fetchone()
    return idx is not None
