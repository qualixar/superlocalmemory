# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.6.15

"""M016 — scope and shared_with columns on core tables (memory.db, deferred).

Adds two columns to each of the 5 core tables for multi-scope memory support:

    scope       TEXT NOT NULL DEFAULT 'personal'  — personal | global
    shared_with TEXT                             — JSON array of profile_ids

Existing data retains scope='personal' (backward compatible). Indexes on
``scope`` and ``(profile_id, scope)`` speed up scope-filtered queries and are
created HERE (not in schema.create_all_tables) so an upgrading DB whose tables
predate the scope column doesn't hit "no such column: scope" when the boot-time
index DDL runs before this migration.

Applied via a conditional apply(conn) rather than static DDL because SQLite has
no ``ADD COLUMN IF NOT EXISTS``: a static ALTER fails on a fresh install (the
column already exists from create_all_tables) and on a deferred boot where a
table isn't created yet. apply() guards every step so it is idempotent and
tolerant of partial/missing tables.

Deferred like M006, M011, and M013 because the core tables are bootstrapped
at engine init, not at migration time. Daemon lifespan calls ``apply_deferred``
right after engine init so these columns materialise on first boot after upgrade.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M016_add_scope_support"
DB_TARGET = "memory"

TABLES = [
    "memories",
    "atomic_facts",
    "canonical_entities",
    "graph_edges",
    "temporal_events",
]

# Retained for the migration log's drift hash and as documentation of intent.
# apply() below is the authoritative, idempotent executor.
DDL = ";".join(
    [f"ALTER TABLE {t} ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal'" for t in TABLES]
    + [f"ALTER TABLE {t} ADD COLUMN shared_with TEXT" for t in TABLES]
    + [f"CREATE INDEX IF NOT EXISTS idx_{t}_scope ON {t}(scope)" for t in TABLES]
    + [
        f"CREATE INDEX IF NOT EXISTS idx_{t}_profile_scope ON {t}(profile_id, scope)"
        for t in TABLES
    ]
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def apply(conn: sqlite3.Connection) -> None:
    """Idempotently add scope/shared_with columns + indexes to every core table.

    Per table: skip if the table doesn't exist yet; add each column only if
    missing (SQLite has no ADD COLUMN IF NOT EXISTS); create indexes with
    IF NOT EXISTS. Safe to run on fresh installs (columns already present),
    upgrades (columns missing), and partial/repeat applies.
    """
    for t in TABLES:
        if not _table_exists(conn, t):
            continue
        cols = _column_names(conn, t)
        if "scope" not in cols:
            conn.execute(
                f"ALTER TABLE {t} ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal'"
            )
        if "shared_with" not in cols:
            conn.execute(f"ALTER TABLE {t} ADD COLUMN shared_with TEXT")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_scope ON {t}(scope)")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{t}_profile_scope ON {t}(profile_id, scope)"
        )


def verify(conn: sqlite3.Connection) -> bool:
    """Applied only when atomic_facts has the scope column AND its scope index.

    Checking the index (not just the column) ensures apply() still runs on a
    fresh install where create_all_tables created the column but not the index.
    """
    # v3.6.15: verify EVERY core table apply() touches — not just atomic_facts.
    # A partial apply (scope added to atomic_facts but not the other tables) must
    # NOT false-pass, or M016 is marked done and the remaining tables are left
    # permanently without the scope column. Absent tables are skipped, matching
    # apply()'s own skip-missing-table contract.
    for t in TABLES:
        try:
            info = conn.execute(f"PRAGMA table_info({t})").fetchall()
        except sqlite3.Error:
            return False
        if not info:
            continue  # table absent on this DB — apply() skips it too
        cols = {r[1] for r in info}
        if "scope" not in cols:
            return False
        idx = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
            (f"idx_{t}_scope",),
        ).fetchone()
        if idx is None:
            return False
    return True
