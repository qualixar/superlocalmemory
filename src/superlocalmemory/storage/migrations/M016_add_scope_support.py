# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.6.13

"""M016 — scope and shared_with columns on core tables (memory.db, deferred).

Adds two columns to each of the 5 core tables for multi-scope memory support:

    scope       TEXT NOT NULL DEFAULT 'personal'  — personal | global
    shared_with TEXT                             — JSON array of profile_ids

Existing data retains scope='personal' (backward compatible). New indexes
on ``scope`` and ``(profile_id, scope)`` speed up scope-filtered queries.

Deferred like M006, M011, and M013 because the core tables are bootstrapped
at engine init, not at migration time. Daemon lifespan calls
``apply_deferred`` right after engine init so these columns materialise
on first boot after upgrade.

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

DDL = ";".join(
    [f"ALTER TABLE {t} ADD COLUMN scope TEXT NOT NULL DEFAULT 'personal'" for t in TABLES]
    + [f"ALTER TABLE {t} ADD COLUMN shared_with TEXT" for t in TABLES]
    + [f"CREATE INDEX IF NOT EXISTS idx_{t}_scope ON {t}(scope)" for t in TABLES]
    + [
        f"CREATE INDEX IF NOT EXISTS idx_{t}_profile_scope ON {t}(profile_id, scope)"
        for t in TABLES
    ]
)


def verify(conn: sqlite3.Connection) -> bool:
    """Check if migration already applied by inspecting atomic_facts columns."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)").fetchall()}
    except sqlite3.Error:
        return False
    return "scope" in cols
