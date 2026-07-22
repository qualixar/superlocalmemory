# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory — per-profile isolation (I-7)

"""M022 — profile_id on entity_aliases (memory.db, deferred).

entity_aliases had no profile_id, so aliases were keyed by entity_id alone. When
the same entity_id appears under two profiles, one profile's aliases were
visible to (and deduped against) the other. Add profile_id and backfill each
alias from its parent canonical entity's profile (the correct owner), falling
back to 'default' for orphans.

Additive column — no UNIQUE change, so a simple ADD COLUMN + backfill. Deferred
because entity_aliases is created at engine init (create_all_tables), after
apply_all runs — same reason as M016.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M022_entity_aliases_profile"
DB_TARGET = "memory"

DDL = "ALTER TABLE entity_aliases ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default'"


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def apply(conn: sqlite3.Connection) -> None:
    """Add profile_id to entity_aliases and backfill from the parent entity.

    No-op on a fresh install (column already present) or a missing table.
    """
    if not _table_exists(conn, "entity_aliases"):
        return

    if "profile_id" not in _cols(conn, "entity_aliases"):
        conn.execute(
            "ALTER TABLE entity_aliases "
            "ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default'"
        )
        # Backfill each alias from its parent canonical entity's profile. If the
        # parent is missing (orphan) the COALESCE keeps 'default'.
        if _table_exists(conn, "canonical_entities"):
            conn.execute(
                "UPDATE entity_aliases SET profile_id = COALESCE("
                " (SELECT ce.profile_id FROM canonical_entities ce "
                "  WHERE ce.entity_id = entity_aliases.entity_id LIMIT 1), 'default')"
            )

    # Always ensure the index (this migration owns it — schema.create_all_tables
    # deliberately does NOT create it, since on an upgrade the column doesn't
    # exist yet at create-all time). Idempotent for fresh installs too.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aliases_profile_entity "
        "ON entity_aliases(profile_id, entity_id)"
    )


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once entity_aliases has profile_id AND its index (or is absent).

    Checking the index (not just the column) means a fresh install — where
    create_all_tables made the column but not the index — still runs apply()
    to create the index.
    """
    if not _table_exists(conn, "entity_aliases"):
        return True
    if "profile_id" not in _cols(conn, "entity_aliases"):
        return False
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='idx_aliases_profile_entity'"
    ).fetchone()
    return idx is not None
