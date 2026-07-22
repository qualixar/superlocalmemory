# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — performance indexes

"""M025 — hot-path indexes confirmed missing by EXPLAIN QUERY PLAN.

* atomic_facts(profile_id, content) — the store_fact dedup check runs on every
  ingest; without this it scans all profile facts for a lifecycle match.
* mesh_events(created_at) — the 7-day cleanup range-scanned the whole table.
* mesh_peers(status, last_heartbeat) — stale-peer cleanup full-scanned.
* mesh_peers(profile_id, last_heartbeat) — list_peers sorted in a temp b-tree.

All additive CREATE INDEX IF NOT EXISTS — idempotent, no data change. Deferred
so the target tables (mesh_* via schema_v343, atomic_facts via engine init)
exist first.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M025_perf_indexes"
DB_TARGET = "memory"

DDL = (
    "CREATE INDEX IF NOT EXISTS idx_facts_content_dedup "
    "ON atomic_facts(profile_id, content);"
    "CREATE INDEX IF NOT EXISTS idx_mesh_events_created_at ON mesh_events(created_at);"
    "CREATE INDEX IF NOT EXISTS idx_mesh_peers_status_heartbeat "
    "ON mesh_peers(status, last_heartbeat);"
    "CREATE INDEX IF NOT EXISTS idx_mesh_peers_profile_heartbeat "
    "ON mesh_peers(profile_id, last_heartbeat);"
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def apply(conn: sqlite3.Connection) -> None:
    """Create the perf indexes, each guarded so a non-standard schema cannot
    abort the migration (and leave verify() falsely reporting done)."""
    if _table_exists(conn, "atomic_facts") and {"profile_id", "content"} <= _cols(conn, "atomic_facts"):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_content_dedup "
            "ON atomic_facts(profile_id, content)"
        )
    if _table_exists(conn, "mesh_events") and "created_at" in _cols(conn, "mesh_events"):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mesh_events_created_at "
            "ON mesh_events(created_at)"
        )
    if _table_exists(conn, "mesh_peers"):
        pc = _cols(conn, "mesh_peers")
        if {"status", "last_heartbeat"} <= pc:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mesh_peers_status_heartbeat "
                "ON mesh_peers(status, last_heartbeat)"
            )
        if {"profile_id", "last_heartbeat"} <= pc:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mesh_peers_profile_heartbeat "
                "ON mesh_peers(profile_id, last_heartbeat)"
            )


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once the two always-creatable indexes exist (mesh_peers/events
    are always present via schema_v343; atomic_facts always via engine init)."""
    names = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    required = {"idx_facts_content_dedup", "idx_mesh_events_created_at"}
    # Only require indexes whose tables exist (fresh vs partial installs).
    ok = True
    if _table_exists(conn, "atomic_facts"):
        ok = ok and "idx_facts_content_dedup" in names
    if _table_exists(conn, "mesh_events"):
        ok = ok and "idx_mesh_events_created_at" in names
    return ok
