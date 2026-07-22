# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory — per-profile isolation (I-6 / C2)

"""M023 — profile_id on all mesh coordination tables (memory.db, deferred).

The mesh broker is a per-agent coordination bus (peers, notification messages,
shared key/value state, file locks, event log). Every mesh table was GLOBAL —
no ``profile_id`` — so in a multi-tenant deployment (profiles = individual /
team / company tenants) tenant A could see tenant B's peers, receive B's
broadcast messages, read/overwrite B's shared state by key collision, and be
blocked by B's file locks. That is a concrete cross-tenant isolation leak.

This migration makes the mesh namespace per-profile:

  * mesh_peers, mesh_messages, mesh_events — additive ``profile_id`` column +
    profile-leading indexes (no primary-key change).
  * mesh_state — rebuilt with PRIMARY KEY(profile_id, key) so the same key can
    exist independently per tenant.
  * mesh_locks — rebuilt with PRIMARY KEY(profile_id, file_path) so one tenant's
    lock on a path never blocks another tenant.

mesh_reads is intentionally left unchanged: it is (message_id, peer_id), both of
which are already profile-scoped by their parent rows, so a cross-profile read
row is unreachable.

Existing rows backfill to the 'default' profile (the default active profile).
Idempotent and tolerant of missing tables; deferred because the mesh tables are
created at engine init (apply_v343_schema), after ``apply_all`` runs — same
reason as M021.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M023_mesh_profile_isolation"
DB_TARGET = "memory"

# Documentation + drift hash. apply() below is the authoritative executor.
DDL = (
    "ALTER TABLE mesh_peers ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default';"
    "ALTER TABLE mesh_messages ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default';"
    "ALTER TABLE mesh_events ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default';"
    "-- rebuild mesh_state PK(profile_id, key); mesh_locks PK(profile_id, file_path)"
)

_NEW_STATE = """
CREATE TABLE mesh_state (
    profile_id TEXT NOT NULL DEFAULT 'default',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    set_by TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, key)
)
"""

_NEW_LOCKS = """
CREATE TABLE mesh_locks (
    profile_id TEXT NOT NULL DEFAULT 'default',
    file_path TEXT NOT NULL,
    locked_by TEXT NOT NULL,
    locked_at TEXT NOT NULL,
    expires_at TEXT NOT NULL DEFAULT '9999-12-31T23:59:59Z',
    PRIMARY KEY (profile_id, file_path)
)
"""


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _add_profile_column(conn: sqlite3.Connection, table: str) -> None:
    """Additive profile_id on a table whose primary key does not change."""
    if not _table_exists(conn, table):
        return
    if "profile_id" not in _cols(conn, table):
        conn.execute(
            f"ALTER TABLE {table} "
            "ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default'"
        )
        # New column defaults 'default' for every legacy row; no extra UPDATE
        # needed. (NOT NULL DEFAULT is applied to existing rows by SQLite.)


def _atomic_rebuild(conn: sqlite3.Connection, table: str, create_sql: str,
                    insert_sql: str) -> None:
    """RENAME→CREATE→copy→DROP as ONE transaction.

    Uses conn.execute (NOT executescript, which force-commits mid-rebuild and
    could leave a zombie *_old table with the only copy of the data if a later
    step fails). BEGIN IMMEDIATE + explicit COMMIT/ROLLBACK guarantees the
    rebuild is all-or-nothing even under the runner's autocommit connection.
    """
    if not _table_exists(conn, table):
        return
    if "profile_id" in _cols(conn, table):
        return
    old = f"_{table}_old"
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"ALTER TABLE {table} RENAME TO {old}")
        conn.execute(create_sql.strip().rstrip(";"))
        conn.execute(insert_sql.replace("__OLD__", old))
        conn.execute(f"DROP TABLE {old}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _rebuild_state(conn: sqlite3.Connection) -> None:
    _atomic_rebuild(
        conn, "mesh_state", _NEW_STATE,
        "INSERT INTO mesh_state (profile_id, key, value, set_by, updated_at) "
        "SELECT 'default', key, value, set_by, updated_at FROM __OLD__",
    )


def _rebuild_locks(conn: sqlite3.Connection) -> None:
    _atomic_rebuild(
        conn, "mesh_locks", _NEW_LOCKS,
        "INSERT INTO mesh_locks "
        "(profile_id, file_path, locked_by, locked_at, expires_at) "
        "SELECT 'default', file_path, locked_by, locked_at, expires_at FROM __OLD__",
    )


def _indexes(conn: sqlite3.Connection) -> None:
    """Profile-leading indexes. Created here (not in schema_v343) so an upgrade
    never runs CREATE INDEX on a column that does not exist yet."""
    if _table_exists(conn, "mesh_peers") and "profile_id" in _cols(conn, "mesh_peers"):
        peer_cols = _cols(conn, "mesh_peers")
        if "session_id" in peer_cols:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mesh_peers_profile "
                "ON mesh_peers(profile_id, session_id)"
            )
        if "status" in peer_cols:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mesh_peers_profile_status "
                "ON mesh_peers(profile_id, status)"
            )
    if _table_exists(conn, "mesh_messages") and "profile_id" in _cols(conn, "mesh_messages"):
        msg_cols = _cols(conn, "mesh_messages")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mesh_messages_profile_to "
            "ON mesh_messages(profile_id, to_peer, read)"
        )
        # target_type / project_path are v3.4.6 additions. In the normal upgrade
        # path they exist (schema_v343 applies its ALTERs before deferred
        # migrations run), but guard anyway so M023 is robust on any DB.
        if {"target_type", "project_path"} <= msg_cols:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mesh_messages_profile_target "
                "ON mesh_messages(profile_id, target_type, project_path)"
            )
    if _table_exists(conn, "mesh_events") and "profile_id" in _cols(conn, "mesh_events"):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mesh_events_profile "
            "ON mesh_events(profile_id, id)"
        )


def apply(conn: sqlite3.Connection) -> None:
    """Profile-scope every mesh table. Idempotent; no-op on a fresh/migrated DB.

    Existing rows backfill to the 'default' profile.
    """
    _add_profile_column(conn, "mesh_peers")
    _add_profile_column(conn, "mesh_messages")
    _add_profile_column(conn, "mesh_events")
    _rebuild_state(conn)
    _rebuild_locks(conn)
    _indexes(conn)


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once every mesh table carries profile_id (or is absent)."""
    for table in ("mesh_peers", "mesh_messages", "mesh_events",
                  "mesh_state", "mesh_locks"):
        if _table_exists(conn, table) and "profile_id" not in _cols(conn, table):
            return False
    return True
