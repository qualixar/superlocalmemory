# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Removing one profile from mesh.db (GDPR Art. 17 and profile deletion).

Up to 4.2.7 the generic memory.db sweep removed the mesh rows, because the
mesh tables sat in memory.db and carried ``profile_id``. In mesh.db they need
their own pass, and the rows with no ``profile_id`` -- reads, envelopes, web
leases and sent-operation ids of the profile's messages, and the profiles of
its peers -- go with their parents.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .mesh_store import connect

#: Tables keyed by message id, removed with the profile's messages.
_BY_MESSAGE = ("mesh_reads", "mesh_message_envelopes", "mesh_web_deliveries", "mesh_sent_ops")
#: Tables that carry the tenant column.
_SCOPED = ("mesh_peers", "mesh_messages", "mesh_state", "mesh_locks", "mesh_events")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _erase(conn: sqlite3.Connection, profile_id: str) -> int:
    have = _tables(conn)
    removed = 0
    messages = "SELECT id FROM mesh_messages WHERE profile_id = ?"
    peers = "SELECT peer_id FROM mesh_peers WHERE profile_id = ?"
    if "mesh_messages" in have:
        for table in (t for t in _BY_MESSAGE if t in have):
            removed += conn.execute(
                f"DELETE FROM {table} WHERE message_id IN ({messages})", (profile_id,)).rowcount
    if "mesh_peers" in have:
        for table in ("mesh_reads", "mesh_peer_profiles"):
            if table in have:
                removed += conn.execute(
                    f"DELETE FROM {table} WHERE peer_id IN ({peers})", (profile_id,)).rowcount
    for table in (t for t in _SCOPED if t in have):
        removed += conn.execute(
            f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,)).rowcount
    return removed


def erase_profile(mesh_db: str | Path, profile_id: str) -> int:
    """Delete every mesh.db row of ``profile_id``; returns the row count. A missing file is 0."""
    path = Path(mesh_db)
    if not path.exists():
        return 0
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            removed = _erase(conn, profile_id)
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        return removed
    finally:
        conn.close()


__all__ = ["erase_profile"]
