# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Stable per-node identity for the SLM mesh distributed protocol (3c-0).

Each daemon owns a durable ``node_id`` (uuid4 hex) persisted once in the mesh
SQLite DB (single-row ``mesh_node_identity`` table). It is the deterministic
tie-breaker for the distributed protocol layer:

* **LWW state convergence** — when two nodes hold the same key at the same
  ``revision``, the winner is the one with the higher ``node_id`` (a total
  order → every node converges on the same value).
* **Distributed lock ordering** — the effective lock holder is the one with the
  higher ``(fencing_token, node_id)``; the fence guarantees single-writer
  safety downstream even during a brief split.

Design:
* Persisted → stable across restarts (a restart must NOT change the tie-break).
* ``INSERT OR IGNORE`` + re-read → two processes racing first-creation converge
  on ONE id.
* **Fail-soft**: any DB error returns a process-stable fallback
  (``hostname-pid``) so callers never crash; the mesh degrades to node-local
  behavior rather than breaking.
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import time
import uuid

logger = logging.getLogger("superlocalmemory.mesh.node_identity")

_CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS mesh_node_identity ("
    " id INTEGER PRIMARY KEY CHECK (id = 1),"
    " node_id TEXT NOT NULL,"
    " created_at REAL NOT NULL)"
)

# Process-stable fallback cache (only used when the DB is unavailable).
_fallback_cache: dict[str, str] = {}


def _fallback(db_path: str) -> str:
    """Return a process-stable, non-persisted id for the fail-soft path."""
    value = _fallback_cache.get(db_path)
    if value is None:
        value = f"{socket.gethostname()}-{os.getpid()}"
        _fallback_cache[db_path] = value
    return value


def get_node_id(db_path: str) -> str:
    """Return this node's stable id, creating it once if absent.

    Args:
        db_path: Path to the mesh SQLite DB (same file the broker uses).

    Returns:
        A stable hex node id (persisted), or a process-stable ``hostname-pid``
        fallback if the DB cannot be read/written.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            conn.execute(_CREATE_SQL)
            row = conn.execute(
                "SELECT node_id FROM mesh_node_identity WHERE id = 1"
            ).fetchone()
            if row is not None:
                return row[0]
            new_id = uuid.uuid4().hex
            # INSERT OR IGNORE lets a concurrent first-creator win; we then
            # re-read so every racer returns the SAME persisted id.
            conn.execute(
                "INSERT OR IGNORE INTO mesh_node_identity (id, node_id, created_at)"
                " VALUES (1, ?, ?)",
                (new_id, time.time()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT node_id FROM mesh_node_identity WHERE id = 1"
            ).fetchone()
            return row[0] if row is not None else new_id
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.error(
            "get_node_id: DB error at %s — using process fallback: %s",
            db_path, exc,
        )
        return _fallback(db_path)
