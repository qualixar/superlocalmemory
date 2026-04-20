# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.4

"""M004 — cross-platform adapter sync log (memory.db).

LLD-05 requirement. SEC-05-04: stores target paths as SHA-256 digests, not
raw strings — keeps plaintext filesystem paths out of the DB. A separate
``target_basename`` column keeps the last path segment for the dashboard
without exposing the full path.
"""

from __future__ import annotations

import sqlite3

NAME = "M004_cross_platform_sync_log"
DB_TARGET = "memory"


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if cross_platform_sync_log table is in place."""
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(cross_platform_sync_log)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return {"adapter_name", "profile_id", "target_path_sha256",
            "target_basename", "last_sync_at", "bytes_written",
            "content_sha256", "success"} <= cols

DDL = """
CREATE TABLE IF NOT EXISTS cross_platform_sync_log (
    adapter_name       TEXT NOT NULL,
    profile_id         TEXT NOT NULL,
    target_path_sha256 TEXT NOT NULL,
    target_basename    TEXT NOT NULL,
    last_sync_at       TEXT NOT NULL,
    bytes_written      INTEGER NOT NULL,
    content_sha256     TEXT NOT NULL,
    success            INTEGER NOT NULL,
    error_msg          TEXT,
    PRIMARY KEY (adapter_name, target_path_sha256)
);
"""
