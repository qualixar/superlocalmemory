# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M020 — forward-only integrity repair for learned model state.

M002's DDL shipped before model-state SHA values were backfilled.  Do not
modify M002: its stored hash is part of every existing user's upgrade record.
This migration deterministically fills only missing digests and never changes
the model payload or ranking metadata.
"""

from __future__ import annotations

import hashlib
import sqlite3

NAME = "M020_model_state_integrity"
DB_TARGET = "learning"

# The runner records a stable fingerprint even though this migration uses a
# Python apply function for binary-safe SHA computation.
DDL = "-- M020: backfill missing learning_model_state.bytes_sha256\n"


def verify(conn: sqlite3.Connection) -> bool:
    """Return True only when every persisted model has an integrity digest."""
    try:
        missing = conn.execute(
            "SELECT 1 FROM learning_model_state "
            "WHERE bytes_sha256 = '' OR bytes_sha256 IS NULL LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return False
    return missing is None


def apply(conn: sqlite3.Connection) -> None:
    """Backfill SHA-256 values without modifying model bytes or metadata."""
    rows = conn.execute(
        "SELECT id, state_bytes FROM learning_model_state "
        "WHERE bytes_sha256 = '' OR bytes_sha256 IS NULL"
    ).fetchall()
    updates = [
        (hashlib.sha256(state_bytes).hexdigest(), row_id)
        for row_id, state_bytes in rows
        if state_bytes is not None
    ]
    if updates:
        conn.executemany(
            "UPDATE learning_model_state SET bytes_sha256 = ? WHERE id = ?",
            updates,
        )
