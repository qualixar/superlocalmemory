# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mesh broker security helpers — startup integrity, state guards, fencing.

Pure functions so they can be unit-tested independently of the broker.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("superlocalmemory.mesh")

# Pattern matches key names that imply secret material.
import re as _re
STATE_SECRET_KEY = _re.compile(
    r"(?:^|[_\-.])(api[_\-.]?key|secret|token|password|credential)(?:$|[_\-.]|$)",
    _re.IGNORECASE,
)

_SCHEMA_ALTERS = (
    "ALTER TABLE mesh_locks ADD COLUMN fencing_token INTEGER DEFAULT 0",
    "ALTER TABLE mesh_state ADD COLUMN revision INTEGER DEFAULT 0",
)
_SENT_OPS_DDL = """
CREATE TABLE IF NOT EXISTS mesh_sent_ops (
    operation_id TEXT PRIMARY KEY,
    message_id   INTEGER NOT NULL,
    created_at   TEXT NOT NULL
)"""


def ensure_db_healthy(db_path: str) -> bool:
    """Return True (degraded) if the DB was corrupt and had to be quarantined.

    Quarantine = rename to ``<name>.quarantine-<ms>``.  The original bytes
    are preserved; the caller receives a fresh empty DB on the same path.
    A missing DB is a normal first-run situation and is not an error.
    """
    path = Path(db_path)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT count(*) FROM sqlite_master")
        conn.close()
        return False
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        ts = int(time.monotonic() * 1_000)
        quarantine = path.with_name(f"{path.name}.quarantine-{ts}")
        try:
            path.rename(quarantine)
        except OSError as rename_err:
            logger.error("mesh db corrupt and could not be quarantined: %s", rename_err)
            return False
        logger.warning(
            "mesh db corrupt (%s); quarantined to %s; starting fresh",
            exc, quarantine.name,
        )
        return True


def apply_security_schema(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema additions (fencing_token, revision, mesh_sent_ops)."""
    for sql in _SCHEMA_ALTERS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    try:
        conn.executescript(_SENT_OPS_DDL)
    except sqlite3.OperationalError:
        pass
    conn.commit()


def reject_secret_state(key: str, value: str) -> dict | None:
    """Return an error dict if key or value looks like a secret, else None."""
    if STATE_SECRET_KEY.search(key):
        return {"ok": False, "error": "mesh state is coordination metadata; secret key names are prohibited"}
    try:
        from superlocalmemory.core.security_primitives import redact_secrets
        if redact_secrets(value) != value:
            return {"ok": False, "error": "mesh state is coordination metadata; secret values are prohibited"}
    except ImportError:
        pass
    return None


def check_cross_profile_sender(
    conn: sqlite3.Connection, from_peer: str, profile_id: str
) -> dict | None:
    """Return an error dict if from_peer is a known peer in a different profile.

    Arbitrary label strings (not registered anywhere) are allowed — they are
    metadata, not identity claims.  Only a server-assigned peer_id that
    belongs to a different profile is rejected (cross-profile impersonation).
    """
    if not from_peer:
        return None
    row = conn.execute(
        "SELECT profile_id FROM mesh_peers WHERE peer_id=? LIMIT 1",
        (from_peer,),
    ).fetchone()
    if row is not None and row["profile_id"] != profile_id:
        return {"ok": False, "error": "from_peer belongs to a different profile"}
    return None


def validate_lock_fence_query(
    conn: sqlite3.Connection,
    file_path: str,
    fencing_token: int,
    profile_id: str,
) -> dict:
    """Compare presented fencing_token against the current lock record."""
    row = conn.execute(
        "SELECT COALESCE(fencing_token, 0) AS fencing_token "
        "FROM mesh_locks WHERE profile_id=? AND file_path=?",
        (profile_id, file_path),
    ).fetchone()
    if row is None:
        return {"ok": False, "error": "no lock held for this resource"}
    current = row["fencing_token"]
    if fencing_token < current:
        return {"ok": False, "error": f"fencing token {fencing_token} is stale; current token is {current}"}
    return {"ok": True, "fencing_token": current}
