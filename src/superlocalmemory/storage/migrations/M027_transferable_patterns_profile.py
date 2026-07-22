# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — isolation correctness (H-01, cycle-3 audit)

"""M027 — add profile_id to transferable_patterns + rebuild UNIQUE constraint (learning.db).

H-01: ``transferable_patterns`` had no ``profile_id`` column; every profile's
consolidation wrote to the same (pattern_type, key) row.  The last writer won,
so any profile's ``get_preferences()`` call returned patterns contaminated by
other profiles' consolidation runs.

Fix:
  1. Add ``profile_id TEXT NOT NULL DEFAULT 'default'`` to the table.
  2. Change the UNIQUE constraint from ``(pattern_type, key)`` to
     ``(profile_id, pattern_type, key)`` so each profile owns its own rows.
  3. Backfill existing rows to ``profile_id = 'default'`` (the only valid value
     before this migration — they are legacy/unattributed patterns).

SQLite cannot ALTER a UNIQUE constraint in place; the table must be rebuilt
(rename → create canonical form → copy → drop old).  Wrapped in
``BEGIN IMMEDIATE / COMMIT`` for atomicity.

Idempotent:
  * If ``profile_id`` already exists with the correct UNIQUE constraint
    (fresh install after the code change), ``apply()`` is a no-op.
  * If the table is absent entirely (CrossProjectAggregator was never run
    against learning.db), ``apply()`` is a no-op — the new _SCHEMA in
    cross_project.py will create it in its canonical form on first use.

DB target: ``learning`` — CrossProjectAggregator is under the learning/
package and is expected to share the learning database.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M027_transferable_patterns_profile"
DB_TARGET = "learning"

DDL = (
    "-- rebuild transferable_patterns with "
    "profile_id TEXT NOT NULL DEFAULT 'default' "
    "and UNIQUE(profile_id, pattern_type, key)"
)

_NEW_TABLE = """
CREATE TABLE transferable_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    pattern_type TEXT NOT NULL DEFAULT 'preference',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    profiles_seen INTEGER DEFAULT 1,
    decay_factor REAL DEFAULT 1.0,
    contradictions TEXT DEFAULT '[]',
    first_seen TEXT,
    last_seen TEXT,
    UNIQUE(profile_id, pattern_type, key)
)
"""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _unique_index_covers_profile(conn: sqlite3.Connection) -> bool:
    """True when transferable_patterns already has the UNIQUE(profile_id, …) constraint."""
    indexes = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='index' AND tbl_name='transferable_patterns'"
    ).fetchall()
    for idx in indexes:
        sql = (idx[1] or "").lower()
        if "profile_id" in sql and "pattern_type" in sql and "key" in sql:
            return True
    # SQLite stores the UNIQUE constraint on the CREATE TABLE statement as well.
    tbl_sql = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' AND name='transferable_patterns'"
    ).fetchone()
    if tbl_sql:
        sql = (tbl_sql[0] or "").lower()
        if "unique(profile_id" in sql or "unique (profile_id" in sql:
            return True
    return False


def apply(conn: sqlite3.Connection) -> None:
    """Rebuild transferable_patterns with profile_id column and correct UNIQUE constraint.

    No-op when the table is absent or already in canonical form.
    Existing data is preserved; legacy rows get profile_id = 'default'.
    """
    if not _table_exists(conn, "transferable_patterns"):
        return  # Table was never created against this DB — fresh install handles it.

    if (
        "profile_id" in _cols(conn, "transferable_patterns")
        and _unique_index_covers_profile(conn)
    ):
        return  # Already in canonical form.

    has_profile_col = "profile_id" in _cols(conn, "transferable_patterns")

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "ALTER TABLE transferable_patterns RENAME TO _tp_old"
        )
        conn.execute(_NEW_TABLE)

        if has_profile_col:
            # Column exists but UNIQUE constraint is wrong — copy as-is.
            conn.execute(
                "INSERT INTO transferable_patterns "
                "(profile_id, pattern_type, key, value, confidence, evidence_count, "
                " profiles_seen, decay_factor, contradictions, first_seen, last_seen) "
                "SELECT profile_id, pattern_type, key, value, confidence, evidence_count, "
                "       profiles_seen, decay_factor, contradictions, first_seen, last_seen "
                "FROM _tp_old"
            )
        else:
            # No profile_id yet — backfill legacy rows to 'default'.
            conn.execute(
                "INSERT INTO transferable_patterns "
                "(profile_id, pattern_type, key, value, confidence, evidence_count, "
                " profiles_seen, decay_factor, contradictions, first_seen, last_seen) "
                "SELECT 'default', pattern_type, key, value, confidence, evidence_count, "
                "       profiles_seen, decay_factor, contradictions, first_seen, last_seen "
                "FROM _tp_old"
            )

        conn.execute("DROP TABLE _tp_old")
        conn.execute("COMMIT")
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def verify(conn: sqlite3.Connection) -> bool:
    """True when transferable_patterns has profile_id with the correct UNIQUE,
    or when the table is absent (fresh install will create it correctly).
    """
    if not _table_exists(conn, "transferable_patterns"):
        return True  # absent — first run of _ensure_schema creates canonical form
    if "profile_id" not in _cols(conn, "transferable_patterns"):
        return False
    return _unique_index_covers_profile(conn)
