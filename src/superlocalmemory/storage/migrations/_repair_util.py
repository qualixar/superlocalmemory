# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shared idempotent-DDL repair for migration modules (4.1.14 #133).

A completed migration whose end-state no longer holds (partial restore,
hand-recreated table) must never be *replayed* — replay would duplicate
rows or drop live tables. The only allowed reconciler is a
module-supplied ``repair(conn)`` hook, and the framework re-runs the
module's own ``verify()`` afterwards: repair is best-effort by
construction, and anything it cannot restore stays a loud failure.

This module is the single implementation behind the mechanical repairs:
re-executing a migration's static DDL idempotently. ``ADD COLUMN`` has
no ``IF NOT EXISTS`` in SQLite, so those statements are guarded by
``PRAGMA table_info``; ``CREATE TABLE/INDEX ... IF NOT EXISTS`` is
natively idempotent. Any other ``OperationalError`` propagates — only
SQLite's own idempotence signals ("already exists", "duplicate column")
are tolerated. Transaction-control statements (``BEGIN``/``COMMIT``)
are skipped: the runner owns the transaction boundary, not the replay.
"""
from __future__ import annotations

import re
import sqlite3

_ADD_COLUMN_RE = re.compile(
    r"^\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<table>[\"'\w\[\] ]+?)"
    r"\s+ADD\s+(?:COLUMN\s+)?(?P<column>[\"'\w\[\]]+)",
    re.IGNORECASE,
)
_TXN_RE = re.compile(
    r"^\s*(BEGIN(?:\s+(?:IMMEDIATE|EXCLUSIVE|DEFERRED))?|COMMIT|ROLLBACK)\s*$",
    re.IGNORECASE,
)
_TOLERATED_RE = re.compile(r"already exists|duplicate column", re.IGNORECASE)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
    except sqlite3.Error:
        return set()


def repair_ddl(conn: sqlite3.Connection, ddl: str) -> None:
    """Re-execute static ``ddl`` idempotently against ``conn``.

    Safe to call on a complete schema (every statement becomes a no-op),
    on a partial one (only the missing pieces apply), and on an empty
    database missing whole tables (``ADD COLUMN`` on a missing table
    raises, honestly — repair cannot invent tables the migration never
    owned; use the migration's own ``apply`` path for those).
    """
    for chunk in ddl.split(";"):
        stmt = chunk.strip()
        if not stmt:
            continue
        if _TXN_RE.match(stmt):
            continue
        match = _ADD_COLUMN_RE.match(stmt)
        if match is not None:
            table = match.group("table").strip().strip("\"'[]")
            column = match.group("column").strip().strip("\"'[]")
            if column in _existing_columns(conn, table):
                continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if _TOLERATED_RE.search(str(exc)):
                continue
            raise
