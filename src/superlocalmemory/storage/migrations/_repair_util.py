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
    r"^\s*(BEGIN(?:\s+(?:IMMEDIATE|EXCLUSIVE|DEFERRED|TRANSACTION))?"
    r"|COMMIT(?:\s+TRANSACTION)?|END(?:\s+TRANSACTION)?|ROLLBACK)\s*$",
    re.IGNORECASE,
)
_TOLERATED_RE = re.compile(r"already exists|duplicate column", re.IGNORECASE)
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    # 4.1.14 audit: the table name is quoted — an unquoted PRAGMA breaks
    # on legacy names with spaces or keywords.
    quoted = '"' + table.replace('"', '""') + '"'
    try:
        return {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({quoted})"
            ).fetchall()
        }
    except sqlite3.Error:
        return set()


def _split_statements(ddl: str) -> list[str]:
    """Split DDL on statement boundaries, respecting quotes and comments.

    4.1.14 audit: a naive ``split(";")`` dies on semicolons inside string
    literals (DEFAULT values) or comments, turning one statement into two
    invalid fragments. This splitter tracks single/double-quoted regions
    plus ``--`` line comments and ``/* */`` block comments.
    """
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    bracket_depth = 0
    line_comment = False
    block_comment = False
    i = 0
    while i < len(ddl):
        two = ddl[i:i + 2]
        char = ddl[i]
        if line_comment:
            current.append(char)
            if char == "\n":
                line_comment = False
        elif block_comment:
            current.append(char)
            if two == "*/":
                current.append(ddl[i + 1])
                i += 1
                block_comment = False
        elif quote is not None:
            current.append(char)
            if char == quote:
                if ddl[i + 1:i + 2] == quote:
                    current.append(quote)
                    i += 1
                else:
                    quote = None
        elif bracket_depth > 0:
            # 4.1.14 audit: [bracketed] identifiers may legally contain
            # semicolons; only the closing bracket ends the region.
            current.append(char)
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
        elif two == "--":
            current.append(two)
            i += 1
            line_comment = True
        elif two == "/*":
            current.append(two)
            i += 1
            block_comment = True
        elif char in ("'", '"', "`"):
            # 4.1.14 audit: backticks quote identifiers like quotes do.
            current.append(char)
            quote = char
        elif char == "[":
            current.append(char)
            bracket_depth = 1
        elif char == ";":
            statements.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def repair_ddl(conn: sqlite3.Connection, ddl: str) -> None:
    """Re-execute static ``ddl`` idempotently against ``conn``.

    Safe to call on a complete schema (every statement becomes a no-op),
    on a partial one (only the missing pieces apply), and on an empty
    database missing whole tables (``ADD COLUMN`` on a missing table
    raises, honestly — repair cannot invent tables the migration never
    owned; use the migration's own ``apply`` path for those).
    """
    for chunk in _split_statements(ddl):
        stmt = chunk.strip()
        if not stmt:
            continue
        if not _COMMENT_RE.sub("", stmt).strip():
            continue  # comment-only chunk (e.g. "-- rebuild for ...")
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
