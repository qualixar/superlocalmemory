# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M037 — additive columns for HMAC manifest/receipt versioning.

Adds ``manifest_version`` to ``completion_manifests`` and
``receipt_version`` to ``erasure_receipts``.  Default value 1 means the
existing rows were written with the unkeyed SHA-256 scheme (v1).  New rows
are written with version 2 (HMAC-SHA256).  The verify path reads the
version column and selects the appropriate verification algorithm.

Dependency: M033 (completion_manifests), M035 (erasure_receipts).
"""

from __future__ import annotations

import sqlite3

NAME = "M037_manifest_hmac_version"
DB_TARGET = "memory"

_ADDED_COLUMNS: dict[str, tuple[str, str]] = {
    "manifest_version_in_completion_manifests": (
        "completion_manifests",
        "manifest_version INTEGER NOT NULL DEFAULT 1",
    ),
    "receipt_version_in_erasure_receipts": (
        "erasure_receipts",
        "receipt_version INTEGER NOT NULL DEFAULT 1",
    ),
}

DDL = "\n".join(
    f"ALTER TABLE {tbl} ADD COLUMN {col_def};"
    for _, (tbl, col_def) in _ADDED_COLUMNS.items()
)


def apply(conn: sqlite3.Connection) -> None:
    for _key, (table, col_def) in _ADDED_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        col_name = col_def.split()[0]
        if col_name not in _columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    for _key, (table, col_def) in _ADDED_COLUMNS.items():
        if not _table_exists(conn, table):
            return False
        col_name = col_def.split()[0]
        if col_name not in _columns(conn, table):
            return False
    return True


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
