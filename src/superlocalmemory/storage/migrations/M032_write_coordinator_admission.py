# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M032 — durable, append-only receipts for canonical write commands.

The receipt ledger is profile-isolated at the client idempotency boundary.
``command_id`` and ``journal_id`` remain globally durable identifiers; a
client-supplied idempotency key is unique only with its target profile.  An
operation id is a projection label (for example, ``update:<fact_id>``), not a
durable command identifier, so it is deliberately indexed but not unique.
"""

from __future__ import annotations

import sqlite3

NAME = "M032_write_coordinator_admission"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS write_commits (
    commit_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id      TEXT NOT NULL UNIQUE,
    journal_id      TEXT NOT NULL UNIQUE,
    command_kind    TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    profile_id      TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    operation_id    TEXT NOT NULL,
    receipt_json    TEXT NOT NULL,
    committed_at    REAL NOT NULL,
    UNIQUE(profile_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_write_commits_committed_at
    ON write_commits (committed_at);
CREATE INDEX IF NOT EXISTS idx_write_commits_operation_id
    ON write_commits (operation_id);
CREATE TRIGGER IF NOT EXISTS trg_write_commits_immutable_update
BEFORE UPDATE ON write_commits
BEGIN
    SELECT RAISE(ABORT, 'write_commits receipts are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_write_commits_immutable_delete
BEFORE DELETE ON write_commits
BEGIN
    SELECT RAISE(ABORT, 'write_commits receipts are immutable');
END;
"""

_CREATE_TABLE = DDL.split(";", 1)[0]
_CREATE_COMMITTED_AT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_write_commits_committed_at "
    "ON write_commits (committed_at)"
)
_CREATE_OPERATION_ID_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_write_commits_operation_id "
    "ON write_commits (operation_id)"
)
_CREATE_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_write_commits_immutable_update
BEFORE UPDATE ON write_commits
BEGIN
    SELECT RAISE(ABORT, 'write_commits receipts are immutable');
END
"""
_CREATE_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_write_commits_immutable_delete
BEFORE DELETE ON write_commits
BEGIN
    SELECT RAISE(ABORT, 'write_commits receipts are immutable');
END
"""


def apply(conn: sqlite3.Connection) -> None:
    """Create or safely upgrade the profile-scoped append-only receipt ledger."""
    if verify(conn):
        return
    if not _table_exists(conn):
        _create_current_schema(conn)
        return
    _rebuild_legacy_schema(conn)


def repair(conn: sqlite3.Connection) -> None:
    """Repair a completed provisional M032 in developer/test databases."""
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    """Return true only when the full profile-safe ledger contract exists."""
    if not _table_exists(conn):
        return False
    columns = {row[1] for row in conn.execute("PRAGMA table_info(write_commits)").fetchall()}
    required = {
        "commit_sequence",
        "command_id",
        "journal_id",
        "command_kind",
        "request_hash",
        "profile_id",
        "idempotency_key",
        "operation_id",
        "receipt_json",
        "committed_at",
    }
    if not required <= columns:
        return False
    if not _has_unique_index(conn, ("command_id",)):
        return False
    if not _has_unique_index(conn, ("journal_id",)):
        return False
    if not _has_unique_index(conn, ("profile_id", "idempotency_key")):
        return False
    if _has_unique_index(conn, ("idempotency_key",)):
        return False
    if _has_unique_index(conn, ("operation_id",)):
        return False
    object_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?, ?)",
            (
                "idx_write_commits_committed_at",
                "idx_write_commits_operation_id",
                "trg_write_commits_immutable_update",
                "trg_write_commits_immutable_delete",
            ),
        ).fetchall()
    }
    return object_names == {
        "idx_write_commits_committed_at",
        "idx_write_commits_operation_id",
        "trg_write_commits_immutable_update",
        "trg_write_commits_immutable_delete",
    }


def _table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='write_commits'"
    ).fetchone() is not None


def _has_unique_index(conn: sqlite3.Connection, columns: tuple[str, ...]) -> bool:
    for index in conn.execute("PRAGMA index_list(write_commits)").fetchall():
        if not index[2]:
            continue
        index_columns = tuple(
            row[2] for row in conn.execute(f"PRAGMA index_info({index[1]})").fetchall()
        )
        if index_columns == columns:
            return True
    return False


def _create_current_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_COMMITTED_AT_INDEX)
    conn.execute(_CREATE_OPERATION_ID_INDEX)
    conn.execute(_CREATE_UPDATE_TRIGGER)
    conn.execute(_CREATE_DELETE_TRIGGER)


def _rebuild_legacy_schema(conn: sqlite3.Connection) -> None:
    """Rebuild only M032's own standalone table under one savepoint."""
    conn.execute("SAVEPOINT m032_profile_scoped_idempotency")
    try:
        conn.execute("DROP TRIGGER IF EXISTS trg_write_commits_immutable_update")
        conn.execute("DROP TRIGGER IF EXISTS trg_write_commits_immutable_delete")
        conn.execute("ALTER TABLE write_commits RENAME TO write_commits_legacy")
        conn.execute("DROP INDEX IF EXISTS idx_write_commits_committed_at")
        conn.execute("DROP INDEX IF EXISTS idx_write_commits_operation_id")
        _create_current_schema(conn)
        conn.execute(
            "INSERT INTO write_commits("
            "commit_sequence, command_id, journal_id, command_kind, request_hash, "
            "profile_id, idempotency_key, operation_id, receipt_json, committed_at"
            ") SELECT commit_sequence, command_id, journal_id, command_kind, request_hash, "
            "profile_id, idempotency_key, operation_id, receipt_json, committed_at "
            "FROM write_commits_legacy"
        )
        conn.execute("DROP TABLE write_commits_legacy")
    except BaseException:
        conn.execute("ROLLBACK TO m032_profile_scoped_idempotency")
        conn.execute("RELEASE m032_profile_scoped_idempotency")
        raise
    conn.execute("RELEASE m032_profile_scoped_idempotency")
