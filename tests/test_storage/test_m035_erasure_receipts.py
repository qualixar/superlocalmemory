# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage.migrations import (
    M035_erasure_receipts as m035,
)


def _memory_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _object_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master").fetchall()
    }


@pytest.fixture
def memory_db(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        from superlocalmemory.storage.migrations import M003_migration_log as m003

        conn.executescript(m003.DDL)
    finally:
        conn.close()
    return path


def test_apply_creates_both_tables(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        names = _object_names(conn)
        assert "erasure_receipts" in names
        assert "projection_tombstones" in names
    finally:
        conn.close()


def test_receipt_table_shape(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        columns = _memory_columns(conn, "erasure_receipts")
        assert {
            "erasure_id",
            "profile_id",
            "subject_type",
            "subject_id",
            "requested_by",
            "fact_count",
            "state",
            "all_erased",
            "owner_evidence_json",
            "audit_hash",
            "requested_at",
            "completed_at",
        } <= columns
    finally:
        conn.close()


def test_tombstone_table_shape(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        columns = _memory_columns(conn, "projection_tombstones")
        assert {"profile_id", "fact_id", "erasure_id", "created_at"} <= columns
    finally:
        conn.close()


def test_receipt_state_and_subject_type_are_constrained(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO erasure_receipts "
                "(erasure_id, profile_id, subject_type, subject_id, state, "
                "audit_hash, requested_at, completed_at) VALUES "
                "('e', 'p', 'not-a-type', 's', 'COMPLETE', 'h', 0, 0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO erasure_receipts "
                "(erasure_id, profile_id, subject_type, subject_id, state, "
                "audit_hash, requested_at, completed_at) VALUES "
                "('e', 'p', 'fact', 's', 'PARTIAL', 'h', 0, 0)"
            )
    finally:
        conn.close()


def test_tombstone_unique_per_profile_fact(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        conn.execute(
            "INSERT INTO projection_tombstones "
            "(profile_id, fact_id, erasure_id, created_at) VALUES "
            "('p', 'f1', 'e1', 0)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO projection_tombstones "
                "(profile_id, fact_id, erasure_id, created_at) VALUES "
                "('p', 'f1', 'e2', 0)"
            )
    finally:
        conn.close()


def test_apply_is_idempotent(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        assert m035.verify(conn) is True
        m035.apply(conn)
        assert m035.verify(conn) is True
    finally:
        conn.close()


def test_repair_restores_missing_table(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m035.apply(conn)
        conn.execute("DROP TABLE projection_tombstones")
        assert m035.verify(conn) is False
        m035.repair(conn)
        assert m035.verify(conn) is True
    finally:
        conn.close()


def test_registered_in_runner_catalogue() -> None:
    names = {m.name for m in mr.MIGRATIONS}
    assert m035.NAME in names


def test_apply_all_runs_m035(tmp_path: Path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    result = mr.apply_all(learning_db, memory_db)
    assert m035.NAME not in result["failed"]
    conn = sqlite3.connect(memory_db)
    try:
        names = _object_names(conn)
        assert "erasure_receipts" in names
        assert "projection_tombstones" in names
    finally:
        conn.close()
