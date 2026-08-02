# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage.migrations import (
    M033_projection_transactions as m033,
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
        m033.apply(conn)
        names = _object_names(conn)
        assert "projection_obligations" in names
        assert "completion_manifests" in names
    finally:
        conn.close()


def test_obligations_table_shape(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        columns = _memory_columns(conn, "projection_obligations")
        assert {
            "obligation_id",
            "operation_id",
            "profile_id",
            "owner",
            "kind",
            "subject_id",
            "revision",
            "state",
            "checksum",
            "detail",
            "attempts",
            "created_at",
            "updated_at",
        } <= columns
    finally:
        conn.close()


def test_manifests_table_shape(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        columns = _memory_columns(conn, "completion_manifests")
        assert {
            "operation_id",
            "profile_id",
            "state",
            "all_met",
            "obligation_count",
            "owner_evidence_json",
            "manifest_hash",
            "created_at",
            "updated_at",
        } <= columns
    finally:
        conn.close()


def test_obligation_kind_and_state_are_constrained(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO projection_obligations "
                "(operation_id, profile_id, owner, kind, subject_id, state, "
                "created_at, updated_at) VALUES "
                "('op', 'p', 'vector', 'not-a-kind', 's', 'pending', 0, 0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO projection_obligations "
                "(operation_id, profile_id, owner, kind, subject_id, state, "
                "created_at, updated_at) VALUES "
                "('op', 'p', 'vector', 'apply', 's', 'not-a-state', 0, 0)"
            )
    finally:
        conn.close()


def test_obligation_unique_per_operation_owner_kind(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        conn.execute(
            "INSERT INTO projection_obligations "
            "(operation_id, profile_id, owner, kind, subject_id, state, "
            "created_at, updated_at) VALUES "
            "('op', 'p', 'vector', 'apply', 's', 'pending', 0, 0)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO projection_obligations "
                "(operation_id, profile_id, owner, kind, subject_id, state, "
                "created_at, updated_at) VALUES "
                "('op', 'p', 'vector', 'apply', 's2', 'pending', 0, 0)"
            )
    finally:
        conn.close()


def test_manifest_state_is_constrained(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO completion_manifests "
                "(operation_id, profile_id, state, all_met, obligation_count, "
                "owner_evidence_json, manifest_hash, created_at, updated_at) "
                "VALUES ('op', 'p', 'PARTIAL', 0, 0, '[]', 'h', 0, 0)"
            )
    finally:
        conn.close()


def test_apply_is_idempotent(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        assert m033.verify(conn) is True
        m033.apply(conn)
        assert m033.verify(conn) is True
    finally:
        conn.close()


def test_repair_restores_missing_table(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m033.apply(conn)
        conn.execute("DROP TABLE completion_manifests")
        assert m033.verify(conn) is False
        m033.repair(conn)
        assert m033.verify(conn) is True
    finally:
        conn.close()


def test_registered_in_runner_catalogue() -> None:
    names = {m.name for m in mr.MIGRATIONS}
    assert m033.NAME in names


def test_apply_all_runs_m033(tmp_path: Path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    result = mr.apply_all(learning_db, memory_db)
    assert m033.NAME not in result["failed"]
    conn = sqlite3.connect(memory_db)
    try:
        names = _object_names(conn)
        assert "projection_obligations" in names
        assert "completion_manifests" in names
    finally:
        conn.close()
