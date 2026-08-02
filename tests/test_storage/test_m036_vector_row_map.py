# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage.migrations import (
    M036_vector_row_map as m036,
)


def _object_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master").fetchall()
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


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


def test_apply_creates_table_and_index(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m036.apply(conn)
        names = _object_names(conn)
        assert "vector_row_map" in names
        assert "idx_vector_row_map_profile" in names
    finally:
        conn.close()


def test_table_shape(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m036.apply(conn)
        cols = _columns(conn, "vector_row_map")
        assert {"fact_id", "profile_id", "vec_rowid"} <= cols
    finally:
        conn.close()


def test_primary_key_on_fact_id(memory_db: Path) -> None:
    # duplicate fact_id inserts must fail
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m036.apply(conn)
        conn.execute("INSERT INTO vector_row_map VALUES ('f1', 'p1', 1)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO vector_row_map VALUES ('f1', 'p1', 2)")
    finally:
        conn.close()


def test_apply_is_idempotent(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m036.apply(conn)
        assert m036.verify(conn) is True
        m036.apply(conn)
        assert m036.verify(conn) is True
    finally:
        conn.close()


def test_repair_restores_missing_table(memory_db: Path) -> None:
    conn = sqlite3.connect(memory_db, isolation_level=None)
    try:
        m036.apply(conn)
        conn.execute("DROP TABLE vector_row_map")
        assert m036.verify(conn) is False
        m036.repair(conn)
        assert m036.verify(conn) is True
    finally:
        conn.close()


def test_registered_in_runner_catalogue() -> None:
    names = {m.name for m in mr.MIGRATIONS}
    assert m036.NAME in names


def test_apply_all_runs_m036(tmp_path: Path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    result = mr.apply_all(learning_db, memory_db)
    assert m036.NAME not in result["failed"], result["details"].get(m036.NAME)
    conn = sqlite3.connect(memory_db)
    try:
        assert "vector_row_map" in _object_names(conn)
    finally:
        conn.close()
