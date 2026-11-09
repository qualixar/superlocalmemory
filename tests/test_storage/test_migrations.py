# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.storage.migrations — schema versioning.

Covers:
  - get_schema_version / set_schema_version
  - needs_migration detection
  - is_v1_database detection
  - backup_database creates copy
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import (
    backup_database,
    get_schema_version,
    is_v1_database,
    needs_migration,
    set_schema_version,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create a minimal database at a temp path."""
    p = tmp_path / "test.db"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE schema_version "
        "(version INTEGER, applied_at TEXT DEFAULT '', description TEXT DEFAULT '')"
    )
    conn.execute(
        "INSERT INTO schema_version (version, applied_at, description) "
        "VALUES (1, datetime('now'), 'initial')"
    )
    conn.commit()
    conn.close()
    return p


@pytest.fixture()
def v1_db_path(tmp_path: Path) -> Path:
    """Create a V1-style database (has episodic_memory, no schema_version)."""
    p = tmp_path / "v1.db"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE episodic_memory "
        "(memory_id TEXT PRIMARY KEY, content TEXT)"
    )
    conn.execute(
        "INSERT INTO episodic_memory VALUES ('m1', 'hello')"
    )
    conn.commit()
    conn.close()
    return p


# ---------------------------------------------------------------------------
# get_schema_version
# ---------------------------------------------------------------------------

class TestGetSchemaVersion:
    def test_returns_version_from_table(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            version = get_schema_version(conn)
            assert version == 1
        finally:
            conn.close()

    def test_returns_zero_when_no_table(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.db"
        conn = sqlite3.connect(str(p))
        try:
            version = get_schema_version(conn)
            assert version == 0
        finally:
            conn.close()

    def test_returns_zero_when_table_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty_table.db"
        conn = sqlite3.connect(str(p))
        conn.execute(
            "CREATE TABLE schema_version "
            "(version INTEGER, applied_at TEXT, description TEXT)"
        )
        conn.commit()
        try:
            version = get_schema_version(conn)
            assert version == 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# set_schema_version
# ---------------------------------------------------------------------------

class TestSetSchemaVersion:
    def test_inserts_new_version(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            set_schema_version(conn, 2)
            conn.commit()
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            assert row[0] == 2
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# needs_migration
# ---------------------------------------------------------------------------

class TestNeedsMigration:
    def test_no_migration_needed_at_current_version(self, db_path: Path) -> None:
        assert needs_migration(db_path) is False

    def test_migration_needed_at_version_zero(self, tmp_path: Path) -> None:
        p = tmp_path / "old.db"
        conn = sqlite3.connect(str(p))
        # Create a DB with version 0 (no schema_version table)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()
        assert needs_migration(p) is True

    def test_nonexistent_db_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist.db"
        assert needs_migration(p) is False


# ---------------------------------------------------------------------------
# is_v1_database
# ---------------------------------------------------------------------------

class TestIsV1Database:
    def test_v1_database_detected(self, v1_db_path: Path) -> None:
        assert is_v1_database(v1_db_path) is True

    def test_innovation_database_not_v1(self, db_path: Path) -> None:
        assert is_v1_database(db_path) is False

    def test_nonexistent_db_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "ghost.db"
        assert is_v1_database(p) is False

    def test_empty_db_is_not_v1(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.db"
        conn = sqlite3.connect(str(p))
        conn.close()
        assert is_v1_database(p) is False


# ---------------------------------------------------------------------------
# backup_database
# ---------------------------------------------------------------------------

class TestBackupDatabase:
    def test_creates_backup_copy(self, db_path: Path) -> None:
        backup_path = backup_database(db_path)
        assert backup_path.exists()
        assert backup_path != db_path
        assert ".backup_" in backup_path.name
        assert backup_path.stat().st_size > 0

    def test_backup_preserves_data(self, db_path: Path) -> None:
        backup_path = backup_database(db_path)
        conn = sqlite3.connect(str(backup_path))
        try:
            row = conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            assert row[0] == 1
        finally:
            conn.close()

    def test_backup_name_format(self, db_path: Path) -> None:
        backup_path = backup_database(db_path)
        # Expected: test.backup_YYYYMMDD_HHMMSS.db
        assert backup_path.suffix == ".db"
        assert "backup_" in backup_path.name
