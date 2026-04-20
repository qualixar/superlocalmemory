# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07

"""Per-migration DDL content tests.

Covers LLD-07 §8.2 — each migration's exact schema effects on a
pre-v3.4.22 database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import (
    M001_add_signal_features_columns as M001,
    M002_model_state_history as M002,
    M003_migration_log as M003,
    M004_cross_platform_sync_log as M004,
    M005_bandit_tables as M005,
)


def _cols(db: Path, table: str) -> list[str]:
    with sqlite3.connect(db) as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _indexes(db: Path, table: str) -> set[str]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table,),
        ).fetchall()
    return {r[0] for r in rows}


@pytest.fixture
def learning_db(tmp_path: Path) -> Path:
    db = tmp_path / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE learning_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                query TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                value REAL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE learning_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                features_json TEXT NOT NULL,
                label REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE learning_model_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL UNIQUE,
                state_bytes BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
    return db


@pytest.fixture
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE profiles (profile_id TEXT PRIMARY KEY);
            CREATE TABLE action_outcomes (
                outcome_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default',
                query TEXT NOT NULL DEFAULT '',
                fact_ids_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT '',
                context_json TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
    return db


def _run_ddl(db: Path, ddl: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(ddl)


# ---------------------------------------------------------------------------
# M001
# ---------------------------------------------------------------------------


def test_M001_adds_signal_columns(learning_db: Path) -> None:
    _run_ddl(learning_db, M001.DDL)
    cols = set(_cols(learning_db, "learning_signals"))
    assert {"query_id", "query_text_hash", "position",
            "channel_scores", "cross_encoder"} <= cols


def test_M001_adds_feature_columns(learning_db: Path) -> None:
    _run_ddl(learning_db, M001.DDL)
    cols = set(_cols(learning_db, "learning_features"))
    assert {"signal_id", "is_synthetic"} <= cols


def test_M001_creates_indexes(learning_db: Path) -> None:
    _run_ddl(learning_db, M001.DDL)
    sig_idx = _indexes(learning_db, "learning_signals")
    assert "idx_signals_profile_time" in sig_idx
    assert "idx_signals_query_id" in sig_idx
    feat_idx = _indexes(learning_db, "learning_features")
    assert "idx_features_signal" in feat_idx


def test_M001_ddl_has_begin_commit() -> None:
    assert "BEGIN IMMEDIATE" in M001.DDL
    assert "COMMIT" in M001.DDL


def test_M001_has_no_destructive_ops() -> None:
    ddl_upper = M001.DDL.upper()
    assert "DROP TABLE" not in ddl_upper
    assert "DROP COLUMN" not in ddl_upper
    assert "DELETE FROM" not in ddl_upper


# ---------------------------------------------------------------------------
# M002
# ---------------------------------------------------------------------------


def test_M002_rebuilds_model_state(learning_db: Path) -> None:
    _run_ddl(learning_db, M002.DDL)
    cols = set(_cols(learning_db, "learning_model_state"))
    assert {"model_version", "bytes_sha256", "trained_on_count",
            "feature_names", "metrics_json", "is_active",
            "trained_at", "updated_at"} <= cols


def test_M002_removes_unique_profile_id(learning_db: Path) -> None:
    _run_ddl(learning_db, M002.DDL)
    with sqlite3.connect(learning_db) as conn:
        # Two rows with same profile_id but different is_active should be fine
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, trained_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("p1", b"x", "t", "t", 0),
        )
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, trained_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("p1", b"y", "t", "t", 1),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state WHERE profile_id='p1'"
        ).fetchone()
    assert rows[0] == 2


def test_M002_unique_active_per_profile(learning_db: Path) -> None:
    _run_ddl(learning_db, M002.DDL)
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, trained_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("p2", b"x", "t", "t", 1),
        )
        conn.commit()
        # Attempt a second active row for same profile → violates partial unique index
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, state_bytes, trained_at, updated_at, is_active) "
                "VALUES (?, ?, ?, ?, ?)",
                ("p2", b"y", "t", "t", 1),
            )
            conn.commit()


def test_M002_preserves_existing_rows(learning_db: Path) -> None:
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "INSERT INTO learning_model_state (profile_id, state_bytes, updated_at) "
            "VALUES (?, ?, ?)",
            ("old", b"legacy", "2026-01-01"),
        )
        conn.commit()
    _run_ddl(learning_db, M002.DDL)
    with sqlite3.connect(learning_db) as conn:
        row = conn.execute(
            "SELECT profile_id, state_bytes, is_active FROM learning_model_state "
            "WHERE profile_id='old'"
        ).fetchone()
    assert row == ("old", b"legacy", 1)


# ---------------------------------------------------------------------------
# M003
# ---------------------------------------------------------------------------


def test_M003_creates_migration_log(learning_db: Path) -> None:
    _run_ddl(learning_db, M003.DDL)
    with sqlite3.connect(learning_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "migration_log" in tables
    cols = set(_cols(learning_db, "migration_log"))
    assert {"name", "applied_at", "ddl_sha256", "rows_affected", "status"} <= cols


def test_M003_idempotent(learning_db: Path) -> None:
    _run_ddl(learning_db, M003.DDL)
    _run_ddl(learning_db, M003.DDL)  # no error


# ---------------------------------------------------------------------------
# M004
# ---------------------------------------------------------------------------


def test_M004_creates_sync_log(memory_db: Path) -> None:
    _run_ddl(memory_db, M004.DDL)
    with sqlite3.connect(memory_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "cross_platform_sync_log" in tables
    cols = set(_cols(memory_db, "cross_platform_sync_log"))
    assert {"adapter_name", "profile_id", "target_path_sha256",
            "target_basename", "last_sync_at", "bytes_written",
            "content_sha256", "success", "error_msg"} <= cols


def test_M004_primary_key(memory_db: Path) -> None:
    _run_ddl(memory_db, M004.DDL)
    with sqlite3.connect(memory_db) as conn:
        conn.execute(
            "INSERT INTO cross_platform_sync_log "
            "(adapter_name, profile_id, target_path_sha256, target_basename, "
            "last_sync_at, bytes_written, content_sha256, success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("cursor", "default", "abc", "rules.mdc", "now", 10, "h", 1),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO cross_platform_sync_log "
                "(adapter_name, profile_id, target_path_sha256, target_basename, "
                "last_sync_at, bytes_written, content_sha256, success) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("cursor", "default", "abc", "rules.mdc", "now", 10, "h", 1),
            )


# ---------------------------------------------------------------------------
# M005
# ---------------------------------------------------------------------------


def test_M005_creates_bandit_tables(learning_db: Path) -> None:
    _run_ddl(learning_db, M005.DDL)
    with sqlite3.connect(learning_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "bandit_arms" in tables
    assert "bandit_plays" in tables


def test_M005_bandit_arms_pk(learning_db: Path) -> None:
    _run_ddl(learning_db, M005.DDL)
    cols = set(_cols(learning_db, "bandit_arms"))
    assert {"profile_id", "stratum", "arm_id", "alpha", "beta",
            "plays", "last_played_at"} <= cols


def test_M005_indexes(learning_db: Path) -> None:
    _run_ddl(learning_db, M005.DDL)
    plays_idx = _indexes(learning_db, "bandit_plays")
    assert "idx_plays_query" in plays_idx
    assert "idx_plays_unsettled" in plays_idx
    assert "idx_plays_retention" in plays_idx


def test_M005_has_no_destructive_ops() -> None:
    u = M005.DDL.upper()
    assert "DROP TABLE" not in u
    assert "DELETE FROM" not in u


# ---------------------------------------------------------------------------
# All migrations share hard rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", [M001, M002, M003, M004, M005])
def test_every_migration_has_name(mod) -> None:
    assert hasattr(mod, "NAME")
    assert isinstance(mod.NAME, str)
    assert mod.NAME.startswith("M0")


@pytest.mark.parametrize("mod", [M001, M002, M003, M004, M005])
def test_every_migration_has_ddl(mod) -> None:
    assert hasattr(mod, "DDL")
    assert isinstance(mod.DDL, str)
    assert len(mod.DDL) > 10


@pytest.mark.parametrize("mod", [M001, M002, M003, M004, M005])
def test_no_pickle_in_ddl(mod) -> None:
    assert "pickle" not in mod.DDL.lower()


@pytest.mark.parametrize("mod", [M001, M003, M004, M005])
def test_no_drop_in_additive_migrations(mod) -> None:
    # M002 legitimately uses DROP to rebuild the model_state table — excluded.
    assert "DROP TABLE" not in mod.DDL.upper()
    assert "DROP COLUMN" not in mod.DDL.upper()
    assert "DELETE FROM" not in mod.DDL.upper()


# ---------------------------------------------------------------------------
# verify() hooks — cover the "idempotent re-apply" safety net
# ---------------------------------------------------------------------------


def test_M001_verify_before_and_after(learning_db: Path) -> None:
    with sqlite3.connect(learning_db) as conn:
        assert M001.verify(conn) is False
    _run_ddl(learning_db, M001.DDL)
    with sqlite3.connect(learning_db) as conn:
        assert M001.verify(conn) is True


def test_M002_verify_before_and_after(learning_db: Path) -> None:
    with sqlite3.connect(learning_db) as conn:
        assert M002.verify(conn) is False
    _run_ddl(learning_db, M002.DDL)
    with sqlite3.connect(learning_db) as conn:
        assert M002.verify(conn) is True


def test_M003_verify_before_and_after(learning_db: Path) -> None:
    with sqlite3.connect(learning_db) as conn:
        assert M003.verify(conn) is False
    _run_ddl(learning_db, M003.DDL)
    with sqlite3.connect(learning_db) as conn:
        assert M003.verify(conn) is True


def test_M004_verify_before_and_after(memory_db: Path) -> None:
    with sqlite3.connect(memory_db) as conn:
        assert M004.verify(conn) is False
    _run_ddl(memory_db, M004.DDL)
    with sqlite3.connect(memory_db) as conn:
        assert M004.verify(conn) is True


def test_M005_verify_before_and_after(learning_db: Path) -> None:
    with sqlite3.connect(learning_db) as conn:
        assert M005.verify(conn) is False
    _run_ddl(learning_db, M005.DDL)
    with sqlite3.connect(learning_db) as conn:
        assert M005.verify(conn) is True


@pytest.mark.parametrize("mod", [M001, M002, M003, M004, M005])
def test_verify_returns_false_on_closed_connection(
    mod, learning_db: Path, memory_db: Path,
) -> None:
    # Opening then closing a conn should make PRAGMA calls raise
    # sqlite3.ProgrammingError (a subclass of sqlite3.Error). The verify
    # helper must return False rather than propagating.
    db = memory_db if getattr(mod, "DB_TARGET", "learning") == "memory" else learning_db
    conn = sqlite3.connect(db)
    conn.close()
    assert mod.verify(conn) is False
