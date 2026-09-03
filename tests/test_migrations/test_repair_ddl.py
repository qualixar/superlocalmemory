# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Unit contracts for the shared migration repair helper (4.1.14 #133)."""
from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage.migrations._repair_util import repair_ddl

_M001_STYLE_DDL = """
BEGIN IMMEDIATE;

ALTER TABLE learning_signals ADD COLUMN query_id TEXT DEFAULT '';
ALTER TABLE learning_signals ADD COLUMN position INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_signals_profile_time
    ON learning_signals(profile_id, created_at);

COMMIT;
"""


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE learning_signals (id INTEGER PRIMARY KEY, "
        "profile_id TEXT, created_at TEXT)"
    )
    yield connection
    connection.close()


def test_repair_applies_missing_pieces(conn) -> None:
    repair_ddl(conn, _M001_STYLE_DDL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(learning_signals)")}
    assert {"query_id", "position"} <= cols
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    assert "idx_signals_profile_time" in indexes


def test_repair_is_noop_on_complete_schema(conn) -> None:
    repair_ddl(conn, _M001_STYLE_DDL)
    repair_ddl(conn, _M001_STYLE_DDL)  # second run must not raise
    cols = {row[1] for row in conn.execute("PRAGMA table_info(learning_signals)")}
    assert {"query_id", "position"} <= cols


def test_repair_tolerates_bare_create_of_existing_table(conn) -> None:
    ddl = "CREATE TABLE learning_signals (id INTEGER PRIMARY KEY);"
    repair_ddl(conn, ddl)  # already exists -> tolerated, not raised


def test_repair_propagates_real_errors(conn) -> None:
    with pytest.raises(sqlite3.OperationalError):
        repair_ddl(conn, "INSERT INTO no_such_table (a) VALUES (1)")


def test_repair_add_column_on_missing_table_raises(conn) -> None:
    # Repair cannot invent tables the migration never owned; the
    # migration's own apply path owns that. Raising here is honesty,
    # and the framework reports it instead of claiming success.
    with pytest.raises(sqlite3.OperationalError):
        repair_ddl(conn, "ALTER TABLE no_such_table ADD COLUMN c TEXT;")
