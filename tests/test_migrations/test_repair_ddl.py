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


def test_splitter_respects_semicolons_in_literals() -> None:
    from superlocalmemory.storage.migrations._repair_util import (
        _split_statements,
    )

    ddl = (
        "ALTER TABLE t ADD COLUMN note TEXT DEFAULT 'a;b';\n"
        "CREATE INDEX IF NOT EXISTS i ON t(note);"
    )
    parts = _split_statements(ddl)
    assert len(parts) == 2
    assert "DEFAULT 'a;b'" in parts[0]


def test_splitter_respects_comments() -> None:
    from superlocalmemory.storage.migrations._repair_util import (
        _split_statements,
        repair_ddl,
    )

    # A semicolon inside a comment is NOT a boundary: comment and
    # statement travel as one chunk, and repair executes it (SQLite
    # skips leading comments itself).
    ddl = "-- rebuild for UNIQUE(a, b);\nALTER TABLE t ADD COLUMN c TEXT;"
    parts = [p for p in _split_statements(ddl) if p.strip()]
    assert len(parts) == 1
    assert parts[0].startswith("--")

    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    repair_ddl(conn, ddl)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(t)")}
    assert "c" in cols
    conn.close()


def test_repair_skips_comment_only_chunks(conn) -> None:
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    repair_ddl(conn, "-- just a note;\nALTER TABLE t ADD COLUMN c TEXT;")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(t)")}
    assert "c" in cols


def test_repair_skips_transaction_variants(conn) -> None:
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    repair_ddl(
        conn,
        "BEGIN TRANSACTION;\n"
        "ALTER TABLE t ADD COLUMN c TEXT;\n"
        "COMMIT TRANSACTION;",
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(t)")}
    assert "c" in cols


def test_repair_quoted_table_in_pragma(conn) -> None:
    conn.execute('CREATE TABLE "odd table" (id INTEGER PRIMARY KEY)')
    repair_ddl(conn, 'ALTER TABLE "odd table" ADD COLUMN c TEXT;')
    repair_ddl(conn, 'ALTER TABLE "odd table" ADD COLUMN c TEXT;')
    cols = {
        row[1] for row in conn.execute('PRAGMA table_info("odd table")')
    }
    assert "c" in cols


def test_splitter_respects_bracket_and_backtick_identifiers() -> None:
    """4.1.14 audit: a semicolon inside a bracketed/backticked identifier
    is not a boundary (SQLite has no dollar-quoting to worry about)."""
    from superlocalmemory.storage.migrations._repair_util import (
        _split_statements,
    )

    ddl = (
        "ALTER TABLE [odd;table] ADD COLUMN c TEXT;\n"
        "ALTER TABLE `other;table` ADD COLUMN d TEXT;"
    )
    parts = [p.strip() for p in _split_statements(ddl) if p.strip()]
    assert len(parts) == 2
    assert parts[0].startswith("ALTER TABLE [odd;table]")
    assert parts[1].startswith("ALTER TABLE `other;table`")
