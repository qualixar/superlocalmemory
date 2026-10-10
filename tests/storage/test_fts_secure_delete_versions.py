# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""FTS5 ``secure-delete`` corrupts the keyword index on SQLite 3.44.0 up to
(not including) 3.46.1: a content update (FTS 'delete' + insert) leaves the
index "malformed" for ``PRAGMA quick_check``. It is never turned on there."""
import sqlite3
from contextlib import closing

import pytest

from superlocalmemory.storage import fts_residue
from tests.test_storage import _upgrade_store as store

_IN_BROKEN_RANGE = (3, 44, 0) <= sqlite3.sqlite_version_info < (3, 46, 1)


@pytest.mark.parametrize("version, supported", [
    ((3, 37, 2), False), ((3, 42, 0), True), ((3, 43, 1), True),
    ((3, 44, 0), False), ((3, 45, 1), False), ((3, 46, 0), False),
    ((3, 46, 1), True), ((3, 51, 1), True),
])
def test_secure_delete_supported_by_version(version, supported):
    assert fts_residue.secure_delete_supported(version) is supported


def _fts() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(content)")
    return conn


def _config_rows(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT v FROM atomic_facts_fts_config WHERE k = 'secure-delete'").fetchall()


def test_broken_sqlite_is_unsupported_and_writes_nothing(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 45, 1))
    monkeypatch.setattr(fts_residue, "_old_sqlite_reported", False)
    conn = _fts()
    assert fts_residue.ensure_secure_delete(conn)["atomic_facts_fts"] == "unsupported"
    assert _config_rows(conn) == []


def test_safe_sqlite_enables_it(monkeypatch):
    if sqlite3.sqlite_version_info < (3, 42, 0):
        pytest.skip("this SQLite has no FTS5 secure-delete at all")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 46, 1))
    conn = _fts()
    assert fts_residue.ensure_secure_delete(conn)["atomic_facts_fts"] == "enabled"
    assert _config_rows(conn) == [(1,)] or _config_rows(conn) == [("1",)]


def test_an_index_already_on_is_left_alone_on_broken_sqlite(monkeypatch):
    if sqlite3.sqlite_version_info < (3, 42, 0):
        pytest.skip("this SQLite has no FTS5 secure-delete at all")
    conn = _fts()
    conn.execute("INSERT INTO atomic_facts_fts(atomic_facts_fts, rank) VALUES('secure-delete', 1)")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 45, 1))
    monkeypatch.setattr(fts_residue, "_old_sqlite_reported", False)
    assert fts_residue.ensure_secure_delete(conn)["atomic_facts_fts"] == "unsupported"
    assert len(_config_rows(conn)) == 1


@pytest.mark.skipif(
    not _IN_BROKEN_RANGE,
    reason="only SQLite 3.44.0 up to (not including) 3.46.1 corrupts the FTS5 index")
def test_content_update_leaves_the_store_intact(tmp_path):
    _, memory_db = store.current_store(tmp_path / "s")
    fids = store.add_memory(memory_db, "m1", ["alpha beta gamma"])
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute("UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
                     ("delta epsilon zeta", fids[0]))
        conn.commit()
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
