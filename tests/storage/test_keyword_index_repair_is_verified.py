# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""GitHub #204 (SQLite 3.45.1): ``slm db repair --apply`` said it had rewritten the
keyword indexes, yet the index stayed malformed and secure-delete stayed on.

A repair of a derived index has to prove itself: rebuild, check again, fall back
to dropping and recreating the table from the stored memories, check again, and
say so plainly when it is still damaged. On a SQLite that damages the index with
secure-delete on, that setting is off when the repair is done, whether or not an
earlier repair had already purged the index. These tests run on any SQLite: the
version is patched where the behaviour depends on it.
"""
import argparse
import sqlite3
from contextlib import closing

import pytest
from tests.test_storage import _upgrade_store as store

from superlocalmemory.storage import fts_residue
from superlocalmemory.storage.integrity_repair import Repair
from superlocalmemory.storage.integrity_scan import keyword_index_state

TABLE = "atomic_facts_fts"
EXPANSION = "fact_expansion_fts"


def _store(root, memories=4):
    _, memory_db = store.current_store(root)
    for i in range(memories):
        store.add_memory(memory_db, f"m{i}", [f"alpha{i} beta gamma", f"delta{i} epsilon zeta"])
    return memory_db


def _cut(memory_db, table):
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute(f"UPDATE {table}_data SET block = substr(block, 1, length(block) - 3) "  # noqa: S608
                     "WHERE id > 10")
        conn.commit()
        assert conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"


def _scalar(memory_db, sql):
    with closing(sqlite3.connect(str(memory_db))) as conn:
        return conn.execute(sql).fetchone()[0]


def _quick_check(memory_db):
    return _scalar(memory_db, "PRAGMA quick_check")


def _hits(memory_db, word):
    return _scalar(memory_db, f"SELECT COUNT(*) FROM {TABLE} WHERE {TABLE} MATCH '{word}'")  # noqa: S608


def test_a_purge_that_was_already_done_still_turns_secure_delete_off_on_a_broken_sqlite(
        tmp_path, monkeypatch):
    """The issue's state: purged_by_repair true and secure_delete_on, so the
    repair used to return at once and leave the setting that damages the index."""
    memory_db = _store(tmp_path / "s")
    if sqlite3.sqlite_version_info < fts_residue.SECURE_DELETE_MIN_SQLITE:
        pytest.skip("this SQLite has no FTS5 secure-delete to set the case up with")
    with closing(sqlite3.connect(str(memory_db))) as conn:
        for table in fts_residue.FTS_TABLES:
            conn.execute(f"INSERT INTO {table}({table}, rank) VALUES('secure-delete', 1)")  # noqa: S608
        from superlocalmemory.storage import integrity_receipts as receipts
        receipts.ensure_tables(conn)
        receipts.receipt(conn, "old", "purge_keyword_index", TABLE, "earlier repair",
                         {"blocks": 1}, {"blocks": 1}, undoable=False)
        conn.commit()
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 45, 1))
    Repair(memory_db).apply()
    with closing(sqlite3.connect(str(memory_db))) as conn:
        assert not fts_residue.secure_delete_on(conn, TABLE)
        assert not fts_residue.secure_delete_on(conn, EXPANSION)
        state = keyword_index_state(conn)
    assert state[TABLE] == "secure_delete_off" and state[EXPANSION] == "secure_delete_off"


def test_secure_delete_is_off_before_the_index_is_rebuilt(tmp_path, monkeypatch):
    """Rebuilding with the damaging setting still on would write the damage back."""
    memory_db = _store(tmp_path / "s")
    _cut(memory_db, TABLE)
    seen = []
    real = fts_residue.rebuild_keyword_index

    def spy(conn, table):
        seen.append(fts_residue.secure_delete_on(conn, table))
        real(conn, table)

    monkeypatch.setattr(fts_residue, "rebuild_keyword_index", spy)
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute(f"INSERT INTO {TABLE}({TABLE}, rank) VALUES('secure-delete', 1)")  # noqa: S608
        conn.commit()
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 45, 1))
    Repair(memory_db).apply()
    assert seen == [False]


def test_a_rebuild_that_leaves_the_index_malformed_falls_back_to_recreating_it(
        tmp_path, monkeypatch):
    memory_db = _store(tmp_path / "s")
    facts = _scalar(memory_db, "SELECT COUNT(*) FROM atomic_facts")
    _cut(memory_db, TABLE)
    monkeypatch.setattr(fts_residue, "rebuild_keyword_index", lambda conn, table: None)
    summary = Repair(memory_db).apply()
    assert summary["done"].get("keyword_index.recreated") == 1
    assert "keyword_index.still_damaged" not in summary["done"]
    assert _quick_check(memory_db) == "ok"
    assert summary["after"]["keyword_index"]["damaged"] == []
    assert _scalar(memory_db, "SELECT COUNT(*) FROM atomic_facts") == facts
    assert _hits(memory_db, "epsilon") == facts // 2
    assert _scalar(memory_db, f"SELECT COUNT(*) FROM {TABLE}_docsize") == facts  # noqa: S608
    with closing(sqlite3.connect(str(memory_db))) as conn:
        action = conn.execute("SELECT reason FROM integrity_repair_receipts "
                              "WHERE action = 'recreate_keyword_index'").fetchone()
    assert action and "recreated from the stored memories" in action[0]


def test_a_standalone_index_keeps_its_rows_when_it_has_to_be_recreated(tmp_path, monkeypatch):
    """fact_expansion_fts is not rebuilt from another table: its own rows are the content."""
    memory_db = _store(tmp_path / "s")
    with closing(sqlite3.connect(str(memory_db))) as conn:
        for i in range(30):
            conn.execute(f"INSERT INTO {EXPANSION}(fact_id, alt_keys) VALUES (?, ?)",  # noqa: S608
                         (f"m{i % 4}-f0", f"alias{i} nickname{i} other{i}"))
        conn.commit()
    _cut(memory_db, EXPANSION)
    monkeypatch.setattr(fts_residue, "rebuild_keyword_index", lambda conn, table: None)
    summary = Repair(memory_db).apply()
    assert summary["done"].get("keyword_index.recreated") == 1
    assert _quick_check(memory_db) == "ok"
    assert _scalar(memory_db, f"SELECT COUNT(*) FROM {EXPANSION}") == 30  # noqa: S608
    assert _scalar(memory_db, f"SELECT COUNT(*) FROM {EXPANSION} WHERE {EXPANSION} "  # noqa: S608
                   "MATCH 'alias7'") == 1


def test_an_index_that_cannot_be_repaired_is_reported_not_claimed_fixed(tmp_path, monkeypatch):
    memory_db = _store(tmp_path / "s")
    _cut(memory_db, TABLE)
    monkeypatch.setattr(fts_residue, "rebuild_keyword_index", lambda conn, table: None)
    monkeypatch.setattr(fts_residue, "recreate_keyword_index", lambda conn, table: None)
    summary = Repair(memory_db).apply()
    assert summary["done"].get("keyword_index.still_damaged") == 1
    assert summary["after"]["keyword_index"]["damaged"] == [TABLE]
    with closing(sqlite3.connect(str(memory_db))) as conn:
        action = conn.execute("SELECT 1 FROM integrity_repair_receipts "
                              "WHERE action IN ('rebuild_keyword_index', 'recreate_keyword_index')"
                              ).fetchone()
    assert action is None  # no receipt says it was fixed


def test_the_command_exits_nonzero_and_says_so_when_the_index_is_still_damaged(
        tmp_path, monkeypatch, capsys):
    from superlocalmemory.cli import integrity_cmd

    root = tmp_path / "s"
    memory_db = _store(root)
    _cut(memory_db, TABLE)
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setattr(fts_residue, "rebuild_keyword_index", lambda conn, table: None)
    monkeypatch.setattr(fts_residue, "recreate_keyword_index", lambda conn, table: None)
    monkeypatch.setattr(integrity_cmd, "_via_daemon", lambda body: None)
    args = argparse.Namespace(apply=True, undo="", root=str(root), batch_size=100,
                              pause_ms=0, max_seconds=None, json=False)
    assert integrity_cmd.cmd_db_repair(args) == 1
    out = capsys.readouterr()
    assert TABLE in out.err and "still damaged" in out.err
