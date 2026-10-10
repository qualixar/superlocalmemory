# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A store that already has FTS5 ``secure-delete`` on under a SQLite that corrupts
it (3.44.0 up to, not including, 3.46.1): the setting is turned off at start,
and ``slm db repair`` rebuilds a damaged keyword index from the stored memories.

Tests that need real corruption run only inside that range; the ones that
monkeypatch the version run anywhere.
"""
import logging
import sqlite3
from contextlib import closing

import pytest

from superlocalmemory.storage import fts_residue
from superlocalmemory.storage.integrity_repair import Repair
from tests.test_storage import _upgrade_store as store

_IN_BROKEN_RANGE = (3, 44, 0) <= sqlite3.sqlite_version_info < (3, 46, 1)
needs_broken_sqlite = pytest.mark.skipif(
    not _IN_BROKEN_RANGE,
    reason="only SQLite 3.44.0 up to (not including) 3.46.1 corrupts the FTS5 index")
TABLE = "atomic_facts_fts"


def _damaged_store(root):
    """A real store whose keyword index is malformed (secure-delete on + content update)."""
    _, memory_db = store.current_store(root)
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute(f"INSERT INTO {TABLE}({TABLE}, rank) VALUES('secure-delete', 1)")
        conn.commit()
    fids = store.add_memory(memory_db, "m1", ["alpha beta gamma"])
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute("UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
                     ("delta epsilon zeta", fids[0]))
        conn.commit()
        assert conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
    return memory_db, fids[0]


def _quick_check(memory_db) -> str:
    with closing(sqlite3.connect(str(memory_db))) as conn:
        return conn.execute("PRAGMA quick_check").fetchone()[0]


def _match(memory_db, word: str) -> list:
    with closing(sqlite3.connect(str(memory_db))) as conn:
        return conn.execute(f"SELECT rowid FROM {TABLE} WHERE {TABLE} MATCH ?",
                            (word,)).fetchall()


@needs_broken_sqlite
def test_start_turns_secure_delete_off_and_warns_once(tmp_path, caplog, monkeypatch):
    memory_db, _ = _damaged_store(tmp_path / "s")
    monkeypatch.setattr(fts_residue, "_old_sqlite_reported", False)
    caplog.clear()  # the store set-up logs its own schema warnings
    with closing(sqlite3.connect(str(memory_db))) as conn:
        with caplog.at_level(logging.INFO, logger=fts_residue.logger.name):
            first = fts_residue.ensure_secure_delete(conn)[TABLE]
        assert first == "disabled"
        assert not fts_residue.secure_delete_on(conn, TABLE)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "slm db repair" in warnings[0].getMessage()
        assert TABLE in warnings[0].getMessage()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=fts_residue.logger.name):
            assert fts_residue.ensure_secure_delete(conn)[TABLE] == "unsupported"
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@needs_broken_sqlite
def test_damage_is_detected_and_a_healthy_store_is_not(tmp_path):
    memory_db, _ = _damaged_store(tmp_path / "bad")
    _, healthy_db = store.current_store(tmp_path / "good")
    store.add_memory(healthy_db, "m1", ["alpha beta gamma"])
    with closing(sqlite3.connect(str(memory_db))) as conn:
        assert fts_residue.keyword_index_damaged(conn, TABLE) is True
    with closing(sqlite3.connect(str(healthy_db))) as conn:
        assert fts_residue.keyword_index_damaged(conn, TABLE) is False


@needs_broken_sqlite
def test_repair_rebuilds_the_index_and_is_idempotent(tmp_path):
    memory_db, fact_id = _damaged_store(tmp_path / "s")
    with closing(sqlite3.connect(str(memory_db))) as conn:
        fts_residue.ensure_secure_delete(conn)  # what store open does
    done = Repair(memory_db).apply()["done"]
    assert done.get("keyword_index.rebuilt") == 1
    assert _quick_check(memory_db) == "ok"
    assert _match(memory_db, "epsilon") != []
    assert _match(memory_db, "alpha") == []
    with closing(sqlite3.connect(str(memory_db))) as conn:
        action = conn.execute("SELECT undoable, reason FROM integrity_repair_receipts "
                              "WHERE action = 'rebuild_keyword_index'").fetchone()
        assert action[0] == 0 and "rebuilt from the stored memories" in action[1]
    assert not any(k.startswith("keyword_index.") for k in Repair(memory_db).apply()["done"])


@needs_broken_sqlite
def test_repair_alone_also_rebuilds_while_secure_delete_is_still_on(tmp_path):
    memory_db, _ = _damaged_store(tmp_path / "s")
    done = Repair(memory_db).apply()["done"]
    assert done.get("keyword_index.rebuilt") == 1
    assert _quick_check(memory_db) == "ok"


@needs_broken_sqlite
def test_health_lists_the_damaged_index(tmp_path):
    from superlocalmemory.storage.integrity_scan import keyword_index_state
    memory_db, _ = _damaged_store(tmp_path / "s")
    with closing(sqlite3.connect(str(memory_db))) as conn:
        assert keyword_index_state(conn)["damaged"] == [TABLE]
    Repair(memory_db).apply()
    with closing(sqlite3.connect(str(memory_db))) as conn:
        assert keyword_index_state(conn)["damaged"] == []


@needs_broken_sqlite
def test_a_snapshot_of_the_repaired_store_succeeds(tmp_path):
    from superlocalmemory.storage import backup
    memory_db, _ = _damaged_store(tmp_path / "s")
    Repair(memory_db).apply()
    dest = tmp_path / "snap" / "memory.db"
    backup._backup_via_sqlite_api(memory_db, dest)
    assert _quick_check(dest) == "ok"


def test_outside_the_broken_range_secure_delete_is_left_on(tmp_path, monkeypatch):
    if sqlite3.sqlite_version_info < (3, 42, 0):
        pytest.skip("this SQLite has no FTS5 secure-delete at all")
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE VIRTUAL TABLE {TABLE} USING fts5(content)")
    conn.execute(f"INSERT INTO {TABLE}({TABLE}, rank) VALUES('secure-delete', 1)")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 46, 1))
    assert fts_residue.ensure_secure_delete(conn)[TABLE] == "on"
    assert fts_residue.secure_delete_on(conn, TABLE)


def test_below_3_42_nothing_is_touched(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE VIRTUAL TABLE {TABLE} USING fts5(content)")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 37, 2))
    monkeypatch.setattr(fts_residue, "_old_sqlite_reported", False)
    assert fts_residue.ensure_secure_delete(conn)[TABLE] == "unsupported"
