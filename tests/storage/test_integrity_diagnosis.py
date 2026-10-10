# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""GitHub #204: ``slm doctor`` printed ``quick_check: <sqlite3.Row object at ...>``
and advised "Backup and recreate database" for a keyword index that
``slm db repair --apply`` rebuilds from the stored memories. ``slm restart``
step 5 must say the same thing.

The damage is made by cutting a keyword-index block, which every SQLite reports
as FTS5 corruption; only 3.44.0 to 3.46.0 with secure-delete on writes it on
its own.
"""
import argparse
import sqlite3
from contextlib import closing
from unittest.mock import patch

from tests.test_storage import _upgrade_store as store

from superlocalmemory.storage import integrity_diagnosis as diag
from superlocalmemory.storage.memory_write import memory_read

TABLE = "atomic_facts_fts"


def damaged_store(root, table=TABLE):
    _, memory_db = store.current_store(root)
    store.add_memory(memory_db, "m1", ["alpha beta gamma", "delta epsilon zeta"])
    with closing(sqlite3.connect(str(memory_db))) as conn:
        conn.execute(f"UPDATE {table}_data SET block = substr(block, 1, length(block) - 3) "  # noqa: S608
                     "WHERE id > 10")
        conn.commit()
        assert conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
    return memory_db


def test_the_message_is_the_sqlite_text_never_a_row_repr(tmp_path):
    memory_db = damaged_store(tmp_path / "s")
    with memory_read(memory_db) as conn:  # rows come back as sqlite3.Row
        result = diag.check_database(conn)
    assert result.ok is False
    text = " ".join(result.messages) + result.summary()
    assert "sqlite3.Row" not in text and "object at 0x" not in text
    assert TABLE in text
    assert result.damaged_indexes == (TABLE,)


def test_a_damaged_keyword_index_is_pointed_at_the_targeted_rebuild(tmp_path):
    memory_db = damaged_store(tmp_path / "s")
    with memory_read(memory_db) as conn:
        result = diag.check_database(conn, root=tmp_path / "s")
    assert "slm db repair --apply" in result.fix
    assert str(tmp_path / "s") in result.fix  # --root is required to change anything
    assert "recreate" not in result.fix.lower()
    assert "nothing is lost" in result.fix.lower()


def test_damage_that_is_not_a_keyword_index_still_gets_the_restore_advice():
    class Conn:
        def execute(self, sql):
            return [("Tree 7 page 7: btreeInitPage() returns error code 11",)]

    result = diag.check_database(Conn())
    assert result.ok is False and result.damaged_indexes == ()
    assert "slm db repair --apply" not in result.fix
    assert "restore" in result.fix.lower()


def test_a_mixed_report_names_both_steps():
    class Conn:
        def execute(self, sql):
            return [(f'malformed inverted index for FTS5 table main.{TABLE}',),
                    ("Tree 7 page 7: btreeInitPage() returns error code 11",)]

    result = diag.check_database(Conn())
    assert result.damaged_indexes == (TABLE,)
    assert "slm db repair --apply" in result.fix and "restore" in result.fix.lower()


def test_both_sqlite_wordings_name_the_table():
    expansion = "malformed inverted index for FTS5 table main.fact_expansion_fts"
    assert diag.damaged_indexes_in([expansion]) == ("fact_expansion_fts",)
    assert diag.damaged_indexes_in(
        ['fts5: corruption found reading blob 5 from table "atomic_facts_fts"']) == (TABLE,)


def test_a_healthy_store_is_ok(tmp_path):
    _, memory_db = store.current_store(tmp_path / "s")
    with memory_read(memory_db) as conn:
        result = diag.check_database(conn)
    assert result.ok and result.messages == ("ok",) and result.fix == ""


def test_doctor_shows_the_message_and_the_targeted_fix(tmp_path, monkeypatch):
    from superlocalmemory.cli.commands import cmd_doctor

    root = tmp_path / "s"
    damaged_store(root)
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    captured: list[dict] = []
    with patch("superlocalmemory.cli.json_output.json_print",
               side_effect=lambda event, data=None, **kw: captured.append(data or {})):
        cmd_doctor(argparse.Namespace(json=True, quick=True))
    check = next(c for c in captured[-1]["checks"] if c["name"] == "Database")
    assert check["status"] == "FAIL"
    assert "sqlite3.Row" not in check["detail"] and TABLE in check["detail"]
    assert "slm db repair --apply" in check["fix"]
    assert "Backup and recreate" not in check["fix"]


def test_restart_step_five_reports_the_message_and_the_fix(tmp_path):
    memory_db = damaged_store(tmp_path / "s")
    status, detail = diag.restart_report(memory_db)
    assert status == "fail"
    assert "sqlite3.Row" not in detail and TABLE in detail
    assert "slm db repair --apply" in detail


def test_restart_step_five_is_healthy_for_a_sound_store(tmp_path):
    _, memory_db = store.current_store(tmp_path / "s")
    store.add_memory(memory_db, "m1", ["alpha beta"])
    status, detail = diag.restart_report(memory_db)
    assert status == "ok"
    assert detail.startswith("integrity=ok, 1 facts")


def test_db_integrity_and_the_repair_preview_name_the_next_step(tmp_path, monkeypatch, capsys):
    from superlocalmemory.cli import integrity_cmd

    root = tmp_path / "s"
    damaged_store(root)
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    assert integrity_cmd.cmd_db_integrity(argparse.Namespace(pages=True, json=False)) == 0
    out = capsys.readouterr().out
    assert f"slm db repair --apply --root {root.resolve()}" in out
    args = argparse.Namespace(apply=False, undo="", root="", batch_size=100, pause_ms=0,
                              max_seconds=None, json=False)
    assert integrity_cmd.cmd_db_repair(args) == 0
    assert f"slm db repair --apply --root {root.resolve()}" in capsys.readouterr().out


def test_a_healthy_store_gets_no_repair_hint(tmp_path, monkeypatch, capsys):
    from superlocalmemory.cli import integrity_cmd

    root = tmp_path / "s"
    store.current_store(root)
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    integrity_cmd.cmd_db_integrity(argparse.Namespace(pages=True, json=False))
    assert "slm db repair --apply" not in capsys.readouterr().out
