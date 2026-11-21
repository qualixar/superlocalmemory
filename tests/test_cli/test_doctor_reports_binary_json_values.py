# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm doctor` counts JSON text columns that hold binary values.

Such a value (bytes bound into a TEXT column, read back as a BLOB) used to
fail every save whose enrichment read the row, with "'utf-8' codec can't
decode ...". The readers now treat one that is not text as empty, so saves
keep working; a later save that writes the value rebuilds it from its source
and keeps the original bytes. Doctor says where the rows are so an agent or a
person can look, and tells apart a BLOB that holds UTF-8 text (it reads
normally) from one that does not (it reads as empty). Read-only: it counts,
it never repairs.
"""

from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

NOT_TEXT = bytes.fromhex("61628163")


def _store(root: Path) -> Path:
    mgr = DatabaseManager(root / "memory.db")
    mgr.initialize(real_schema)
    return root / "memory.db"


def _damage(db: Path, rows: dict[str, list[tuple[str, int]]]) -> None:
    """Insert minimal rows whose named JSON column holds ``NOT_TEXT``."""
    with sqlite3.connect(str(db)) as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1','default','x')")
        for n in range(rows.get("atomic_facts", 0)):
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, entities_json) "
                "VALUES (?, 'm1', 'default', ?, ?)", (f"f{n}", f"fact {n}", NOT_TEXT))
        for n in range(rows.get("memory_scenes", 0)):
            conn.execute(
                "INSERT INTO memory_scenes (scene_id, profile_id, entity_ids_json) "
                "VALUES (?, 'default', ?)", (f"s{n}", NOT_TEXT))


def _doctor_rows(root: Path, monkeypatch) -> dict[str, dict]:
    from tests.helpers.doctor import run_doctor_report

    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    captured: list[dict] = []
    with patch("superlocalmemory.cli.json_output.json_print",
               side_effect=lambda event, data=None, **kw: captured.append(data or {})):
        run_doctor_report(Namespace(json=True, quick=True))
    assert captured, "doctor did not run"
    return {c["name"]: c for c in captured[-1]["checks"]}


def test_a_clean_store_passes(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    status, detail, fix = binary_json_values_finding(_store(tmp_path))
    assert status == "PASS" and fix == ""


def test_binary_values_are_counted_per_column_and_warned(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import (
        binary_json_values,
        binary_json_values_finding,
    )

    db = _store(tmp_path)
    _damage(db, {"atomic_facts": 3, "memory_scenes": 1})
    assert binary_json_values(db) == {
        "atomic_facts.entities_json": 3, "memory_scenes.entity_ids_json": 1}
    status, detail, fix = binary_json_values_finding(db)
    assert status == "WARN"  # never FAIL: the rows are read safely
    assert "atomic_facts.entities_json: 3" in detail and "memory_scenes.entity_ids_json: 1" in detail
    assert "read safely" in fix
    assert "typeof(entities_json)='blob'" in fix  # the exact query that shows them
    # The hint says what is true: nothing is changed by the check, a later save
    # that writes the value rebuilds it, and the bytes are kept.
    assert "nothing is changed" not in fix
    assert "atomic_facts.entities_json: rebuilt from the fact" in fix
    assert "memory_scenes.entity_ids_json: rebuilt from the scene's facts" in fix
    assert "change ledger" in fix and "version row" in fix
    assert "read as empty" in detail


def test_every_tolerated_column_is_checked(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values
    from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

    db = _store(tmp_path)
    # doc_pages lives in media.db, which this store does not have.
    assert set(binary_json_values(db, include_zero=True)) == {
        f"{t}.{c}" for t, c in TOLERATED_COLUMNS if t != "doc_pages"}


def test_a_store_without_some_tables_is_not_a_finding(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    db = tmp_path / "memory.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, entities_json TEXT)")
    assert binary_json_values_finding(db)[0] == "PASS"


def test_the_check_never_writes(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    db = _store(tmp_path)
    _damage(db, {"atomic_facts": 2})
    before = db.read_bytes()
    binary_json_values_finding(db)
    assert db.read_bytes() == before


def test_slm_doctor_json_carries_the_row(tmp_path, monkeypatch) -> None:
    db = _store(tmp_path)
    _damage(db, {"atomic_facts": 2})
    row = _doctor_rows(tmp_path, monkeypatch)["Stored JSON values"]
    assert row["status"] == "WARN"
    assert "atomic_facts.entities_json: 2" in row["detail"]


def test_slm_doctor_passes_a_clean_store(tmp_path, monkeypatch) -> None:
    _store(tmp_path)
    assert _doctor_rows(tmp_path, monkeypatch)["Stored JSON values"]["status"] == "PASS"


def test_a_blob_holding_utf8_json_is_told_apart_from_one_that_is_not_text(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import (
        binary_json_value_kinds,
        binary_json_values_finding,
    )

    db = _store(tmp_path)
    _damage(db, {"atomic_facts": 2})
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, entities_json) "
            "VALUES ('fx', 'm1', 'default', 'fact x', ?)", ('["Tokyo"]'.encode(),))
    assert binary_json_value_kinds(db) == {"atomic_facts.entities_json": (1, 2)}
    status, detail, _fix = binary_json_values_finding(db)
    assert status == "WARN"
    assert "2 stored values are not text and read as empty" in detail
    assert "1 more holds UTF-8 JSON and reads normally" in detail


def test_only_blobs_that_hold_utf8_json_is_a_pass_that_says_so(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    db = _store(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1','default','x')")
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, entities_json) "
            "VALUES ('fx', 'm1', 'default', 'fact x', ?)", ('["Tokyo"]'.encode(),))
    status, detail, _fix = binary_json_values_finding(db)
    assert status == "PASS"
    assert "holds UTF-8 JSON and reads normally" in detail and "as empty" not in detail


def test_the_check_scans_each_table_once(tmp_path, monkeypatch) -> None:
    from superlocalmemory.cli import doctor_stored_values as dsv
    from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

    db = _store(tmp_path)
    _damage(db, {"atomic_facts": 2})
    seen: list[str] = []
    real = dsv._readonly

    class _Recording:
        def __init__(self, conn) -> None:
            self._conn = conn

        def execute(self, sql, *args):
            seen.append(sql)
            return self._conn.execute(sql, *args)

        def close(self) -> None:
            self._conn.close()

    monkeypatch.setattr(dsv, "_readonly", lambda path: _Recording(real(path)))
    dsv.binary_json_values(db, include_zero=True)

    scans = [q for q in seen if "typeof" in q]
    assert len(scans) == len({t for t, _c in TOLERATED_COLUMNS if t != "doc_pages"})  # one per table, not per column


def test_a_database_path_with_a_quote_gives_a_runnable_hint(tmp_path) -> None:
    import shlex

    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    folder = tmp_path / "it's here"
    folder.mkdir()
    db = _store(folder)
    _damage(db, {"atomic_facts": 1})
    status, _detail, fix = binary_json_values_finding(db)
    assert status == "WARN"
    command = fix[fix.index("sqlite3 -readonly"):fix.index(" (the same")]
    argv = shlex.split(command)
    assert argv[:2] == ["sqlite3", "-readonly"] and argv[2] == str(db)
    assert argv[3] == "SELECT rowid FROM atomic_facts WHERE typeof(entities_json)='blob'"


def test_the_fix_text_says_per_column_what_a_writer_does(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import _FATE, binary_json_values_finding
    from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

    assert {f"{t}.{c}" for t, c in TOLERATED_COLUMNS} <= set(_FATE)  # no column goes unsaid
    db = _store(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("INSERT INTO memories (memory_id, profile_id, content, metadata_json) "
                     "VALUES ('m1','default','x',?)", (NOT_TEXT,))
    status, _detail, fix = binary_json_values_finding(db)
    assert status == "WARN"
    assert "memories.metadata_json: left as it is: writers skip that row" in fix
    assert "rebuilt" not in fix and "ledger" not in fix  # it does not claim a rebuild it never does


def test_a_damaged_doc_page_list_is_counted_from_the_media_store(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values_finding

    db = _store(tmp_path)
    with sqlite3.connect(str(tmp_path / "media.db")) as conn:
        conn.execute("CREATE TABLE doc_pages (document_id TEXT, page_no INTEGER, fact_ids_json TEXT)")
        conn.execute("INSERT INTO doc_pages VALUES ('d1', 1, ?)", (NOT_TEXT,))
    status, detail, fix = binary_json_values_finding(db)
    assert status == "WARN" and "doc_pages.fact_ids_json: 1" in detail
    assert "media.db" in fix  # the query is run against the file the table is in


def test_the_hint_is_quoted_for_the_platform(tmp_path, monkeypatch) -> None:
    import shlex
    import subprocess

    from superlocalmemory.cli import doctor_stored_values as dsv

    folder = tmp_path / "it's here"
    folder.mkdir()
    db = _store(folder)
    _damage(db, {"atomic_facts": 1})
    sql = "SELECT rowid FROM atomic_facts WHERE typeof(entities_json)='blob'"

    monkeypatch.setattr(dsv, "_is_windows", lambda: False)
    posix = dsv._show_query(db, "atomic_facts.entities_json")
    assert shlex.split(posix) == ["sqlite3", "-readonly", str(db), sql]

    monkeypatch.setattr(dsv, "_is_windows", lambda: True)
    windows = dsv._show_query(db, "atomic_facts.entities_json")
    assert windows == f'sqlite3 -readonly {subprocess.list2cmdline([str(db)])} "{sql}"'
    assert f'"{db}"' in windows  # a path with a space is in double quotes, not shlex single quotes
