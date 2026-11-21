# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm doctor` counts JSON text columns that hold binary values.

Such a value (bytes bound into a TEXT column, read back as a BLOB) used to
fail every save whose enrichment read the row, with "'utf-8' codec can't
decode ...". The readers now treat one that is not text as empty, so saves
keep working; doctor says where the rows are so an agent or a person can look.
Read-only: it counts, it never repairs.
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


def test_every_tolerated_column_is_checked(tmp_path) -> None:
    from superlocalmemory.cli.doctor_stored_values import binary_json_values
    from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

    db = _store(tmp_path)
    assert set(binary_json_values(db, include_zero=True)) == {
        f"{t}.{c}" for t, c in TOLERATED_COLUMNS}


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
