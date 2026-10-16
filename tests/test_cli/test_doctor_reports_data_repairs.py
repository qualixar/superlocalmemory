# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm doctor` says what the automatic 4.2.8 data repairs did (M065, M066)."""

from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.cli import doctor_repairs
from superlocalmemory.storage import data_repair_undo as undo_log
from superlocalmemory.storage.migrations import M065_fabricated_dates_are_removed as M065
from tests.test_migrations._repair_store import (
    add_fact,
    first_start,
    forget_migration,
    migrated_store,
)


def _repaired_store(tmp_path: Path) -> Path:
    memory, learning = migrated_store(tmp_path)
    conn = sqlite3.connect(memory)
    add_fact(conn, "f1", "The file a.py was modified on 2024-01-15",
             referenced_date="2024-01-15")
    add_fact(conn, "f2", "2024-01-15 was a Tuesday", referenced_date="2024-01-15")
    conn.commit()
    conn.close()
    forget_migration(memory, M065.NAME)
    first_start(memory, learning)
    return memory


def test_nothing_to_report_on_a_store_the_repairs_never_touched(tmp_path: Path) -> None:
    memory, _ = migrated_store(tmp_path)
    status, detail, fix = doctor_repairs.data_repairs_finding(memory)
    assert status == "PASS" and "nothing" in detail.lower() and fix == ""


def test_reports_what_m065_changed(tmp_path: Path) -> None:
    memory = _repaired_store(tmp_path)
    status, detail, fix = doctor_repairs.data_repairs_finding(memory)
    assert status == "PASS"
    assert "made-up dates" in detail and "2 memories" in detail and "1 sentence" in detail
    assert "archived" in detail  # M066 line is always there once it has run
    assert fix == ""


def test_reports_an_undo(tmp_path: Path) -> None:
    memory = _repaired_store(tmp_path)
    conn = sqlite3.connect(memory, isolation_level=None)
    undo_log.undo(conn, M065.NAME)
    conn.close()
    _status, detail, _fix = doctor_repairs.data_repairs_finding(memory)
    assert "undone" in detail


def test_a_missing_store_is_not_a_finding(tmp_path: Path) -> None:
    status, _detail, _fix = doctor_repairs.data_repairs_finding(tmp_path / "memory.db")
    assert status == "PASS"


def test_doctor_has_a_data_repairs_row(tmp_path: Path, monkeypatch) -> None:
    from tests.helpers.doctor import run_doctor_report

    _repaired_store(tmp_path)
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    captured: list[dict] = []
    with patch("superlocalmemory.cli.json_output.json_print",
               side_effect=lambda event, data=None, **kw: captured.append(data or {})):
        run_doctor_report(Namespace(json=True, quick=True))
    rows = {c["name"]: c for c in captured[-1]["checks"]}
    assert rows["Data repairs"]["status"] == "PASS"
    assert "made-up dates" in rows["Data repairs"]["detail"]
