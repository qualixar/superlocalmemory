# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm doctor` reports the health of mesh.db (4.2.8: the mesh's own file)."""

from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.cli import doctor_mesh
from superlocalmemory.mesh import mesh_store


def _rows(root: Path, monkeypatch) -> dict[str, dict]:
    from tests.helpers.doctor import run_doctor_report

    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    captured: list[dict] = []
    with patch("superlocalmemory.cli.json_output.json_print",
               side_effect=lambda event, data=None, **kw: captured.append(data or {})):
        run_doctor_report(Namespace(json=True, quick=True))
    return {c["name"]: c for c in captured[-1]["checks"]}


def test_healthy_mesh_db_passes(tmp_path: Path) -> None:
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    status, detail, fix = doctor_mesh.mesh_finding(tmp_path)
    assert status == "PASS" and "mesh.db" in detail and fix == ""


def test_missing_mesh_db_is_not_a_failure(tmp_path: Path) -> None:
    status, detail, _ = doctor_mesh.mesh_finding(tmp_path)
    assert status == "PASS" and "not yet created" in detail
    assert not (tmp_path / "mesh.db").exists()  # doctor creates nothing


def test_corrupt_mesh_db_fails_with_a_fix(tmp_path: Path) -> None:
    (tmp_path / "mesh.db").write_bytes(b"garbage" * 500)
    status, _detail, fix = doctor_mesh.mesh_finding(tmp_path)
    assert status == "FAIL" and "restart" in fix


def test_old_tables_left_in_memory_db_warn(tmp_path: Path) -> None:
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute("CREATE TABLE mesh_peers (peer_id TEXT)")
    conn.close()
    status, detail, _ = doctor_mesh.mesh_finding(tmp_path)
    assert status == "WARN" and "memory.db" in detail


def test_doctor_has_a_mesh_row(tmp_path: Path, monkeypatch) -> None:
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    rows = _rows(tmp_path, monkeypatch)
    assert rows["Mesh store"]["status"] == "PASS"


def test_busy_mesh_db_warns_and_never_suggests_a_rebuild(tmp_path: Path, monkeypatch) -> None:
    """A locked mesh.db is not damaged: no FAIL, no 'restart to rebuild' advice."""
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    monkeypatch.setattr(mesh_store, "inspect", lambda _p: {
        "status": "unavailable", "path": "", "detail": "database is locked"})
    status, detail, fix = doctor_mesh.mesh_finding(tmp_path)
    assert status == "WARN" and "database is locked" in detail
    assert "quarantine" not in fix
