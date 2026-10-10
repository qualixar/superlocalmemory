# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm doctor`` re-files only the memories it flags as wrongly hidden."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.core.memory_health import (
    MemoryHealth,
    classify,
    describe,
    measure,
    refile_hidden,
)

_P = "default"


def _store(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    mgr = DatabaseManager(path)
    mgr.initialize(schema)
    mgr.close()
    conn = sqlite3.connect(str(path))
    try:
        rows = [("flagged", 0.95, "archive")]
        rows += [(f"old{i}", 0.2, "archive") for i in range(50)]
        rows += [(f"live{i}", 0.9, "active") for i in range(50)]
        for fid, score, zone in rows:
            mirror = "archived" if zone == "archive" else "active"
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
                " lifecycle, created_at) VALUES (?, 'm', ?, ?, ?, '2026-03-01')",
                (fid, _P, f"memory text for {fid} " * 5, mirror),
            )
            conn.execute(
                "INSERT INTO fact_retention (fact_id, profile_id, retention_score,"
                " lifecycle_zone) VALUES (?, ?, ?, ?)", (fid, _P, score, zone),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _dump(path: Path) -> dict:
    conn = sqlite3.connect(str(path))
    try:
        ret = conn.execute(
            "SELECT fact_id, profile_id, retention_score, lifecycle_zone "
            "FROM fact_retention ORDER BY fact_id").fetchall()
        mir = conn.execute(
            "SELECT fact_id, content, lifecycle FROM atomic_facts "
            "ORDER BY fact_id").fetchall()
        return {"ret": ret, "mir": mir}
    finally:
        conn.close()


class TestClassify:
    def test_one_inconsistent_memory_is_a_warning_with_a_targeted_fix(self) -> None:
        status, fix = classify(MemoryHealth(
            live_facts=100, findable_by_meaning=100, inconsistently_hidden=1))
        assert status == "WARN"
        assert fix == "slm doctor --refile-hidden"
        assert "decay" not in fix

    def test_low_reachability_still_fails(self) -> None:
        assert classify(MemoryHealth(live_facts=100, findable_by_meaning=50,
                                     missing_vector=50)) == ("FAIL", "slm restart")

    def test_missing_vectors_only_warns(self) -> None:
        assert classify(MemoryHealth(live_facts=100, findable_by_meaning=95,
                                     missing_vector=5)) == (
            "WARN", "slm db reembed --missing-only")

    def test_healthy_passes(self) -> None:
        assert classify(MemoryHealth(live_facts=3, findable_by_meaning=3))[0] == "PASS"

    def test_description_names_the_targeted_fix_not_decay(self) -> None:
        text = " ".join(describe(MemoryHealth(
            live_facts=10, findable_by_meaning=10, inconsistently_hidden=2)))
        assert "slm doctor --refile-hidden" in text
        assert "decay" not in text
        assert "nothing else moves" in text


class TestRefile:
    def test_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        db = _store(tmp_path)
        before = _dump(db)
        report = refile_hidden(db, dry_run=True)
        assert report.fact_ids == ("flagged",)
        assert report.moved == 0
        assert _dump(db) == before

    def test_apply_moves_only_the_flagged_memory(self, tmp_path: Path) -> None:
        db = _store(tmp_path)
        before = _dump(db)
        assert measure(db).inconsistently_hidden == 1
        report = refile_hidden(db, dry_run=False)
        assert report.moved == 1
        after = _dump(db)
        assert len(after["ret"]) == len(before["ret"]) == 101
        changed = [(b, a) for b, a in zip(before["ret"], after["ret"]) if b != a]
        assert len(changed) == 1
        assert changed[0][1][0] == "flagged" and changed[0][1][3] == "active"
        assert changed[0][1][2] == 0.95
        mir = {r[0]: r[2] for r in after["mir"]}
        assert mir["flagged"] == "active"
        changed_mir = [(b, a) for b, a in zip(before["mir"], after["mir"]) if b != a]
        assert len(changed_mir) == 1
        assert sum(1 for r in after["ret"] if r[3] == "archive") == 50
        assert measure(db).inconsistently_hidden == 0


class TestDoctorCommand:
    def _run(self, db: Path, **flags) -> list[dict]:
        from superlocalmemory.cli.commands import cmd_doctor

        captured: list[dict] = []
        cfg = type("C", (), {"db_path": db})
        with patch(
            "superlocalmemory.cli.json_output.json_print",
            side_effect=lambda event, data=None, **kw: captured.append(data or {}),
        ), patch("superlocalmemory.core.config.SLMConfig.load", return_value=cfg):
            cmd_doctor(argparse.Namespace(json=True, quick=True, **flags))
        return captured

    def test_refile_flag_is_a_dry_run_without_yes(self, tmp_path: Path) -> None:
        db = _store(tmp_path)
        before = _dump(db)
        out = self._run(db, refile_hidden=True, yes=False)
        assert out[-1]["refile_hidden"]["fact_ids"] == ["flagged"]
        assert out[-1]["refile_hidden"]["applied"] is False
        assert _dump(db) == before

    def test_refile_flag_with_yes_applies(self, tmp_path: Path) -> None:
        db = _store(tmp_path)
        out = self._run(db, refile_hidden=True, yes=True)
        assert out[-1]["refile_hidden"]["moved"] == 1
        assert measure(db).inconsistently_hidden == 0
