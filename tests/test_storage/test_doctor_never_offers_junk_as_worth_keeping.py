# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm doctor --refile-hidden`` offers only real memories, named by profile.

16 Oct 2026: it listed six memories "hidden even though they are scored to
keep". Five were junk carrying the default score 1.0. The sixth lived in
another profile, so ``slm delete <id>`` answered NOT_FOUND and nothing on the
line said which profile to ask.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.core.memory_health import measure, refile_hidden

_ROWS = (
    # fact_id, profile, content, score, zone
    ("real", "default", "Varun prefers the dark theme on the dashboard", 1.0, "archive"),
    ("junk-date", "default", "2024-01-15 File was modified", 1.0, "archive"),
    ("junk-test", "default", "2024-01-15 was used as a controlled test string", 1.0, "archive"),
    ("junk-empty", "default", "The session ended at 16:07", 0.95, "forgotten"),
    ("other", "slm-video-demo", "The demo film opens on the recall latency chart", 1.0, "archive"),
    ("faded", "default", "An old memory that has genuinely faded away", 0.1, "archive"),
)


def _store(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    mgr = DatabaseManager(path)
    mgr.initialize(schema)
    mgr.close()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) "
                     "VALUES ('slm-video-demo', 'slm-video-demo')")
        for fid, profile, content, score, zone in _ROWS:
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
                " lifecycle, created_at) VALUES (?, 'm', ?, ?, 'archived', '2026-03-01')",
                (fid, profile, content),
            )
            conn.execute(
                "INSERT INTO fact_retention (fact_id, profile_id, retention_score,"
                " lifecycle_zone) VALUES (?, ?, ?, ?)", (fid, profile, score, zone),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _zones(path: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(path))
    try:
        return dict(conn.execute("SELECT fact_id, lifecycle_zone FROM fact_retention"))
    finally:
        conn.close()


def test_junk_is_not_counted_as_wrongly_hidden(tmp_path: Path) -> None:
    assert measure(_store(tmp_path)).inconsistently_hidden == 2


def test_refile_lists_real_memories_with_their_profile(tmp_path: Path) -> None:
    report = refile_hidden(_store(tmp_path), dry_run=True)
    assert report.fact_ids == ("other", "real")
    assert report.profiles == ("slm-video-demo", "default")
    assert report.previews[0].startswith("The demo film")


def test_apply_never_resurrects_junk(tmp_path: Path) -> None:
    db = _store(tmp_path)
    report = refile_hidden(db, dry_run=False)
    assert report.moved == 2
    zones = _zones(db)
    assert zones["real"] == zones["other"] == "active"
    assert zones["junk-date"] == zones["junk-test"] == "archive"
    assert zones["junk-empty"] == "forgotten"
    assert measure(db).inconsistently_hidden == 0


def _doctor(db: Path, capsys, **flags) -> str:
    from tests.helpers.doctor import run_doctor_report

    cfg = type("C", (), {"db_path": db})
    with patch("superlocalmemory.core.config.SLMConfig.load", return_value=cfg):
        run_doctor_report(argparse.Namespace(json=False, quick=True, **flags))
    return capsys.readouterr().out


def test_doctor_prints_profile_and_the_delete_command(tmp_path: Path, capsys) -> None:
    out = _doctor(_store(tmp_path), capsys, refile_hidden=True, yes=False)
    line = next(ln for ln in out.splitlines() if "other" in ln and "demo film" in ln)
    assert "slm-video-demo" in line
    assert "slm delete <id> --profile <profile>" in out
    assert "File was modified" not in out


def test_doctor_json_carries_profiles(tmp_path: Path) -> None:
    from tests.helpers.doctor import run_doctor_report

    captured: list[dict] = []
    cfg = type("C", (), {"db_path": _store(tmp_path)})
    with patch(
        "superlocalmemory.cli.json_output.json_print",
        side_effect=lambda event, data=None, **kw: captured.append(data or {}),
    ), patch("superlocalmemory.core.config.SLMConfig.load", return_value=cfg):
        run_doctor_report(argparse.Namespace(json=True, quick=True,
                                             refile_hidden=True, yes=False))
    refile = captured[-1]["refile_hidden"]
    assert refile["fact_ids"] == ["other", "real"]
    assert refile["profiles"] == ["slm-video-demo", "default"]
