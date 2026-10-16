# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm doctor` reports hook events waiting in the spool and any it dropped.

A hook event is never dropped silently: when the spool is full the drop is
counted, and doctor is where the user sees the count.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from superlocalmemory.cli import doctor_hook_spool
from superlocalmemory.hooks import event_spool


def _ev(i: int) -> dict:
    return {"kind": "tool_event", "event_id": f"{i:032x}", "profile_id": "default",
            "session_id": "s", "tool_name": "Read"}


def test_no_spool_is_a_pass_and_creates_nothing(tmp_path: Path) -> None:
    status, detail, fix = doctor_hook_spool.hook_spool_finding(tmp_path)
    assert status == "PASS" and fix == ""
    assert not (tmp_path / "hook_spool").exists()


def test_waiting_events_are_reported(tmp_path: Path) -> None:
    event_spool.append_events([_ev(1), _ev(2)], profile="default",
                              root=tmp_path / "hook_spool")
    status, detail, _ = doctor_hook_spool.hook_spool_finding(tmp_path)
    assert status == "PASS" and "2 hook event" in detail


def test_dropped_events_are_a_warning_with_the_count(tmp_path: Path) -> None:
    root = tmp_path / "hook_spool"
    event_spool.append_events([_ev(i) for i in range(10)], profile="default", root=root,
                              max_bytes=300)
    dropped = event_spool.read_stats(root).dropped
    status, detail, fix = doctor_hook_spool.hook_spool_finding(tmp_path)
    assert dropped > 0
    assert status == "WARN" and f"{dropped} hook event" in detail and "dropped" in detail
    assert "slm" in fix


def test_unreadable_lines_are_a_warning(tmp_path: Path) -> None:
    event_spool.record_counts(tmp_path / "hook_spool", "default", invalid=2)
    status, detail, _ = doctor_hook_spool.hook_spool_finding(tmp_path)
    assert status == "WARN" and "2" in detail and "unreadable" in detail


def test_doctor_has_a_hook_events_row(tmp_path: Path, monkeypatch) -> None:
    from tests.helpers.doctor import run_doctor_report

    event_spool.record_counts(tmp_path / "hook_spool", "default", dropped=3)
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    captured: list[dict] = []
    with patch("superlocalmemory.cli.json_output.json_print",
               side_effect=lambda event, data=None, **kw: captured.append(data or {})):
        run_doctor_report(Namespace(json=True, quick=True))
    rows = {c["name"]: c for c in captured[-1]["checks"]}
    assert rows["Hook events"]["status"] == "WARN"
    assert "3 hook event" in rows["Hook events"]["detail"]
