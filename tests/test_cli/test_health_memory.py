# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-5: ``slm health --memory`` reads the store read-only and prints the one report."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.cli.commands import cmd_health
from superlocalmemory.storage.migrations import M075_injection_receipts as M075


def _store(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    M075.apply(conn)
    at = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="microseconds")
    conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path,"
                 " design, holdout_state, epsilon, eligible_count, candidates, delivered_at)"
                 " VALUES ('d1', 'work', 's', 'hook_recall', 'one-out-v2', 'off', 0, 0, 0, ?)",
                 (at,))
    conn.commit()
    conn.close()
    return path


def _args(**kw) -> Namespace:
    return Namespace(**{"memory": True, "json": False, "db": None, "profile": None, **kw})


def test_the_report_is_printed_in_plain_words_without_changing_the_store(tmp_path, capsys):
    path = _store(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cmd_health(_args(db=str(path), profile="work"))
    out = capsys.readouterr().out
    assert out.startswith("Memory Health (profile work")
    assert "Deliveries in the last 30 days: 1 (hook_recall 1)" in out
    assert "Judge certificates: 0 active" in out
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_json_carries_the_report_for_agents(tmp_path, capsys):
    path = _store(tmp_path)
    cmd_health(_args(db=str(path), profile="work", json=True))
    body = json.loads(capsys.readouterr().out)
    report = body["data"]["memory"]
    assert report["profile_id"] == "work"
    assert report["ledger"]["deliveries"] == {"hook_recall": 1}
    assert {"effects", "use", "certificates", "second_opinions"} <= set(report)


def test_a_missing_store_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cmd_health(_args(db=str(tmp_path / "nope.db")))
    assert exc.value.code == 2


def test_the_command_line_reaches_the_report(tmp_path):
    """The real parser and dispatch: ``slm health --memory --db ... --json``."""
    import os
    import subprocess
    import sys

    path = _store(tmp_path)
    env = {**os.environ, "SLM_DATA_DIR": str(tmp_path)}
    run = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "health", "--memory",
                          "--db", str(path), "--profile", "work", "--json"],
                         capture_output=True, text=True, timeout=120, env=env)
    assert run.returncode == 0, run.stderr
    body = json.loads(run.stdout)
    assert body["data"]["memory"]["ledger"]["deliveries"] == {"hook_recall": 1}
