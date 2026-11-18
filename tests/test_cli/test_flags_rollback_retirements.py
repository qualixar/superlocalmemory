# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm flags kill`` and ``slm flags rollback-retirements [--dry-run]`` (D-23)."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.cli import flags_cmd
from superlocalmemory.core.feature_flags import KILL_ENV, FeatureFlags
from tests.test_truth.test_slot_pass import _reconcile, _retired, _settled_by_a_person, store  # noqa: F401


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.setattr(flags_cmd, "_home", lambda: tmp_path / "home")
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: False)
    return tmp_path / "home"


def _args(db, **kw):
    return Namespace(action="rollback-retirements", dry_run=False, db=str(db.db_path),
                     profile="default", **kw)


def _retired_store(store):  # noqa: F811
    db, old, new, _tmp = store
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    assert _retired(db, old)[2] == new
    return db, old


def test_kill_prints_what_it_does_and_what_it_does_not(home, capsys) -> None:
    assert flags_cmd.cmd_flags(Namespace(action="kill", value="on")) == 0
    out = capsys.readouterr().out
    assert "kill switch on" in out and "does not bring back" in out
    assert FeatureFlags(home, environ={}).killed


def test_a_dry_run_says_how_many_would_come_back_and_changes_nothing(home, store, capsys) -> None:  # noqa: F811
    db, old = _retired_store(store)
    args = _args(db)
    args.dry_run = True
    assert flags_cmd.cmd_flags(args) == 0
    out = capsys.readouterr().out
    assert "1 memory would come back" in out and "Nothing was changed" in out
    assert _retired(db, old)[2] is not None


def test_the_command_says_the_count_first_then_brings_them_back_and_names_the_undo(
        home, store, capsys) -> None:  # noqa: F811
    db, old = _retired_store(store)
    assert flags_cmd.cmd_flags(_args(db)) == 0
    out = capsys.readouterr().out
    assert out.index("1 memory would come back") < out.index("Brought back 1 memory")
    assert "slm decisions" in out or "Undo" in out
    assert _retired(db, old) == (None, None, None)
    rows = db.execute("SELECT kind, channel FROM decision_log")
    assert [(r["kind"], r["channel"]) for r in rows] == [("rollback_retirements", "cli")]


def test_with_nothing_retired_it_says_so(home, store, capsys) -> None:  # noqa: F811
    assert flags_cmd.cmd_flags(_args(store[0])) == 0
    assert "No memory was retired by 4.2.30" in capsys.readouterr().out


def test_a_missing_store_is_a_plain_refusal(home, tmp_path, capsys) -> None:
    args = Namespace(action="rollback-retirements", dry_run=True, db=str(tmp_path / "none.db"),
                     profile="default")
    assert flags_cmd.cmd_flags(args) == 2
    assert "No store at" in capsys.readouterr().err


def test_with_the_daemon_running_the_request_goes_through_it(home, monkeypatch, capsys) -> None:
    sent = []

    def fake(method, path, body=None, **kw):
        sent.append((method, path, body))
        return {"success": True, "would_come_back": 3, "came_back": 0 if body["dry_run"] else 3,
                "decision_id": "" if body["dry_run"] else "d1", "refused": 0}

    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", fake)
    args = Namespace(action="rollback-retirements", dry_run=False, db=None, profile="")
    assert flags_cmd.cmd_flags(args) == 0
    assert [(m, p, b["dry_run"]) for m, p, b in sent] == [
        ("POST", "/api/flags/rollback-retirements", True),
        ("POST", "/api/flags/rollback-retirements", False)]
    assert "3 memories would come back" in capsys.readouterr().out
