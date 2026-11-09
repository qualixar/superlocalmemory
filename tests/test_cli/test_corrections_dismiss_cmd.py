# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm corrections dismiss`` (issue #309): dry run by default, ``--apply`` to act.

The command only talks to the daemon (SLM stays the only writer of its store);
the daemon route and the move are covered in tests/test_server and tests/core.
"""

from __future__ import annotations

import argparse
import json

import pytest

from superlocalmemory.cli import corrections_cmd as cc

DRY = {"success": True, "profile_id": "default", "dry_run": True, "eligible": 12,
       "by_reason": {"temporal_contradiction": 9, "consolidation_update": 3},
       "oldest": "2026-10-01T08:00:00+00:00", "newest": "2026-11-08T21:30:00+00:00"}
DONE = {**DRY, "dry_run": False, "dismissed": 11, "skipped_raced": 1, "skipped_changed": 0,
        "backup": "/data/memory.before-save-repair-1-x.db"}


def _parse(*argv: str) -> argparse.Namespace:
    root = argparse.ArgumentParser()
    cc.register_corrections_parsers(root.add_subparsers(dest="command"))
    return root.parse_args(["corrections", *argv])


@pytest.fixture
def daemon(monkeypatch):
    calls: list[tuple] = []
    answer = {"value": DRY}

    def fake(method, path, body=None, **kw):
        calls.append((method, path, body, kw))
        return answer["value"]

    monkeypatch.setattr(cc, "_daemon", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", fake)
    monkeypatch.setattr("superlocalmemory.core.admission.gate_cli_mutation",
                        lambda *a, **k: None)
    return calls, answer


def test_dry_run_is_the_default_and_names_counts_and_dates(daemon, capsys) -> None:
    calls, _ = daemon
    assert cc.cmd_corrections(_parse("dismiss")) == 0

    [(method, path, body, _)] = calls
    assert (method, path) == ("POST", "/api/overtaken-corrections/dismiss")
    assert body["apply"] is False
    out = capsys.readouterr().out
    assert "12" in out and "temporal_contradiction" in out and "consolidation_update" in out
    assert "2026-10-01" in out and "2026-11-08" in out
    assert "Nothing changed" in out and "--apply" in out


def test_filters_are_sent(daemon) -> None:
    calls, _ = daemon
    args = _parse("dismiss", "--reason", "temporal_contradiction", "--reason",
                  "consolidation_update", "--older-than", "30", "--before", "2026-11-01",
                  "--profile", "work")
    assert cc.cmd_corrections(args) == 0
    [(_, _, body, _)] = calls
    assert body == {"apply": False, "reasons": ["temporal_contradiction", "consolidation_update"],
                    "older_than_days": 30, "before": "2026-11-01", "profile_id": "work"}


def test_apply_prints_the_backup_and_how_to_undo(daemon, capsys) -> None:
    calls, answer = daemon
    answer["value"] = DONE
    assert cc.cmd_corrections(_parse("dismiss", "--apply")) == 0

    [(_, _, body, kw)] = calls
    assert body["apply"] is True and kw["timeout_seconds"] >= 600
    out = capsys.readouterr().out
    assert "11" in out and "/data/memory.before-save-repair-1-x.db" in out
    assert "restore-overtaken" in out and "1" in out


def test_json_output(daemon, capsys) -> None:
    assert cc.cmd_corrections(_parse("dismiss", "--json")) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["eligible"] == 12


@pytest.mark.parametrize("argv,needle", [
    (("--reason", "replaced_by_caller"), "unknown reason"),
    (("--older-than", "-2"), "--older-than"),
    (("--before", "next-week"), "--before"),
])
def test_bad_filters_are_refused_before_the_daemon_is_called(daemon, capsys, argv, needle) -> None:
    calls, _ = daemon
    assert cc.cmd_corrections(_parse("dismiss", *argv)) == 1
    assert needle in capsys.readouterr().err
    assert calls == []


def test_the_daemons_refusal_is_shown(daemon, monkeypatch, capsys) -> None:
    from superlocalmemory.cli.daemon import DaemonUnprocessable

    def refuse(*a, **k):
        raise DaemonUnprocessable("INVALID", "no such thing")

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", refuse)
    assert cc.cmd_corrections(_parse("dismiss", "--apply")) == 1
    assert "no such thing" in capsys.readouterr().err


def test_the_other_subcommands_are_unchanged(daemon, capsys) -> None:
    calls, answer = daemon
    answer["value"] = {"success": True, "overtaken": []}
    assert cc.cmd_corrections(_parse("overtaken")) == 0
    assert calls[0][:2] == ("GET", "/api/overtaken-corrections?limit=50")
    assert "No correction case was overtaken" in capsys.readouterr().out
