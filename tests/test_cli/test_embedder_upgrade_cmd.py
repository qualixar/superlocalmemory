# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm embedder upgrade [--yes] [--json]``: show the plan, ask, then start."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import embedder_cmd

PLAN = {"available": True, "reason": "", "already": False, "needs_media": False, "turn_on_command": "",
        "from": {"provider": "sentence-transformers", "model": "nomic-ai/nomic-embed-text-v1.5", "dimension": 768},
        "to": {"provider": "slm-media", "model": "google/embeddinggemma-2", "dimension": 768},
        "memories": 1200, "ram_mb": 3000, "disk_new_mb": 4, "disk_kept_mb": 4, "disk_mb": 8,
        "minutes": 10, "minutes_label": "about 10 minutes", "rollback": True, "label": "upgrade",
        "explain": "Your memories are re-read with the new engine in the background. Recall keeps working "
                   "on the current engine until it finishes, nothing is deleted, and you can roll back "
                   "to the previous engine afterwards."}
JOB = {"job_id": 4, "kind": "switch", "state": "queued", "from": "nomic::768",
       "to": "google/embeddinggemma-2::768", "done": 0, "total": 1200}


@pytest.fixture()
def daemon(monkeypatch):
    seen, state = [], {"plan": dict(PLAN)}

    def fake(method, path, body=None, **kw):
        seen.append((method, path, body, kw))
        if method == "GET" and path.endswith("/upgrade"):
            return state["plan"]
        if method == "POST":
            return {"success": True, "accepted": True, "label": "upgrade", "job": JOB,
                    "detail": "Upgrading your memory engine."}
        return {"job": JOB}

    monkeypatch.setattr(embedder_cmd, "daemon_request", fake)
    monkeypatch.setattr(embedder_cmd.time, "sleep", lambda s: None)
    return seen, state


def _run(**kw):
    base = {"embedder_command": "upgrade", "json": False, "yes": False}
    base.update(kw)
    embedder_cmd.cmd_embedder(Namespace(**base))


def _tty(monkeypatch, value, answer=""):
    monkeypatch.setattr(embedder_cmd, "_is_tty", lambda: value)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()) if answer is None else answer)


def _posts(seen):
    return [s for s in seen if s[0] == "POST"]


def test_the_plan_names_what_changes_ram_disk_time_and_the_way_back(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, False)
    _run()
    out = capsys.readouterr().out
    assert "nomic-ai/nomic-embed-text-v1.5" in out and "google/embeddinggemma-2" in out
    assert "1200" in out and "2.9 GB" in out and "8 MB" in out
    assert "about 10 minutes" in out
    assert "recall keeps working" in out.lower() and "roll back" in out.lower()
    assert "slm embedder rollback" in out
    assert "re-index" not in out.lower()
    assert _posts(seen) == []


def test_off_a_terminal_without_yes_it_never_starts_and_exits_zero(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, False)
    _run()
    assert _posts(seen) == []
    assert "--yes" in capsys.readouterr().out


def test_on_a_terminal_yes_to_the_question_starts_it(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, True, "y")
    _run(no_wait=True)
    (method, path, body, kw), = _posts(seen)
    assert path == "/api/v3/embedding/reindex/upgrade" and body == {}
    out = capsys.readouterr().out
    assert "Upgrading your memory engine" in out


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", None])
def test_on_a_terminal_anything_but_yes_changes_nothing(daemon, monkeypatch, capsys, answer):
    seen, _ = daemon
    _tty(monkeypatch, True, answer)
    _run()
    assert _posts(seen) == []
    assert "Nothing changed" in capsys.readouterr().out


def test_yes_starts_without_asking(daemon, monkeypatch):
    seen, _ = daemon
    _tty(monkeypatch, False)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked despite --yes"))
    _run(yes=True, no_wait=True)
    assert len(_posts(seen)) == 1


def test_unavailable_refuses_with_the_plans_reason_and_the_command(daemon, monkeypatch, capsys):
    seen, state = daemon
    state["plan"] = {**PLAN, "available": False, "needs_media": True, "turn_on_command": "slm media enable",
                     "reason": "Turn on images and documents first (about 1.5 GB): slm media enable"}
    _tty(monkeypatch, True, "y")
    with pytest.raises(SystemExit) as exc:
        _run(yes=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Turn on images and documents first" in err and "slm media enable" in err
    assert _posts(seen) == []


def test_already_upgraded_is_a_friendly_exit_zero(daemon, monkeypatch, capsys):
    seen, state = daemon
    state["plan"] = {**PLAN, "available": False, "already": True,
                     "reason": "Your memories already use the new engine."}
    _run(yes=True)
    assert "already use the new engine" in capsys.readouterr().out
    assert _posts(seen) == []


def test_daemon_down_says_so(monkeypatch, capsys):
    monkeypatch.setattr(embedder_cmd, "daemon_request", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        _run(yes=True)
    assert "slm serve" in capsys.readouterr().err


def test_json_prints_the_plan_and_never_starts_without_yes(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, True, "y")
    _run(json=True)
    data = json.loads(capsys.readouterr().out)
    assert data["data"]["plan"]["available"] is True and data["data"]["started"] is False
    assert _posts(seen) == []


def test_json_with_yes_starts_and_reports_the_job(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _run(json=True, yes=True)
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["started"] is True and data["job"]["job_id"] == 4
    assert len(_posts(seen)) == 1


def test_the_parser_accepts_yes_json_and_no_wait():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    embedder_cmd.register_embedder_parser(sub)
    ns = parser.parse_args(["embedder", "upgrade", "--yes", "--json", "--no-wait"])
    assert ns.embedder_command == "upgrade" and ns.yes and ns.json and ns.no_wait
    assert parser.parse_args(["embedder", "upgrade"]).yes is False
