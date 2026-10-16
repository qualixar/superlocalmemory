# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm delete <id> --profile <name>`` reaches a memory outside the active profile.

16 Oct 2026: ``slm doctor --refile-hidden`` listed 43089c745e6e4738, which lives
in profile ``slm-video-demo``. ``slm delete 43089c745e6e4738 --yes`` answered
NOT_FOUND because delete only ever asked about the active profile, and there
was no way to name another. The daemon route already took ``profile_id``; the
command never sent it. The active profile is not switched.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _run(args: Namespace, responses: list) -> list:
    from superlocalmemory.cli.commands import cmd_delete

    replies = iter(responses)
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request",
              side_effect=lambda *_a, **_k: next(replies)) as request,
    ):
        cmd_delete(args)
    return request.call_args_list


def test_profile_is_sent_with_the_delete(capsys) -> None:
    calls = _run(Namespace(fact_id="43089c745e6e4738", yes=True, json=True,
                           profile="slm-video-demo"),
                 [{"success": True, "deleted": "43089c745e6e4738"}])
    assert [c.args for c in calls] == [
        ("DELETE", "/api/memories/43089c745e6e4738?profile_id=slm-video-demo")]
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["deleted"] == "43089c745e6e4738"
    assert payload["data"]["profile"] == "slm-video-demo"


def test_profile_is_sent_with_the_preview_too(capsys) -> None:
    calls = _run(Namespace(fact_id="f 1", yes=False, json=True, profile="demo & co"),
                 [{"content": "The demo film opens on the latency chart"}])
    assert [c.args for c in calls] == [
        ("GET", "/api/facts/f%201?profile_id=demo%20%26%20co")]
    assert "latency chart" in json.loads(capsys.readouterr().out)["data"]["content"]


def test_without_profile_the_request_is_unchanged(capsys) -> None:
    calls = _run(Namespace(fact_id="fact-1", yes=True, json=True),
                 [{"success": True, "deleted": "fact-1"}])
    assert [c.args for c in calls] == [("DELETE", "/api/memories/fact-1")]


def _parsed(argv: list[str]) -> Namespace:
    from superlocalmemory.cli.main import main

    with patch("sys.argv", ["slm", *argv]), \
            patch("superlocalmemory.cli.commands.cmd_delete") as handler:
        main()
    return handler.call_args.args[0]


def test_the_parser_accepts_profile() -> None:
    args = _parsed(["delete", "43089c745e6e4738", "--profile", "slm-video-demo", "--yes"])
    assert (args.fact_id, args.profile, args.yes) == (
        "43089c745e6e4738", "slm-video-demo", True)
    assert _parsed(["delete", "x"]).profile == ""


@pytest.fixture()
def dashboard(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    db = engine._db
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('demo', 'demo')")
    db.execute("INSERT INTO memories(memory_id,profile_id,content) VALUES "
               "('m-demo','demo','source')")
    db.execute("INSERT INTO atomic_facts(fact_id,memory_id,profile_id,content,fact_type,"
               "created_at) VALUES ('f-demo','m-demo','demo','in the demo profile',"
               "'semantic',datetime('now'))")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    descriptor = app.state.daemon_descriptor
    headers = {"X-SLM-Daemon-Capability": descriptor.capability,
               "X-SLM-Target-Instance": descriptor.instance_id}
    yield TestClient(app), headers, engine


def test_fact_detail_reads_the_named_profile(dashboard) -> None:
    client, headers, engine = dashboard
    assert engine.profile_id != "demo"
    assert client.get("/api/facts/f-demo", headers=headers).status_code == 404
    named = client.get("/api/facts/f-demo?profile_id=demo", headers=headers)
    assert named.status_code == 200, named.text
    assert named.json()["content"] == "in the demo profile"
    unknown = client.get("/api/facts/f-demo?profile_id=nope", headers=headers)
    assert unknown.status_code == 404
    assert engine.profile_id != "demo"


def test_not_found_names_the_profile_flag(capsys) -> None:
    from superlocalmemory.cli.commands import _memory_not_found

    with pytest.raises(SystemExit):
        _memory_not_found("delete", "43089c745e6e4738", True)
    hint = json.loads(capsys.readouterr().out)["error"]["hint"]
    assert "--profile <name>" in hint
