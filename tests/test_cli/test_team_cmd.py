# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm team policy``: the owner's way back into a workspace that requires login.

In company mode the dashboard install token cannot administer, so a locked-out
administrator needs a door that proves access to this user's files: the CLI
sends the daemon capability, which is never served over HTTP.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.cli import team_cmd
from superlocalmemory.cli.daemon import DaemonRefused


@pytest.fixture
def world(engine_with_mock_deps):
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    app.state.rbac = RbacEngine(str(engine._config.db_path))
    bind_profile_runtime(app.state, engine, engine._config)
    rbac = app.state.rbac
    rbac.create_user("admin1", "password-1234")
    rbac.set_require_login(True)
    return app, TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 5000))


def _bind(monkeypatch, app, tc, *, capability: bool) -> None:
    d = app.state.daemon_descriptor
    cap = {"X-SLM-Daemon-Capability": d.capability, "X-SLM-Target-Instance": d.instance_id}

    def fake(method, path, body=None, **_kw):
        response = tc.request(method, path, json=body, headers=cap if capability else {})
        if response.status_code in (401, 403):
            raise DaemonRefused(response.status_code, path)
        return response.json() if response.status_code < 400 else None

    monkeypatch.setattr(team_cmd, "daemon_request", fake)


def _args(**kw) -> Namespace:
    return Namespace(**{"json": False, "team_command": "policy", "require_login": None, **kw})


def test_the_owner_can_switch_login_off_from_the_command_line(world, monkeypatch, capsys):
    app, tc = world
    _bind(monkeypatch, app, tc, capability=True)
    team_cmd.cmd_team(_args(require_login="off"))
    assert app.state.rbac.require_login() is False
    assert "off" in capsys.readouterr().out.lower()


def test_without_the_capability_the_command_is_refused_and_changes_nothing(world, monkeypatch, capsys):
    app, tc = world
    _bind(monkeypatch, app, tc, capability=False)
    with pytest.raises(SystemExit) as stop:
        team_cmd.cmd_team(_args(require_login="off"))
    assert stop.value.code == 1
    assert app.state.rbac.require_login() is True


def test_status_reports_the_policy_as_json(world, monkeypatch, capsys):
    app, tc = world
    _bind(monkeypatch, app, tc, capability=True)
    team_cmd.cmd_team(_args(team_command="status", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["require_login"] is True and payload["data"]["user_count"] == 1


def test_a_missing_daemon_is_said_plainly(monkeypatch, capsys):
    monkeypatch.setattr(team_cmd, "daemon_request", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        team_cmd.cmd_team(_args(require_login="on"))
    assert "not running" in capsys.readouterr().out


def test_the_parser_takes_on_and_off_only():
    import argparse

    root = argparse.ArgumentParser()
    team_cmd.register_team_parser(root.add_subparsers(dest="command"))
    assert root.parse_args(["team", "policy", "--require-login", "off"]).require_login == "off"
    with pytest.raises(SystemExit):
        root.parse_args(["team", "policy", "--require-login", "maybe"])
