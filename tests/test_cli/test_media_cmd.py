# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm media`` and ``slm features`` talk to the daemon; the install never runs in the CLI."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import features_cmd, media_cmd
from superlocalmemory.runtimes import features as feat

STATUS = {"media": {"enabled": False, "requested": False, "env_state": "not_installed", "progress": 0.0,
                    "step": "", "restart_required": False,
                    "precheck": {"disk_ok": True, "free_bytes": 50 * 1024 ** 3}},
          "mesh": {"apps_with_mesh": 2}}


@pytest.fixture(autouse=True)
def never_in_process(monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("the CLI must not run the install itself")

    monkeypatch.setattr(feat, "enable_media", fail)
    monkeypatch.setattr(feat, "disable_media", fail)


@pytest.fixture()
def daemon(monkeypatch):
    seen = []
    state = {"reply": STATUS}

    def fake(method, path, body=None, **_kw):
        seen.append((method, path, body))
        return state["reply"]

    for mod in (media_cmd, features_cmd):
        monkeypatch.setattr(mod, "daemon_request", fake)
    return seen, state


def _args(**kw):
    base = {"json": False, "media_command": "status", "yes": False, "remove_files": False}
    base.update(kw)
    return Namespace(**base)


def _tty(monkeypatch, value):
    monkeypatch.setattr(media_cmd, "_is_tty", lambda: value)


def test_status_json(daemon, capsys):
    seen, _ = daemon
    media_cmd.cmd_media(_args(json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is True and out["data"]["media"]["enabled"] is False
    assert seen == [("GET", "/api/v3/features", None)]


def test_enable_with_yes_posts_to_the_daemon(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, False)
    media_cmd.cmd_media(_args(media_command="enable", yes=True))
    assert ("POST", "/api/v3/features/media/enable", {"yes": True}) in seen
    assert "1.5 GB" in capsys.readouterr().out


def test_enable_without_yes_off_a_terminal_exits_2(daemon, monkeypatch, capsys):
    seen, _ = daemon
    _tty(monkeypatch, False)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(media_command="enable"))
    assert exc.value.code == 2
    assert not any(m == "POST" for m, *_ in seen)
    assert "--yes" in capsys.readouterr().out


def test_enable_on_a_terminal_asks_and_defaults_to_no(daemon, monkeypatch):
    seen, _ = daemon
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    media_cmd.cmd_media(_args(media_command="enable"))
    assert not any(m == "POST" for m, *_ in seen)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    media_cmd.cmd_media(_args(media_command="enable"))
    assert any(m == "POST" for m, *_ in seen)


def test_daemon_down_exits_3_with_a_restart_hint(daemon, monkeypatch, capsys):
    _, state = daemon
    state["reply"] = None
    _tty(monkeypatch, False)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(media_command="enable", yes=True))
    assert exc.value.code == 3
    assert "slm restart" in capsys.readouterr().out


def test_disable_remove_files(daemon):
    seen, _ = daemon
    media_cmd.cmd_media(_args(media_command="disable", remove_files=True))
    assert ("POST", "/api/v3/features/media/disable", {"remove_files": True}) in seen


def test_features_json_and_text(daemon, capsys):
    features_cmd.cmd_features(Namespace(json=True))
    assert json.loads(capsys.readouterr().out)["data"]["mesh"]["apps_with_mesh"] == 2
    features_cmd.cmd_features(Namespace(json=False))
    text = capsys.readouterr().out
    assert "Images & documents" in text and "slm media enable" in text


def test_features_daemon_down_exits_3(daemon, capsys):
    _, state = daemon
    state["reply"] = None
    with pytest.raises(SystemExit) as exc:
        features_cmd.cmd_features(Namespace(json=False))
    assert exc.value.code == 3


def test_the_modules_never_import_the_in_process_switch():
    import inspect
    for mod in (media_cmd, features_cmd):
        src = inspect.getsource(mod)
        assert "enable_media" not in src and "disable_media" not in src


def test_parsers_are_wired():
    from superlocalmemory.cli.main import _NO_DAEMON_COMMANDS
    assert {"media", "features"} <= _NO_DAEMON_COMMANDS


def test_doctor_asks_the_daemon_whether_a_restart_is_needed(daemon):
    seen, state = daemon
    state["reply"] = {"media": {"restart_required": True}}
    assert features_cmd.running_daemon_media() == {"restart_required": True}
    state["reply"] = None
    assert features_cmd.running_daemon_media() == {}
