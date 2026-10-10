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
    monkeypatch.setattr(media_cmd, "is_daemon_running", lambda: True)  # the fake daemon is up
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
    assert ("POST", "/api/v3/features/media/enable", {"yes": True, "source": "cli"}) in seen
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


def test_enable_starts_slm_when_it_is_not_running(daemon, monkeypatch, capsys):
    """First run: `slm media enable` starts the daemon itself instead of saying 'run slm restart'."""
    seen, state = daemon
    state["reply"] = None
    started = []

    def start():
        started.append(True)
        state["reply"] = STATUS
        return True

    monkeypatch.setattr(media_cmd, "is_daemon_running", lambda: False)
    monkeypatch.setattr(media_cmd, "ensure_daemon", start)
    _tty(monkeypatch, False)
    media_cmd.cmd_media(_args(media_command="enable", yes=True))
    assert started == [True]
    assert "Starting SuperLocalMemory" in capsys.readouterr().out
    assert ("POST", "/api/v3/features/media/enable", {"yes": True, "source": "cli"}) in seen


def test_daemon_down_exits_3_with_a_restart_hint(daemon, monkeypatch, capsys):
    _, state = daemon
    state["reply"] = None
    monkeypatch.setattr(media_cmd, "is_daemon_running", lambda: False)
    monkeypatch.setattr(media_cmd, "ensure_daemon", lambda: False)  # could not start it
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


def _reply(state, step="Checking this computer"):
    media = dict(STATUS["media"], enabled=True, env_state=state, step=step)
    return {"media": media}


def _enable_with_reply(daemon, monkeypatch, reply):
    _, st = daemon
    st["reply"] = reply
    _tty(monkeypatch, False)
    media_cmd.cmd_media(_args(media_command="enable", yes=True))


def test_enable_unsupported_is_honest(daemon, monkeypatch, capsys):
    _enable_with_reply(daemon, monkeypatch, _reply("unsupported"))
    out = capsys.readouterr().out
    assert "runs in the background" not in out
    assert "can't be set up" in out and "Nothing is downloaded" in out


def test_enable_failed_names_the_failure(daemon, monkeypatch, capsys):
    _enable_with_reply(daemon, monkeypatch, _reply("failed"))
    out = capsys.readouterr().out
    assert "runs in the background" not in out
    assert "set-up failed" in out and "slm doctor" in out


def test_enable_installing_keeps_the_background_text(daemon, monkeypatch, capsys):
    _enable_with_reply(daemon, monkeypatch, _reply("installing"))
    assert "runs in the background" in capsys.readouterr().out


def test_media_line_unsupported_and_failed():
    assert "can't be set up on this computer yet" in features_cmd.media_line(_reply("unsupported")["media"])
    line = features_cmd.media_line(_reply("failed")["media"])
    assert "set-up failed" in line and "slm doctor" in line


MESSAGE = ("Images and documents need a computer with at least 16 GB of memory; this one has 4.0 GB. "
           "Your text memories keep working.")


def _small_machine(state):
    media = dict(STATUS["media"], ram_ok=False, ram_message=MESSAGE)
    state["reply"] = {"media": media, "mesh": STATUS["mesh"]}


@pytest.mark.parametrize("yes", [False, True])
def test_enable_on_a_small_machine_prints_the_message_exits_nonzero_and_asks_nothing(daemon, monkeypatch, capsys, yes):
    seen, state = daemon
    _small_machine(state)
    _tty(monkeypatch, True)

    def no_prompt(*_a):
        raise AssertionError("must not ask to turn on a machine that will be refused")

    monkeypatch.setattr("builtins.input", no_prompt)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(media_command="enable", yes=yes))
    assert exc.value.code not in (0, None)
    out = capsys.readouterr().out
    assert MESSAGE in out and "Turn on" not in out and "1.5 GB" not in out
    assert not any(m == "POST" for m, *_ in seen)


def test_enable_on_a_small_machine_in_json_is_an_error_object(daemon, monkeypatch, capsys):
    _, state = daemon
    _small_machine(state)
    _tty(monkeypatch, False)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(media_command="enable", yes=True, json=True))
    assert exc.value.code not in (0, None)
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False and out["error"]["message"] == MESSAGE


def test_a_refusal_that_arrives_with_the_turn_on_request_is_printed_too(daemon, monkeypatch, capsys):
    from superlocalmemory.cli.daemon import DaemonConflict

    def conflict(method, path, body=None, **_kw):
        if method == "POST":
            raise DaemonConflict(MESSAGE)
        return STATUS

    monkeypatch.setattr(media_cmd, "daemon_request", conflict)
    _tty(monkeypatch, False)
    with pytest.raises(SystemExit) as exc:
        media_cmd.cmd_media(_args(media_command="enable", yes=True))
    assert exc.value.code not in (0, None)
    assert MESSAGE in capsys.readouterr().out


def test_status_on_a_small_machine_still_works(daemon, capsys):
    _, state = daemon
    _small_machine(state)
    media_cmd.cmd_media(_args())
    assert "Images & documents:" in capsys.readouterr().out


def test_enable_prints_no_memory_line_on_a_big_machine(daemon, monkeypatch, capsys):
    _tty(monkeypatch, False)
    media_cmd.cmd_media(_args(media_command="enable", yes=True))
    assert "memory" not in capsys.readouterr().out.lower().replace("memories", "")


GC_REPORT = {"dry_run": True, "rows_without_memory": ["m1", "m2"], "files_without_row": ["f1"],
             "memories_without_row": [], "files_skipped_young": 3, "rows_removed": 0, "files_removed": 0}


def test_gc_reports_by_default_and_removes_nothing(daemon, capsys):
    seen, state = daemon
    state["reply"] = GC_REPORT
    media_cmd.cmd_media(_args(media_command="gc", apply=False))
    assert ("POST", "/api/v3/media/gc", {"dry_run": True}) in seen
    out = capsys.readouterr().out
    assert "2 picture records" in out and "1 file" in out and "slm media gc --apply" in out


def test_gc_apply_removes_and_says_what_went(daemon, capsys):
    seen, state = daemon
    state["reply"] = {**GC_REPORT, "dry_run": False, "rows_removed": 2, "files_removed": 1}
    media_cmd.cmd_media(_args(media_command="gc", apply=True))
    assert ("POST", "/api/v3/media/gc", {"dry_run": False}) in seen
    assert "Removed 2 picture records and 1 file" in capsys.readouterr().out


def test_gc_parser_is_wired():
    import argparse

    parser = argparse.ArgumentParser()
    media_cmd.register_media_parser(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["media", "gc"]).apply is False
    assert parser.parse_args(["media", "gc", "--apply"]).apply is True
