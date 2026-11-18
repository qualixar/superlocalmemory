# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm import``: the command line end to end, through the real routes and the real writer."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import daemon, import_cmd
from superlocalmemory.core.feature_flags import env_name
from tests.test_portability import fixtures as fx


@pytest.fixture
def cli(world, monkeypatch, tmp_path):
    """``daemon_request`` answered by the real daemon routes, raising what the real one raises."""

    def request(method, path, body=None, **flags):
        r = world.client.request(method, path, json=body)
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else ""
        if r.status_code in (401, 403):
            raise daemon.DaemonRefused(r.status_code, path)
        if r.status_code == 409:
            raise daemon.DaemonConflict(str(detail))
        if r.status_code == 404:
            raise daemon.not_found_from({"detail": detail}, path)
        if r.status_code == 422:
            raise daemon.DaemonUnprocessable("", str(detail))
        return r.json()

    monkeypatch.setattr(daemon, "daemon_request", request)
    monkeypatch.setattr(import_cmd, "_daemon", lambda: True)
    monkeypatch.setattr(import_cmd.time, "sleep", lambda _s: None)
    world.files = tmp_path / "exports"
    world.files.mkdir()
    return world


def args(sub, **kw):
    base = dict(import_command=sub, profile="", json=False, format="", file="", run_id="", wait=0.0)
    return Namespace(**{**base, **kw})


def write(world, name, data):
    path = world.files / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return str(path)


def test_preview_says_what_is_in_the_file_and_saves_nothing(cli, capsys) -> None:
    path = write(cli, "conversations.json", fx.dumps(fx.chatgpt_list()))
    assert import_cmd.cmd_import(args("preview", file=path)) == 0
    out = capsys.readouterr().out
    assert "We found 2 conversations from ChatGPT, November 2023." in out
    assert "Nothing has been saved yet." in out and f"slm import run {path}" in out
    assert cli.count("memories") == 0 and cli.count("import_runs") == 0


def test_run_wait_status_list_and_undo_round_trip(cli, capsys) -> None:
    path = write(cli, "conversations.json", fx.dumps(fx.chatgpt_list()))
    assert import_cmd.cmd_import(args("run", file=path, wait=60.0)) == 0
    out = capsys.readouterr().out
    assert "4 of 4 saved" in out and "Take it back any time: slm import undo imp_" in out
    assert "left out: 1 replies you edited or regenerated away" in out
    run_id = out.split("slm import undo ")[1].split()[0]
    assert import_cmd.cmd_import(args("status", run_id=run_id)) == 0
    assert "4 of 4 saved" in capsys.readouterr().out
    assert import_cmd.cmd_import(args("list")) == 0
    assert run_id in capsys.readouterr().out
    assert import_cmd.cmd_import(args("undo", run_id=run_id, wait=30.0)) == 0
    assert "Undone. The memory is as it was before this import." in capsys.readouterr().out
    assert cli.count("memories") == 0


def test_json_output_is_for_agents(cli, capsys) -> None:
    path = write(cli, "m.json", fx.dumps(fx.MEM0_ITEMS))
    assert import_cmd.cmd_import(args("preview", file=path, json=True)) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["data"]["would_save"] == 2 and preview["data"]["format"] == "mem0"
    assert import_cmd.cmd_import(args("run", file=path, json=True, wait=60.0)) == 0
    ran = json.loads(capsys.readouterr().out)["data"]["run"]
    assert ran["state"] == "done" and ran["actor"] and ran["channel"] in ("cli", "http")


def test_off_is_refused_with_the_switch_named(cli, monkeypatch, capsys) -> None:
    monkeypatch.setenv(env_name("portability.import"), "off")
    path = write(cli, "m.json", fx.dumps(fx.MEM0_ITEMS))
    for sub in ("preview", "run"):
        assert import_cmd.cmd_import(args(sub, file=path)) == 1
        assert "slm flags set portability.import on" in capsys.readouterr().err
    assert import_cmd.cmd_import(args("list")) == 0
    assert "Importing from other apps is off. Turn it on: slm flags set portability.import on" in \
        capsys.readouterr().out


def test_a_file_in_no_known_format_is_refused_naming_the_formats(cli, capsys) -> None:
    path = write(cli, "x.json", '{"hello": 1}')
    assert import_cmd.cmd_import(args("preview", file=path)) == 1
    err = capsys.readouterr().err
    assert err.startswith("Refused: That file is not in a format this version")
    for name in ("ChatGPT", "Claude", "Mem0", "Letta", "OKF", "Markdown"):
        assert name in err
    assert import_cmd.cmd_import(args("preview", file=str(cli.files / "missing.json"))) == 1
    assert "not found" in capsys.readouterr().err


def test_a_folder_of_markdown_imports_and_the_format_can_be_named(cli, capsys) -> None:
    write(cli, "AGENTS.md", fx.AGENTS_MD)
    assert import_cmd.cmd_import(args("run", file=str(cli.files), wait=60.0, format="markdown")) == 0
    assert "3 of 3 saved" in capsys.readouterr().out


def test_unknown_runs_are_reported_in_words(cli, capsys) -> None:
    for sub in ("status", "undo", "cancel"):
        assert import_cmd.cmd_import(args(sub, run_id="imp_nope")) == 1
        assert "No such import." in capsys.readouterr().err


def test_the_parser_knows_only_the_listed_formats_and_commands(capsys) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    import_cmd.register_import_parsers(parser.add_subparsers(dest="command"))
    ok = parser.parse_args(["import", "run", "x.json", "--format", "mem0", "--wait"])
    assert ok.format == "mem0" and ok.wait == 600.0 and ok.import_command == "run"
    for bad in (["import", "run", "x", "--format", "notion"], ["import", "explode"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_the_command_is_in_help_and_dispatch() -> None:
    from superlocalmemory.cli import commands

    assert "_cmd_import_dispatch" in dir(commands)
    source = open(commands.__file__, encoding="utf-8").read()
    assert '"import": _cmd_import_dispatch' in source and '("import", "Bring memories in' in source


def test_with_no_daemon_it_says_so(monkeypatch, capsys) -> None:
    monkeypatch.setattr(import_cmd, "_daemon", lambda: False)
    monkeypatch.setattr("superlocalmemory.cli.commands._daemon_unavailable",
                        lambda command, use_json: print("daemon down"))
    assert import_cmd.cmd_import(args("list")) == 1
    assert "daemon down" in capsys.readouterr().out
