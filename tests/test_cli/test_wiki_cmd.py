# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm wiki``: any caller reads, builds, rebuilds or syncs; each is recorded (WS-6, D-17, D-49)."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import wiki_cmd

STATUS = {"success": True, "flag": "on", "status": {
    "dir": "/w/default", "pages": 2, "by_kind": {"topic": 1, "current_state": 1}, "pending_sets": 1,
    "files": 2, "differences": {"edited": 1}, "edits_waiting": 3}, "pages": []}
BUILD = {"success": True, "dir": "/w/default", "dry_run": False, "written": ["topic/a.md"],
         "unchanged": [], "kept_edited": ["topic/b.md"], "removed": []}
SYNC = {"success": True, "found": 1, "dry_run": False, "files": 2, "deleted_lines": 1,
        "queued": [{"set_id": "s1", "path": "topic/a.md"}],
        "skipped": [{"path": "topic/b.md", "why": "this line cites several memories"}]}


def _wire(monkeypatch, *, tty=True, answers=()):
    from superlocalmemory.cli import daemon

    sent = []

    def fake(method, path, body=None, **flags):
        sent.append((method, path, body))
        if path.startswith("/api/wiki/pages"):
            return STATUS
        if path.startswith("/api/wiki/diff"):
            return {"success": True, "items": [{"path": "topic/a.md", "state": "edited"}]}
        if path.endswith("/sync"):
            return SYNC
        if path.endswith("/rebuild"):
            return {"success": True, "pages": 2, "set_id": "x", "files": BUILD}
        return BUILD

    monkeypatch.setenv("SLM_AGENT_ID", "alice")  # the client name on the record, never a real user
    monkeypatch.setattr(daemon, "daemon_request", fake)
    monkeypatch.setattr(wiki_cmd, "_daemon", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty)
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    return sent


def _args(sub="status", **kw):
    return Namespace(**{**dict(wiki_command=sub, profile="", json=False, path="", default=False), **kw})


def test_status_is_the_default_and_works_anywhere(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert wiki_cmd.cmd_wiki(_args(wiki_command=None)) == 0
    out = capsys.readouterr().out
    assert "Markdown wiki: on" in out and "Pages: 2" in out and "proposals from edits: 3" in out
    assert sent == [("GET", "/api/wiki/pages", None)]


def test_off_says_how_to_turn_it_on(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: {**STATUS, "flag": "off"})
    monkeypatch.setattr(wiki_cmd, "_daemon", lambda: True)
    assert wiki_cmd.cmd_wiki(_args()) == 0
    assert "slm flags set compiler.wiki on" in capsys.readouterr().out


def test_diff_and_build_work_without_a_terminal_and_json_is_for_agents(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert wiki_cmd.cmd_wiki(_args("diff")) == 0
    assert "you edited it" in capsys.readouterr().out
    assert wiki_cmd.cmd_wiki(_args("build", json=True)) == 0
    assert json.loads(capsys.readouterr().out)["data"]["written"] == ["topic/a.md"]
    assert sent[-1] == ("POST", "/api/wiki/build", {"by": "agent:alice"})
    assert wiki_cmd.cmd_wiki(_args("build")) == 0
    assert "kept topic/b.md: you edited it" in capsys.readouterr().out


@pytest.mark.parametrize("sub", ["rebuild", "sync"])
def test_without_a_terminal_rebuild_and_sync_work_and_never_ask(monkeypatch, capsys, sub) -> None:
    """D-49: no command needs a terminal or a confirmation; the daemon records who decided."""
    sent = _wire(monkeypatch, tty=False, answers=())  # input() would raise StopIteration
    assert wiki_cmd.cmd_wiki(_args(sub)) == 0
    assert [(m, p) for m, p, _ in sent] == [("POST", f"/api/wiki/{sub}")]
    assert "asks a person" not in capsys.readouterr().err


def test_sync_goes_out_with_a_client_name_and_the_channel_comes_from_the_connection(
        monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch)
    monkeypatch.setenv("SLM_AGENT_ID", "claude-code")
    assert wiki_cmd.cmd_wiki(_args("sync", profile="work")) == 0
    out = capsys.readouterr().out
    assert "Proposed 1 edit(s)" in out and "a deleted line forgets nothing" in out
    assert "left alone: topic/b.md" in out and "slm proposals" in out
    assert sent == [("POST", "/api/wiki/sync", {"by": "agent:claude-code", "profile_id": "work"})]


def test_rebuild_says_it_was_applied_and_can_be_rolled_back(monkeypatch, capsys) -> None:
    _wire(monkeypatch)
    assert wiki_cmd.cmd_wiki(_args("rebuild")) == 0
    out = capsys.readouterr().out
    assert "Compiled and applied 2 page(s)" in out and "slm proposals rollback x" in out


def test_the_daemons_refusal_is_a_plain_answer(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch)

    def refuse(*a, **k):
        raise daemon.DaemonRefused(403, "/api/wiki/sync")

    monkeypatch.setattr(daemon, "daemon_request", refuse)
    assert wiki_cmd.cmd_wiki(_args("sync")) == 1
    err = capsys.readouterr().err
    assert "decided by a person" not in err and "role" in err


def test_dir_shows_and_chooses_the_folder_in_the_data_folder_never_the_real_one(
        monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SLM_WIKI_DIR", raising=False)
    assert wiki_cmd.cmd_wiki(_args("dir")) == 0
    assert str(tmp_path / "data" / "wiki") in capsys.readouterr().out
    chosen = tmp_path / "notes"
    assert wiki_cmd.cmd_wiki(_args("dir", path=str(chosen), json=True)) == 0
    assert json.loads(capsys.readouterr().out)["data"]["dir"] == str(chosen.resolve())
    from superlocalmemory.compiler import wiki

    assert wiki.wiki_root() == chosen.resolve()
    (tmp_path / "afile").write_text("x")
    assert wiki_cmd.cmd_wiki(_args("dir", path=str(tmp_path / "afile"))) == 1
    assert wiki_cmd.cmd_wiki(_args("dir", default=True)) == 0
    assert wiki.wiki_root() == tmp_path / "data" / "wiki"


def test_the_parser_registers_every_subcommand() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    wiki_cmd.register_wiki_parsers(parser.add_subparsers(dest="command"))
    for sub in ("status", "diff", "build", "rebuild", "sync", "dir"):
        assert parser.parse_args(["wiki", sub]).wiki_command == sub
