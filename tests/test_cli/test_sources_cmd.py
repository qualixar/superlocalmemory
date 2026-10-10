"""``slm sources`` talks to the daemon only; a fake daemon records what it was asked."""

from __future__ import annotations

import io
import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import sources_cmd
from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound

PREVIEW = {"source_id": "src1", "root": "/notes", "kind": "folder", "files_by_type": {".md": 3, ".pdf": 1},
           "skipped_by_rule": {"hidden": 2}, "est_bytes": 2048, "quarantined_count": 1, "est_seconds": 1.5,
           "estimate_note": "Estimate only.", "capped": False, "warnings": []}


class FakeDaemon:
    def __init__(self, replies=None, down=False):
        self.calls: list[tuple] = []
        self.replies = replies or {}
        self.down = down

    def __call__(self, method, path, body=None, **_kw):
        self.calls.append((method, path, body))
        if self.down:
            return None
        reply = self.replies.get((method, path.split("?")[0].rstrip("/") if "{" not in path else path))
        if isinstance(reply, Exception):
            raise reply
        return reply if reply is not None else {}


def run(monkeypatch, daemon, *, tty=True, typed=None, **kw):
    monkeypatch.setattr(sources_cmd, "daemon_request", daemon)
    monkeypatch.setattr(sources_cmd, "_is_tty", lambda: tty)

    def ask(prompt=""):
        assert typed is not None, "asked a question it should not have asked"
        return typed

    monkeypatch.setattr("builtins.input", ask)
    base = dict(json=False, sources_command=None, yes=False, purge=False, kind=None)
    base.update(kw)
    return sources_cmd.cmd_sources(Namespace(**base))


def daemon_for_add():
    return FakeDaemon({("POST", "/api/v3/sources"): PREVIEW,
                       ("POST", "/api/v3/sources/src1/confirm"): {"confirmed": True, "source_id": "src1"}})


def test_add_shows_the_preview_and_asks_before_connecting(monkeypatch, capsys, tmp_path):
    d = daemon_for_add()
    rc = run(monkeypatch, d, typed="y", sources_command="add", path=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0 and "3" in out and ".md" in out and "hidden" in out and "1" in out
    assert [c[1] for c in d.calls] == ["/api/v3/sources", "/api/v3/sources/src1/confirm"]
    assert d.calls[0][2]["path"] == str(tmp_path)


def test_add_declined_does_not_connect(monkeypatch, capsys, tmp_path):
    d = daemon_for_add()
    rc = run(monkeypatch, d, typed="n", sources_command="add", path=str(tmp_path))
    assert rc == 1 and [c[1] for c in d.calls] == ["/api/v3/sources"]
    assert "not connected" in capsys.readouterr().out.lower()


def test_add_yes_skips_the_question(monkeypatch, tmp_path):
    d = daemon_for_add()
    assert run(monkeypatch, d, tty=False, yes=True, sources_command="add", path=str(tmp_path)) == 0
    assert d.calls[-1][1].endswith("/confirm")


def test_add_without_tty_and_without_yes_refuses_before_any_request(monkeypatch, capsys, tmp_path):
    d = daemon_for_add()
    rc = run(monkeypatch, d, tty=False, sources_command="add", path=str(tmp_path))
    assert rc == 2 and d.calls == []
    assert "--yes" in capsys.readouterr().out


def test_relative_path_is_sent_absolute(monkeypatch, tmp_path):
    d = daemon_for_add()
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, d, yes=True, sources_command="add", path=".")
    assert d.calls[0][2]["path"] == str(tmp_path)


def test_daemon_down_prints_a_clear_message(monkeypatch, capsys):
    for cmd, extra in (("list", {}), ("report", {"source_id": "s1"}), ("rescan", {"source_id": "s1"}),
                       ("remove", {"source_id": "s1"}), ("add", {"path": "/x", "yes": True})):
        rc = run(monkeypatch, FakeDaemon(down=True), sources_command=cmd, **extra)
        assert rc == 1
        assert "daemon is not running" in capsys.readouterr().out


def test_list_text_and_json(monkeypatch, capsys):
    reply = {"sources": [{"source_id": "src1", "kind": "obsidian", "root_path": "/v", "state": "active",
                          "files": {"indexed": 4}, "last_scan_at": "2026-01-01T00:00:00Z"}]}
    d = FakeDaemon({("GET", "/api/v3/sources"): reply})
    assert run(monkeypatch, d, sources_command="list") == 0
    text = capsys.readouterr().out
    assert "src1" in text and "active" in text and "/v" in text
    run(monkeypatch, d, sources_command="list", json=True)
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["sources"][0]["source_id"] == "src1"


def test_list_empty(monkeypatch, capsys):
    run(monkeypatch, FakeDaemon({("GET", "/api/v3/sources"): {"sources": []}}), sources_command="list")
    assert "no folder" in capsys.readouterr().out.lower()


def test_rescan_and_report(monkeypatch, capsys):
    d = FakeDaemon({("POST", "/api/v3/sources/s1/rescan"): {"job_id": "j", "state": "queued", "done": 0, "total": 0},
                    ("GET", "/api/v3/sources/s1/report"): {
                        "source_id": "s1", "state": "active", "counts": {"indexed": 2},
                        "skipped_by_rule": {"hidden": 1}, "quarantined": [{"relpath": "k.txt", "reason": "credential:aws"}],
                        "cloud_only": ["c.md"], "errors": [], "last_scan_at": None, "paused_reason": None,
                        "capped": False, "offline_reason": None, "watch": 0}})
    assert run(monkeypatch, d, sources_command="rescan", source_id="s1") == 0
    assert "queued" in capsys.readouterr().out.lower()
    assert run(monkeypatch, d, sources_command="report", source_id="s1") == 0
    text = capsys.readouterr().out
    assert "k.txt" in text and "credential:aws" in text and "c.md" in text and "15 minutes" in text


def test_remove_keeps_memories_without_purge(monkeypatch):
    d = FakeDaemon({("DELETE", "/api/v3/sources/s1"): {"removed": True}})
    assert run(monkeypatch, d, sources_command="remove", source_id="s1") == 0
    assert d.calls == [("DELETE", "/api/v3/sources/s1", None)]


def test_purge_needs_the_typed_word(monkeypatch, capsys):
    d = FakeDaemon({("DELETE", "/api/v3/sources/s1"): {"removed": True, "purged": True}})
    assert run(monkeypatch, d, typed="yes", sources_command="remove", source_id="s1", purge=True) == 1
    assert d.calls == []
    assert run(monkeypatch, d, typed="erase", sources_command="remove", source_id="s1", purge=True) == 0
    assert d.calls == [("DELETE", "/api/v3/sources/s1?purge=true", None)]


def test_purge_with_yes_needs_no_word_and_without_tty_refuses(monkeypatch, capsys):
    d = FakeDaemon({("DELETE", "/api/v3/sources/s1"): {"removed": True}})
    assert run(monkeypatch, d, tty=False, sources_command="remove", source_id="s1", purge=True) == 2
    assert d.calls == []
    assert run(monkeypatch, d, tty=False, yes=True, sources_command="remove", source_id="s1", purge=True) == 0
    assert d.calls[0][1].endswith("?purge=true")


def test_refusals_are_shown_in_words(monkeypatch, capsys):
    err = DaemonConflict("{'code': 'remote_access_on', 'message': 'Turn remote access off first.'}")
    d = FakeDaemon({("POST", "/api/v3/sources"): err})
    assert run(monkeypatch, d, yes=True, sources_command="add", path="/x") == 1
    assert "Turn remote access off first." in capsys.readouterr().out
    d = FakeDaemon({("GET", "/api/v3/sources/s1/report"): DaemonNotFound(404, "not_found", "Not found.", "p")})
    assert run(monkeypatch, d, sources_command="report", source_id="s1") == 1
    assert "Not found." in capsys.readouterr().out


def test_bad_source_id_never_reaches_the_daemon(monkeypatch, capsys):
    d = FakeDaemon()
    assert run(monkeypatch, d, sources_command="report", source_id="../x") == 2
    assert d.calls == []


def test_parser_wires_the_command():
    import argparse

    p = argparse.ArgumentParser()
    sources_cmd.register_sources_parser(p.add_subparsers(dest="command"))
    ns = p.parse_args(["sources", "remove", "abc", "--purge", "--yes", "--json"])
    assert (ns.sources_command, ns.source_id, ns.purge, ns.yes, ns.json) == ("remove", "abc", True, True, True)
    ns = p.parse_args(["sources", "add", "/n"])
    assert ns.path == "/n" and ns.yes is False


def test_forget_empty_needs_a_yes_and_posts_nothing_without_one(monkeypatch, capsys):
    d = FakeDaemon({("POST", "/api/v3/sources/s1/forget-empty"): {"source_id": "s1", "forgotten": 2, "state": "active"}})
    assert run(monkeypatch, d, tty=False, sources_command="forget-empty", source_id="s1") == 2
    assert d.calls == []
    assert run(monkeypatch, d, typed="n", sources_command="forget-empty", source_id="s1") == 1
    assert d.calls == []
    assert run(monkeypatch, d, typed="y", sources_command="forget-empty", source_id="s1") == 0
    assert d.calls == [("POST", "/api/v3/sources/s1/forget-empty", None)]
    assert "Forgot 2 file(s) of s1; the folder is active again." in capsys.readouterr().out


def test_forget_empty_yes_and_json(monkeypatch, capsys):
    d = FakeDaemon({("POST", "/api/v3/sources/s1/forget-empty"): {"source_id": "s1", "forgotten": 2, "state": "active"}})
    assert run(monkeypatch, d, tty=False, yes=True, json=True, sources_command="forget-empty", source_id="s1") == 0
    assert json.loads(capsys.readouterr().out)["data"] == {"source_id": "s1", "forgotten": 2, "state": "active"}


def test_forget_empty_refusal_is_shown(monkeypatch, capsys):
    err = DaemonConflict("{'code': 'folder_not_empty', 'message': 'The folder has files again.'}")
    d = FakeDaemon({("POST", "/api/v3/sources/s1/forget-empty"): err})
    assert run(monkeypatch, d, yes=True, sources_command="forget-empty", source_id="s1") == 1
    assert "files again" in capsys.readouterr().out


def test_list_and_report_hint_at_forget_empty(monkeypatch, capsys):
    row = {"source_id": "src1", "kind": "folder", "root_path": "/v", "state": "offline", "files": {"indexed": 4},
           "offline_reason": "empty_folder"}
    run(monkeypatch, FakeDaemon({("GET", "/api/v3/sources"): {"sources": [row]}}), sources_command="list")
    assert "slm sources forget-empty src1" in capsys.readouterr().out
    rep = {"source_id": "src1", "state": "offline", "counts": {}, "skipped_by_rule": {}, "quarantined": [],
           "cloud_only": [], "errors": [], "last_scan_at": None, "paused_reason": None, "capped": False,
           "offline_reason": "empty_folder", "watch": 0}
    run(monkeypatch, FakeDaemon({("GET", "/api/v3/sources/src1/report"): rep}), sources_command="report", source_id="src1")
    assert "slm sources forget-empty src1" in capsys.readouterr().out


def test_forget_empty_parses():
    import argparse

    p = argparse.ArgumentParser()
    sources_cmd.register_sources_parser(p.add_subparsers(dest="command"))
    ns = p.parse_args(["sources", "forget-empty", "abc", "--yes", "--json"])
    assert (ns.sources_command, ns.source_id, ns.yes, ns.json) == ("forget-empty", "abc", True, True)


# --- a daemon that answered with an error is not a daemon that is down -------

def test_a_daemon_server_error_prints_its_message_not_the_not_running_text(monkeypatch, capsys):
    from superlocalmemory.cli.daemon import DaemonServerError

    seen = {}

    def daemon(method, path, body=None, **kw):
        seen.update(kw)
        raise DaemonServerError(503, "writer_not_ready", "The memory writer is not ready; try again shortly.")

    rc = run(monkeypatch, daemon, sources_command="list")
    out = capsys.readouterr().out
    assert rc == 1
    assert seen["preserve_server_error"] is True
    assert "The memory writer is not ready; try again shortly." in out
    assert "not running" not in out


def test_a_none_result_still_means_not_running(monkeypatch, capsys):
    rc = run(monkeypatch, FakeDaemon(down=True), sources_command="list")
    assert rc == 1
    assert "daemon is not running" in capsys.readouterr().out
