# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""JSON mode, audit fixes: streaming commands, tool parity, dry runs, error shapes.

In-process, with the test's own HOME and data folder and the daemon mocked away.
"""

from __future__ import annotations

import argparse
import json

import pytest
from tests.cli.test_json_everywhere import no_daemon, one_document, run  # noqa: F401

from superlocalmemory.cli import json_mode, tool_cmd


def _store() -> None:
    """A real memory.db with the default profile, in the test's data folder."""
    import os
    from pathlib import Path

    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(Path(os.environ["SLM_DATA_DIR"]) / "memory.db")
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
               ("default", "Default Profile"))


def test_gdpr_export_streams_its_own_envelope(no_daemon, capsys):  # noqa: F811
    """The export writes bytes to stdout's buffer; JSON mode must not swap stdout for text."""
    _store()
    rc, out, _ = run(["--json", "gdpr", "export", "--profile", "default"], capsys)
    doc = json.loads(out)
    assert rc == 0 and doc["success"] is True
    assert doc["data"]["profile_id"] == "default"


def test_gdpr_export_under_the_env_var(no_daemon, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SLM_JSON", "1")
    _store()
    rc, out, _ = run(["gdpr", "export", "--profile", "default"], capsys)
    assert rc == 0 and json.loads(out)["data"]["profile_id"] == "default"


def test_a_successful_tool_call_is_exactly_the_tools_result(no_daemon, capsys,  # noqa: F811
                                                         monkeypatch):
    monkeypatch.setattr(tool_cmd, "_cmd_call",
                        lambda name, rest: print(json.dumps({"facts": [1]})))
    rc, out, _ = run(["--json", "tool", "recall", "--query", "x"], capsys)
    assert rc == 0 and out == '{"facts": [1]}\n'


def test_a_tool_failure_carries_an_error_object_and_keeps_code(no_daemon, capsys):  # noqa: F811
    rc, out, _ = run(["--json", "tool", "no_such_tool_anywhere"], capsys)
    doc = one_document(out)
    assert rc == 2
    assert doc["code"] == "unknown_tool"  # old parsers read the top-level code
    assert doc["error"]["code"] == "unknown_tool" and doc["error"]["message"]


def test_tool_fail_body_without_json_mode_has_error_object(capsys):
    with pytest.raises(SystemExit):
        tool_cmd._fail("usage", "bad thing", tool="recall")
    doc = json.loads(capsys.readouterr().out)
    assert doc["code"] == "usage" and doc["error"] == {"code": "usage", "message": "bad thing"}


def test_complete_turns_a_string_error_into_an_object():
    doc = json_mode.normalize("x", 1, json.dumps({"success": False, "code": "nope",
                                                  "error": "it broke"}))
    assert doc["error"] == {"code": "nope", "message": "it broke"} and doc["code"] == "nope"
    doc = json_mode.normalize("x", 1, json.dumps({"error": "plain"}))
    assert doc["error"] == {"code": "error", "message": "plain"}


def test_backup_failure_is_an_error_object(capsys):
    from superlocalmemory.cli import backup_cmd

    with pytest.raises(SystemExit) as exc:
        backup_cmd._fail(argparse.Namespace(json=True), "no key")
    doc = json.loads(capsys.readouterr().out)
    assert exc.value.code == 1 and doc["success"] is False
    assert doc["error"]["message"] == "no key" and doc["error"]["code"]


@pytest.mark.parametrize("rest,args_json", [
    (["--query", "x", "--json"], None),
    (["--json", "--query", "x"], None),
    (["--json", '{"query": "x"}'], '{"query": "x"}'),
])
def test_a_bare_trailing_json_on_a_tool_call_means_json_on(rest, args_json):
    call = tool_cmd._split_call(rest)
    assert call.options.get("json") == args_json
    if args_json is None:
        assert call.tool_tokens == ["--query", "x"]


def _settings(tmp_path, monkeypatch):
    from superlocalmemory.hooks import claude_code_hooks

    path = tmp_path / "settings.json"
    path.write_text('{"theme": "dark"}\n', encoding="utf-8")
    monkeypatch.setattr(claude_code_hooks, "CLAUDE_SETTINGS", path)
    return path


@pytest.mark.parametrize("action", ["install", "remove"])
def test_hooks_dry_run_json_writes_nothing(no_daemon, capsys, tmp_path, monkeypatch,  # noqa: F811
                                           action):
    path = _settings(tmp_path, monkeypatch)
    rc, out, _ = run(["hooks", action, "--dry-run", "--json"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["dry_run"] is True
    assert path.read_text(encoding="utf-8") == '{"theme": "dark"}\n'
    if action == "install":
        assert doc["data"]["would_change"] is True and doc["data"]["hooks_added"]


def test_plain_hooks_dry_run_writes_nothing(no_daemon, capsys, tmp_path, monkeypatch):  # noqa: F811
    path = _settings(tmp_path, monkeypatch)
    rc, out, _ = run(["hooks", "install", "--dry-run"], capsys)
    assert rc == 0 and "would" in out.lower()
    assert path.read_text(encoding="utf-8") == '{"theme": "dark"}\n'


def test_evidence_without_a_subcommand_is_usage(no_daemon, capsys):  # noqa: F811
    rc, out, _ = run(["evidence"], capsys)
    assert rc == 2
    rc, out, _ = run(["--json", "evidence"], capsys)
    assert rc == 2 and one_document(out)["error"]["code"] == "usage"


def test_non_ascii_is_escaped(capsys):
    json_mode.run_json("x", lambda: print(json.dumps({"name": "café ✓"}, ensure_ascii=False)))
    out = capsys.readouterr().out
    assert out.isascii() and json.loads(out)["name"] == "café ✓"


def test_session_commands_ignore_the_env_var():
    assert {"mcp", "session-context", "session"} <= json_mode.ENV_EXEMPT
