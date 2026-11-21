# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every ``slm`` command answers in JSON: the switch, the one-document rule, no prompts.

Runs ``main()`` in-process with the test's own HOME and data folder (the
isolation plugin), the daemon mocked away, no server started, no browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.cli.test_json_ratchet import leaf_commands

from superlocalmemory.cli import json_mode
from superlocalmemory.cli.main import build_parser, main


@pytest.fixture
def no_daemon(monkeypatch):
    """No daemon answers, none is started, nothing stops, no browser opens."""
    from superlocalmemory.cli import daemon

    calls: list[str] = []
    monkeypatch.setattr(daemon, "ensure_daemon", lambda *a, **k: calls.append("ensure") or False)
    monkeypatch.setattr(daemon, "is_daemon_running", lambda *a, **k: False)
    monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: None)
    monkeypatch.setattr(daemon, "stop_daemon", lambda *a, **k: False)
    monkeypatch.setattr(daemon, "owned_daemon_process_alive", lambda *a, **k: False)

    def _no_browser(*_a, **_k):
        raise AssertionError("a browser was opened")

    monkeypatch.setattr("webbrowser.open", _no_browser)
    monkeypatch.delenv(json_mode.ENV_VAR, raising=False)
    return calls


def run(argv: list[str], capsys) -> tuple[int, str, str]:
    """``slm <argv>`` in-process: (exit code, stdout, stderr)."""
    rc = 0
    with patch.object(sys, "argv", ["slm", *argv]):
        try:
            main()
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    out, err = capsys.readouterr()
    return rc, out, err


def one_document(out: str) -> dict:
    """stdout is exactly one JSON object on one line, with the contract's keys."""
    assert out.endswith("\n") and out.count("\n") == 1, out
    doc = json.loads(out)
    assert isinstance(doc, dict)
    for key in ("success", "command", "version"):
        assert key in doc, (key, doc)
    if doc["success"] is False and isinstance(doc.get("error"), dict):
        assert doc["error"].get("code"), doc
    return doc


# -- the switch ------------------------------------------------------------------------


def test_extract_global_json_only_before_the_command():
    argv, found = json_mode.extract_global_json(["slm", "--json", "tool", "remember",
                                                 "--json", "{}"])
    assert found is True
    assert argv == ["slm", "tool", "remember", "--json", "{}"]
    assert json_mode.extract_global_json(["slm", "recall", "x", "--json"]) == (
        ["slm", "recall", "x", "--json"], False)


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("ON", True),
                                            ("0", False), ("", False), ("no", False)])
def test_env_values(value, expected):
    assert json_mode.env_requests_json({"SLM_JSON": value}) is expected


def test_global_flag_before_the_command(no_daemon, capsys):
    rc, out, _ = run(["--json", "flags"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["success"] is True
    assert "kill_switch" in doc  # the command's own keys are kept as they were


def test_env_var_turns_json_on(no_daemon, capsys, monkeypatch):
    monkeypatch.setenv("SLM_JSON", "1")
    rc, out, _ = run(["enable"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["command"] == "enable" and doc["data"]["enabled"] is True


def test_global_flag_equals_the_command_flag(no_daemon, capsys):
    _, a, _ = run(["--json", "clear-cache"], capsys)
    _, b, _ = run(["clear-cache", "--json"], capsys)
    assert json.loads(a)["data"] == json.loads(b)["data"]


def test_env_var_never_turns_mcp_into_json(no_daemon, capsys, monkeypatch):
    """An MCP server launched from an environment with SLM_JSON=1 still starts."""
    from superlocalmemory.cli import commands

    started: list[bool] = []
    monkeypatch.setattr(commands, "cmd_mcp", lambda _a: started.append(True))
    monkeypatch.setenv("SLM_JSON", "1")
    rc, out, _ = run(["mcp"], capsys)
    assert started == [True] and out == "" and rc == 0


def test_mcp_json_describes_and_never_starts_the_server(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import commands

    monkeypatch.setattr(commands, "cmd_mcp", lambda _a: pytest.fail("server started"))
    rc, out, _ = run(["mcp", "--json"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["started"] is False
    assert doc["data"]["stdio"]["args"] == ["mcp"]


def test_usage_error_is_a_json_envelope(no_daemon, capsys):
    rc, out, _ = run(["--json", "recall"], capsys)
    doc = one_document(out)
    assert rc == 2 and doc["error"]["code"] == "usage"


def test_no_command_gives_the_tree(no_daemon, capsys):
    rc, out, _ = run(["--json"], capsys)
    assert rc == 0 and one_document(out)["command"] == "help"


def test_dash_help_under_json_gives_that_subtree(no_daemon, capsys):
    rc, out, _ = run(["--json", "db", "--help"], capsys)
    doc = one_document(out)
    paths = [c["path"] for c in doc["data"]["commands"]]
    assert rc == 0 and paths[0] == "db" and "db migrate" in paths
    assert all(p == "db" or p.startswith("db ") for p in paths)


# -- help --json: the whole tree ---------------------------------------------------------


def test_help_json_lists_every_leaf_with_options(no_daemon, capsys):
    rc, out, _ = run(["help", "--json"], capsys)
    doc = one_document(out)
    tree = {c["path"]: c for c in doc["data"]["commands"]}
    leaves = [" ".join(p) for p, _ in leaf_commands(build_parser("t"))]
    assert rc == 0 and set(leaves) <= set(tree)
    recall = {o["name"]: o for o in tree["recall"]["options"]}
    assert recall["query"]["required"] is True and recall["query"]["positional"] is True
    assert recall["--limit"]["type"] == "integer"
    assert "search" in tree["recall"]["aliases"]
    assert all({"type", "required", "default"} <= set(o) for c in tree.values()
               for o in c["options"])


# -- no prompts ------------------------------------------------------------------------------


def _parser_with(*flags: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slm demo")
    for f in flags:
        p.add_argument(f, action="store_true")
    return p


def test_a_confirmation_question_becomes_confirmation_required(capsys):
    def asks():
        input("Delete 3 memories? [y/N] ")

    with pytest.raises(SystemExit) as exc:
        json_mode.run_json("demo", asks, _parser_with("--yes"))
    doc = one_document(capsys.readouterr().out)
    assert exc.value.code == 1
    assert doc["error"]["code"] == "confirmation_required" and doc["error"]["flag"] == "--yes"


def test_a_password_question_becomes_input_required(capsys):
    import getpass

    with pytest.raises(SystemExit):
        json_mode.run_json("demo", lambda: getpass.getpass("Password: "), _parser_with())
    assert one_document(capsys.readouterr().out)["error"]["code"] == "input_required"


def test_prompt_functions_come_back_after_a_run(capsys):
    import builtins

    real = builtins.input
    json_mode.run_json("demo", lambda: print("{}"))
    capsys.readouterr()
    assert builtins.input is real


@pytest.mark.parametrize("argv,flag", [(["--json", "setup"], "--auto"),
                                       (["--json", "init"], "--auto"),
                                       (["--json", "reconfigure"], "--profile=NAME")])
def test_wizards_name_the_missing_flags(no_daemon, capsys, argv, flag):
    rc, out, _ = run(argv, capsys)
    doc = one_document(out)
    assert rc == 1 and doc["error"]["code"] == "input_required"
    assert flag in doc["error"]["missing_flags"]


def test_migrate_without_yes_is_confirmation_required(no_daemon, capsys, monkeypatch):
    class _Migrator:
        def detect_v2(self):
            return True

        def is_already_migrated(self):
            return False

        def get_v2_stats(self):
            return {"memory_count": 3}

        def migrate(self):
            raise AssertionError("migrated without --yes")

    monkeypatch.setattr("superlocalmemory.storage.v2_migrator.V2Migrator", _Migrator)
    rc, out, _ = run(["--json", "migrate"], capsys)
    doc = one_document(out)
    assert rc == 1 and doc["error"]["code"] == "confirmation_required"
    assert doc["error"]["flag"] == "--yes"


def test_team_reset_password_never_prompts(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import team_cmd

    monkeypatch.setattr(team_cmd, "daemon_request",
                        lambda *a, **k: {"users": [{"username": "ana", "user_id": "u1"}]})
    rc, out, _ = run(["team", "reset-password", "ana", "--json"], capsys)
    doc = one_document(out)
    assert rc == 1 and doc["error"]["code"] == "input_required"
    assert doc["error"]["missing_flags"] == ["--password-stdin"]


def test_team_group_flag_is_kept_by_the_leaf(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import team_cmd

    monkeypatch.setattr(team_cmd, "daemon_request",
                        lambda *a, **k: {"require_login": False, "user_count": 0})
    for argv in (["team", "--json", "status"], ["team", "status", "--json"]):
        rc, out, _ = run(argv, capsys)
        assert rc == 0 and one_document(out)["data"]["user_count"] == 0


# -- secrets -------------------------------------------------------------------------------


def test_token_show_carries_the_token_under_its_name_only(no_daemon, capsys):
    from superlocalmemory.core.security_primitives import ensure_install_token

    token = ensure_install_token()
    rc, out, err = run(["--json", "token", "show"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"] == {"install_token": token}
    assert token not in err


def test_rotate_token_shows_only_the_last_four(no_daemon, capsys):
    from superlocalmemory.core.security_primitives import ensure_install_token

    ensure_install_token()
    rc, out, err = run(["--json", "rotate-token"], capsys)
    doc = one_document(out)
    from superlocalmemory.core.security_primitives import ensure_install_token as now

    assert rc == 0 and doc["data"]["new_token_last4"] == now()[-4:]
    assert now() not in out and now() not in err


# -- every command that gained --json ------------------------------------------------------

#: argv for each command that did not take --json before; each must answer with one
#: JSON object. Exit codes are checked against ``success``.
NEW_COMMANDS: list[list[str]] = [
    ["init", "--json"], ["setup", "--json"], ["provider", "--json"],
    ["upgrade-hosts", "--json"], ["migrate", "--json"],
    ["db", "migrate", "--status", "--json"], ["db", "scale", "status", "--json"],
    ["db", "compact", "--json"], ["mesh", "status", "--json"], ["help", "--json"],
    ["help", "modes", "--json"], ["wrap", "--json", "--list"], ["wrap", "--json", "claude"],
    ["mcp", "--json"], ["tool", "--json", "list"],
    ["flags", "set", "truth.slots", "shadow", "--json"], ["flags", "kill", "off", "--json"],
    ["flags", "rollback-retirements", "--dry-run", "--json"],
    ["dashboard", "--json"], ["serve", "status", "--json"], ["serve", "stop", "--json"],
    ["hooks", "--json"], ["codex", "--json"], ["session", "open", "--json"],
    ["session", "close", "--json"], ["observe", "", "--json"], ["adapters", "--json", "list"],
    ["evolve", "--json"], ["disable", "--json"], ["enable", "--json"],
    ["clear-cache", "--json"], ["reconfigure", "--json"],
    ["diagnostics", "reliability", "--json"], ["rotate-token", "--json"],
    ["token", "show", "--json"], ["remote", "enable", "--listen", "127.0.0.1:9443", "--json"],
    ["remote", "disable", "--json"], ["remote", "keys", "add", "laptop", "--json"],
    ["remote", "keys", "revoke", "nobody", "--json"], ["help-optimize", "--json"],
    ["team", "status", "--json"], ["team", "policy", "--require-login", "off", "--json"],
    ["team", "admin", "ana", "--json"], ["team", "reset-password", "ana", "--json"],
    ["signing", "resign", "--json"],
]


@pytest.mark.parametrize("argv", NEW_COMMANDS, ids=lambda a: " ".join(x for x in a if x))
def test_new_command_answers_with_one_json_object(no_daemon, capsys, argv):
    rc, out, _ = run(argv, capsys)
    doc = one_document(out)
    assert rc in (0, 1, 2)
    assert (rc == 0) == (doc["success"] is True), (rc, doc)


def test_new_commands_cover_every_leaf_that_lacked_json():
    covered = {" ".join(w for w in a if not w.startswith("-") and w)
               for a in NEW_COMMANDS}
    before = {"init", "setup", "provider", "upgrade-hosts", "migrate", "db migrate",
              "db scale", "db regraph", "db compact", "mesh", "help", "wrap", "mcp", "tool",
              "flags set", "flags kill", "flags rollback-retirements", "warmup", "dashboard",
              "serve", "hooks", "codex", "session open", "session close", "observe",
              "adapters", "evolve", "disable", "enable", "clear-cache", "reconfigure",
              "diagnostics reliability", "rotate-token", "token show", "remote tls init",
              "remote enable", "remote disable", "remote keys add", "remote keys revoke",
              "help-optimize", "team status", "team policy", "team admin",
              "team reset-password", "signing resign"}
    # db regraph, warmup and remote tls init have their own tests below.
    missing = {b for b in before if not any(c == b or c.startswith(b + " ") for c in covered)}
    assert missing == {"db regraph", "warmup", "remote tls init"}


def test_warmup_json_reports_ready(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import daemon, warmup_readiness

    monkeypatch.setattr(daemon, "is_daemon_running", lambda *a, **k: True)
    monkeypatch.setattr(warmup_readiness, "wait_until_warm", lambda fetch, timeout, **k:
                        SimpleNamespace(ready=True, reachable=True, detail="",
                                        waited_seconds=0.0))
    rc, out, _ = run(["warmup", "--json", "--timeout", "0"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["ready"] is True and doc["data"]["via"] == "daemon"


def test_warmup_json_not_ready_exits_1(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import daemon, warmup_readiness

    monkeypatch.setattr(daemon, "is_daemon_running", lambda *a, **k: True)
    monkeypatch.setattr(warmup_readiness, "wait_until_warm", lambda fetch, timeout, **k:
                        SimpleNamespace(ready=False, reachable=True, detail="loading",
                                        waited_seconds=0.0))
    rc, out, _ = run(["--json", "warmup", "--timeout", "0"], capsys)
    doc = one_document(out)
    assert rc == 1 and doc["error"]["code"] == "not_ready"


def test_db_regraph_json(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import commands

    monkeypatch.setattr(commands, "_cmd_db_regraph", lambda _a: print("graph copy: in step"))
    rc, out, _ = run(["db", "regraph", "--check", "--json"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["output"] == "graph copy: in step"


def test_remote_tls_init_json(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import remote_commands

    monkeypatch.setattr(remote_commands, "tls_init", lambda *a: {
        "server_cert": "/x/server.pem", "ca": "/x/ca.pem", "ca_sha256": "ab", "expires": "2027"})
    rc, out, _ = run(["remote", "tls", "init", "--name", "h", "--json"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["ca"] == "/x/ca.pem"


def test_dashboard_json_starts_nothing(no_daemon, capsys):
    rc, out, _ = run(["--json", "dashboard"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["opened_browser"] is False and doc["data"]["running"] is False
    assert no_daemon == []  # ensure_daemon was never called


def test_wrap_json_never_launches(no_daemon, capsys, monkeypatch):
    from superlocalmemory.optimize.adapters import wrap

    monkeypatch.setattr(wrap, "wrap_agent", lambda *a, **k: pytest.fail("agent launched"))
    rc, out, _ = run(["wrap", "--json", "claude", "-p", "hi"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["data"]["launched"] is False
    assert doc["data"]["agent_args"] == ["-p", "hi"]


# -- the normaliser, for commands that already printed JSON or text -------------------------


def test_normalize_adds_contract_keys_without_renaming():
    doc = json_mode.normalize("ops list", 0, json.dumps({"failures": [], "command": "x"},
                                                       indent=2))
    assert doc["success"] is True and doc["failures"] == []
    assert doc["command"] == "x"  # an existing key is never overwritten


def test_normalize_wraps_text_and_failures():
    ok = json_mode.normalize("enable", 0, "SLM enabled.\n")
    assert ok["data"] == {"output": "SLM enabled."}
    bad = json_mode.normalize("mesh", 1, "", "Mesh: cannot reach the daemon\n")
    assert bad["success"] is False and bad["error"]["code"] == "command_failed"
    assert bad["error"]["message"] == "Mesh: cannot reach the daemon"


def test_normalize_moves_stray_text_off_stdout(capsys):
    doc = json_mode.normalize("x", 0, 'warming up\n{"a": 1}\n')
    assert doc["a"] == 1 and "warming up" in capsys.readouterr().err


def test_ops_list_json_gains_success_additively(no_daemon, capsys, monkeypatch):
    from superlocalmemory.cli import ops_cmd

    monkeypatch.setattr(ops_cmd, "cmd_ops", lambda _a: print(json.dumps(
        {"profile": None, "failures": []}, indent=2)))
    rc, out, _ = run(["ops", "list", "--json"], capsys)
    doc = one_document(out)
    assert rc == 0 and doc["failures"] == [] and doc["profile"] is None
