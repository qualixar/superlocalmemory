# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Hermes plugin, ``connection: remote``: Hermes's MCP client only, and no silent loss."""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import pathlib
import re
import socket
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PLUGIN = REPO / "hermes-plugin"


def _load():
    spec = importlib.util.spec_from_file_location("slm_hermes_remote_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _release(module) -> str:
    return ".".join(str(p) for p in module._RELEASE_SLM_VERSION)


class Ctx:
    """A Hermes plugin context: settings plus a scripted ``call_mcp``."""

    def __init__(self, module, connection="remote", answers=None, settings=None):
        self.module = module
        self.settings = {"connection": connection, **(settings or {})}
        self.calls: list[tuple[str, str, dict, float]] = []
        self.answers = answers or {}
        self.skills: dict = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def call_mcp(self, server, tool, arguments=None, timeout=30):
        self.calls.append((server, tool, dict(arguments or {}), timeout))
        answer = self.answers.get(tool)
        if callable(answer):
            return answer(arguments)
        if answer is not None:
            return answer
        if tool == "get_status":
            return {"ok": True, "result": json.dumps({"success": True,
                                                      "version": _release(self.module)})}
        return {"ok": True, "result": json.dumps({"success": True, "fact_ids": ["f1"]})}

    def register_skill(self, name, path):
        self.skills[name] = path

    def register_hook(self, *a):
        pass

    def register_command(self, *a):
        pass

    def register_tool(self, *a, **k):
        pass


@pytest.fixture()
def module():
    return _load()


# -- opt-in ------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "local", "Remote", "REMOTE", "yes", True, "remote ",
                                   "https://evil.example/mcp"])
def test_remote_mode_is_an_explicit_opt_in(module, value, monkeypatch) -> None:
    ctx = Ctx(module, connection=value)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    out = module.SlmHermesPlugin(ctx).slash_router("status")
    assert out.startswith("SLM CLI is unavailable on this machine")
    assert "settings.connection: remote" in out
    assert ctx.calls == []


def test_unreadable_settings_fall_back_to_local_and_all_skills(module) -> None:
    class NoConfig(Ctx):
        def get_config(self, key, default=None):
            raise RuntimeError("config not readable yet")

    ctx = NoConfig(module)
    module.register(ctx)
    inventory = json.loads((PLUGIN / "command-inventory.json").read_text(encoding="utf-8"))
    assert set(ctx.skills) == set(inventory["skills"])


def test_remote_mode_registers_only_remote_capable_skills(module) -> None:
    ctx = Ctx(module)
    module.register(ctx)
    assert set(ctx.skills) == {"slm-cache", "slm-compress", "slm-recall", "slm-remember",
                               "slm-scope", "slm-session", "slm-status"}


# -- no sockets, no programs ------------------------------------------------------------------


def test_remote_mode_never_opens_sockets_or_subprocesses(module, monkeypatch) -> None:
    def forbidden(*a, **k):
        raise AssertionError("remote mode touched the network or ran a program")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(module.shutil, "which", forbidden)
    ctx = Ctx(module)
    plugin = module.SlmHermesPlugin(ctx)
    argv_for = {"status": "", "health": "", "recall": "a question", "remember": "a fact",
                "list": "", "delete": "f1 CONFIRM", "update": "f1 new text", "summary": "",
                "trace": "a question", "kinds": "", "help": ""}
    assert set(argv_for) == set(module._REMOTE.REMOTE_COMMANDS)
    for command, rest in argv_for.items():
        plugin.slash_router(f"{command} {rest}".strip())
    plugin.on_session_start(session_id="s1")
    plugin.pre_llm_call(session_id="s1", user_message="what did we decide?")
    plugin.post_tool_call(session_id="s1", tool_name="t", args={}, result="ok")
    assert ctx.calls and all(server == "superlocalmemory" for server, *_ in ctx.calls)


def test_crit_ssrf_a_url_in_plugin_settings_is_never_used(module, monkeypatch) -> None:
    """CRIT 3: the plugin has no URL setting; whatever a config says, the only
    destination is Hermes's configured superlocalmemory MCP server."""
    def forbidden(*a, **k):
        raise AssertionError("network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    ctx = Ctx(module, settings={"url": "http://169.254.169.254/latest/meta-data",
                                "remote_url": "http://127.0.0.1:8765/internal/token",
                                "server": "evil", "mcp_server": "evil"})
    plugin = module.SlmHermesPlugin(ctx)
    plugin.slash_router("remember a fact")
    plugin.slash_router("recall something")
    assert {server for server, *_ in ctx.calls} == {"superlocalmemory"}
    read_keys = set()
    original = ctx.get_config
    ctx.get_config = lambda key, default=None: (read_keys.add(key), original(key, default))[1]
    plugin.slash_router("status")
    plugin.post_llm_call(session_id="s", user_message="u", assistant_response="a")
    assert read_keys <= {"connection", "capture_turns"}


def test_plugin_source_has_no_network_stack_or_local_secrets() -> None:
    for name in ("remote.py", "__init__.py"):
        source = (PLUGIN / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in getattr(node, "names", [])}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & {"socket", "urllib", "http", "httpx", "requests", "aiohttp",
                               "ssl", "superlocalmemory"}, (name, imported)
        assert ".install_token" not in source and "X-SLM-Hook-Token" not in source
        assert "SLM_DAEMON_URL" not in source
    remote_source = (PLUGIN / "remote.py").read_text(encoding="utf-8")
    assert not re.search(r"https?://", remote_source)
    assert "subprocess" not in remote_source and "shutil" not in remote_source


def test_the_product_never_reads_slm_daemon_url() -> None:
    """#142: no process-wide variable reroutes memory traffic."""
    for path in (REPO / "src").rglob("*.py"):
        assert "SLM_DAEMON_URL" not in path.read_text(encoding="utf-8"), path


# -- NOT SAVED -------------------------------------------------------------------------


def test_remote_down_remember_reports_not_saved(module) -> None:
    ctx = Ctx(module, answers={"remember": {"ok": False, "error": "connection refused"}})
    out = module.SlmHermesPlugin(ctx).slash_router("remember the deploy key rotates monthly")
    assert out.startswith("NOT SAVED"), out
    assert "connection refused" in out


def test_server_unreachable_before_remember_is_not_saved(module) -> None:
    ctx = Ctx(module, answers={"get_status": {"ok": False, "error": "timed out"}})
    out = module.SlmHermesPlugin(ctx).slash_router("remember x")
    assert out.startswith("NOT SAVED")
    assert [tool for _, tool, *_ in ctx.calls] == ["get_status"]


def test_a_transport_exception_is_not_saved(module) -> None:
    def boom(_args):
        raise TimeoutError("read timed out")

    out = module.SlmHermesPlugin(Ctx(module, answers={"remember": boom})).slash_router("remember x")
    assert out.startswith("NOT SAVED") and "read timed out" in out


def test_missing_mcp_grant_is_not_saved_with_guidance(module) -> None:
    def denied(_args):
        raise PermissionError("not allowed")

    out = module.SlmHermesPlugin(Ctx(module, answers={"remember": denied})).slash_router("remember x")
    assert out.startswith("NOT SAVED") and "mcp_allowlist" in out


@pytest.mark.parametrize("result", [
    {"success": False, "error": "disk full"},
    {"fact_ids": ["f1"]},
    {"success": "true"},
    "Saved!",
    None,
    [],
])
def test_anything_but_success_true_is_not_saved(module, result) -> None:
    answer = {"ok": True, "result": json.dumps(result) if not isinstance(result, str) else result}
    out = module.SlmHermesPlugin(Ctx(module, answers={"remember": answer})).slash_router(
        "remember x")
    assert out.startswith("NOT SAVED"), (result, out)


def test_policy_refusal_from_a_read_only_key_is_not_saved(module) -> None:
    refusal = {"ok": False, "error": "'remember' changes memory, and remote key 'viewer' is "
               "read-only. [remote_key_read_only]"}
    plugin = module.SlmHermesPlugin(Ctx(module, answers={"remember": refusal}))
    out = plugin.slash_router("remember x")
    assert out.startswith("NOT SAVED") and "read-only" in out


def test_pending_write_is_reported_honestly(module) -> None:
    answer = {"ok": True, "result": json.dumps({"success": True, "fact_ids": ["a", "b"],
                                                "pending": True})}
    out = module.SlmHermesPlugin(Ctx(module, answers={"remember": answer})).slash_router(
        "remember x")
    assert out.startswith("Saved (2 fact(s); queryable now")


def test_remember_sends_content_as_written_and_the_idempotency_key_is_stable(module) -> None:
    ctx = Ctx(module)
    plugin = module.SlmHermesPlugin(ctx)
    secret_like = "token=abc123secretvalue sk-ABCDEFGHIJKLMNOPQRST"
    plugin.slash_router(f"remember '{secret_like}' --tags ops")
    plugin.slash_router(f"remember '{secret_like}' --tags ops")
    plugin.slash_router(f"remember '{secret_like}' --tags other")
    sent = [args for _, tool, args, _ in ctx.calls if tool == "remember"]
    assert sent[0]["content"] == secret_like
    assert sent[0]["idempotency_key"] == sent[1]["idempotency_key"]
    assert sent[0]["idempotency_key"] != sent[2]["idempotency_key"]
    assert len(sent[0]["idempotency_key"]) == 32


def test_version_mismatch_refuses_commands(module) -> None:
    ctx = Ctx(module, answers={"get_status": {"ok": True, "result": json.dumps(
        {"success": True, "version": "0.0.1"})}})
    plugin = module.SlmHermesPlugin(ctx)
    assert "requires exactly SLM" in plugin.slash_router("recall x")
    assert plugin.slash_router("remember x").startswith("NOT SAVED")
    assert [tool for _, tool, *_ in ctx.calls] == ["get_status"]  # cached, then refused


# -- command surface ---------------------------------------------------------------------


@pytest.mark.parametrize("command", ["serve", "backup", "remote", "profile", "mode", "forget",
                                     "gdpr", "restart", "mesh", "loop", "connect"])
def test_host_only_commands_are_refused_with_guidance(module, command) -> None:
    ctx = Ctx(module)
    out = module.SlmHermesPlugin(ctx).slash_router(f"{command} x CONFIRM")
    assert "manages the SLM computer" in out and f"slm {command}" in out
    assert ctx.calls == []


def test_high_impact_still_needs_confirm_in_remote_mode(module) -> None:
    ctx = Ctx(module)
    plugin = module.SlmHermesPlugin(ctx)
    assert plugin.slash_router("delete f1").startswith("Preview required")
    assert ctx.calls == []
    assert plugin.slash_router("remember x --replaces f1").startswith("Preview required")
    assert plugin.slash_router("delete f1 CONFIRM") == "Done."


def test_search_alias_maps_to_recall_with_bounded_arguments(module) -> None:
    ctx = Ctx(module)
    plugin = module.SlmHermesPlugin(ctx)
    plugin.slash_router("search where is the runbook --limit 5")
    _, tool, args, timeout = ctx.calls[-1]
    assert tool == "recall" and args["query"] == "where is the runbook" and args["limit"] == 5
    assert "Usage error" in plugin.slash_router("recall x --limit 500")
    assert "Usage error" in plugin.slash_router("recall x --profile_id other")
    assert "Usage error" in plugin.slash_router("recall")


# -- automatic capture during an outage -----------------------------------------------------


def test_lifecycle_failures_are_counted_and_warned_once(module, caplog) -> None:
    caplog.set_level(logging.INFO, logger="superlocalmemory.hermes.remote")
    state = {"down": True}

    def maybe(_args):
        return {"ok": False, "error": "connection refused"} if state["down"] else \
            {"ok": True, "result": "{}"}

    ctx = Ctx(module, answers={"log_tool_event": maybe})
    plugin = module.SlmHermesPlugin(ctx)
    for _ in range(5):
        plugin.post_tool_call(session_id="s", tool_name="t", args={}, result="r")
    sent = [tool for _, tool, *_ in ctx.calls if tool == "log_tool_event"]
    assert len(sent) == 1, "the circuit should stop calls after the first failure"
    warnings = [r for r in caplog.records if "unreachable" in r.getMessage()]
    assert len(warnings) == 1
    status = plugin.slash_router("status")
    assert "failures since start: 1 " in status
    assert "lifecycle captures skipped while unreachable: 4" in status
    # Recovery: the circuit reopens after its window; success logs INFO once.
    state["down"] = False
    plugin._remote_health._open_until = 0.0
    plugin.post_tool_call(session_id="s", tool_name="t", args={}, result="r")
    assert any("reachable again" in r.getMessage() for r in caplog.records)


def test_a_read_only_key_stops_lifecycle_writes_after_one_refusal(module) -> None:
    refusal = {"ok": False, "error": "... read-only. [remote_key_read_only]"}
    ctx = Ctx(module, answers={"log_tool_event": refusal, "session_init": refusal})
    plugin = module.SlmHermesPlugin(ctx)
    plugin.on_session_start(session_id="s")
    for _ in range(4):
        plugin.post_tool_call(session_id="s", tool_name="t", args={}, result="r")
    writes = [tool for _, tool, *_ in ctx.calls if tool in ("log_tool_event", "session_init")]
    assert writes == ["session_init"]
    assert plugin.pre_llm_call(session_id="s", user_message="what?") is not None  # recall works
    assert "read-only key, captures not sent: 4" in plugin.slash_router("status")


def test_remote_recall_timeout_allows_for_the_network(module) -> None:
    ctx = Ctx(module)
    module.SlmHermesPlugin(ctx).pre_llm_call(session_id="s", user_message="q")
    assert [t for _, tool, _, t in ctx.calls if tool == "recall"] == [5]
    local = Ctx(module, connection="local")
    module.SlmHermesPlugin(local).pre_llm_call(session_id="s", user_message="q")
    assert [t for _, tool, _, t in local.calls if tool == "recall"] == [3]


def test_host_only_advisors_are_refused_remotely(module) -> None:
    plugin = module.SlmHermesPlugin(Ctx(module))
    for role in ("governance", "loop"):
        result = plugin.agent_tool(role=role, goal="g")
        assert result["ok"] is False and "SLM computer" in result["error"]
