"""Contract tests for the generated native Hermes SLM plugin."""

from __future__ import annotations

import json
import importlib.util
import pathlib
import re
import subprocess
import sys
import types


REPO = pathlib.Path(__file__).resolve().parents[2]
PLUGIN = REPO / "hermes-plugin"


def _manifest() -> str:
    return (PLUGIN / "plugin.yaml").read_text(encoding="utf-8")


def _runtime() -> str:
    return (PLUGIN / "__init__.py").read_text(encoding="utf-8")


def test_native_hermes_plugin_declares_additive_contract() -> None:
    manifest = _manifest()
    assert "name: superlocalmemory" in manifest
    assert "manifest_version: 1" in manifest
    assert "api_version: 1" in manifest
    assert "python_dependencies: []" in manifest
    assert "superlocalmemory" in manifest


def test_plugin_registers_all_skills_agents_and_lifecycle_hooks() -> None:
    runtime = _runtime()
    for skill in json.loads((PLUGIN / "command-inventory.json").read_text())["skills"]:
        assert f'ctx.register_skill("{skill}"' in runtime
    for role in ("memory", "governance", "optimize", "loop"):
        assert role in runtime
    for hook in (
        "on_session_start", "pre_llm_call", "post_tool_call",
        "post_llm_call", "on_session_end", "on_session_finalize", "on_session_reset",
        "on_skill_lifecycle", "subagent_start", "subagent_stop",
    ):
        assert f'"{hook}"' in runtime
    assert "ctx.register_hook(hook, getattr(plugin, hook))" in runtime


def test_all_cli_commands_have_router_and_generated_slash_aliases() -> None:
    inventory = json.loads((PLUGIN / "command-inventory.json").read_text(encoding="utf-8"))
    commands = inventory["primary_commands"]
    assert len(commands) == 60
    assert len(set(commands)) == len(commands)
    runtime = _runtime()
    assert 'ctx.register_command("slm", plugin.slash_router' in runtime
    assert "for command in COMMANDS:" in runtime
    assert 'ctx.register_command(f"slm-{command}"' in runtime


def test_inventory_matches_the_real_slm_parser_tree() -> None:
    """A CLI addition cannot ship without Hermes reachability."""
    inventory = json.loads((PLUGIN / "command-inventory.json").read_text(encoding="utf-8"))
    result = subprocess.run(
        [str(REPO / ".venv" / "bin" / "slm"), "--help"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    match = re.search(r"\{([^}]+)\} \.\.\.", result.stdout)
    assert match, result.stdout[:500]
    parser_commands = set(match.group(1).split(","))
    expected = set(inventory["primary_commands"]) | set(inventory["aliases"])
    assert parser_commands == expected


def test_runtime_uses_allowlisted_mcp_and_never_imports_ambient_slm() -> None:
    runtime = _runtime()
    assert 'ctx.call_mcp("superlocalmemory"' in runtime
    assert "subprocess.run([binary, *argv]" in runtime
    assert "shell=True" not in runtime
    assert "import superlocalmemory" not in runtime
    assert "close_session" in runtime
    assert "session_init" in runtime


def test_lifecycle_payloads_use_explicit_session_identity_and_existing_mcp_shapes() -> None:
    spec = importlib.util.spec_from_file_location("slm_hermes_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Context:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def call_mcp(self, server: str, tool: str, arguments: dict, timeout: float = 0) -> dict:
            assert server == "superlocalmemory"
            self.calls.append((tool, arguments))
            return {"ok": True, "result": "one recalled fact"}

        def get_config(self, key: str, default: bool) -> bool:
            return default

    context = Context()
    plugin = module.SlmHermesPlugin(context)
    plugin.on_session_start(session_id="hermes-a", project_path="/repo")
    injected = plugin.pre_llm_call(session_id="hermes-a", user_message="What did we decide?")
    plugin.post_llm_call(session_id="hermes-a", user_message="Remember this", assistant_response="Done")
    plugin.on_session_end(session_id="hermes-a", turn_id="turn-1")

    assert injected and "UNTRUSTED SLM EVIDENCE" in injected["context"]
    names = [name for name, _ in context.calls]
    assert names == ["session_init", "recall", "settle_session_outcomes"]
    init = context.calls[0][1]
    assert set(init) == {"session_id", "agent_id", "project_path", "query"}
    settle = context.calls[2][1]
    assert settle == {"session_id": "hermes-a", "agent_id": "hermes"}
    plugin.on_session_finalize(session_id="hermes-a")
    assert context.calls[3][1] == {"session_id": "hermes-a", "agent_id": "hermes", "finalize": True}


def test_plugin_redacts_sensitive_hook_payloads_and_capture_is_explicit() -> None:
    spec = importlib.util.spec_from_file_location("slm_hermes_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Context:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def call_mcp(self, _server, tool, arguments, timeout=0):
            self.calls.append((tool, arguments))
            return {"ok": True}

        def get_config(self, key, default):
            return default

    context = Context()
    plugin = module.SlmHermesPlugin(context)
    canary = "sk-live-DO_NOT_STORE_1234567890"
    plugin.post_tool_call(session_id="s", tool_name="terminal", args={"token": canary, "email": "person@example.com"}, result=canary)
    plugin.post_llm_call(session_id="s", user_message=canary, assistant_response=canary)
    plugin.pre_llm_call(session_id="s", user_message=canary)
    assert [name for name, _ in context.calls] == ["log_tool_event", "recall"]
    payload = context.calls[0][1]
    assert canary not in repr(payload)
    assert "person@example.com" not in repr(payload)
    assert "[REDACTED" in payload["input_summary"]
    assert canary not in repr(context.calls[1][1])


def test_lifecycle_telemetry_uses_only_public_log_tool_event_fields() -> None:
    spec = importlib.util.spec_from_file_location("slm_hermes_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Context:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def call_mcp(self, _server, tool, arguments, timeout=0):
            self.calls.append((tool, arguments))
            return {"ok": True}

        def get_config(self, _key, default):
            return default

    plugin = module.SlmHermesPlugin(Context())
    plugin.post_tool_call(session_id="s", tool_name="terminal", tool_call_id="call-1", status="complete")
    plugin.on_skill_lifecycle(session_id="s", skill_name="slm-recall", status="complete")
    plugin.subagent_start(parent_session_id="s", child_role="leaf")
    plugin.subagent_stop(parent_session_id="s", child_role="leaf", child_status="SUCCEEDED", child_summary="ok")

    allowed = {
        "tool_name", "event_type", "input_summary", "output_summary", "duration_ms",
        "metadata", "session_id", "agent_id", "project_path",
    }
    for tool, payload in plugin.ctx.calls:
        if tool == "log_tool_event":
            assert set(payload) <= allowed


def test_register_uses_real_hermes_hook_and_tool_shapes() -> None:
    spec = importlib.util.spec_from_file_location("slm_hermes_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Context:
        def __init__(self):
            self.hooks, self.tools, self.commands, self.skills = {}, {}, {}, {}
        def register_hook(self, name, callback): self.hooks[name] = callback
        def register_tool(self, name, namespace, schema, callback, description): self.tools[name] = (namespace, schema, callback, description)
        def register_command(self, name, callback, description, arguments): self.commands[name] = callback
        def register_skill(self, name, path): self.skills[name] = path

    context = Context()
    module.register(context)
    assert "pre_tool_call" not in context.hooks
    assert "slm_agent_result" in context.tools
    assert context.tools["slm_agent_result"][1]["required"] == ["handle"]
    assert "slm-agent-result" in context.commands


def test_advisor_children_inherit_parent_dynamic_mcp_toolsets() -> None:
    """Hermes cannot name dynamic mcp-* toolsets in a narrowed child request."""
    runtime = _runtime()
    assert "allowed_toolsets=_ROLE_TOOLSETS[role]" not in runtime
    assert "inherits_parent_mcp" in runtime


def test_subagent_handle_round_trips_and_cli_refuses_stale_slm(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("slm_hermes_test", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Handle:
        def to_dict(self):
            return {"contract_version": 1, "subagent_id": "child", "parent_session_id": "s", "correlation_id": "c", "created_at": 1.0, "provider": None, "model": None, "role": "leaf", "depth": 1, "capability": "opaque"}

    class Lifecycle:
        def __init__(self): self.seen = None
        def status(self, handle): self.seen = handle; return {"state": "RUNNING"}
        def cancel(self, handle, reason): self.seen = handle; return {"accepted": True}
        def result(self, handle): self.seen = handle; return {"ready": True}

    class Context:
        def __init__(self): self.subagent_lifecycle = Lifecycle()
        def get_config(self, key, default): return default

    class HermesHandle:
        @classmethod
        def from_dict(cls, value):
            instance = cls()
            instance.__dict__.update(value)
            return instance

    agent_module = types.ModuleType("agent")
    lifecycle_module = types.ModuleType("agent.subagent_lifecycle")
    lifecycle_module.SubagentHandle = HermesHandle
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.subagent_lifecycle", lifecycle_module)

    context = Context()
    plugin = module.SlmHermesPlugin(context)
    handle = Handle().to_dict()
    assert plugin.agent_status_tool(handle)["state"] == "RUNNING"
    assert context.subagent_lifecycle.seen.subagent_id == "child"
    assert plugin.agent_cancel_tool(handle)["accepted"] is True
    assert plugin.agent_result_tool(handle)["ready"] is True
    monkeypatch.setattr(module.shutil, "which", lambda _: "/tmp/slm")
    monkeypatch.setattr(module, "_slm_version", lambda _: "4.1.11")
    assert "requires exactly SLM CLI 4.1.12" in plugin.slash_router("status")
    monkeypatch.setattr(module, "_slm_version", lambda _: "4.1.13")
    assert "requires exactly SLM CLI 4.1.12" in plugin.slash_router("status")
