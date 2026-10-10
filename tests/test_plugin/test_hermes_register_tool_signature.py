"""Issue #155: Hermes evaluates each plugin tool's check_fn; a description passed in that slot breaks it.

Hermes' PluginContext.register_tool is (name, toolset, schema, handler, check_fn=None, requires_env=None,
is_async=False, description="", emoji="", override=False). The advisor tools passed their description as
the 5th positional argument, so Hermes called a string: "TypeError: 'str' object is not callable".
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


class HermesContext:
    """The real register_tool signature from hermes_cli/plugins.py; check_fn must be callable or None."""

    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}

    def register_tool(self, name, toolset, schema, handler, check_fn=None, requires_env=None,
                      is_async=False, description="", emoji="", override=False):
        assert check_fn is None or callable(check_fn), f"{name}: check_fn is {check_fn!r}"
        assert callable(handler), name
        self.tools[name] = {"description": description, "check_fn": check_fn}

    def register_command(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):  # any other hook the plugin may register
        return lambda *a, **k: None


@pytest.mark.parametrize("folder", ["plugin-src/hermes", "hermes-plugin"])
def test_every_tool_registers_with_hermes_real_signature(folder):
    spec = importlib.util.spec_from_file_location(f"slm_hermes_{folder.replace('/', '_')}",
                                                  REPO / folder / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = HermesContext()
    module.register(ctx)
    for name in ("slm_agent", "slm_agent_status", "slm_agent_cancel", "slm_agent_result"):
        assert name in ctx.tools, name
        assert ctx.tools[name]["description"], f"{name} lost its description"
        assert ctx.tools[name]["check_fn"] is None


@pytest.mark.parametrize("folder", ["hermes-plugin"])  # the built plugin carries the advisor prompts
def test_slash_agent_outside_a_conversation_says_what_to_do(folder, monkeypatch):
    """Issue #155 part 2: Hermes launches an advisor only inside a running agent turn."""
    import sys
    import types

    class LifecycleError(Exception):
        pass

    class Lifecycle:
        def launch(self, request):
            raise LifecycleError("No active Hermes parent session is available.")

    fake = types.ModuleType("agent.subagent_lifecycle")
    fake.SubagentLaunchRequest = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.subagent_lifecycle", fake)
    spec = importlib.util.spec_from_file_location(f"slm_hermes_agent_{folder.replace('/', '_')}",
                                                  REPO / folder / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = HermesContext()
    ctx.subagent_lifecycle = Lifecycle()
    plugin = module.SlmHermesPlugin(ctx)
    import json as _json

    out = _json.loads(plugin.slash_agent("memory tidy my recent notes"))
    assert out["ok"] is False
    assert "parent session" not in out["error"]
    assert "slm_agent" in out["error"] and "chat" in out["error"]
