# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""End-to-end registration contract for the MCP product surface.

MCP exposure and memory processing mode are independent controls:

* exposure selects which registered tools an MCP client can see;
* Mode A/B/C selects how the engine processes memory after dispatch.

These tests keep those contracts separate while exercising the real functions
stored by FastMCP, not copies of their implementations.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


class _StrictToolServer:
    """Minimal server that rejects a second registration of the same name."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        del args, kwargs

        def decorator(fn):
            if fn.__name__ in self.tools:
                raise AssertionError(f"duplicate MCP tool registration: {fn.__name__}")
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class _SandboxPool:
    """Deterministic canonical worker boundary used by registered tools."""

    def store(self, content: str, metadata: dict) -> dict:
        assert content
        assert metadata["idempotency_key"].startswith("mcp:")
        return {
            "ok": True,
            "fact_ids": ["fact-sandbox"],
            "count": 1,
            "operation_id": "operation-sandbox",
            "pending_id": None,
            "materialization_state": "complete",
        }

    def recall(self, query: str, **kwargs) -> dict:
        assert query
        assert kwargs["limit"] > 0
        return {
            "ok": True,
            "results": [],
            "result_count": 0,
            "query_type": "sandbox",
            "score_contract_version": "2",
            "calibration_status": "uncalibrated",
            "no_confident_match": True,
        }


def _fresh_server(monkeypatch: pytest.MonkeyPatch, profile: str = "core"):
    for key in ("SLM_MCP_ALL_TOOLS", "SLM_MCP_TOOLS", "SLM_MCP_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")
    monkeypatch.setenv("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    monkeypatch.setenv("SLM_MCP_MESH_TOOLS", "1")
    if profile:
        monkeypatch.setenv("SLM_MCP_PROFILE", profile)

    module_name = "superlocalmemory.mcp.server"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _register_every_tool(target) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_mesh import register_mesh_tools
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools
    from superlocalmemory.mcp.tools_loops import register_loop_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools

    get_engine = lambda: None  # noqa: E731 - registration never executes it
    register_core_tools(target, get_engine)
    register_v28_tools(target, get_engine)
    register_v3_tools(target, get_engine)
    register_active_tools(target, get_engine)
    register_v33_tools(target, get_engine)
    register_code_graph_tools(target, get_engine)
    register_mesh_tools(target, get_engine)
    register_learning_tools(target, get_engine)
    register_evolution_tools(target, get_engine)
    register_optimize_tools(target)
    register_loop_tools(target, get_engine)


@pytest.mark.parametrize(
    ("exposure", "profile", "expected_count"),
    (
        ("essential", "", 42),
        ("named-core", "core", 14),
        ("whole", "whole", 84),
    ),
)
def test_registration_exposure_is_exact_and_duplicate_free(
    monkeypatch: pytest.MonkeyPatch,
    exposure: str,
    profile: str,
    expected_count: int,
) -> None:
    """The default, named-profile and raw surfaces register exactly once."""
    mod = _fresh_server(monkeypatch, profile)
    strict = _StrictToolServer()

    if exposure == "essential":
        expected = mod._ESSENTIAL_TOOLS
        target = mod._FilteredServer(strict, expected)
    elif exposure == "named-core":
        expected = mod._PROFILE_DEFINITIONS["core"]
        target = mod._FilteredServer(strict, expected)
    else:
        expected = None
        target = strict

    _register_every_tool(target)

    assert len(strict.tools) == expected_count
    if expected is not None:
        assert set(strict.tools) == expected

    actual_names = [tool.name for tool in mod.server._tool_manager.list_tools()]
    assert len(actual_names) == expected_count
    assert len(actual_names) == len(set(actual_names))
    assert set(actual_names) == set(strict.tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("product_mode", ("A", "B", "C"))
async def test_core_registered_callables_work_in_each_product_mode(
    monkeypatch: pytest.MonkeyPatch,
    product_mode: str,
) -> None:
    """Remember, recall and session_init dispatch from the real core surface."""
    mod = _fresh_server(monkeypatch, "core")
    pool = _SandboxPool()
    engine = SimpleNamespace(
        profile_id="sandbox-profile",
        mode=product_mode,
        db=SimpleNamespace(get_pinned=lambda profile_id: []),
        _adaptive_learner=SimpleNamespace(get_feedback_count=lambda profile_id: 0),
    )
    mod._engine = engine

    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: False)
    monkeypatch.setattr("superlocalmemory.mcp._daemon_proxy.choose_pool", lambda: pool)
    monkeypatch.setattr("superlocalmemory.mcp.tools_core._emit_event", lambda *a, **k: None)
    monkeypatch.setattr("superlocalmemory.mcp.tools_active._emit_event", lambda *a, **k: None)
    from superlocalmemory.hooks.rules_engine import RulesEngine

    monkeypatch.setattr(RulesEngine, "should_recall", lambda self, event: True)
    monkeypatch.setattr(
        RulesEngine,
        "get_recall_config",
        lambda self: {"relevance_threshold": 0.3},
    )

    registered = {
        tool.name: tool.fn for tool in mod.server._tool_manager.list_tools()
    }
    assert {"remember", "recall", "session_init"} <= registered.keys()

    remembered = await registered["remember"]("sandbox memory")
    recalled = await registered["recall"]("sandbox query")
    session = await registered["session_init"](query="sandbox session")

    assert remembered == {
        "success": True,
        "fact_ids": ["fact-sandbox"],
        "count": 1,
        "pending": False,
        "pending_id": None,
        "operation_id": "operation-sandbox",
        "materialization_state": "complete",
        "message": "Stored through canonical local ingestion.",
    }
    assert recalled["success"] is True
    assert recalled["results"] == []
    assert recalled["no_confident_match"] is True
    assert session["success"] is True
    assert session["retrieval_mode"] == "hybrid_candidate_fusion"
    assert session["memory_count"] == 0
    assert session["session_id"].startswith("slm-")
