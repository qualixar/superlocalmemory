# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP `session_init` tool.

The hot path should do one fast recall through the pool adapter and build
both the formatted context and structured memory list from that response.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from superlocalmemory.mcp._pool_adapter import (
    PoolFact,
    PoolRecallItem,
    PoolRecallResponse,
)


class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


def _get_session_init_tool():
    """Register active tools on a mock server and return session_init."""
    from superlocalmemory.mcp.tools_active import register_active_tools

    srv = _MockServer()
    get_engine = MagicMock()
    register_active_tools(srv, get_engine)
    return srv._tools["session_init"], get_engine


def _make_engine_mock(profile_id="default", feedback_count=10):
    """Build a MemoryEngine mock with adaptive_learner."""
    engine = MagicMock()
    engine.profile_id = profile_id
    engine._adaptive_learner.get_feedback_count.return_value = feedback_count
    return engine


def _make_rules_mock(should_recall=True, threshold=0.3):
    """Build a RulesEngine mock."""
    rules = MagicMock()
    rules.should_recall.return_value = should_recall
    rules.get_recall_config.return_value = {
        "enabled": True,
        "relevance_threshold": threshold,
        "max_memories_injected": 10,
    }
    return rules


def _make_response(count: int = 2) -> PoolRecallResponse:
    """Build a typed pool recall response."""
    return PoolRecallResponse(results=[
        PoolRecallItem(
            fact=PoolFact(
                fact_id=f"f-{i}",
                content=f"memory {i} content",
                memory_id=f"m-{i}",
            ),
            score=0.9 - (i * 0.1),
        )
        for i in range(count)
    ])


class TestSessionInitTool:
    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_returns_context_and_memories_from_one_fast_recall(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock()
        rules = _make_rules_mock()
        MockRulesEngine.return_value = rules
        mock_pool_recall.return_value = _make_response(2)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init(project_path="/my/project"))

        assert result["success"] is True
        assert "memory 0 content" in result["context"]
        assert result["memory_count"] == 2
        assert len(result["memories"]) == 2
        mock_pool_recall.assert_called_once_with(
            "project context /my/project", limit=10, fast=None,
        )

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_uses_query_override(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock()
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(1)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init(query="what is Q-CLAW"))

        mock_pool_recall.assert_called_once_with(
            "what is Q-CLAW", limit=10, fast=None,
        )

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_returns_learning_status(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock(feedback_count=75)
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(0)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["success"] is True
        learning = result["learning"]
        assert learning["feedback_signals"] == 75
        assert learning["phase"] == 2
        assert learning["status"] == "learning"

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_respects_max_results(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock()
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(20)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init(max_results=3))

        assert result["success"] is True
        assert len(result["memories"]) == 3
        mock_pool_recall.assert_called_once_with(
            "recent important decisions", limit=3, fast=None,
        )

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_filters_by_relevance_threshold(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock()
        MockRulesEngine.return_value = _make_rules_mock(threshold=0.85)
        mock_pool_recall.return_value = _make_response(2)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["memory_count"] == 1
        assert result["memories"][0]["fact_id"] == "f-0"


class TestSessionInitGating:
    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    def test_session_init_disabled_by_rules(
        self, MockRulesEngine, mock_register, mock_emit,
    ):
        engine = _make_engine_mock()
        MockRulesEngine.return_value = _make_rules_mock(should_recall=False)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["success"] is True
        assert result["context"] == ""
        assert result["memories"] == []
        assert "disabled" in result["message"].lower()


class TestSessionInitIntegration:
    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.mcp._pool_adapter.pool_recall")
    def test_session_init_registers_agent_from_env(
        self, mock_pool_recall, MockRulesEngine, mock_register, mock_emit,
        monkeypatch,
    ):
        monkeypatch.setenv("SLM_AGENT_ID", "codex")
        engine = _make_engine_mock(profile_id="varun")
        MockRulesEngine.return_value = _make_rules_mock()
        mock_pool_recall.return_value = _make_response(1)

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init(project_path="/slm"))

        mock_register.assert_called_once_with("codex", "varun")
        mock_emit.assert_called_once()
        payload = mock_emit.call_args[0][1]
        assert payload["agent_id"] == "codex"
        assert payload["project_path"] == "/slm"
        assert payload["memory_count"] == 1

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    def test_session_init_error_returns_failure(self, mock_register, mock_emit):
        session_init, get_engine = _get_session_init_tool()
        get_engine.side_effect = RuntimeError("engine init failed")

        result = asyncio.run(session_init())

        assert result["success"] is False
        assert "engine init failed" in result["error"]
