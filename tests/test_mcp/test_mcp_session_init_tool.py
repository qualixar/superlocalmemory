# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP `session_init` tool — Phase 0 Safety Net.

Covers:
    - Success path: returns context, memories, learning status
    - project_path and query forwarding
    - RulesEngine gating (should_recall=False -> empty)
    - max_results limiting
    - Agent registration and event emission
    - Error path: exception -> success=False

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self):
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


def _make_auto_recall_mock(context="# Context", memories=None):
    """Build an AutoRecall mock returning controlled data."""
    auto = MagicMock()
    auto.get_session_context.return_value = context
    auto.get_query_context.return_value = memories or []
    return auto


def _make_rules_mock(should_recall=True):
    """Build a RulesEngine mock."""
    rules = MagicMock()
    rules.should_recall.return_value = should_recall
    rules.get_recall_config.return_value = {
        "enabled": True,
        "relevance_threshold": 0.3,
        "max_memories_injected": 10,
    }
    return rules


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestSessionInitTool:
    """Core behavior of the session_init MCP tool."""

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.mcp.tools_active.RulesEngine", create=True)
    @patch("superlocalmemory.mcp.tools_active.AutoRecall", create=True)
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_returns_context(
        self, MockAutoRecall, MockRulesEngine,
        _ar_create, _re_create, mock_register, mock_emit,
    ):
        """session_init returns success=True with a context string."""
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock(context="# Relevant Memory Context")
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        with patch("superlocalmemory.hooks.auto_recall.AutoRecall", return_value=auto), \
             patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=rules):
            result = asyncio.run(session_init())

        assert result["success"] is True
        assert "context" in result

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_returns_memories(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """session_init returns a memories list."""
        memories = [
            {"fact_id": "f-1", "content": "decision X", "score": 0.9},
            {"fact_id": "f-2", "content": "bug fix Y", "score": 0.8},
        ]
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock(memories=memories)
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["success"] is True
        assert result["memory_count"] == 2
        assert len(result["memories"]) == 2

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_returns_learning_status(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """session_init returns learning.phase based on feedback count."""
        engine = _make_engine_mock(feedback_count=75)
        auto = _make_auto_recall_mock()
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["success"] is True
        learning = result["learning"]
        assert learning["feedback_signals"] == 75
        assert learning["phase"] == 2  # 50 <= 75 < 200 -> phase 2
        assert learning["status"] == "learning"

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_uses_project_path(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """project_path is forwarded to AutoRecall.get_session_context()."""
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock()
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init(project_path="/my/project"))

        auto.get_session_context.assert_called_once_with(
            project_path="/my/project", query="",
        )

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_uses_query_override(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """Explicit query param is forwarded to AutoRecall."""
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock()
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init(query="what is Q-CLAW"))

        auto.get_session_context.assert_called_once_with(
            project_path="", query="what is Q-CLAW",
        )


# ---------------------------------------------------------------------------
# Tests: gating
# ---------------------------------------------------------------------------

class TestSessionInitGating:
    """RulesEngine gating for session_init."""

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_disabled_by_rules(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """When should_recall returns False, response is empty context."""
        engine = _make_engine_mock()
        rules = _make_rules_mock(should_recall=False)
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init())

        assert result["success"] is True
        assert result["context"] == ""
        assert result["memories"] == []
        assert "disabled" in result["message"].lower()

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_respects_max_results(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """max_results limits how many memories are returned."""
        many_memories = [
            {"fact_id": f"f-{i}", "content": f"mem {i}", "score": 0.9}
            for i in range(20)
        ]
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock(memories=many_memories)
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        result = asyncio.run(session_init(max_results=3))

        assert result["success"] is True
        # Memories are sliced to max_results
        assert len(result["memories"]) <= 3


# ---------------------------------------------------------------------------
# Tests: integration-like (agent registration, events, errors)
# ---------------------------------------------------------------------------

class TestSessionInitIntegration:
    """Agent registration, event emission, and error handling."""

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_registers_agent(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """_register_agent is called with 'mcp_client' and the profile id."""
        engine = _make_engine_mock(profile_id="varun")
        auto = _make_auto_recall_mock()
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init())

        mock_register.assert_called_once_with("mcp_client", "varun")

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    @patch("superlocalmemory.hooks.rules_engine.RulesEngine")
    @patch("superlocalmemory.hooks.auto_recall.AutoRecall")
    def test_session_init_emits_agent_connected(
        self, MockAutoRecall, MockRulesEngine, mock_register, mock_emit,
    ):
        """Event 'agent.connected' is emitted with project_path."""
        engine = _make_engine_mock()
        auto = _make_auto_recall_mock()
        rules = _make_rules_mock()

        MockAutoRecall.return_value = auto
        MockRulesEngine.return_value = rules

        session_init, get_engine = _get_session_init_tool()
        get_engine.return_value = engine

        asyncio.run(session_init(project_path="/slm"))

        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "agent.connected"
        payload = args[0][1]
        assert payload["project_path"] == "/slm"
        assert payload["agent_id"] == "mcp_client"

    @patch("superlocalmemory.mcp.tools_active._emit_event")
    @patch("superlocalmemory.mcp.tools_active._register_agent")
    def test_session_init_error_returns_failure(self, mock_register, mock_emit):
        """When get_engine raises, tool returns success=False with error."""
        session_init, get_engine = _get_session_init_tool()
        get_engine.side_effect = RuntimeError("engine init failed")

        result = asyncio.run(session_init())

        assert result["success"] is False
        assert "engine init failed" in result["error"]
