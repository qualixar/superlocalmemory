# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP `remember` tool — Phase 0 Safety Net.

Covers:
    - Success path: store returns fact_ids, count
    - Failure path: store error propagated
    - WorkerPool.shared().store() called with correct args
    - Event emission on success
    - Metadata forwarding (tags, project, importance, agent_id)
    - Edge cases: empty content, pool exception

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: capture tool functions registered on a mock server
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


def _get_remember_tool():
    """Register core tools on a mock server and return the remember function."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _MockServer()
    get_engine = MagicMock()
    register_core_tools(srv, get_engine)
    return srv._tools["remember"]


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestRememberTool:
    """Core behavior of the remember MCP tool."""

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    @patch("superlocalmemory.mcp.tools_core.WorkerPool", create=True)
    @patch("superlocalmemory.core.worker_pool.WorkerPool")
    def test_remember_success_returns_fact_ids(self, mock_wp_mod, _wp_create, mock_emit):
        """Successful store returns success=True with fact_ids list."""
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["f-001", "f-002"],
            "count": 2,
        }
        mock_wp_mod.shared.return_value = pool

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            result = asyncio.run(remember("Test content about Python"))

        assert result["success"] is True
        assert result["fact_ids"] == ["f-001", "f-002"]
        assert result["count"] == 2

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_failure_returns_error(self, mock_emit):
        """When pool.store returns ok=False, tool returns success=False with error."""
        pool = MagicMock()
        pool.store.return_value = {"ok": False, "error": "DB locked"}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            result = asyncio.run(remember("some content"))

        assert result["success"] is False
        assert "DB locked" in result["error"]

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_calls_worker_pool_store(self, mock_emit):
        """pool.store() is called with the content and metadata dict."""
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": ["f-x"], "count": 1}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            asyncio.run(remember("important fact", tags="python", project="slm"))

        pool.store.assert_called_once()
        call_args = pool.store.call_args
        assert call_args[0][0] == "important fact"
        meta = call_args[1]["metadata"] if "metadata" in call_args[1] else call_args[0][1]
        assert meta["tags"] == "python"
        assert meta["project"] == "slm"

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_emits_memory_created_event(self, mock_emit):
        """On success, _emit_event('memory.created', ...) is called."""
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": ["f-1"], "count": 1}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            asyncio.run(remember("event test content"))

        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "memory.created"
        payload = args[0][1]
        assert "content_preview" in payload
        assert payload["fact_count"] == 1

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_passes_metadata(self, mock_emit):
        """tags, project, importance are forwarded in the metadata dict."""
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": ["f-m"], "count": 1}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            asyncio.run(remember(
                "meta test", tags="ai,ml", project="qclaw",
                importance=9, agent_id="test-agent",
            ))

        meta = pool.store.call_args[1]["metadata"]
        assert meta["tags"] == "ai,ml"
        assert meta["project"] == "qclaw"
        assert meta["importance"] == 9
        assert meta["agent_id"] == "test-agent"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestRememberEdgeCases:
    """Edge case handling for the remember tool."""

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_empty_content_handled(self, mock_emit):
        """Empty string content does not crash the tool."""
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": [], "count": 0}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            result = asyncio.run(remember(""))

        # Should not raise; pool.store still called
        assert result["success"] is True
        pool.store.assert_called_once()

    def test_remember_worker_pool_exception(self):
        """When WorkerPool.shared() raises, tool returns error dict."""
        remember = _get_remember_tool()

        with patch(
            "superlocalmemory.core.worker_pool.WorkerPool.shared",
            side_effect=RuntimeError("worker crashed"),
        ):
            result = asyncio.run(remember("boom"))

        assert result["success"] is False
        assert "worker crashed" in result["error"]

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_agent_id_forwarded(self, mock_emit):
        """agent_id parameter is included in the metadata and event."""
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": ["f-a"], "count": 1}

        remember = _get_remember_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            asyncio.run(remember("agent test", agent_id="claude-opus"))

        meta = pool.store.call_args[1]["metadata"]
        assert meta["agent_id"] == "claude-opus"
        # Event also carries agent_id
        event_payload = mock_emit.call_args[0][1]
        assert event_payload["agent_id"] == "claude-opus"
