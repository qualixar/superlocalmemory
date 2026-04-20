# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP `recall` tool — Phase 0 Safety Net.

Covers:
    - Success path: recall returns results list
    - Failure path: pool error propagated
    - WorkerPool.shared().recall() called with query + limit
    - Event emission on success
    - Implicit feedback recording (_record_recall_hits)
    - Edge cases: empty query, limit forwarded, feedback failure non-blocking

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

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


def _get_recall_tool():
    """Register core tools on a mock server and return the recall function."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _MockServer()
    get_engine = MagicMock()
    register_core_tools(srv, get_engine)
    return srv._tools["recall"], get_engine


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestRecallTool:
    """Core behavior of the recall MCP tool."""

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_success_returns_results(self, mock_emit, mock_record):
        """Successful recall returns success=True with results list."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True,
            "results": [
                {"fact_id": "f-1", "content": "Python is great", "score": 0.9},
            ],
            "result_count": 1,
            "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            result = asyncio.run(recall("tell me about Python"))

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["count"] == 1
        assert result["query_type"] == "semantic"

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_failure_returns_error(self, mock_emit, mock_record):
        """When pool.recall returns ok=False, tool returns success=False."""
        pool = MagicMock()
        pool.recall.return_value = {"ok": False, "error": "Index corrupted"}

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            result = asyncio.run(recall("any query"))

        assert result["success"] is False
        assert "Index corrupted" in result["error"]

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_calls_worker_pool_recall(self, mock_emit, mock_record):
        """pool.recall() is called with the query and limit."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        # S9-DASH-10: registry lookup must return None in tests so the
        # final fallback ``mcp:<agent_id>`` is used. Without the patch
        # the test picks up a real live session from the CI/dev registry.
        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            asyncio.run(recall("architecture patterns", limit=5))

        # S9-DASH-02: recall now threads session_id to pool.recall so the
        # outcome-queue producer can correlate hook signals. Default
        # session_id when the caller doesn't supply one is ``mcp:<agent_id>``.
        pool.recall.assert_called_once_with(
            "architecture patterns", limit=5, session_id="mcp:mcp_client",
        )

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_emits_memory_recalled_event(self, mock_emit, mock_record):
        """On success, _emit_event('memory.recalled', ...) is called."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "fts",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool):
            asyncio.run(recall("event check"))

        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "memory.recalled"
        payload = args[0][1]
        assert "query" in payload
        assert payload["result_count"] == 0
        assert payload["query_type"] == "fts"

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_records_implicit_feedback(self, mock_emit):
        """_record_recall_hits is called with get_engine, query, and results."""
        pool = MagicMock()
        results_data = [{"fact_id": "f-10", "content": "x", "score": 0.8}]
        pool.recall.return_value = {
            "ok": True, "results": results_data, "result_count": 1,
            "query_type": "semantic",
        }

        recall, get_engine = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool), \
             patch("superlocalmemory.mcp.tools_core._record_recall_hits") as mock_record:
            asyncio.run(recall("feedback query"))

        mock_record.assert_called_once_with(get_engine, "feedback query", results_data)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestRecallEdgeCases:
    """Edge case handling for the recall tool."""

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_empty_query_handled(self, mock_emit, mock_record):
        """Empty string query does not crash the tool."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "unknown",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            result = asyncio.run(recall(""))

        assert result["success"] is True
        pool.recall.assert_called_once_with(
            "", limit=10, session_id="mcp:mcp_client",
        )

    @patch("superlocalmemory.mcp.tools_core._record_recall_hits")
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_limit_forwarded(self, mock_emit, mock_record):
        """Custom limit=5 is forwarded to pool.recall()."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            asyncio.run(recall("limit test", limit=5))

        pool.recall.assert_called_once_with(
            "limit test", limit=5, session_id="mcp:mcp_client",
        )

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_recall_feedback_failure_non_blocking(self, mock_emit):
        """If _record_recall_hits raises, recall still returns successfully."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True,
            "results": [{"fact_id": "f-err", "content": "x"}],
            "result_count": 1,
            "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool), \
             patch(
                 "superlocalmemory.mcp.tools_core._record_recall_hits",
                 side_effect=RuntimeError("feedback DB broken"),
             ):
            result = asyncio.run(recall("should still work"))

        # Recall must succeed even when feedback recording fails
        assert result["success"] is True
        assert result["count"] == 1
