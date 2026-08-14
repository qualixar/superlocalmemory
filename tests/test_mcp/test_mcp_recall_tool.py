# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP `recall` tool — Phase 0 Safety Net.

Covers:
    - Success path: recall returns results list
    - Failure path: pool error propagated
    - choose_pool().recall() called with query + limit + fast
    - Recall has no post-query persistence side effects
    - Edge cases: empty query and limit forwarding

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch):
    """Run asyncio.to_thread inline so these unit tests never spawn threads."""
    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _run_inline)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        # v3.4.26 Phase 1: tools now carry ToolAnnotations kwargs
        # (readOnlyHint, destructiveHint, idempotentHint). The mock
        # ignores them — behaviour tests don't need the metadata.
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

    def test_recall_success_returns_results(self):
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

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
            result = asyncio.run(recall("tell me about Python"))

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["count"] == 1
        assert result["query_type"] == "semantic"

    def test_recall_failure_returns_error(self):
        """When pool.recall returns ok=False, tool returns success=False."""
        pool = MagicMock()
        pool.recall.return_value = {"ok": False, "error": "Index corrupted"}

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
            result = asyncio.run(recall("any query"))

        assert result["success"] is False
        assert "Index corrupted" in result["error"]

    def test_recall_calls_pool_recall(self):
        """pool.recall() is called with the query and limit."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        # S9-DASH-10: registry lookup must return None in tests so the
        # final fallback ``mcp:<agent_id>`` is used. Without the patch
        # the test picks up a real live session from the CI/dev registry.
        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            asyncio.run(recall("architecture patterns", limit=5))

        # The response shape preserves session_id forwarding for transport
        # compatibility, but recall itself no longer schedules outcomes.
        # v3.8.2 client-driven agentic: the tool forwards fast=None when the
        # caller omits it, so the daemon resolves the configured default.
        pool.recall.assert_called_once_with(
            "architecture patterns", limit=5, session_id="mcp:mcp_client",
            fast=None, include_global=None, include_shared=None, window=None,
            as_of=None, known_as_of=None, valid_at=None, include_unknown=False,
        )

    def test_recall_forwards_fast_flag(self):
        """fast=True is forwarded to the selected pool implementation."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            asyncio.run(recall("architecture patterns", limit=5, fast=True))

        pool.recall.assert_called_once_with(
            "architecture patterns", limit=5, session_id="mcp:mcp_client",
            fast=True, include_global=None, include_shared=None, window=None,
            as_of=None, known_as_of=None, valid_at=None, include_unknown=False,
        )

    def test_recall_emits_no_persistent_event(self):
        """CQS: a successful recall cannot persist an event."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "fts",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool), \
             patch("superlocalmemory.mcp.tools_core._emit_event") as mock_emit:
            asyncio.run(recall("event check"))

        mock_emit.assert_not_called()

    def test_recall_has_no_implicit_feedback_writer(self):
        """CQS: learning occurs only through explicit feedback commands."""
        pool = MagicMock()
        results_data = [{"fact_id": "f-10", "content": "x", "score": 0.8}]
        pool.recall.return_value = {
            "ok": True, "results": results_data, "result_count": 1,
            "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
            asyncio.run(recall("feedback query"))

        assert not hasattr(
            __import__("superlocalmemory.mcp.tools_core", fromlist=["*"]),
            "_record_recall_hits",
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestRecallEdgeCases:
    """Edge case handling for the recall tool."""

    def test_recall_empty_query_handled(self):
        """Empty string query does not crash the tool."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "unknown",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            result = asyncio.run(recall(""))

        assert result["success"] is True
        # WP-02 D9: default limit is now CANONICAL_RECALL_LIMIT (20)
        from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT
        pool.recall.assert_called_once_with(
            "", limit=CANONICAL_RECALL_LIMIT, session_id="mcp:mcp_client", fast=None,
            include_global=None, include_shared=None, window=None,
            as_of=None, known_as_of=None, valid_at=None, include_unknown=False,
        )

    def test_recall_limit_forwarded(self):
        """Custom limit=5 is forwarded to pool.recall()."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True, "results": [], "result_count": 0, "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool), \
             patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None), \
             patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None):
            asyncio.run(recall("limit test", limit=5))

        pool.recall.assert_called_once_with(
            "limit test", limit=5, session_id="mcp:mcp_client", fast=None,
            include_global=None, include_shared=None, window=None,
            as_of=None, known_as_of=None, valid_at=None, include_unknown=False,
        )

    def test_recall_returns_even_when_no_implicit_feedback_exists(self):
        """No best-effort memory write is attached to recall success."""
        pool = MagicMock()
        pool.recall.return_value = {
            "ok": True,
            "results": [{"fact_id": "f-err", "content": "x"}],
            "result_count": 1,
            "query_type": "semantic",
        }

        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
            result = asyncio.run(recall("should still work"))

        assert result["success"] is True
        assert result["count"] == 1

    def test_recall_forwards_explicit_two_clock_boundaries(self):
        pool = MagicMock()
        pool.recall.return_value = {"ok": True, "results": [], "result_count": 0, "query_type": "factual"}
        recall, _ = _get_recall_tool()

        with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool):
            result = asyncio.run(recall(
                "historical state",
                known_as_of="2026-01-01T00:00:00Z",
                valid_at="2025-06-01T00:00:00Z",
                include_unknown=True,
            ))

        assert result["success"] is True
        assert pool.recall.call_args.kwargs["known_as_of"] == "2026-01-01T00:00:00+00:00"
        assert pool.recall.call_args.kwargs["valid_at"] == "2025-06-01T00:00:00+00:00"
        assert pool.recall.call_args.kwargs["include_unknown"] is True
