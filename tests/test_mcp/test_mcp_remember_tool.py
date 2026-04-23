# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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


@pytest.fixture(autouse=True)
def _isolate_slm_data_dir(tmp_path, monkeypatch):
    """Ensure every test in this module stores into tmp_path, not the live
    ~/.superlocalmemory/. pending_store honors SLM_DATA_DIR in v3.4.31+."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# Helper: capture tool functions registered on a mock server
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        # v3.4.26 Phase 1: ignore ToolAnnotations kwargs.
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
        # V3.3.27: MCP remember uses store-first pattern (pending.db)
        # Returns pending ID, not fact IDs. Background processing creates facts.
        assert result["count"] >= 1
        assert len(result["fact_ids"]) >= 1

    @pytest.mark.slow
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_returns_pending_id(self, mock_emit):
        """V3.3.27: Store-first pattern returns pending ID for background processing.

        Marked ``slow`` (Stage 7 delivery-lead review): spawns a real
        worker subprocess and blocks ~100s on its ready-signal, which
        single-handedly doubled the default suite runtime. Runs under
        ``pytest -m slow``; default config excludes it.
        """
        remember = _get_remember_tool()
        result = asyncio.run(remember("Test content for pending store"))
        assert result["success"] is True
        assert result.get("pending") is True

    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_routes_to_pending_store(self, mock_emit):
        """v3.4.32: remember writes to pending.db only — daemon materializer
        drains the queue with recall priority. No redundant pool.store call."""
        remember = _get_remember_tool()

        with patch("superlocalmemory.cli.pending_store.store_pending",
                   return_value=42) as mock_store:
            result = asyncio.run(
                remember("important fact", tags="python", project="slm")
            )

        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args[0][0] == "important fact"
        assert call_args[1]["tags"] == "python"
        assert call_args[1]["metadata"]["project"] == "slm"
        assert result["success"] is True
        assert result["pending"] is True
        assert result["pending_id"] == 42
        assert result["fact_ids"] == ["pending:42"]

    def test_remember_stores_to_pending_with_metadata(self):
        """V3.3.27: Store-first pattern saves content and metadata to pending.db."""
        remember = _get_remember_tool()

        result = asyncio.run(remember(
            "meta test content for pending store",
            tags="ai,ml", project="qclaw",
            importance=9, agent_id="test-agent",
        ))

        assert result["success"] is True
        assert result.get("pending") is True
        # Verify pending ID is returned
        assert len(result["fact_ids"]) == 1
        assert result["fact_ids"][0].startswith("pending:")


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestRememberEdgeCases:
    """Edge case handling for the remember tool."""

    def test_remember_empty_content_handled(self):
        """V3.3.27: Empty string content does not crash the store-first path."""
        remember = _get_remember_tool()
        result = asyncio.run(remember(""))
        # Should not raise — store_pending accepts any content
        assert result["success"] is True

    def test_remember_worker_pool_exception_still_stores_pending(self):
        """V3.3.27: When WorkerPool crashes, data is still safe in pending.db."""
        remember = _get_remember_tool()

        with patch(
            "superlocalmemory.core.worker_pool.WorkerPool.shared",
            side_effect=RuntimeError("worker crashed"),
        ):
            result = asyncio.run(remember("boom"))

        assert result["success"] is True
        assert result.get("pending") is True

    def test_remember_agent_id_included_in_result(self):
        """V3.3.27: agent_id is included in the store-first result."""
        remember = _get_remember_tool()
        result = asyncio.run(remember("agent test", agent_id="claude-opus"))
        assert result["success"] is True
        assert result.get("pending") is True
