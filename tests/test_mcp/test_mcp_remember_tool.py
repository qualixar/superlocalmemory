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

@pytest.fixture(autouse=True)
def _daemon_offline(monkeypatch):
    """v3.5.5: MCP remember now routes through the daemon (write-through) when
    available, falling back to pending.db only when the daemon is offline.
    These tests validate the pending fallback, so force daemon-offline."""
    import superlocalmemory.cli.daemon as _d
    monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: False)


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
    def test_remember_routes_to_canonical_worker(self, mock_emit):
        """Daemon-offline remember uses the capability-owned worker."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["fact-42"],
            "count": 1,
        }

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(
                remember("important fact", tags="python", project="slm")
            )

        pool.store.assert_called_once()
        call_args = pool.store.call_args
        assert call_args.args[0] == "important fact"
        assert call_args.args[1]["tags"] == "python"
        assert call_args.args[1]["project"] == "slm"
        assert call_args.args[1]["idempotency_key"].startswith("mcp:")
        assert result["success"] is True
        assert result["pending"] is False
        assert result["pending_id"] is None
        assert result["fact_ids"] == ["fact-42"]

    def test_remember_sends_metadata_to_canonical_worker(self):
        """Offline canonical ingestion preserves untrusted source metadata."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["fact-meta"],
            "count": 1,
        }

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(remember(
                "meta test content for canonical store",
                tags="ai,ml", project="qclaw",
                importance=9, agent_id="test-agent",
            ))

        assert result["success"] is True
        assert result.get("pending") is False
        metadata = pool.store.call_args.args[1]
        assert metadata["agent_id"] == "test-agent"
        assert metadata["project"] == "qclaw"
        assert metadata["importance"] == 9


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestRememberEdgeCases:
    """Edge case handling for the remember tool."""

    def test_remember_empty_content_handled(self):
        """Empty string rejection is returned without raw staging."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": [], "count": 0}
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(remember(""))
        assert result["success"] is True

    def test_remember_worker_pool_exception_fails_without_raw_persistence(self):
        """Worker failure is explicit; it cannot bypass write authorization."""
        remember = _get_remember_tool()

        with patch(
            "superlocalmemory.core.worker_pool.WorkerPool.shared",
            side_effect=RuntimeError("worker crashed"),
        ):
            result = asyncio.run(remember("boom"))

        assert result["success"] is False
        assert "worker crashed" in result["error"]

    def test_remember_agent_id_is_untrusted_worker_metadata(self):
        """Caller agent ID is audit metadata, not the trusted actor."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.return_value = {"ok": True, "fact_ids": ["fact-a"], "count": 1}
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(remember("agent test", agent_id="claude-opus"))
        assert result["success"] is True
        assert pool.store.call_args.args[1]["agent_id"] == "claude-opus"


class TestRememberWriteThrough:
    """v3.5.5: when the daemon is up, remember routes through it (write-through)."""

    def test_remember_routes_through_daemon_when_online(self, monkeypatch):
        import superlocalmemory.cli.daemon as _d
        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(
            _d, "daemon_request",
            lambda method, path, body=None: {
                "ok": True, "fact_ids": ["abc123"], "count": 1, "status": "stored",
            },
        )
        remember = _get_remember_tool()
        result = asyncio.run(remember("write-through fact", tags="t"))
        assert result["success"] is True
        assert result["fact_ids"] == ["abc123"]
        assert result["pending"] is False
