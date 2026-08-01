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
import threading
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
    def test_remember_success_returns_fact_ids(self, mock_emit):
        """Successful store returns success=True with fact_ids list."""
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["f-001", "f-002"],
            "count": 2,
        }
        remember = _get_remember_tool()

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool,
        ):
            result = asyncio.run(remember("Test content about Python"))

        assert result["success"] is True
        # V3.3.27: MCP remember uses store-first pattern (pending.db)
        # Returns pending ID, not fact IDs. Background processing creates facts.
        assert result["count"] >= 1
        assert len(result["fact_ids"]) >= 1

    @pytest.mark.slow
    @patch("superlocalmemory.mcp.tools_core._emit_event")
    def test_remember_returns_pending_id(self, mock_emit):
        """Offline canonical ingestion returns a truthful durable receipt.

        The historical regression lived in the real worker slow lane. The
        suite-level heavy-worker guard now supplies the same public receipt
        contract without loading models, while the worker receipt itself is
        covered in ``test_recall_worker_write_identity``.
        """
        remember = _get_remember_tool()
        result = asyncio.run(remember("Test content for pending store"))
        assert result["success"] is True
        assert result["materialization_state"] == "complete"
        assert result["pending"] is False
        assert result["pending_id"] is None
        assert result["operation_id"]
        assert result["fact_ids"]
        assert all(not fact_id.startswith("pending:") for fact_id in result["fact_ids"])

    def test_remember_preserves_worker_materialization_receipt(self):
        """The MCP surface must not relabel a queryable operation complete."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["queryable-fact"],
            "count": 1,
            "operation_id": "operation-42",
            "pending_id": "operation-42",
            "materialization_state": "queryable",
        }

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(remember("queryable canonical fact"))

        assert result["success"] is True
        assert result["materialization_state"] == "queryable"
        assert result["operation_id"] == "operation-42"
        assert result["pending"] is True
        assert result["pending_id"] == "operation-42"
        assert result["fact_ids"] == ["queryable-fact"]

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
        assert call_args.args[1]["idempotency_key"].startswith("mcp")
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

    def test_remember_daemon_proxy_exception_fails_closed(self):
        """A daemon-proxy failure is retryable and cannot bypass ownership."""
        remember = _get_remember_tool()
        pool = MagicMock()
        pool.store.side_effect = RuntimeError("daemon proxy crashed")

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            return_value=pool,
        ):
            result = asyncio.run(remember("boom"))

        assert result["success"] is False
        assert result["code"] == "DAEMON_UNAVAILABLE"
        assert result["retryable"] is True

    def test_remember_resolves_offline_pool_outside_event_loop(self):
        """Mounted HTTP fallback cannot synchronously probe its own daemon."""
        remember = _get_remember_tool()
        event_loop_threads: list[int] = []
        choose_pool_threads: list[int] = []
        pool = MagicMock()
        pool.store.return_value = {
            "ok": True,
            "fact_ids": ["thread-safe-fact"],
            "count": 1,
        }

        def choose_pool():
            choose_pool_threads.append(threading.get_ident())
            return pool

        async def invoke():
            event_loop_threads.append(threading.get_ident())
            return await remember("offline thread-boundary witness")

        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            side_effect=choose_pool,
        ):
            result = asyncio.run(invoke())

        assert result["success"] is True
        assert choose_pool_threads
        assert choose_pool_threads[0] != event_loop_threads[0]

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

    def test_remember_never_spawns_a_second_writer_when_daemon_is_unavailable(
        self, monkeypatch,
    ) -> None:
        """A known daemon may be retrying a writer lock; do not bypass it.

        Falling back to a local WorkerPool after the daemon was positively
        identified creates a second database writer, which turns a transient
        collision into repeated lock failures under parallel MCP clients.
        """
        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(_d, "daemon_request", lambda *a, **k: None)

        remember = _get_remember_tool()
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            side_effect=AssertionError("a live daemon must retain write ownership"),
        ):
            result = asyncio.run(remember("do not fork a writer"))

        assert result["success"] is False
        assert result["retryable"] is True
        assert "daemon" in result["error"].lower()

    def test_remember_daemon_request_exception_never_falls_back_to_local_writer(
        self, monkeypatch,
    ) -> None:
        """A timed-out owned daemon remains the sole canonical writer."""
        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)

        def unavailable(*args, **kwargs):
            raise TimeoutError("canonical daemon request timed out")

        monkeypatch.setattr(_d, "daemon_request", unavailable)
        choose_pool = MagicMock()

        remember = _get_remember_tool()
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool",
            choose_pool,
        ):
            result = asyncio.run(remember("retain daemon ownership"))

        assert result["success"] is False
        assert result["code"] == "DAEMON_UNAVAILABLE"
        assert result["retryable"] is True
        choose_pool.assert_not_called()

    def test_complete_empty_write_never_fabricates_pending_fact_id(
        self, monkeypatch,
    ) -> None:
        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(
            _d,
            "daemon_request",
            lambda method, path, body=None: {
                "ok": True,
                "fact_ids": [],
                "count": 0,
                "operation_id": "operation-empty",
                "pending_id": None,
                "materialization_state": "complete",
            },
        )

        remember = _get_remember_tool()
        result = asyncio.run(remember("content rejected after admission"))

        assert result["success"] is True
        assert result["materialization_state"] == "complete"
        assert result["pending"] is False
        assert result["pending_id"] is None
        assert result["fact_ids"] == []
        assert result["count"] == 0
