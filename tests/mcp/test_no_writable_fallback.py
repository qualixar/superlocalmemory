"""Fail-closed contract for MCP clients when the owned daemon is absent."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock


class _Server:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def register(function):
            self.tools[function.__name__] = function
            return function
        return register


def _remember_tool():
    from superlocalmemory.mcp.tools_core import register_core_tools

    server = _Server()
    register_core_tools(server, MagicMock())
    return server.tools["remember"]


def test_mcp_remember_fails_closed_without_worker_or_writable_sqlite(
    monkeypatch,
) -> None:
    """An absent daemon is a retryable error, never an embedded writer."""
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.mcp import _daemon_proxy

    def forbidden(*args, **kwargs):
        raise AssertionError("MCP client must not construct a local writer")

    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.is_daemon_running", lambda: False,
    )
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.ensure_daemon", lambda: False,
    )
    monkeypatch.setattr(WorkerPool, "shared", forbidden)
    monkeypatch.setattr(MemoryEngine, "__init__", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    pool = _daemon_proxy.choose_pool()
    rejected = pool.store("the owned daemon is down")

    assert rejected == {
        "ok": False,
        "code": "DAEMON_UNAVAILABLE",
        "retryable": True,
        "error": "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later.",
    }

    result = asyncio.run(_remember_tool()("the owned daemon is down"))

    assert result == {
        "success": False,
        "code": "DAEMON_UNAVAILABLE",
        "retryable": True,
        "error": "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later.",
    }
