"""Inside the daemon, MCP tools use the daemon's engine instead of building a second one.

Found live: a remote MCP get_status after a restart built a second MemoryEngine inside the
daemon, on the event loop (deferred migrations, every migration's verify()), so /health went
unanswered for minutes on a 32k-fact store.
"""

from __future__ import annotations

import pytest

import superlocalmemory.mcp.server as mcp_server
import superlocalmemory.server.unified_daemon as daemon


@pytest.fixture
def no_second_engine(monkeypatch):
    import superlocalmemory.core.engine as engine_mod

    def refuse(*_a, **_kw):
        raise AssertionError("the embedded MCP server must not build its own engine")

    monkeypatch.setattr(engine_mod, "MemoryEngine", refuse)
    monkeypatch.setattr(mcp_server, "_engine", None)
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")


def test_embedded_tools_get_the_resident_engine(no_second_engine, monkeypatch):
    resident = object()
    monkeypatch.setattr(daemon, "_engine", resident)
    assert mcp_server.get_engine() is resident


def test_a_mode_switch_engine_is_picked_up(no_second_engine, monkeypatch):
    first, second = object(), object()
    monkeypatch.setattr(daemon, "_engine", first)
    assert mcp_server.get_engine() is first
    monkeypatch.setattr(daemon, "_engine", second)
    assert mcp_server.get_engine() is second
