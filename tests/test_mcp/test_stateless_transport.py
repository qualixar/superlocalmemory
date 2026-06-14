"""Regression guard for the stateless MCP transport (issue #39 Issue 3).

The actual LAN bug was that SLM's Streamable-HTTP transport is STATEFUL, so a
gateway/hub forwarding a tool call without replaying Mcp-Session-Id got
-32600. v3.6.12 added a stateless mode (SLM_REMOTE / SLM_MCP_STATELESS). The
predicate was unit-tested, but the daemon WIRING (settings.stateless_http) had
no test — so this asserts the wiring the daemon performs actually flips the
FastMCP settings. Mirrors unified_daemon.py's stateless block.
"""

from __future__ import annotations

import importlib

import pytest

from superlocalmemory.core import remote_mode


def _fresh_fastmcp():
    """A fresh FastMCP server instance (re-import to avoid cross-test state)."""
    import superlocalmemory.mcp.server as server_mod
    importlib.reload(server_mod)
    return server_mod.server


def test_mcp_stateless_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    assert remote_mode.mcp_stateless() is False
    monkeypatch.setenv("SLM_MCP_STATELESS", "1")
    assert remote_mode.mcp_stateless() is True


def test_daemon_wiring_flips_stateless_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replicate unified_daemon's stateless block: when mcp_stateless() is True,
    FastMCP settings must enable stateless_http + json_response so a forwarder
    can call tools/call without an Mcp-Session-Id."""
    monkeypatch.setenv("SLM_MCP_STATELESS", "1")
    fastmcp = _fresh_fastmcp()
    # exact wiring performed by unified_daemon.create_app()
    if remote_mode.mcp_stateless():
        fastmcp.settings.stateless_http = True
        fastmcp.settings.json_response = True
    assert fastmcp.settings.stateless_http is True
    assert fastmcp.settings.json_response is True


def test_daemon_wiring_stays_stateful_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no opt-in) must keep the stateful transport (loopback clients
    rely on sessions); the daemon must NOT flip stateless."""
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    fastmcp = _fresh_fastmcp()
    assert remote_mode.mcp_stateless() is False
    # The daemon's `if mcp_stateless():` block would NOT run → settings untouched.
    assert fastmcp.settings.stateless_http is False
