"""Regression guard for the stateless MCP transport (issue #39 Issue 3).

The actual LAN bug was that SLM's Streamable-HTTP transport is STATEFUL, so a
gateway/hub forwarding a tool call without replaying Mcp-Session-Id got
-32600. v3.6.12 added a stateless mode (SLM_REMOTE / SLM_MCP_STATELESS). The
predicate was unit-tested, but the daemon WIRING (settings.stateless_http) had
no test — so this asserts the wiring the daemon performs actually flips the
FastMCP settings. Mirrors unified_daemon.py's stateless block.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core import remote_mode


def _fresh_fastmcp_settings():
    """Minimal settings double; rebuilding FastMCP in-process is unsafe."""
    return SimpleNamespace(
        settings=SimpleNamespace(stateless_http=False, json_response=False),
    )


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
    from superlocalmemory.server.unified_daemon import (
        _configure_mcp_transport_settings,
    )

    fastmcp = _fresh_fastmcp_settings()
    _configure_mcp_transport_settings(fastmcp)
    assert fastmcp.settings.stateless_http is True
    assert fastmcp.settings.json_response is True


def test_daemon_wiring_stays_stateful_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no opt-in) must keep the stateful transport (loopback clients
    rely on sessions); the daemon must NOT flip stateless."""
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    from superlocalmemory.server.unified_daemon import (
        _configure_mcp_transport_settings,
    )

    fastmcp = _fresh_fastmcp_settings()
    # Prove the helper resets a singleton that was stateless in a prior app.
    fastmcp.settings.stateless_http = True
    fastmcp.settings.json_response = True
    _configure_mcp_transport_settings(fastmcp)
    assert remote_mode.mcp_stateless() is False
    assert fastmcp.settings.stateless_http is False
    assert fastmcp.settings.json_response is False
