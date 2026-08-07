"""Regression guard for the fully-stateless MCP transport (mcp 2.0.0).

Fully-stateless is the DEFAULT. ``_configure_mcp_transport_settings()`` returns
a kwargs dict for ``streamable_http_app()`` — it no longer mutates FastMCP
settings. Opt out with ``SLM_MCP_STATEFUL=1``.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import remote_mode


def test_mcp_stateless_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)
    assert remote_mode.mcp_stateless() is True


def test_mcp_stateless_opt_out_stateful(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_MCP_STATEFUL", "1")
    assert remote_mode.mcp_stateless() is False


def test_mcp_stateless_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)
    monkeypatch.setenv("SLM_MCP_STATELESS", "0")
    assert remote_mode.mcp_stateless() is False


def test_configure_returns_stateless_kwargs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    from superlocalmemory.server.unified_daemon import (
        _configure_mcp_transport_settings,
    )

    kwargs = _configure_mcp_transport_settings()
    assert isinstance(kwargs, dict)
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True
    assert kwargs["event_store"] is None
    assert kwargs["streamable_http_path"] == "/"
    assert "session_idle_timeout" not in kwargs


def test_configure_stateful_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_MCP_STATEFUL", "1")
    from superlocalmemory.server.unified_daemon import (
        _configure_mcp_transport_settings,
    )

    kwargs = _configure_mcp_transport_settings()
    assert kwargs["stateless_http"] is False
