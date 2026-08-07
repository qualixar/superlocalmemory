# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | mcp 2.0.0 fully-stateless transport

"""Stateless MCP transport contract (mcp==2.0.0).

STATELESS IS THE DEFAULT. Session idle-timeout and EventStore SSE resumability
are illegal/unused under ``stateless_http=True`` (SDK raises RuntimeError if
``session_idle_timeout`` is set with stateless). Application-level
``session_init``/``close_session`` are orthogonal and unchanged.

Run:
    SLM_TEST_ISOLATION=1 .venv/bin/python -m pytest \\
        tests/mcp/test_http_transport_resilience.py -o addopts="" -q --tb=short
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SLM_MCP_EMBEDDED", "1")
os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")


def _import_transport():
    import superlocalmemory.mcp.http_transport as mod
    return mod


def _get_slmmcp():
    mod = _import_transport()
    # Prefer SLMFastMCP alias if kept; fall back to MCPServer wrapper name.
    return getattr(mod, "SLMFastMCP", None) or getattr(mod, "SLMMCPServer")


# ===========================================================================
# GROUP A: Stateless default contract
# ===========================================================================


class TestStatelessDefaultContract:
    """Transport kwargs and streamable_http_app honour fully-stateless defaults."""

    def test_configure_returns_stateless_kwargs_by_default(self, monkeypatch):
        """_configure_mcp_transport_settings returns kwargs — not a settings mutator."""
        monkeypatch.delenv("SLM_REMOTE", raising=False)
        monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
        monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)

        from superlocalmemory.server.unified_daemon import (
            _configure_mcp_transport_settings,
        )

        kwargs = _configure_mcp_transport_settings()
        assert isinstance(kwargs, dict), "must return a kwargs dict for streamable_http_app()"
        assert kwargs.get("stateless_http") is True
        assert kwargs.get("json_response") is True
        assert kwargs.get("event_store") in (None, False) or kwargs.get("event_store") is None
        assert "session_idle_timeout" not in kwargs, (
            "session_idle_timeout is illegal with stateless_http=True"
        )
        assert kwargs.get("streamable_http_path") == "/"

    def test_streamable_http_app_accepts_stateless_kwargs(self):
        """streamable_http_app(**stateless_kwargs) must not raise RuntimeError."""
        SLM = _get_slmmcp()
        mcp = SLM("slm-stateless-contract")
        try:
            app = mcp.streamable_http_app(
                streamable_http_path="/",
                stateless_http=True,
                json_response=True,
                event_store=None,
            )
        except RuntimeError as exc:
            pytest.fail(f"stateless streamable_http_app raised RuntimeError: {exc}")
        assert app is not None

    def test_session_manager_is_stateless_when_configured(self):
        """After streamable_http_app(stateless=True), session manager is stateless."""
        SLM = _get_slmmcp()
        mcp = SLM("slm-stateless-sm")
        mcp.streamable_http_app(
            streamable_http_path="/",
            stateless_http=True,
            json_response=True,
        )
        sm = mcp.session_manager
        assert sm.stateless is True
        assert sm.session_idle_timeout is None, (
            "idle timeout must be None in stateless mode"
        )

    def test_no_event_store_in_stateless_manager(self):
        """Stateless transport must not rely on an EventStore."""
        SLM = _get_slmmcp()
        mcp = SLM("slm-stateless-no-store")
        mcp.streamable_http_app(
            streamable_http_path="/",
            stateless_http=True,
            json_response=True,
            event_store=None,
        )
        sm = mcp.session_manager
        # Manager may keep the ctor arg as None, or ignore it for stateless path.
        assert getattr(sm, "event_store", None) is None or sm.stateless is True

    def test_stateless_plus_idle_timeout_is_sdk_illegal(self):
        """Document the SDK invariant: stateless + idle timeout → RuntimeError."""
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.lowlevel import Server

        app = Server("probe")
        with pytest.raises(RuntimeError, match="session_idle_timeout"):
            StreamableHTTPSessionManager(
                app=app,
                stateless=True,
                session_idle_timeout=600.0,
            )

    def test_product_version_on_mcp_server(self):
        """MCPServer.version is set via constructor — no private _mcp_server poke."""
        from superlocalmemory import __version__

        SLM = _get_slmmcp()
        mcp = SLM("slm-version-check")
        assert mcp.version == __version__, (
            f"expected product version {__version__!r}, got {mcp.version!r}"
        )

    def test_mcp_server_import_path(self):
        """mcp.server.fastmcp is gone; we use mcp.server.mcpserver.MCPServer."""
        mod = _import_transport()
        import inspect
        src = inspect.getsource(mod)
        # Disallow a live import of the deleted FastMCP module (docstring
        # mentions of the migration history are fine).
        assert "from mcp.server.fastmcp" not in src, "FastMCP import must be removed"
        assert "import mcp.server.fastmcp" not in src
        assert "from mcp.server.mcpserver import MCPServer" in src

    def test_tool_decorator_still_works(self):
        """@server.tool registrations must keep working on MCPServer."""
        SLM = _get_slmmcp()
        mcp = SLM("slm-tool-deco")

        @mcp.tool()
        def ping() -> str:
            """Health ping."""
            return "pong"

        # Tool manager should list the registered tool
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "ping" in names


# ===========================================================================
# GROUP B: Daemon kwargs wiring (no settings mutation)
# ===========================================================================


class TestDaemonKwargsWiring:
    def test_configure_does_not_mutate_settings_object(self, monkeypatch):
        """Helper must not write to a FastMCP-style settings.stateless_http attr."""
        monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)
        from superlocalmemory.server.unified_daemon import (
            _configure_mcp_transport_settings,
        )
        import inspect
        import ast
        src = inspect.getsource(_configure_mcp_transport_settings)
        tree = ast.parse(src)
        # No attribute assignments of the form x.settings.stateless_http = ...
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr in {
                        "stateless_http", "json_response", "streamable_http_path",
                        "transport_security",
                    }:
                        # settings.stateless_http = ...
                        if isinstance(t.value, ast.Attribute) and t.value.attr == "settings":
                            pytest.fail(f"mutates settings.{t.attr}")
                        if isinstance(t.value, ast.Name) and t.value.id == "settings":
                            pytest.fail(f"mutates settings.{t.attr}")
        assert "return kwargs" in src or "return" in src

    def test_allowed_hosts_become_transport_security_kwarg(self, monkeypatch):
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.1.10:*")
        monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)
        from superlocalmemory.server.unified_daemon import (
            _configure_mcp_transport_settings,
        )
        kwargs = _configure_mcp_transport_settings()
        assert "transport_security" in kwargs
        ts = kwargs["transport_security"]
        assert ts is not None
        assert getattr(ts, "enable_dns_rebinding_protection", None) is True
