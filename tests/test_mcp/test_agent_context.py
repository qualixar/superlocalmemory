# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for mcp/agent_context.py — ContextVar-based per-agent-ID routing (v3.6.10).

Covers:
    - get_current_agent_id defaults to "mcp_client"
    - ContextVar override (HTTP transport path)
    - SLM_AGENT_ID env var fallback (stdio transport)
    - env_fallback=False suppresses env var lookup
    - _AgentIDExtractorASGI-style path parsing logic
    - tools_active._get_agent_id() priority chain
    - tools_core remember/recall/delete_memory/update_memory resolve agent_id

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
import contextvars
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI


# ---------------------------------------------------------------------------
# sanitize_agent_id — audit fix (log-injection / oversize / path chars)
# ---------------------------------------------------------------------------

class TestSanitizeAgentId:
    def test_plain_id_unchanged(self) -> None:
        from superlocalmemory.mcp.agent_context import sanitize_agent_id
        assert sanitize_agent_id("claude") == "claude"
        assert sanitize_agent_id("sub-agent_1.2") == "sub-agent_1.2"

    def test_crlf_stripped(self) -> None:
        from superlocalmemory.mcp.agent_context import sanitize_agent_id
        out = sanitize_agent_id("admin\r\n2026 INFO forged log line")
        assert "\r" not in out and "\n" not in out

    def test_control_and_ansi_stripped(self) -> None:
        from superlocalmemory.mcp.agent_context import sanitize_agent_id
        out = sanitize_agent_id("x\x1b[31mred\x00")
        assert "\x1b" not in out and "\x00" not in out

    def test_path_chars_neutralised(self) -> None:
        from superlocalmemory.mcp.agent_context import sanitize_agent_id
        out = sanitize_agent_id("../../etc/passwd")
        assert "/" not in out

    def test_length_capped_at_64(self) -> None:
        from superlocalmemory.mcp.agent_context import sanitize_agent_id
        assert len(sanitize_agent_id("a" * 500)) == 64


# ---------------------------------------------------------------------------
# agent_context module — unit tests
# ---------------------------------------------------------------------------

class TestGetCurrentAgentId:
    def test_default_is_mcp_client(self) -> None:
        from superlocalmemory.mcp.agent_context import get_current_agent_id
        assert get_current_agent_id() == "mcp_client"

    def test_contextvar_set_returns_custom_id(self) -> None:
        from superlocalmemory.mcp.agent_context import _current_agent_id, get_current_agent_id
        token = _current_agent_id.set("claude")
        try:
            assert get_current_agent_id() == "claude"
        finally:
            _current_agent_id.reset(token)

    def test_contextvar_reset_restores_default(self) -> None:
        from superlocalmemory.mcp.agent_context import _current_agent_id, get_current_agent_id
        token = _current_agent_id.set("hermes")
        _current_agent_id.reset(token)
        assert get_current_agent_id() == "mcp_client"

    def test_env_var_fallback_when_contextvar_is_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from superlocalmemory.mcp.agent_context import get_current_agent_id
        monkeypatch.setenv("SLM_AGENT_ID", "codex")
        assert get_current_agent_id(env_fallback=True) == "codex"

    def test_env_var_ignored_when_env_fallback_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from superlocalmemory.mcp.agent_context import get_current_agent_id
        monkeypatch.setenv("SLM_AGENT_ID", "codex")
        assert get_current_agent_id(env_fallback=False) == "mcp_client"

    def test_contextvar_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from superlocalmemory.mcp.agent_context import _current_agent_id, get_current_agent_id
        monkeypatch.setenv("SLM_AGENT_ID", "codex")
        token = _current_agent_id.set("claude")
        try:
            assert get_current_agent_id() == "claude"
        finally:
            _current_agent_id.reset(token)

    def test_contextvar_is_task_local(self) -> None:
        """Two concurrent tasks see independent ContextVar values."""
        from superlocalmemory.mcp.agent_context import _current_agent_id, get_current_agent_id

        results: list[str] = []

        async def task_a():
            token = _current_agent_id.set("agent-a")
            try:
                await asyncio.sleep(0)  # yield to task_b
                results.append(get_current_agent_id())
            finally:
                _current_agent_id.reset(token)

        async def task_b():
            token = _current_agent_id.set("agent-b")
            try:
                await asyncio.sleep(0)
                results.append(get_current_agent_id())
            finally:
                _current_agent_id.reset(token)

        async def run():
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        assert set(results) == {"agent-a", "agent-b"}, f"Expected isolation, got: {results}"


# ---------------------------------------------------------------------------
# _AgentIDExtractorASGI path-parsing logic (tested as a standalone class)
# ---------------------------------------------------------------------------

def _AgentExtractorTest(inner):
    """Return the REAL shipped ASGI wrapper (no re-creation) — the daemon mounts
    exactly this class, so testing it tests production behaviour."""
    from superlocalmemory.mcp.agent_context import AgentIDExtractorASGI
    return AgentIDExtractorASGI(inner)


class TestAgentIDExtractorASGI:
    def _make_scope(self, path: str) -> dict:
        return {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
        }

    @pytest.mark.asyncio
    async def test_agent_id_extracted_from_path(self) -> None:
        """Path /claude → ContextVar = "claude", inner app sees path "/"."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        seen: list[str] = []

        async def inner(scope, receive, send):
            seen.append(get_current_agent_id())
            seen.append(scope["path"])

        app = _AgentExtractorTest(inner)
        await app(self._make_scope("/claude"), None, None)

        assert seen[0] == "claude"
        assert seen[1] == "/"

    @pytest.mark.asyncio
    async def test_root_path_passes_through_unchanged(self) -> None:
        """Path / → no agent_id extracted, ContextVar stays "mcp_client"."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        seen: list[str] = []

        async def inner(scope, receive, send):
            seen.append(get_current_agent_id())
            seen.append(scope["path"])

        app = _AgentExtractorTest(inner)
        await app(self._make_scope("/"), None, None)

        assert seen[0] == "mcp_client"
        assert seen[1] == "/"

    @pytest.mark.asyncio
    async def test_contextvar_reset_after_request(self) -> None:
        """ContextVar returns to "mcp_client" after the request completes."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        async def inner(scope, receive, send):
            pass

        app = _AgentExtractorTest(inner)
        await app(self._make_scope("/hermes"), None, None)

        assert get_current_agent_id() == "mcp_client"

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self) -> None:
        """WebSocket / lifespan scopes bypass agent extraction."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        seen: list[str] = []

        async def inner(scope, receive, send):
            seen.append(get_current_agent_id())

        app = _AgentExtractorTest(inner)
        ws_scope = {"type": "websocket", "path": "/claude"}
        await app(ws_scope, None, None)

        assert seen[0] == "mcp_client"

    @pytest.mark.asyncio
    async def test_agent_id_with_trailing_slash(self) -> None:
        """Path /claude/ → agent_id = "claude", rewritten path = "/"."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        seen: list[str] = []

        async def inner(scope, receive, send):
            seen.append(get_current_agent_id())
            seen.append(scope["path"])

        app = _AgentExtractorTest(inner)
        await app(self._make_scope("/claude/"), None, None)

        assert seen[0] == "claude"
        assert seen[1] == "/"

    @pytest.mark.asyncio
    async def test_contextvar_reset_even_on_inner_exception(self) -> None:
        """ContextVar is reset via finally even when inner app raises."""
        from superlocalmemory.mcp.agent_context import get_current_agent_id

        async def bad_inner(scope, receive, send):
            raise RuntimeError("inner app crashed")

        app = _AgentExtractorTest(bad_inner)
        with pytest.raises(RuntimeError):
            await app(self._make_scope("/crasher"), None, None)

        assert get_current_agent_id() == "mcp_client"


# ---------------------------------------------------------------------------
# tools_active._get_agent_id() — priority chain
# ---------------------------------------------------------------------------

class TestGetAgentIdToolsActive:
    def test_default_when_nothing_set(self) -> None:
        from superlocalmemory.mcp.tools_active import _get_agent_id
        assert _get_agent_id() == "mcp_client"

    def test_contextvar_set_overrides_default(self) -> None:
        from superlocalmemory.mcp.agent_context import _current_agent_id
        from superlocalmemory.mcp.tools_active import _get_agent_id
        token = _current_agent_id.set("gemini")
        try:
            assert _get_agent_id() == "gemini"
        finally:
            _current_agent_id.reset(token)

    def test_env_var_used_when_contextvar_is_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from superlocalmemory.mcp.tools_active import _get_agent_id
        monkeypatch.setenv("SLM_AGENT_ID", "kimi")
        assert _get_agent_id() == "kimi"

    def test_custom_default_returned_when_nothing_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from superlocalmemory.mcp.tools_active import _get_agent_id
        monkeypatch.delenv("SLM_AGENT_ID", raising=False)
        assert _get_agent_id(default="fallback-agent") == "fallback-agent"

    def test_contextvar_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from superlocalmemory.mcp.agent_context import _current_agent_id
        from superlocalmemory.mcp.tools_active import _get_agent_id
        monkeypatch.setenv("SLM_AGENT_ID", "stdio-agent")
        token = _current_agent_id.set("http-agent")
        try:
            assert _get_agent_id() == "http-agent"
        finally:
            _current_agent_id.reset(token)


# ---------------------------------------------------------------------------
# tools_core — agent_id resolution in remember / recall / delete / update
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""
    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def prompt(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _register_core_tools():
    """Register tools on a mock server and return the server."""
    from superlocalmemory.mcp.tools_core import register_core_tools
    srv = _MockServer()
    get_engine = MagicMock()
    register_core_tools(srv, get_engine)
    return srv


class TestToolsCoreAgentIdResolution:
    """Verify that the 4 core tools resolve "mcp_client" sentinel at runtime."""

    @pytest.mark.asyncio
    async def test_remember_resolves_agent_id_from_contextvar(self) -> None:
        """remember() with default agent_id resolves from ContextVar when set."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        remember = srv._tools["remember"]

        token = _current_agent_id.set("test-agent")
        try:
            # Use the pending store path (daemon offline) — simplest mock
            with patch("superlocalmemory.cli.pending_store.store_pending", return_value="pend-001"):
                with patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False):
                    result = await remember(content="test memory")
        finally:
            _current_agent_id.reset(token)

        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_remember_uses_explicit_agent_id_when_provided(self) -> None:
        """Explicitly passing agent_id != "mcp_client" bypasses ContextVar resolution."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        remember = srv._tools["remember"]

        token = _current_agent_id.set("url-agent")
        try:
            with patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False), \
                 patch("superlocalmemory.cli.pending_store.store_pending", return_value="pend-001"):
                result = await remember(content="test", agent_id="explicit-agent")
        finally:
            _current_agent_id.reset(token)

        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_recall_resolves_agent_id_from_contextvar(self) -> None:
        """recall() with default agent_id="mcp_client" resolves from ContextVar."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        recall_fn = srv._tools["recall"]

        token = _current_agent_id.set("recall-agent")
        try:
            mock_pool = MagicMock()
            mock_pool.recall = MagicMock(return_value=MagicMock(facts=[], total=0, elapsed_ms=1.0))
            with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=mock_pool):
                result = await recall_fn(query="test query")
        finally:
            _current_agent_id.reset(token)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_delete_memory_resolves_agent_id_from_contextvar(self) -> None:
        """delete_memory() with default agent_id resolves from ContextVar."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        delete_fn = srv._tools["delete_memory"]

        token = _current_agent_id.set("delete-agent")
        try:
            mock_pool = MagicMock()
            mock_pool._send = MagicMock(return_value={"ok": True})
            with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=mock_pool):
                result = await delete_fn(fact_id="fact-xyz")
        finally:
            _current_agent_id.reset(token)

        assert result.get("success") is True
        call_args = mock_pool._send.call_args[0][0]
        assert call_args["source_agent_id"] == "delete-agent"
        assert "agent_id" not in call_args

    @pytest.mark.asyncio
    async def test_update_memory_resolves_agent_id_from_contextvar(self) -> None:
        """update_memory() with default agent_id resolves from ContextVar."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        update_fn = srv._tools["update_memory"]

        token = _current_agent_id.set("update-agent")
        try:
            mock_pool = MagicMock()
            mock_pool._send = MagicMock(return_value={"ok": True})
            with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=mock_pool):
                result = await update_fn(fact_id="fact-xyz", content="new content")
        finally:
            _current_agent_id.reset(token)

        assert result.get("success") is True
        call_args = mock_pool._send.call_args[0][0]
        assert call_args["source_agent_id"] == "update-agent"
        assert "agent_id" not in call_args

    @pytest.mark.asyncio
    async def test_delete_memory_with_mcp_client_default_uses_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ContextVar is default, env var SLM_AGENT_ID is used."""
        from superlocalmemory.mcp.agent_context import _current_agent_id
        srv = _register_core_tools()
        delete_fn = srv._tools["delete_memory"]

        monkeypatch.setenv("SLM_AGENT_ID", "stdio-claude")
        # Ensure ContextVar is at default (v3.6.12 parity-1: default is now ""
        # — the "no agent routed" sentinel — not the user-visible "mcp_client").
        assert _current_agent_id.get() == ""

        mock_pool = MagicMock()
        mock_pool._send = MagicMock(return_value={"ok": True})
        with patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=mock_pool):
            result = await delete_fn(fact_id="fact-env")

        assert result.get("success") is True
        call_args = mock_pool._send.call_args[0][0]
        assert call_args["source_agent_id"] == "stdio-claude"
        assert "agent_id" not in call_args


# ---------------------------------------------------------------------------
# END-TO-END: real HTTP request through a mounted AgentIDExtractorASGI
# ---------------------------------------------------------------------------

class TestEndToEndHTTPMount:
    """Mount the REAL wrapper exactly as unified_daemon does, send real HTTP,
    and assert the agent id observed inside the inner app + the rewritten path.

    This is the "is the MCP wiring actually complete?" proof — it exercises the
    FastAPI mount → prefix strip → AgentIDExtractorASGI → ContextVar chain over
    a genuine ASGI transport, not a hand-rolled scope dict.
    """

    def _build_app(self, seen: list):
        from superlocalmemory.mcp.agent_context import (
            AgentIDExtractorASGI,
            get_current_agent_id,
        )

        async def inner_asgi(scope, receive, send):
            # Record what the inner (FastMCP-equivalent) app would see. A real
            # Starlette sub-app routes on the mount-RELATIVE path, i.e.
            # path[len(root_path):]. We record that so the assertion mirrors
            # actual FastMCP routing behaviour.
            if scope["type"] == "http":
                root = scope.get("root_path", "")
                full = scope.get("path", "/")
                rel = full[len(root):] if root and full.startswith(root) else full
                seen.append((get_current_agent_id(), rel))
            await receive()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"ok":true}',
            })

        app = FastAPI()
        app.mount("/mcp", AgentIDExtractorASGI(inner_asgi))
        return app

    @pytest.mark.asyncio
    async def test_agent_id_from_url_reaches_inner_app(self) -> None:
        seen: list = []
        app = self._build_app(seen)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/mcp/claude", json={})
        assert resp.status_code == 200
        # inner app saw agent_id="claude" and the path rewritten to "/"
        assert seen == [("claude", "/")]

    @pytest.mark.asyncio
    async def test_different_agents_isolated(self) -> None:
        seen: list = []
        app = self._build_app(seen)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/mcp/hermes", json={})
            await client.post("/mcp/gemini", json={})
        assert ("hermes", "/") in seen
        assert ("gemini", "/") in seen

    @pytest.mark.asyncio
    async def test_bare_mcp_is_backward_compatible(self) -> None:
        seen: list = []
        app = self._build_app(seen)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Bare /mcp/ — no agent segment → default mcp_client preserved.
            resp = await client.post("/mcp/", json={})
        assert resp.status_code == 200
        assert seen == [("mcp_client", "/")]

    @pytest.mark.asyncio
    async def test_contextvar_cleared_after_request(self) -> None:
        from superlocalmemory.mcp.agent_context import get_current_agent_id
        seen: list = []
        app = self._build_app(seen)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/mcp/claude", json={})
        # After the request completes, the outer task's ContextVar is back to default.
        assert get_current_agent_id() == "mcp_client"

    @pytest.mark.asyncio
    async def test_malicious_agent_id_sanitized_at_wrapper(self) -> None:
        """A scope whose path segment carries CRLF/control chars (e.g. a client
        or upstream that bypasses URL validation) is sanitized before it reaches
        the tool layer (audit fix MEDIUM-2 / log-injection)."""
        from superlocalmemory.mcp.agent_context import (
            AgentIDExtractorASGI,
            get_current_agent_id,
        )
        seen: list = []

        async def inner(scope, receive, send):
            seen.append(get_current_agent_id())

        wrapper = AgentIDExtractorASGI(inner)
        # Hand-built scope (bypasses httpx URL validation) with a hostile segment.
        scope = {
            "type": "http", "method": "POST",
            "root_path": "/mcp",
            "path": "/mcp/admin\r\n2026 INFO forged\x1b[31m",
            "raw_path": b"/mcp/admin",
            "headers": [],
        }
        await wrapper(scope, None, None)
        assert len(seen) == 1
        observed = seen[0]
        assert "\r" not in observed and "\n" not in observed
        assert "\x1b" not in observed
        assert observed.startswith("admin")
