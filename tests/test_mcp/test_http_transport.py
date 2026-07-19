# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for v3.6.7 MCP Streamable-HTTP transport.

Validates:
  (a) /mcp route is mounted on the daemon's FastAPI application.
  (b) MCP initialize handshake returns serverInfo over HTTP.
  (c) tools/list returns at least the core SLM tools.
  (d) tools/call recall round-trip completes without deadlock.
  (e) Lifespan guard: requests before lifespan-start raise the expected error.

Run:  .venv/bin/python -m pytest tests/test_mcp/test_http_transport.py -v
"""

from __future__ import annotations

import json
import os

import pytest

# Suppress embedded-daemon background threads BEFORE any mcp.server import.
os.environ.setdefault("SLM_MCP_EMBEDDED", "1")
os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_slm_server_state():
    """Reset global SLM server state between tests.

    Two resets are required:
    1. _session_manager = None: StreamableHTTPSessionManager.run() can only
       be called once per instance. Without this, a second TestClient enter
       fails with "run() can only be called once per instance".
    2. streamable_http_path = "/mcp": create_app() sets this to "/" so the
       endpoint lands at /mcp after the FastAPI mount prefix is stripped.
       Tests (c)/(d) call streamable_http_app() directly and POST to /mcp —
       if the path is still "/" from a prior create_app() call they get 404.
    """
    yield
    try:
        from superlocalmemory.mcp.server import server as slm_server
        slm_server._session_manager = None
        slm_server.settings.streamable_http_path = "/mcp"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_body_to_dict(body: str) -> dict:
    """Parse the first 'data:' line from an SSE body into a dict."""
    for line in body.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    return {}


def _terminate_mcp_session(client, path: str, session_id: str) -> None:
    """Orderly MCP session shutdown required by the stateful HTTP protocol."""
    response = client.delete(
        path,
        headers={
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
    )
    assert response.status_code == 200, (
        f"session termination failed: {response.status_code} {response.text[:300]}"
    )


# ---------------------------------------------------------------------------
# (a) /mcp route is mounted on application
# ---------------------------------------------------------------------------

def test_mcp_route_mounted_on_application(tmp_path, monkeypatch):
    """create_app() must mount an ASGI sub-app at the /mcp prefix."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")

    from superlocalmemory.server import unified_daemon
    app = unified_daemon.create_app()

    # Walk application routes; a Mount at path "/mcp" must exist.
    from starlette.routing import Mount
    mcp_mounts = [
        r for r in app.routes
        if isinstance(r, Mount) and r.path == "/mcp"
    ]
    assert mcp_mounts, "/mcp Mount not found in application.routes"

    # The module-level _mcp_app must be set after create_app() runs.
    assert unified_daemon._mcp_app is not None, "_mcp_app not set after create_app()"


# ---------------------------------------------------------------------------
# (b) MCP initialize handshake returns serverInfo
# ---------------------------------------------------------------------------

def test_mcp_initialize_returns_server_info():
    """POST /mcp with initialize method returns 200 + serverInfo in SSE body."""
    from starlette.testclient import TestClient

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    s = SLMFastMCP("slm-test")
    mcp_app = s.streamable_http_app()

    session_id = None
    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        try:
            resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "1"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            session_id = resp.headers.get("mcp-session-id")
        finally:
            if session_id:
                _terminate_mcp_session(client, "/mcp", session_id)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert session_id, "mcp-session-id header missing"

    body = _sse_body_to_dict(resp.text)
    server_info = body.get("result", {}).get("serverInfo", {})
    assert server_info.get("name"), f"serverInfo.name missing in: {body}"
    assert server_info.get("version") == "3.7.7", server_info


# ---------------------------------------------------------------------------
# (c) tools/list returns the core SLM tools
# ---------------------------------------------------------------------------

def test_mcp_tools_list_returns_core_tools():
    """tools/list must return at least the 'recall' and 'remember' tools."""
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient

    # Use the real SLM FastMCP server which has all tools registered.
    from superlocalmemory.mcp.server import server as slm_server

    mcp_app = slm_server.streamable_http_app()

    session_id = None
    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        try:
            # Step 1: initialize
            r1 = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "1"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert r1.status_code == 200
            session_id = r1.headers.get("mcp-session-id")
            assert session_id

            # Step 2: tools/list
            r2 = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers={
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )
        finally:
            if session_id:
                _terminate_mcp_session(client, "/mcp", session_id)
    assert r2.status_code == 200
    body = _sse_body_to_dict(r2.text)
    tools = body.get("result", {}).get("tools", [])
    tool_names = {t["name"] for t in tools}

    assert "recall" in tool_names, f"'recall' not in tool_names: {sorted(tool_names)}"
    assert "remember" in tool_names, f"'remember' not in tool_names: {sorted(tool_names)}"
    assert "session_init" in tool_names, f"'session_init' not in tool_names"


# ---------------------------------------------------------------------------
# (d) tools/call recall round-trip (mock pool — no live daemon required)
# ---------------------------------------------------------------------------

def test_mcp_tools_call_recall_does_not_deadlock(monkeypatch):
    """tools/call recall must complete without hanging (mock pool path)."""
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient
    from unittest.mock import MagicMock

    # Patch choose_pool at the module level so the tool doesn't reach 127.0.0.1:8765.
    mock_pool = MagicMock()
    mock_pool.recall.return_value = {
        "ok": True,
        "results": [{"content": "test fact", "score": 0.9}],
        "result_count": 1,
        "query_type": "semantic",
        "channel_weights": {},
        "no_confident_match": False,
    }

    import superlocalmemory.mcp._daemon_proxy as _dp
    monkeypatch.setattr(_dp, "choose_pool", lambda: mock_pool)

    from superlocalmemory.mcp.server import server as slm_server
    mcp_app = slm_server.streamable_http_app()

    session_id = None
    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        try:
            r1 = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "1"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert r1.status_code == 200
            session_id = r1.headers.get("mcp-session-id")
            assert session_id

            r2 = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "recall",
                        "arguments": {"query": "test", "limit": 5},
                    },
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )
        finally:
            if session_id:
                _terminate_mcp_session(client, "/mcp", session_id)

    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text[:300]}"
    body = _sse_body_to_dict(r2.text)
    # Either a result or an error is acceptable — what we forbid is a hang.
    assert "result" in body or "error" in body, f"Unexpected body: {body}"


# ---------------------------------------------------------------------------
# (e) Lifespan guard: requests before lifespan-start raise RuntimeError
# ---------------------------------------------------------------------------

def test_mcp_requests_fail_before_lifespan_start():
    """session_manager.handle_request raises before run() is entered."""
    import asyncio
    from mcp.server.fastmcp import FastMCP

    s = FastMCP("lifespan-test")
    # streamable_http_app() lazily creates the session_manager.
    s.streamable_http_app()
    session_mgr = s.session_manager

    # Simulate a minimal ASGI HTTP scope without the task group initialized.
    async def _probe():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "query_string": b"",
        }

        async def _receive():
            return {"type": "http.request", "body": b"{}"}

        async def _send(msg):
            pass

        await session_mgr.handle_request(scope, _receive, _send)

    with pytest.raises(RuntimeError, match="[Tt]ask group"):
        asyncio.run(_probe())


def test_mcp_delete_closes_stateful_transport_streams():
    """Explicit session termination closes every AnyIO transport endpoint."""
    from starlette.testclient import TestClient

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    s = SLMFastMCP("session-close-test")
    mcp_app = s.streamable_http_app()

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        init = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        session_id = init.headers.get("mcp-session-id")
        assert session_id
        transport = s.session_manager._server_instances[session_id]

        _terminate_mcp_session(client, "/mcp", session_id)

        assert transport.is_terminated
        for stream_name in (
            "_read_stream_writer", "_read_stream",
            "_write_stream_reader", "_write_stream",
        ):
            stream = getattr(transport, stream_name)
            assert stream is not None
            assert stream._closed, f"{stream_name} remained open after DELETE"


def test_sse_response_closes_owned_body_iterator():
    """The product SSE response closes its per-request receive iterator."""
    import anyio

    from superlocalmemory.mcp.http_transport import ClosingEventSourceResponse

    send_stream, receive_stream = anyio.create_memory_object_stream[dict](1)

    async def _probe() -> None:
        await send_stream.send({"data": "complete"})
        await send_stream.aclose()
        response = ClosingEventSourceResponse(receive_stream, ping=60)

        async def _receive():
            await anyio.sleep_forever()

        async def _send(_message):
            return None

        await response(
            {"type": "http", "method": "GET", "path": "/mcp"},
            _receive,
            _send,
        )

    anyio.run(_probe)
    assert receive_stream._closed


# ---------------------------------------------------------------------------
# (f) v3.6.10 per-agent-ID routing: real FastMCP via /mcp/{agent_id}
# ---------------------------------------------------------------------------

def test_mcp_per_agent_url_initialize_real_fastmcp():
    """POST /mcp/claude on the REAL FastMCP app (wrapped + mounted exactly like
    unified_daemon) must 200 AND set the agent-id ContextVar to 'claude'.

    This is the end-to-end proof that the per-agent URL wiring is complete:
    FastAPI mount → AgentIDExtractorASGI (root_path-aware) → FastMCP route '/'.
    """
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from superlocalmemory.mcp.agent_context import (
        AgentIDExtractorASGI,
        get_current_agent_id,
    )
    from superlocalmemory.mcp.http_transport import SLMFastMCP

    seen_agent: list[str] = []

    s = SLMFastMCP("slm-peragent-test")

    @s.tool()
    async def whoami() -> dict:
        """Return the resolved agent id (proves ContextVar reached the tool)."""
        aid = get_current_agent_id()
        seen_agent.append(aid)
        return {"agent_id": aid}

    # Mirror create_app(): streamable route is '/', mounted under /mcp.
    s.settings.streamable_http_path = "/"
    mcp_app = s.streamable_http_app()

    # Mirror unified_daemon.lifespan(): start FastMCP's session manager via the
    # inner app's lifespan_context (the wrapper is a transparent ASGI passthrough).
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(lifespan=_lifespan)
    app.mount("/mcp", AgentIDExtractorASGI(mcp_app))

    session_id = None
    with TestClient(app, base_url="http://localhost:8765") as client:
        try:
            init = client.post(
                "/mcp/claude",
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "1"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert init.status_code == 200, (
                f"init failed: {init.status_code} {init.text[:300]}"
            )
            session_id = init.headers.get("mcp-session-id")
            assert session_id, "no session id from /mcp/claude initialize"

            # Notifications/initialized then the tool call.
            client.post(
                "/mcp/claude",
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers={
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )
            call = client.post(
                "/mcp/claude",
                json={
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "whoami", "arguments": {}},
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )
        finally:
            if session_id:
                _terminate_mcp_session(client, "/mcp/claude", session_id)

    assert call.status_code == 200, f"tools/call failed: {call.status_code} {call.text[:300]}"
    # The tool ran with agent_id resolved from the URL path, not 'mcp_client'.
    assert "claude" in seen_agent, f"agent_id not propagated to tool: {seen_agent}"
