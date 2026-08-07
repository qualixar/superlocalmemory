# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Tests for MCP Streamable-HTTP transport under mcp==2.0.0 fully-stateless.

Validates:
  (a) /mcp route is mounted on the daemon's FastAPI application.
  (b) MCP initialize handshake returns serverInfo over HTTP (json_response).
  (c) tools/list returns at least the core SLM tools.
  (d) tools/call recall round-trip completes without deadlock.
  (e) Lifespan guard: requests before lifespan-start raise the expected error.
  (f) Per-agent-ID routing still works.

Run:  .venv/bin/python -m pytest tests/test_mcp/test_http_transport.py -v -o addopts=""
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("SLM_MCP_EMBEDDED", "1")
os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")


def _stateless_kwargs(**overrides) -> dict:
    """Default production kwargs for streamable_http_app under mcp 2.0.0."""
    base = {
        "streamable_http_path": "/",
        "stateless_http": True,
        "json_response": True,
        "event_store": None,
        "host": "127.0.0.1",
    }
    base.update(overrides)
    return base


def _mcp_response_to_dict(body: str) -> dict:
    """Parse either JSON or SSE Streamable-HTTP response representation."""
    body = body.strip()
    if body.startswith("{"):
        return json.loads(body)
    for line in body.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    return {}


def _post_mcp(client, path: str, payload: dict, session_id: str | None = None):
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    return client.post(path, json=payload, headers=headers)


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


def _stream_is_closed(stream) -> bool:
    """True if an AnyIO (or mcp Context* wrapper) memory stream is closed.

    mcp 2.0 wraps MemoryObject streams in ContextSendStream/ContextReceiveStream
    that expose ``_inner`` rather than ``_closed`` directly.
    """
    closed = getattr(stream, "_closed", None)
    if closed is not None:
        return bool(closed)
    inner = getattr(stream, "_inner", None)
    if inner is not None:
        closed = getattr(inner, "_closed", None)
        if closed is not None:
            return bool(closed)
    raise AssertionError(
        f"cannot determine closed state for stream type {type(stream)!r}"
    )


# ---------------------------------------------------------------------------
# (a) /mcp route is mounted on application
# ---------------------------------------------------------------------------

def test_mcp_route_mounted_on_application(tmp_path, monkeypatch):
    """create_app() must mount an ASGI sub-app at the /mcp prefix."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")
    monkeypatch.delenv("SLM_MCP_STATEFUL", raising=False)

    from superlocalmemory.server import unified_daemon
    app = unified_daemon.create_app()

    from starlette.routing import Mount
    mcp_mounts = [
        r for r in app.routes
        if isinstance(r, Mount) and r.path == "/mcp"
    ]
    assert mcp_mounts, "/mcp Mount not found in application.routes"
    assert unified_daemon._mcp_app is not None, "_mcp_app not set after create_app()"


# ---------------------------------------------------------------------------
# (b) MCP initialize handshake returns serverInfo
# ---------------------------------------------------------------------------

def test_mcp_initialize_returns_server_info():
    """POST / with initialize returns 200 + serverInfo (stateless json_response)."""
    from starlette.testclient import TestClient

    from superlocalmemory.mcp.http_transport import SLMFastMCP
    from superlocalmemory import __version__

    s = SLMFastMCP("slm-test")
    mcp_app = s.streamable_http_app(**_stateless_kwargs())

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        resp = _post_mcp(
            client,
            "/",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            },
        )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = _mcp_response_to_dict(resp.text)
    server_info = body.get("result", {}).get("serverInfo", {})
    assert server_info.get("name"), f"serverInfo.name missing in: {body}"
    assert server_info.get("version") == __version__, server_info


# ---------------------------------------------------------------------------
# (c) tools/list returns the core SLM tools
# ---------------------------------------------------------------------------

def test_mcp_tools_list_returns_core_tools():
    """tools/list must return at least the 'recall' and 'remember' tools."""
    from starlette.testclient import TestClient

    from superlocalmemory.mcp.server import server as slm_server

    mcp_app = slm_server.streamable_http_app(**_stateless_kwargs())

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        r1 = _post_mcp(
            client,
            "/",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            },
        )
        assert r1.status_code == 200, r1.text[:300]

        r2 = _post_mcp(
            client,
            "/",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    assert r2.status_code == 200, r2.text[:300]
    body = _mcp_response_to_dict(r2.text)
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
    from starlette.testclient import TestClient
    from unittest.mock import MagicMock

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
    mcp_app = slm_server.streamable_http_app(**_stateless_kwargs())

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        r1 = _post_mcp(
            client,
            "/",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            },
        )
        assert r1.status_code == 200

        r2 = _post_mcp(
            client,
            "/",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "recall",
                    "arguments": {"query": "test", "limit": 5},
                },
            },
        )

    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text[:300]}"
    body = _mcp_response_to_dict(r2.text)
    assert "result" in body or "error" in body, f"Unexpected body: {body}"


# ---------------------------------------------------------------------------
# (e) Lifespan guard: requests before lifespan-start raise RuntimeError
# ---------------------------------------------------------------------------

def test_mcp_requests_fail_before_lifespan_start():
    """session_manager.handle_request raises before run() is entered."""
    import asyncio

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    s = SLMFastMCP("lifespan-test")
    s.streamable_http_app(**_stateless_kwargs())
    session_mgr = s.session_manager

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


def test_mcp_delete_closes_stateful_transport_streams(monkeypatch):
    """DELETE /mcp with Mcp-Session-Id terminates the stateful transport.

    Fully-stateless is the production default, but ``SLM_MCP_STATEFUL=1``
    remains a supported opt-out. Under that path DELETE is the MCP-spec
    session-termination mechanism and must close every transport stream.
    """
    monkeypatch.setenv("SLM_MCP_STATEFUL", "1")

    from starlette.testclient import TestClient

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    s = SLMFastMCP("session-close-test")
    # Explicit stateful kwargs (mirrors _configure_mcp_transport_settings
    # when SLM_MCP_STATEFUL=1): sessions exist, DELETE is meaningful.
    mcp_app = s.streamable_http_app(
        streamable_http_path="/",
        stateless_http=False,
        json_response=False,
        event_store=None,
        host="127.0.0.1",
    )

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
        init = client.post(
            "/",
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
        session_id = init.headers.get("mcp-session-id")
        assert session_id, f"stateful initialize must mint Mcp-Session-Id: {init.headers}"
        assert session_id in s.session_manager._server_instances
        transport = s.session_manager._server_instances[session_id]
        assert not transport.is_terminated

        _terminate_mcp_session(client, "/", session_id)

        assert transport.is_terminated
        for stream_name in (
            "_read_stream_writer",
            "_read_stream",
            "_write_stream_reader",
            "_write_stream",
        ):
            stream = getattr(transport, stream_name)
            assert stream is not None
            assert _stream_is_closed(stream), (
                f"{stream_name} remained open after DELETE"
            )


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
# (f) per-agent-ID routing
# ---------------------------------------------------------------------------

def test_mcp_per_agent_url_initialize_real_mcpserver():
    """POST /mcp/claude on the real MCPServer app must 200 and set agent id."""
    from contextlib import asynccontextmanager

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

    mcp_app = s.streamable_http_app(**_stateless_kwargs())

    @asynccontextmanager
    async def _lifespan(_app):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(lifespan=_lifespan)
    app.mount("/mcp", AgentIDExtractorASGI(mcp_app))

    with TestClient(app, base_url="http://localhost:8765") as client:
        init = _post_mcp(
            client,
            "/mcp/claude",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            },
        )
        assert init.status_code == 200, (
            f"init failed: {init.status_code} {init.text[:300]}"
        )

        call = _post_mcp(
            client,
            "/mcp/claude",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "whoami", "arguments": {}},
            },
        )
        assert call.status_code == 200, call.text[:300]

    assert seen_agent, "whoami tool never ran"
    assert seen_agent[-1] == "claude", f"expected agent_id=claude, got {seen_agent}"
