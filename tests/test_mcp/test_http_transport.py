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
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient

    s = FastMCP("slm-test")
    mcp_app = s.streamable_http_app()

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
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
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.headers.get("mcp-session-id"), "mcp-session-id header missing"

    body = _sse_body_to_dict(resp.text)
    server_info = body.get("result", {}).get("serverInfo", {})
    assert server_info.get("name"), f"serverInfo.name missing in: {body}"


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

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
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

    with TestClient(mcp_app, base_url="http://localhost:8765") as client:
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
