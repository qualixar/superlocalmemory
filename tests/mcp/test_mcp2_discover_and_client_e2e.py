# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | mcp 2.0.0 migration regressions

"""mcp 2.0.0 migration regressions:

1. Pre-initialize ``server/discover`` must NOT close the connection.
2. Official Client(mode=\"auto\") connects, lists tools, and CALLS one.

These are the load-bearing proofs that SLM speaks MCP 2026-07-28 on mcp==2.0.0.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("SLM_MCP_EMBEDDED", "1")
os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")


def _build_server_with_tool():
    """Minimal MCPServer (via SLM wrapper) with one callable tool."""
    from superlocalmemory.mcp.http_transport import SLMFastMCP

    server = SLMFastMCP("slm-mcp2-e2e")

    @server.tool()
    def echo(text: str = "hi") -> dict:
        """Echo a string — used by the e2e client call."""
        return {"echo": text, "ok": True}

    return server


# ---------------------------------------------------------------------------
# 1. Pre-initialize server/discover must keep the connection open
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_server_discover_pre_initialize_does_not_close_connection():
    """POST server/discover before initialize must return 200 + DiscoverResult.

    On mcp 1.x / legacy FastMCP, unsolicited methods before initialize often
    closed the Streamable-HTTP session. SDK 2.0.0 registers server/discover on
    the low-level server; a pre-init discover is the mode=auto probe and MUST
    leave the transport usable for a subsequent initialize / tools/list.
    """
    from starlette.testclient import TestClient

    server = _build_server_with_tool()
    app = server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
    )

    from mcp import types as mcp_types

    version = "2026-07-28"
    # Modern discover envelope (matches ClientSession.send_discover)
    discover_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "server/discover",
        "params": {
            "_meta": {
                mcp_types.PROTOCOL_VERSION_META_KEY: version,
                mcp_types.CLIENT_INFO_META_KEY: {
                    "name": "discover-probe",
                    "version": "1",
                },
                mcp_types.CLIENT_CAPABILITIES_META_KEY: {},
            }
        },
    }

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        r1 = client.post(
            "/",
            json=discover_body,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Protocol-Version": version,
                "mcp-method": "server/discover",
            },
        )
        assert r1.status_code == 200, (
            f"server/discover closed or failed: {r1.status_code} {r1.text[:500]}"
        )
        # Must be a JSON-RPC *result*, not a connection drop / hard close.
        if r1.headers.get("content-type", "").startswith("application/json"):
            payload = r1.json()
        else:
            # SSE path — still open if we can parse a data line
            payload = None
            for line in r1.text.splitlines():
                if line.startswith("data:"):
                    import json as _json
                    payload = _json.loads(line[5:].strip())
                    break
        assert payload is not None, f"empty discover body: {r1.text[:500]}"
        assert "result" in payload, f"discover returned error: {payload}"
        result = payload["result"]
        assert (
            "supportedVersions" in result
            or "supported_versions" in result
            or "capabilities" in result
        ), result

        # Connection must still accept initialize after discover
        r2 = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "discover-probe", "version": "1"},
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        assert r2.status_code == 200, (
            f"initialize after discover failed — connection was closed: "
            f"{r2.status_code} {r2.text[:500]}"
        )


# ---------------------------------------------------------------------------
# 2. mcp 2.0.0 Client(mode="auto") — list tools + call one
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp2_client_mode_auto_lists_and_calls_tool():
    """Official Client with mode=auto must list tools and call one successfully.

    Not an import check — a real round-trip through negotiation + tools/call.
    """
    import importlib.metadata as md
    from mcp.client import Client

    assert md.version("mcp").startswith("2."), (
        f"e2e requires mcp 2.x; got {md.version('mcp')}"
    )

    server = _build_server_with_tool()

    async with Client(server, mode="auto") as client:
        # Negotiation must complete (auto uses discover when available)
        assert client.session is not None
        listed = await client.list_tools()
        names = [t.name for t in listed.tools]
        assert "echo" in names, f"echo tool missing from {names}"

        result = await client.call_tool("echo", {"text": "mcp2-e2e"})
        # call_tool returns a CallToolResult-like object with content blocks
        assert result is not None
        # Prefer structured content / text that mentions our payload
        text_blob = ""
        if hasattr(result, "content") and result.content:
            for block in result.content:
                text_blob += getattr(block, "text", "") or str(block)
        elif hasattr(result, "data"):
            text_blob = str(result.data)
        else:
            text_blob = str(result)

        assert "mcp2-e2e" in text_blob or "ok" in text_blob.lower(), (
            f"tool call did not return expected payload: {result!r}"
        )


@pytest.mark.asyncio
async def test_mcp2_http_client_mode_auto_over_streamable_http():
    """HTTP path: Client(url, mode=auto) against a real Streamable-HTTP app."""
    import asyncio
    import contextlib
    import socket
    import importlib.metadata as md

    import uvicorn
    from fastapi import FastAPI
    from mcp.client import Client

    assert md.version("mcp").startswith("2.")

    server = _build_server_with_tool()
    mcp_app = server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        host="127.0.0.1",
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/mcp", mcp_app)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    _, port = probe.getsockname()
    probe.close()

    uv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(uv.serve())
    try:
        for _ in range(100):
            if uv.started:
                break
            await asyncio.sleep(0.02)
        assert uv.started, "uvicorn failed to start"

        url = f"http://127.0.0.1:{port}/mcp"
        async with Client(url, mode="auto") as client:
            listed = await client.list_tools()
            names = [t.name for t in listed.tools]
            assert "echo" in names, names
            result = await client.call_tool("echo", {"text": "http-auto"})
            text_blob = ""
            if hasattr(result, "content") and result.content:
                for block in result.content:
                    text_blob += getattr(block, "text", "") or str(block)
            else:
                text_blob = str(result)
            assert "http-auto" in text_blob or "ok" in text_blob.lower(), result
    finally:
        uv.should_exit = True
        await asyncio.wait_for(task, timeout=10)
