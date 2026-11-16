# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A relayed MCP request's deadline becomes the recall's time budget.

The laptop stamps ``x-slm-deadline-ms`` (epoch milliseconds on this computer's
clock) on every relayed request. The MCP layer remembers it for the request,
the ``recall`` tool turns what is left into ``budget_s`` and the daemon proxy
sends it to ``GET /recall``. Without the header nothing is sent, so a local
recall's request is exactly what it was before.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp import tools_core
from superlocalmemory.mcp.request_deadline import (
    DEADLINE_HEADER,
    current_deadline_ms,
    parse_deadline_header,
    remaining_budget_s,
    request_deadline,
)


class _Server:
    def __init__(self) -> None:
        self.captured: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture
def recall_tool(monkeypatch):
    from superlocalmemory.mcp import _daemon_proxy

    seen: list[dict] = []

    class _Pool:
        def recall(self, *args, **kwargs):
            seen.append(kwargs)
            return {"ok": True, "results": [], "result_count": 0}

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _Pool())
    server = _Server()
    tools_core.register_core_tools(server, lambda: SimpleNamespace(profile_id="default"))
    return server.captured["recall"], seen


def _now_ms() -> int:
    return int(time.time() * 1000)


def test_the_header_name_is_the_one_the_laptop_stamps():
    assert DEADLINE_HEADER == "x-slm-deadline-ms"


def test_a_deadline_on_the_request_becomes_the_recall_budget(recall_tool):
    tool, seen = recall_tool
    with request_deadline(_now_ms() + 12_000):
        asyncio.run(tool("q"))
    budget = seen[0]["budget_s"]
    assert 10.0 < budget <= 12.0, budget


def test_a_deadline_already_past_still_sends_a_small_positive_budget(recall_tool):
    tool, seen = recall_tool
    with request_deadline(_now_ms() - 5_000):
        asyncio.run(tool("q"))
    assert 0 < seen[0]["budget_s"] <= 0.01


def test_no_deadline_sends_nothing_so_the_call_is_what_it_was(recall_tool):
    tool, seen = recall_tool
    asyncio.run(tool("q"))
    assert "budget_s" not in seen[0]
    assert current_deadline_ms() is None
    # The set of arguments is exactly the pre-existing set.
    assert set(seen[0]) == {
        "session_id", "fast", "include_global", "include_shared", "window", "as_of",
        "known_as_of", "valid_at", "include_unknown", "limit"}


def test_the_deadline_does_not_outlive_its_request():
    with request_deadline(_now_ms() + 1000):
        assert current_deadline_ms() is not None
    assert current_deadline_ms() is None
    assert remaining_budget_s() is None


@pytest.mark.parametrize("raw, expected", [
    ("1760000000000", 1760000000000), (b"1760000000000", 1760000000000),
    ("", None), ("abc", None), ("-5", None), ("0", None), ("1.5", None),
    ("12 34", None), ("9" * 40, None), (None, None), ("1e12", None),
])
def test_only_a_plain_epoch_milliseconds_integer_is_a_deadline(raw, expected):
    assert parse_deadline_header(raw) == expected


def test_the_proxy_sends_budget_s_only_when_given(monkeypatch):
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths: list[str] = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q")
    assert paths[-1] == "/recall?q=q&limit=10&session_id=&via=mcp"  # + the W3-1 path name
    proxy.recall("q", budget_s=7.25)
    assert paths[-1] == "/recall?q=q&limit=10&session_id=&via=mcp&budget_s=7.250"


@pytest.mark.asyncio
async def test_the_asgi_wrapper_hands_the_header_to_the_request_and_clears_it():
    from superlocalmemory.mcp.agent_context import AgentIDExtractorASGI

    seen: list = []

    async def inner(scope, receive, send):
        seen.append(current_deadline_ms())

    app = AgentIDExtractorASGI(inner)
    for path, headers, expected in [
        ("/mcp/", [(b"x-slm-deadline-ms", b"1760000000123")], 1760000000123),
        ("/mcp/claude", [(b"x-slm-deadline-ms", b"1760000000124")], 1760000000124),
        ("/mcp/", [], None),
        ("/mcp/claude", [(b"x-slm-deadline-ms", b"soon")], None),
    ]:
        await app({"type": "http", "root_path": "/mcp", "path": path, "headers": headers},
                  None, None)
        assert seen[-1] == expected, (path, headers)
        assert current_deadline_ms() is None


@pytest.mark.asyncio
async def test_a_relayed_request_reaches_a_real_mcp_tool_with_its_deadline():
    """origin -> ASGI wrapper -> the SDK's own transport -> the tool body."""
    import base64
    import json

    from fastapi import FastAPI
    from mcp.server.mcpserver import MCPServer

    from superlocalmemory.mcp.agent_context import AgentIDExtractorASGI
    from superlocalmemory.remote_connections.credentials import ConnectorCredential
    from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin

    server = MCPServer("deadline transport verification")

    @server.tool()
    async def probe() -> dict:
        return {"deadline": current_deadline_ms(), "budget": remaining_budget_s()}

    mcp_app = server.streamable_http_app(
        stateless_http=True, json_response=True, streamable_http_path="/", host="127.0.0.1")
    app = FastAPI()
    app.mount("/mcp", AgentIDExtractorASGI(mcp_app))
    now = time.time()
    credential = ConnectorCredential("install-a", "owner-a", "default", "a" * 32, 1,
                                     int(now * 1000) + 60000, "a" * 64, "slmr_" + "b" * 43)
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "probe", "arguments": {}}}
    frame = {"v": 1, "kind": "request", "id": "x", "generation": 1,
             "deadlineAt": int(now * 1000) + 20_000,
             "headers": [["content-type", "application/json"],
                         ["accept", "application/json, text/event-stream"]],
             "bodyBase64": base64.b64encode(json.dumps(payload).encode()).decode()}
    async with mcp_app.router.lifespan_context(mcp_app):
        response = await CanonicalMcpOrigin(app)(frame, credential)
    assert response.status == 200, response.body
    result = json.loads(json.loads(response.body)["result"]["content"][0]["text"])
    assert abs(result["deadline"] - (int(now * 1000) + 16_000)) < 1500, result
    assert 13.0 < result["budget"] <= 16.0, result


# -- 4.2.3: a turn-blocking caller names its own budget (Hermes prefetch) -------

def test_a_callers_own_budget_is_sent(recall_tool):
    tool, seen = recall_tool
    asyncio.run(tool("q", budget_s=2.5))
    assert seen[0]["budget_s"] == 2.5


def test_a_callers_budget_only_shortens_the_relays(recall_tool):
    tool, seen = recall_tool
    with request_deadline(_now_ms() + 1_000):
        asyncio.run(tool("q", budget_s=5.0))
    assert seen[0]["budget_s"] <= 1.0


def test_no_budget_keeps_a_question_on_the_full_clock(recall_tool):
    tool, seen = recall_tool
    for bad in (0, -1, float("nan")):
        asyncio.run(tool("q", budget_s=bad))
    assert all("budget_s" not in kw for kw in seen)


def test_the_hermes_prefetch_names_a_budget_inside_its_timeout():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "plugin-src" / "hermes" / "__init__.py").read_text()
    assert '"budget_s": timeout - 0.5' in src
