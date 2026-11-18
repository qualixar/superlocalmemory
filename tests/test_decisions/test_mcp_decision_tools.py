# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49 over MCP: an agent settles, reopens, labels and retracts through the EXISTING tools
(core 20 / code 40 tool caps), and the channel is the connection's, never a body field."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp.tools_core import register_core_tools


class _Tools:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture
def tools(monkeypatch):
    from superlocalmemory.cli import daemon

    sent = []

    def fake(method, path, body=None, **flags):
        sent.append((method, path, body))
        return {"success": True}

    monkeypatch.setattr(daemon, "daemon_request", fake)
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    server = _Tools()
    register_core_tools(server, lambda: SimpleNamespace())
    return server.tools, sent


def _call(fn, **kw):
    return asyncio.run(fn(**kw))


def test_settle_sends_no_channel_the_connection_says_it(tools) -> None:
    fns, sent = tools
    assert _call(fns["settle_dispute"], dispute_id="d1", keep="f2")["success"]
    method, path, body = sent[-1]
    assert (method, path) == ("POST", "/api/disputes/d1/settle")
    assert body == {"keep": "f2"}  # no "channel": "agent" claim


def test_keep_reopen_takes_an_answer_back(tools) -> None:
    fns, sent = tools
    _call(fns["settle_dispute"], dispute_id="d1", keep="reopen")
    assert sent[-1][:2] == ("POST", "/api/disputes/d1/reopen") and "keep" not in sent[-1][2]


def test_an_audit_label_goes_through_the_same_tool_with_the_agents_model(tools) -> None:
    fns, sent = tools
    _call(fns["settle_dispute"], dispute_id="audit:abc", keep="replaces", model="claude-sonnet")
    assert sent[-1] == ("POST", "/api/audit/abc/answer",
                        {"label": "replaces", "model": "claude-sonnet"})
    _call(fns["settle_dispute"], dispute_id="audit:abc", keep="retract")
    assert sent[-1][:2] == ("POST", "/api/audit/abc/retract")


def test_the_audit_queue_is_listed_through_list_disputes(tools) -> None:
    fns, sent = tools
    _call(fns["list_disputes"], audit=True, model="claude sonnet", limit=5)
    method, path, _body = sent[-1]
    assert method == "GET" and path.startswith("/api/audit/pending?limit=5")
    assert "model=claude%20sonnet" in path
    _call(fns["list_disputes"])
    assert sent[-1][1].startswith("/api/truth/attention")


def test_no_new_tool_was_added_so_the_tool_caps_hold() -> None:
    from superlocalmemory.mcp import profiles

    assert len(profiles._PROFILE_CORE) <= 20 and len(profiles._PROFILE_CODE) <= 40
