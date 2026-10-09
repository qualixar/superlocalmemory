"""Per-request peer seam: remote callers get their own peer and a web origin."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.mcp import tools_mesh
from superlocalmemory.mcp.remote_caller import RemotePeer, current_remote_peer, remote_peer


class _Collector:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorate(func):
            self.tools[func.__name__] = func
            return func
        return decorate


@pytest.fixture()
def tools(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []

    def request(method, path, body=None):
        calls.append((method, path, body))
        if path == "/register":
            return {"peer_id": "web-peer-id" if body.get("session_id") == "ref-1" else "other-web", "pending_messages": 0}
        if "/wait" in path:
            return {"messages": [{"id": 3, "read": False, "content": "hello"}], "timed_out": False}
        if path.startswith("/inbox/"):
            return {"messages": [{"id": 4, "read": False, "content": "x"}]}
        return {"ok": True, "id": 1}

    monkeypatch.setattr(tools_mesh, "_mesh_request", request)
    monkeypatch.setattr(tools_mesh, "_start_heartbeat", lambda: None)
    monkeypatch.setattr(tools_mesh, "_REGISTERED", True)
    monkeypatch.setattr(tools_mesh, "_PEER_ID", "local-peer")
    monkeypatch.setattr(tools_mesh, "_PROJECT_PATH", "/p")
    tools_mesh._SEND_CIRCUIT.reset()
    c = _Collector()
    tools_mesh.register_mesh_tools(c, lambda: None)
    return c.tools, calls


def test_seam_default_is_unset() -> None:
    assert current_remote_peer() is None
    with remote_peer(RemotePeer("r", "a", "A")):
        assert current_remote_peer().peer_ref == "r"
    assert current_remote_peer() is None


def test_local_send_uses_process_peer_and_same_body(tools) -> None:
    fns, calls = tools
    asyncio.run(fns["mesh_send"]("other", "hi"))
    assert ("POST", "/send", {"from_peer": "local-peer", "to_peer": "other", "content": "hi"}) in calls
    assert not any(p == "/register" for _, p, _ in calls)


def test_remote_send_registers_a_distinct_web_peer(tools) -> None:
    fns, calls = tools
    with remote_peer(RemotePeer("ref-1", "notes", "My Notes")):
        asyncio.run(fns["mesh_send"]("other", "hi", refs=["fact:abcdef"], reply_to=9))
    reg = [b for _, p, b in calls if p == "/register"][0]
    assert reg["session_id"] == "ref-1" and reg["agent_type"] == "notes"
    send = [b for _, p, b in calls if p == "/send"][0]
    assert send["from_peer"] == "web-peer-id"
    assert send["refs"] == ["fact:abcdef"] and send["reply_to"] == 9
    assert "origin" not in send and "kind" not in send and "app" not in send


def test_two_remote_peers_get_two_identities(tools) -> None:
    fns, calls = tools
    with remote_peer(RemotePeer("ref-1", "notes", "N")):
        asyncio.run(fns["mesh_send"]("x", "a"))
    with remote_peer(RemotePeer("ref-2", "mail", "M")):
        asyncio.run(fns["mesh_send"]("x", "b"))
    froms = [b["from_peer"] for _, p, b in calls if p == "/send"]
    assert froms == ["web-peer-id", "other-web"]


def test_inbox_adds_preface_and_keeps_keys(tools) -> None:
    fns, calls = tools
    out = asyncio.run(fns["mesh_inbox"]())
    assert out["preface"].startswith("These are messages from other bots")
    assert set(out) >= {"messages", "count", "unread", "preface"}
    assert ("POST", "/inbox/local-peer/read", {"message_ids": [4]}) in calls


def test_wait_tool(tools) -> None:
    fns, calls = tools
    out = asyncio.run(fns["mesh_wait"](timeout_s=5))
    assert out["timed_out"] is False and out["messages"][0]["content"] == "hello"
    assert out["preface"].startswith("These are messages from other bots")
    wait_calls = [p for m, p, _ in calls if "/wait" in p]
    assert wait_calls and "timeout_s=5" in wait_calls[0] and "/inbox/local-peer/wait" in wait_calls[0]
    assert ("POST", "/inbox/local-peer/read", {"message_ids": [3]}) in calls


def test_wait_tool_clamps_and_survives_daemon_down(tools, monkeypatch) -> None:
    fns, calls = tools
    asyncio.run(fns["mesh_wait"](timeout_s=999))
    assert any("timeout_s=20" in p for _, p, _ in calls if "/wait" in p)
    monkeypatch.setattr(tools_mesh, "_mesh_request", lambda *a, **k: None)
    out = asyncio.run(fns["mesh_wait"]())
    assert out["messages"] == [] and out["timed_out"] is True
