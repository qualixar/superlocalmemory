"""Sandbox contracts for the eight daemon-backed Mesh MCP tools."""

from __future__ import annotations

import asyncio


class _ToolCollector:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorate(func):
            self.tools[func.__name__] = func
            return func

        return decorate


def test_mesh_mcp_tools_use_one_broker_identity_and_unread_delivery(monkeypatch) -> None:
    """Every exposed Mesh tool uses the broker identity minted at registration."""
    import superlocalmemory.mcp.tools_mesh as mesh_tools

    collector = _ToolCollector()
    mesh_tools.register_mesh_tools(collector, lambda: None)

    calls: list[tuple[str, str, dict | None]] = []

    def request(method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body))
        if path == "/register":
            return {"peer_id": "broker-peer", "pending_messages": 0}
        if path.startswith("/inbox/broker-peer?"):
            return {
                "messages": [
                    {"id": 7, "content": "one delivery", "read": False},
                    {"content": "malformed receipt", "read": False},
                ]
            }
        if path == "/peers":
            return {"peers": [{"peer_id": "other-peer"}]}
        if path == "/events":
            return {"events": [{"event_type": "message_sent"}]}
        if path == "/status":
            return {"broker_up": True, "peer_count": 2}
        if path == "/state" and method == "GET":
            return {"state": {"release": {"value": "v3.7"}}}
        if path == "/state/release":
            return {"key": "release", "value": "v3.7"}
        return {"ok": True}

    monkeypatch.setattr(mesh_tools, "_mesh_request", request)
    monkeypatch.setattr(mesh_tools, "_start_heartbeat", lambda: None)
    monkeypatch.setattr(mesh_tools, "_REGISTERED", False)
    monkeypatch.setattr(mesh_tools, "_PEER_ID", "local-provisional")
    monkeypatch.setattr(mesh_tools, "_HEARTBEAT_THREAD", None)
    monkeypatch.setattr(mesh_tools, "_PROJECT_PATH", "/sandbox/project")
    monkeypatch.setenv("PROJECT_PATH", "/sandbox/project")

    async def exercise() -> None:
        assert (await collector.tools["mesh_summary"]("audit mesh"))["peer_id"] == "broker-peer"
        assert (await collector.tools["mesh_peers"]())["count"] == 1
        assert (await collector.tools["mesh_send"]("other-peer", "ready"))["ok"] is True
        inbox = await collector.tools["mesh_inbox"]()
        assert inbox["unread"] == 1
        assert inbox["count"] == 2
        assert (await collector.tools["mesh_state"]())["state"]["release"]["value"] == "v3.7"
        assert (await collector.tools["mesh_state"]("release"))["value"] == "v3.7"
        assert (await collector.tools["mesh_state"]("release", "v3.7", "set"))["ok"] is True
        assert (await collector.tools["mesh_lock"]("/sandbox/file.py", "acquire"))["ok"] is True
        assert len((await collector.tools["mesh_events"]())["events"]) == 1
        assert (await collector.tools["mesh_status"]())["my_peer_id"] == "broker-peer"

    asyncio.run(exercise())

    assert calls[0] == (
        "POST",
        "/register",
        {
            "peer_id": "local-provisional",
            "session_id": "local-provisional",
            "summary": "audit mesh",
            "project_path": "/sandbox/project",
            "agent_type": "claude_code",
        },
    )
    assert ("POST", "/send", {"from_peer": "broker-peer", "to_peer": "other-peer", "content": "ready"}) in calls
    assert ("POST", "/inbox/broker-peer/read", {"message_ids": [7]}) in calls
    assert ("POST", "/state", {"key": "release", "value": "v3.7", "set_by": "broker-peer"}) in calls
    assert ("POST", "/lock", {"file_path": "/sandbox/file.py", "action": "acquire", "locked_by": "broker-peer"}) in calls
