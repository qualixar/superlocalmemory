"""L-10: stdio sessions join at startup, with broker identity and project."""

import asyncio

from superlocalmemory.mcp import tools_mesh
from tests.mcp.test_loop_tools import _Capture


def test_warmup_registers_with_heartbeat_and_broker_id(monkeypatch):
    monkeypatch.setenv("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    from superlocalmemory.mcp import server

    calls = []
    monkeypatch.setattr("superlocalmemory.cli.daemon.ensure_daemon", lambda: True)
    monkeypatch.setattr(tools_mesh, "_REGISTERED", False)
    monkeypatch.setattr(tools_mesh, "_PROJECT_PATH", "")  # another test may have cached one
    monkeypatch.setenv("SLM_SESSION_ID", "host-session")
    monkeypatch.setenv("SLM_PROJECT_PATH", "/work/project")
    monkeypatch.setattr(
        tools_mesh,
        "_mesh_request",
        lambda method, path, body=None: calls.append((path, body)) or {"peer_id": "broker-peer"},
    )
    monkeypatch.setattr(tools_mesh, "_start_heartbeat", lambda: calls.append(("heartbeat", None)))
    monkeypatch.setenv("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "0")
    server._eager_warmup()
    assert calls[0][0] == "/register"
    assert calls[0][1]["session_id"] == "host-session"
    assert calls[0][1]["project_path"] == "/work/project"
    assert tools_mesh._PEER_ID == "broker-peer"
    assert calls[-1][0] == "heartbeat"


def test_status_registers_before_reporting_my_peer_id(monkeypatch):
    monkeypatch.setattr(tools_mesh, "_REGISTERED", False)
    monkeypatch.setattr(tools_mesh, "_PEER_ID", "old-session-id")
    monkeypatch.setattr(tools_mesh, "_start_heartbeat", lambda: None)
    monkeypatch.setattr(
        tools_mesh,
        "_mesh_request",
        lambda method, path, body=None: (
            {"peer_id": "actual-peer"} if path == "/register" else {"broker_up": True}
        ),
    )
    server = _Capture()
    tools_mesh.register_mesh_tools(server, lambda: None)
    result = asyncio.run(server.fns["mesh_status"]())
    assert result["my_peer_id"] == "actual-peer"


def test_root_cwd_does_not_claim_to_be_a_project(monkeypatch):
    for key in ("SLM_PROJECT_PATH", "CLAUDE_PROJECT_DIR", "PROJECT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tools_mesh.os, "getcwd", lambda: "/")
    assert tools_mesh._detect_project_path() == ""


def test_project_binding_updates_the_same_registered_session(monkeypatch):
    calls = []
    monkeypatch.setattr(tools_mesh, "_REGISTERED", False)
    monkeypatch.setattr(tools_mesh, "_PEER_ID", "initial-id")
    monkeypatch.setattr(tools_mesh, "_start_heartbeat", lambda: None)
    monkeypatch.setattr(
        tools_mesh,
        "_mesh_request",
        lambda method, path, body=None: calls.append(body) or {"peer_id": "broker-peer"},
    )
    tools_mesh._ensure_registered(project_path="/work/first")
    tools_mesh._ensure_registered(project_path="/work/second")
    assert calls[-1]["project_path"] == "/work/second"
    assert calls[0]["session_id"] == calls[-1]["session_id"]
    assert tools_mesh._PEER_ID == "broker-peer"
