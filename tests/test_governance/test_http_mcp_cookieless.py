"""Cookie-less MCP owners either retain their switch or get an honest refusal."""
# ruff: noqa: F811
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_governance.test_role_surface_matrix_mcp import store  # noqa: F401 - pytest fixture, used by name


@pytest.mark.parametrize("connection", ["transport-session", "explicit-binding", "unidentified"])
def test_cookieless_personal_switch_has_connection_continuity(store, monkeypatch, connection):
    from superlocalmemory.core.caller_session import CallerSessionASGI
    from superlocalmemory.mcp import server as srv
    from superlocalmemory.mcp.http_transport import SLMFastMCP
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.server.caller_workspace import CallerWorkspaceMiddleware

    rbac, _ = store
    rbac.set_require_login(False)
    engine = srv.get_engine()
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: False)
    probe = SLMFastMCP("cookie-less owner")
    register_core_tools(probe, srv.get_engine)

    @probe.tool()
    async def active_workspace() -> dict:
        return {"profile_id": srv.get_engine().profile_id}

    mcp_app = probe.streamable_http_app(
        streamable_http_path="/", stateless_http=connection != "transport-session",
        json_response=True,
    )

    @asynccontextmanager
    async def lifespan(_):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(lifespan=lifespan)
    app.state.rbac, app.state.engine = rbac, engine
    app.add_middleware(CallerWorkspaceMiddleware, app_state=app.state)
    app.mount("/mcp", CallerSessionASGI(mcp_app))
    headers = {"Accept": "application/json, text/event-stream",
               "Content-Type": "application/json", "MCP-Protocol-Version": "2025-06-18"}
    with TestClient(app, base_url="http://127.0.0.1:9911", client=("127.0.0.1", 5000)) as client:
        def request(method, params, bound_headers):
            client.cookies.clear()
            return client.post("/mcp/", headers={**headers, **bound_headers},
                               json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})

        def initialize():
            response = request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "cookie-less-test", "version": "1"}}, {})
            assert response.status_code == 200
            return {"Mcp-Session-Id": response.headers["mcp-session-id"]}

        def call(name, args, binding):
            response = request("tools/call", {"name": name, "arguments": args}, binding)
            assert response.status_code == 200
            return json.loads(response.json()["result"]["content"][0]["text"])

        if connection == "transport-session":
            first, second = initialize(), initialize()
        elif connection == "explicit-binding":
            first = {"X-SLM-Workspace-Binding": "first-personal-connection"}
            second = {"X-SLM-Workspace-Binding": "second-personal-connection"}
        else:
            first, second = {}, {}
        switched = call("switch_profile", {"profile_id": "sales"}, first)
        if connection == "unidentified":
            assert switched["success"] is False
            assert "connection" in switched["error"].lower()
            assert call("active_workspace", {}, first)["profile_id"] == "default"
        else:
            assert switched["success"] is True
            assert call("active_workspace", {}, first)["profile_id"] == "sales"
        assert call("active_workspace", {}, second)["profile_id"] == "default"
    assert engine.profile_id == engine._config.active_profile == "default"
