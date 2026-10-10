"""A web app's mesh calls through the real remote policy, MCP server and in-process broker."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.mcp import tools_mesh
from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_grant
from superlocalmemory.remote_connections.grant import RemoteGrant, peer_ref
from superlocalmemory.server.remote_access import PRINCIPAL_SCOPE_KEY, RemotePrincipal
from superlocalmemory.server.remote_tool_policy import RemoteToolScopeASGI
from tests.test_mesh.conftest import make_peer, rows

CID = "a" * 32
OTHER_CID = "b" * 32
KEY = RemotePrincipal("remote-key", "rk_000000a1", "web-" + CID, "write", "default")
WORK_KEY = RemotePrincipal("remote-key", "rk_000000a2", "web-" + CID, "write", "work")


@dataclass(frozen=True)
class _Row:
    key_id: str
    extras: frozenset


class _Keys:
    def __init__(self, *rows_: tuple[RemotePrincipal, tuple]) -> None:
        self._rows = tuple(_Row(p.key_id, frozenset(e)) for p, e in rows_)

    def list(self):
        return self._rows


class _Runtime:
    snapshot = SimpleNamespace(profile_id="default")

    def acquire_operation(self):
        return self.snapshot

    def release_operation(self) -> None:
        return None


def _grant(aid: str = "auth-1", app: str = "client-1", *scopes: str,
           cid: str = CID) -> RemoteGrant:
    return RemoteGrant(connection_id=cid, authorization_id=aid, authorization_version=1,
                       app=app, scopes=frozenset({"slm:read", "slm:write", *scopes}),
                       folders_visible=False, key_version=1)


class _Daemon:
    """What the HTTP interface of the remote connection hands the policy layer."""

    def __init__(self, broker, keys, principal=KEY) -> None:
        mcp = SLMFastMCP("mesh-e2e")
        tools_mesh.register_mesh_tools(mcp, lambda: None)
        self.mcp_app = mcp.streamable_http_app(
            streamable_http_path="/", stateless_http=True, json_response=True,
            event_store=None, host="127.0.0.1")
        guarded = RemoteToolScopeASGI(self.mcp_app, runtime_for=lambda scope: _Runtime(),
                                      key_store=keys)
        app = SimpleNamespace(state=SimpleNamespace(mesh_broker=broker, config=None))
        self.principal = principal

        async def entry(scope, receive, send):
            await guarded(dict(scope, app=app, **{PRINCIPAL_SCOPE_KEY: self.principal}),
                          receive, send)

        self.entry = entry
        self._id = 0

    async def call(self, tool: str, arguments: dict | None = None, *,
                   grant: RemoteGrant | None = None) -> dict:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                "params": {"name": tool, "arguments": arguments or {}}}
        transport = httpx.ASGITransport(app=self.entry)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
            with remote_grant(grant):
                resp = await client.post(
                    "/", content=json.dumps(body),
                    headers={"content-type": "application/json",
                             "accept": "application/json, text/event-stream"})
        assert resp.status_code == 200, resp.text
        return resp.json()["result"]


@contextlib.asynccontextmanager
async def _daemon(broker, keys, principal=KEY):
    daemon = _Daemon(broker, keys, principal)
    async with daemon.mcp_app.router.lifespan_context(daemon.mcp_app):
        yield daemon


def _data(result: dict) -> dict:
    assert not result.get("isError"), result
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_http(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a web caller must not use the daemon's HTTP interface")

    monkeypatch.setattr(tools_mesh, "_mesh_request", refuse)
    tools_mesh._SEND_CIRCUIT.reset()


MESH = ("slm:mesh",)


def test_peers_send_and_inbox_work_with_a_grant_and_an_opted_in_key(broker) -> None:
    local = make_peer(broker, "sess-1", "claude_code")
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            a, b = _grant("auth-a", "app-a", *MESH), _grant("auth-b", "app-b", *MESH)
            peers = _data(await d.call("mesh_peers", grant=a))
            await d.call("mesh_peers", grant=b)
            sent = _data(await d.call("mesh_send", {"to": peer_ref(CID, "auth-b"),
                                                    "message": "hi B"}, grant=a))
            to_local = _data(await d.call("mesh_send", {"to": local, "message": "hi"}, grant=b))
            inbox = _data(await d.call("mesh_inbox", grant=_grant("auth-b", "app-b", *MESH)))
            return peers, sent, to_local, inbox

    peers, sent, to_local, inbox = _run(scenario())
    assert peers["my_peer_id"] == peer_ref(CID, "auth-a")
    assert {p["kind"] for p in peers["peers"]} == {"local", "web"}
    assert sent["ok"] is True and to_local["ok"] is True
    env = inbox["messages"][0]["envelope"]
    assert env["from"] == {"peer_id": peer_ref(CID, "auth-a"), "app": "app-a", "kind": "web"}
    assert inbox["messages"][0]["content"] == "hi B"
    local_msg = broker.get_inbox(local, "", "default")[0]
    assert local_msg["envelope"]["from"]["app"] == "app-b"
    assert local_msg["envelope"]["from"]["kind"] == "web"


def test_wait_and_state_work_through_the_policy(broker) -> None:
    broker.set_state("release", "v9", "owner")
    local = make_peer(broker, "sess-1")
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            g = _grant("auth-a", "app-a", *MESH)
            await d.call("mesh_peers", grant=g)
            broker.send_message(local, peer_ref(CID, "auth-a"), "wake")
            waited = _data(await d.call("mesh_wait", {"timeout_s": 2}, grant=g))
            state = _data(await d.call("mesh_state", {"key": "release"}, grant=g))
            return waited, state

    waited, state = _run(scenario())
    assert waited["messages"][0]["content"] == "wake" and waited["timed_out"] is False
    assert state["value"] == "v9"


def test_remote_mesh_state_cannot_write(broker) -> None:
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            return await d.call("mesh_state", {"key": "k", "value": "x", "action": "set"},
                                grant=_grant("auth-a", "app-a", *MESH))

    result = _run(scenario())
    assert result["isError"] is True and broker.get_state_key("k") is None


@pytest.mark.parametrize("grant_scopes,extras", [((), ("mesh",)), (MESH, ()), ((), ())])
def test_without_the_scope_or_the_key_opt_in_the_call_is_refused(
        broker, grant_scopes, extras) -> None:
    keys = _Keys((KEY, extras))

    async def scenario():
        async with _daemon(broker, keys) as d:
            return await d.call("mesh_peers", grant=_grant("auth-a", "app-a", *grant_scopes))

    result = _run(scenario())
    assert result["isError"] is True
    assert "remote_tool_not_allowed" in result["content"][0]["text"]
    assert rows(broker, "SELECT * FROM mesh_peers") == []


def test_without_a_grant_the_call_is_refused(broker) -> None:
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            return await d.call("mesh_peers")

    assert _run(scenario())["isError"] is True
    assert rows(broker, "SELECT * FROM mesh_peers") == []


def test_a_grant_for_another_connection_is_refused(broker) -> None:
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            return await d.call("mesh_peers", grant=_grant("auth-a", "app-a", *MESH,
                                                           cid=OTHER_CID))

    assert _run(scenario())["isError"] is True
    assert rows(broker, "SELECT * FROM mesh_peers") == []


def test_the_call_is_held_to_the_profile_of_the_key(broker) -> None:
    make_peer(broker, "sess-default")
    broker.register_peer("sess-work", "", profile_id="work")
    keys = _Keys((WORK_KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys, WORK_KEY) as d:
            return _data(await d.call("mesh_peers", grant=_grant("auth-a", "app-a", *MESH)))

    peers = _run(scenario())
    assert peers["count"] == 2
    assert rows(broker, "SELECT profile_id FROM mesh_peers WHERE peer_id=?",
                (peer_ref(CID, "auth-a"),))[0][0] == "work"


def test_a_revoked_apps_next_call_is_refused_and_its_mail_is_gone(broker) -> None:
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            g = _grant("auth-a", "app-a", *MESH)
            await d.call("mesh_peers", grant=g)
            broker.retire_missing_web_peers(CID, set())
            return await d.call("mesh_peers", grant=g)

    result = _run(scenario())
    assert "retired" in json.dumps(result)


def test_the_arguments_are_accepted_by_the_remote_binding(broker) -> None:
    keys = _Keys((KEY, ("mesh",)))

    async def scenario():
        async with _daemon(broker, keys) as d:
            g = _grant("auth-a", "app-a", *MESH)
            out = await d.call("mesh_send", {"to": "nobody", "message": "x", "refs": [],
                                             "reply_to": None}, grant=g)
            bad = await d.call("mesh_send", {"to": "nobody", "message": "x",
                                             "profile_id": "other"}, grant=g)
            return out, bad

    out, bad = _run(scenario())
    assert "not accepted" not in json.dumps(out)
    assert bad["isError"] is True
