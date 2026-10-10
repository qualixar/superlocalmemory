# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Default-deny tool policy for MCP callers on other computers."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from superlocalmemory.server import remote_tool_policy as policy
from superlocalmemory.server.remote_access import PRINCIPAL_SCOPE_KEY, RemotePrincipal

REPO = Path(__file__).resolve().parents[2]
READ_KEY = RemotePrincipal("remote-key", "rk_00000001", "viewer", "read", "default")
WRITE_KEY = RemotePrincipal("remote-key", "rk_00000002", "hermes", "write", "default")


def _registered_tools() -> dict[str, dict]:
    """Every tool the MCP server can register (all profiles), with annotations."""
    code = (
        "import asyncio, json\n"
        "from superlocalmemory.mcp.server import server\n"
        "tools = asyncio.run(server.list_tools())\n"
        "print(json.dumps({t.name: (t.annotations.model_dump() if t.annotations else {})"
        " for t in tools}))\n"
    )
    env = {**os.environ, "SLM_MCP_ALL_TOOLS": "1", "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=240, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def registered() -> dict[str, dict]:
    return _registered_tools()


def test_every_registered_tool_is_classified_exactly_once(registered) -> None:
    names = set(registered)
    read, write_only, host = policy.READ_TOOLS, policy.WRITE_ONLY_TOOLS, policy.HOST_ONLY_TOOLS
    mesh, media = policy.MESH_TOOLS, policy.MEDIA_TOOLS
    groups = [read, write_only, host, mesh, media]
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            assert not left & right
    classified = read | write_only | host | mesh | media
    assert names - classified == set(), (
        f"Classify new MCP tools in server/remote_tool_policy.py: {sorted(names - classified)}")
    # Media tools may be classified before the tools that serve them are registered.
    assert (classified - names) - media == set(), (
        f"Stale policy entries: {sorted((classified - names) - media)}")


def test_no_destructive_tool_is_readable_with_a_read_key(registered) -> None:
    for name in policy.READ_TOOLS:
        annotations = registered[name]
        assert annotations.get("destructiveHint") is not True, name


def test_host_only_tools_are_refused_for_every_scope() -> None:
    for name in policy.HOST_ONLY_TOOLS:
        assert not policy.tool_allowed("read", name)
        assert not policy.tool_allowed("write", name)
    for name in ("switch_profile", "build_code_graph", "forget", "run_maintenance",
                 "set_mode", "mesh_lock", "slm_loop_run"):
        assert name in policy.HOST_ONLY_TOOLS


# -- the ASGI enforcement -----------------------------------------------------------------


class _StubMcp:
    """Stands in for the MCP app: records what reached it and answers JSON."""

    def __init__(self, tools: list[str] | None = None) -> None:
        self.reached: list[dict] = []
        self.peers: list = []
        self._tools = tools or sorted(policy.WRITE_TOOLS | policy.HOST_ONLY_TOOLS)

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        request = json.loads(body)
        self.reached.append(request)
        from superlocalmemory.mcp.remote_caller import current_remote_peer

        self.peers.append(current_remote_peer())
        if request["method"] == "tools/list":
            result = {"tools": [{"name": n, "inputSchema": {}} for n in self._tools]}
        else:
            result = {"content": [{"type": "text", "text": "ran"}], "isError": False}
        payload = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})


def _run(body: bytes, principal=WRITE_KEY, stub: _StubMcp | None = None, chunks: int = 1,
         method: str = "POST", sent_out: list | None = None, key_store=None):
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    stub = stub or _StubMcp()
    runtime = ProfileRuntime("default")
    app = policy.RemoteToolScopeASGI(stub, runtime_for=lambda _scope: runtime,
                                     key_store=key_store)
    scope = {"type": "http", "method": method, "path": "/mcp/hermes", "root_path": "/mcp",
             "headers": [], "client": ("remote-listener-peer", 1),
             "slm_remote_listener": True}
    if principal is not None:
        scope[PRINCIPAL_SCOPE_KEY] = principal
    size = max(1, len(body) // chunks + 1)
    parts = [body[i:i + size] for i in range(0, len(body), size)] or [b""]
    queue = [{"type": "http.request", "body": p, "more_body": i < len(parts) - 1}
             for i, p in enumerate(parts)]
    sent: list[dict] = [] if sent_out is None else sent_out

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, (json.loads(raw) if raw else None), stub


def _call(tool, rid=1, **params) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                       "params": {"name": tool, "arguments": {}, **params}}).encode()


def test_read_key_cannot_remember() -> None:
    status, body, stub = _run(_call("remember"), READ_KEY)
    assert status == 200
    assert body["result"]["isError"] is True
    assert "read-only" in body["result"]["content"][0]["text"]
    assert stub.reached == []


def test_read_key_can_recall_and_write_key_can_remember() -> None:
    assert _run(_call("recall"), READ_KEY)[2].reached
    assert _run(_call("remember"), WRITE_KEY)[2].reached


def test_no_key_can_switch_profile_or_build_code_graph() -> None:
    for tool in ("switch_profile", "build_code_graph", "forget", "run_maintenance"):
        for principal in (READ_KEY, WRITE_KEY):
            status, body, stub = _run(_call(tool), principal)
            assert status == 200 and body["result"]["isError"] is True, tool
            assert stub.reached == [], tool


@pytest.mark.parametrize("alias", ["Remember", "remember ", " remember", "REMEMBER",
                                   "remember​", "remеmber", "remember/",
                                   "superlocalmemory.remember", "mcp__superlocalmemory__remember"])
def test_crit_read_key_cannot_reach_a_write_tool_by_an_alias(alias) -> None:
    """CRIT 2: names are matched exactly; an alias of a write tool is unknown, so refused."""
    status, body, stub = _run(_call(alias), READ_KEY)
    assert status == 200 and body["result"]["isError"] is True
    assert stub.reached == []


def test_crit_duplicate_name_key_cannot_smuggle_a_write_tool() -> None:
    """CRIT 2: {"name":"recall","name":"remember"} - parsers disagree on which wins."""
    body = (b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
            b'"params":{"name":"recall","name":"remember","arguments":{}}}')
    status, payload, stub = _run(body, READ_KEY)
    assert status == 400 and payload["error"] == "duplicate_json_key"
    assert stub.reached == []


def test_crit_non_string_tool_name_is_refused() -> None:
    for name in (["remember"], {"x": "remember"}, None, 5):
        status, payload, stub = _run(_call(name), READ_KEY)
        assert status == 400 and stub.reached == []


def test_batch_with_one_denied_call_is_refused_whole() -> None:
    batch = json.dumps([json.loads(_call("recall", 1)), json.loads(_call("switch_profile", 2))])
    status, payload, stub = _run(batch.encode(), WRITE_KEY)
    assert status == 400 and payload["error"] == "batch_not_supported"
    assert stub.reached == []


@pytest.mark.parametrize("method", ["resources/read", "resources/list", "prompts/get",
                                    "completion/complete", "logging/setLevel",
                                    "sampling/createMessage", "tools/call2"])
def test_methods_other_than_tools_are_refused(method) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": method, "params": {}}).encode()
    status, payload, stub = _run(body, WRITE_KEY)
    assert status == 200 and payload["error"]["code"] == -32601
    assert stub.reached == []


def test_tools_list_hides_tools_the_key_cannot_call() -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).encode()
    _, read_list, _ = _run(body, READ_KEY)
    _, write_list, _ = _run(body, WRITE_KEY)
    read_names = {t["name"] for t in read_list["result"]["tools"]}
    write_names = {t["name"] for t in write_list["result"]["tools"]}
    assert read_names == set(policy.READ_TOOLS)
    assert write_names == set(policy.WRITE_TOOLS)
    assert "switch_profile" not in write_names and "remember" not in read_names


def test_body_over_1_mib_is_413() -> None:
    big = _call("remember", content="x" * (policy.MAX_BODY_BYTES + 10))
    status, payload, stub = _run(big, WRITE_KEY, chunks=8)
    assert status == 413 and stub.reached == []


def test_non_json_body_is_400() -> None:
    status, payload, stub = _run(b"not json", WRITE_KEY)
    assert status == 400 and payload["error"] == "invalid_jsonrpc"
    assert stub.reached == []


def test_policy_denial_is_a_tool_error_with_http_200() -> None:
    status, body, _ = _run(_call("forget", rid=42), WRITE_KEY)
    assert status == 200
    assert body["id"] == 42 and body["result"]["isError"] is True
    assert body["result"]["structuredContent"]["error"] == policy.DENIAL_CODE


def test_a_network_request_without_a_principal_fails_closed() -> None:
    status, payload, stub = _run(_call("recall"), principal=None)
    assert status == 401 and stub.reached == []


def test_audit_line_names_the_key_and_tool_but_never_arguments(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="superlocalmemory.remote.audit")
    _run(_call("remember", arguments={"content": "SECRET-CONTENT-123"}), WRITE_KEY)
    _run(_call("switch_profile"), WRITE_KEY)
    text = caplog.text
    assert "key_name=hermes" in text and "tool=remember" in text and "decision=allow" in text
    assert "tool=switch_profile" in text and "decision=deny" in text
    assert "SECRET-CONTENT-123" not in text


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
def test_non_post_from_a_remote_caller_is_405_without_reaching_mcp(method) -> None:
    """A GET used to open an event stream that never carries a message (stateless)."""
    sent: list[dict] = []
    status, payload, stub = _run(b"", WRITE_KEY, method=method, sent_out=sent)
    assert status == 405 and payload["error"] == "remote_method_not_allowed"
    assert stub.reached == []
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert (b"allow", b"POST") in start["headers"]


def test_the_mcp_app_sees_which_remote_key_is_calling() -> None:
    """Per-agent stores key on this, not on the caller-chosen /mcp/<agent> segment."""
    from superlocalmemory.mcp.remote_caller import current_remote_key_id

    seen: list[str | None] = []

    class _Recording(_StubMcp):
        async def __call__(self, scope, receive, send) -> None:
            seen.append(current_remote_key_id())
            await super().__call__(scope, receive, send)

    _run(_call("slm_cache_get", arguments={"key": "k"}), READ_KEY, stub=_Recording())
    _run(_call("slm_cache_set", arguments={"key": "k", "value": "v"}), WRITE_KEY,
         stub=_Recording())
    assert seen == [READ_KEY.key_id, WRITE_KEY.key_id]
    assert current_remote_key_id() is None


# -- mesh and media tools: signed grant x key opt-in ----------------------------------------

from superlocalmemory.mcp.remote_caller import remote_grant  # noqa: E402
from superlocalmemory.remote_connections.grant import RemoteGrant, peer_ref  # noqa: E402

CID = "a" * 32
WEB_WRITE = RemotePrincipal("remote-key", "rk_00000003", "web-" + CID, "write", "default")
WEB_READ = RemotePrincipal("remote-key", "rk_00000004", "web-" + CID, "read", "default")


def _grant(*scopes: str, cid: str = CID) -> RemoteGrant:
    return RemoteGrant(connection_id=cid, authorization_id="auth-1", authorization_version=1,
                       app="client-1", scopes=frozenset({"slm:read", *scopes}),
                       folders_visible=False, key_version=1)


class _Keys:
    def __init__(self, principal: RemotePrincipal, extras=()) -> None:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Row:
            key_id: str
            extras: frozenset

        self._rows = (Row(principal.key_id, frozenset(extras)),)

    def list(self):
        return self._rows


MESH_ALL = sorted(policy.MESH_TOOLS)
MEDIA_ALL = sorted(policy.MEDIA_TOOLS)


def test_mesh_and_media_sets_match_the_design() -> None:
    assert policy.MESH_TOOLS == {"mesh_peers", "mesh_send", "mesh_inbox", "mesh_wait",
                                 "mesh_state"}
    assert policy.MEDIA_TOOLS == {"remember_media", "get_media", "remember_document",
                                  "media_status"}
    assert not hasattr(policy, "REMOTE_MESH_TOOLS_ENABLED")
    for host_only in ("mesh_lock", "mesh_events", "mesh_status", "mesh_summary"):
        assert host_only in policy.HOST_ONLY_TOOLS


@pytest.mark.parametrize("tool", MESH_ALL)
def test_mesh_tool_needs_grant_scope_and_key_opt_in(tool) -> None:
    mesh = _grant("slm:mesh")
    for scope in ("read", "write"):
        assert policy.tool_allowed(scope, tool, mesh, frozenset({"mesh"}))
        assert not policy.tool_allowed(scope, tool)                      # no grant
        assert not policy.tool_allowed(scope, tool, mesh)                # key not opted in
        assert not policy.tool_allowed(scope, tool, mesh, frozenset({"media"}))
        assert not policy.tool_allowed(scope, tool, _grant(), frozenset({"mesh"}))
        assert not policy.tool_allowed(scope, tool, _grant("slm:media"), frozenset({"mesh"}))


@pytest.mark.parametrize("tool", ["get_media", "media_status"])
def test_media_read_tools_need_media_scope_and_opt_in(tool) -> None:
    media = _grant("slm:media")
    assert policy.tool_allowed("read", tool, media, frozenset({"media"}))
    assert policy.tool_allowed("write", tool, media, frozenset({"media"}))
    assert not policy.tool_allowed("read", tool, media)
    assert not policy.tool_allowed("read", tool, _grant("slm:mesh"), frozenset({"media"}))
    assert not policy.tool_allowed("read", tool, None, frozenset({"media"}))


@pytest.mark.parametrize("tool", ["remember_media", "remember_document"])
def test_media_save_tools_also_need_a_write_key_and_write_scope(tool) -> None:
    full = _grant("slm:write", "slm:media")
    extras = frozenset({"media"})
    assert policy.tool_allowed("write", tool, full, extras)
    assert not policy.tool_allowed("read", tool, full, extras)
    assert not policy.tool_allowed("write", tool, _grant("slm:media"), extras)
    assert not policy.tool_allowed("write", tool, _grant("slm:write"), extras)
    assert not policy.tool_allowed("write", tool, full)


def test_unknown_scope_or_name_type_is_refused_with_a_grant() -> None:
    assert not policy.tool_allowed("admin", "mesh_peers", _grant("slm:mesh"), frozenset({"mesh"}))
    assert not policy.tool_allowed("write", ["mesh_peers"], _grant("slm:mesh"),
                                   frozenset({"mesh"}))


def _tools_list(principal, keys, stub_tools):
    body = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).encode()
    _, answer, _ = _run(body, principal, stub=_StubMcp(stub_tools), key_store=keys)
    return {t["name"] for t in answer["result"]["tools"]}


def test_tools_list_matches_tools_call_for_every_combination() -> None:
    every = sorted(policy.MESH_TOOLS | policy.MEDIA_TOOLS
                   | {"recall", "remember", "switch_profile", "mesh_lock"})
    for principal in (WEB_READ, WEB_WRITE):
        for scopes in ((), ("slm:mesh",), ("slm:media",), ("slm:write", "slm:media", "slm:mesh")):
            for extras in ((), ("mesh",), ("media",), ("mesh", "media")):
                keys = _Keys(principal, extras)
                with remote_grant(_grant(*scopes)):
                    listed = _tools_list(principal, keys, every)
                    for tool in every:
                        _, body, stub = _run(_call(tool, arguments=_ok_args(tool)),
                                             principal, key_store=keys)
                        ran = bool(stub.reached)
                        assert ran == (tool in listed), (tool, principal.scope, scopes, extras)


def _ok_args(tool: str) -> dict:
    return {"mesh_state": {"key": "k"}}.get(tool, {})


def test_listing_without_a_grant_is_exactly_today() -> None:
    every = sorted(policy.WRITE_TOOLS | policy.MESH_TOOLS | policy.MEDIA_TOOLS)
    assert _tools_list(WEB_WRITE, _Keys(WEB_WRITE, ("mesh", "media")), every) == set(
        policy.WRITE_TOOLS)
    assert _tools_list(WEB_READ, _Keys(WEB_READ, ("mesh",)), every) == set(policy.READ_TOOLS)


def test_a_grant_for_another_connection_is_ignored(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="superlocalmemory.remote.audit")
    keys = _Keys(WEB_WRITE, ("mesh",))
    with remote_grant(_grant("slm:mesh", cid="b" * 32)):
        _, body, stub = _run(_call("mesh_peers"), WEB_WRITE, key_store=keys)
    assert stub.reached == [] and body["result"]["isError"] is True
    assert "grant_mismatch" in caplog.text


def test_a_grant_is_ignored_for_a_key_that_is_not_a_web_connection_key() -> None:
    keys = _Keys(WRITE_KEY, ("mesh",))
    with remote_grant(_grant("slm:mesh")):
        _, body, stub = _run(_call("mesh_peers"), WRITE_KEY, key_store=keys)
    assert stub.reached == []


def test_mesh_denial_text_names_the_missing_pieces() -> None:
    with remote_grant(_grant()):
        _, body, _ = _run(_call("mesh_peers"), WEB_WRITE, key_store=_Keys(WEB_WRITE))
    text = body["result"]["content"][0]["text"]
    assert "other bots" in text and "slm remote keys allow web-" + CID + " mesh" in text
    assert "[remote_tool_not_allowed]" in text
    _, body, _ = _run(_call("get_media"), WEB_WRITE, key_store=_Keys(WEB_WRITE))
    text = body["result"]["content"][0]["text"]
    assert "images and documents" in text and "keys allow web-" + CID + " media" in text


def test_a_mesh_call_runs_as_the_web_peer_and_the_audit_names_it(caplog) -> None:
    import logging

    from superlocalmemory.remote_connections import peer_names

    caplog.set_level(logging.INFO, logger="superlocalmemory.remote.audit")
    keys = _Keys(WEB_WRITE, ("mesh",))
    ref = peer_ref(CID, "auth-1")
    peer_names.set_names(CID, {})
    with remote_grant(_grant("slm:mesh")):
        _, _, stub = _run(_call("mesh_peers"), WEB_WRITE, key_store=keys)
        peer_names.set_names(CID, {"auth-1": "ChatGPT"})
        _, _, named = _run(_call("mesh_peers"), WEB_WRITE, key_store=keys)
        _, _, plain = _run(_call("recall"), WEB_WRITE, key_store=keys)
    assert stub.peers[0].peer_ref == ref and stub.peers[0].app == "client-1"
    assert stub.peers[0].display_name == "Web app " + ref[2:8]
    assert named.peers[0].display_name == "ChatGPT"
    assert plain.peers == [None]
    assert f"app={ref}" in caplog.text
    peer_names.set_names(CID, {})


@pytest.mark.parametrize("arguments", [
    {"key": "k", "action": "set", "value": "v"}, {"key": "k", "action": "delete"},
    {"key": ""}, {}, {"key": 5}, {"key": "x" * 257}, {"action": "get"},
])
def test_remote_mesh_state_is_get_only_with_a_real_key(arguments) -> None:
    keys = _Keys(WEB_WRITE, ("mesh",))
    with remote_grant(_grant("slm:mesh")):
        _, body, stub = _run(_call("mesh_state", arguments=arguments), WEB_WRITE, key_store=keys)
    assert stub.reached == [] and body["result"]["isError"] is True
    assert "remote_argument_not_allowed" in body["result"]["content"][0]["text"]


def test_remote_mesh_state_get_is_allowed() -> None:
    keys = _Keys(WEB_WRITE, ("mesh",))
    with remote_grant(_grant("slm:mesh")):
        for arguments in ({"key": "k"}, {"key": "k", "action": "get"}):
            _, _, stub = _run(_call("mesh_state", arguments=arguments), WEB_WRITE,
                              key_store=keys)
            assert stub.reached
