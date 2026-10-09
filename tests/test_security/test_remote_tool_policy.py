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
    assert not read & write_only and not read & host and not write_only & host
    classified = read | write_only | host
    assert names - classified == set(), (
        f"Classify new MCP tools in server/remote_tool_policy.py: {sorted(names - classified)}")
    assert classified - names == set(), f"Stale policy entries: {sorted(classified - names)}"
    assert len(names) == len(read) + len(write_only) + len(host)


def test_no_destructive_tool_is_readable_with_a_read_key(registered) -> None:
    for name in policy.READ_TOOLS:
        annotations = registered[name]
        assert annotations.get("destructiveHint") is not True, name


def test_host_only_tools_are_refused_for_every_scope() -> None:
    for name in policy.HOST_ONLY_TOOLS:
        assert not policy.tool_allowed("read", name)
        assert not policy.tool_allowed("write", name)
    for name in ("switch_profile", "build_code_graph", "forget", "run_maintenance",
                 "set_mode", "mesh_send", "slm_loop_run"):
        assert name in policy.HOST_ONLY_TOOLS


# -- the ASGI enforcement -----------------------------------------------------------------


class _StubMcp:
    """Stands in for the MCP app: records what reached it and answers JSON."""

    def __init__(self, tools: list[str] | None = None) -> None:
        self.reached: list[dict] = []
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
         method: str = "POST", sent_out: list | None = None):
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    stub = stub or _StubMcp()
    runtime = ProfileRuntime("default")
    app = policy.RemoteToolScopeASGI(stub, runtime_for=lambda _scope: runtime)
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


@pytest.mark.parametrize("tool", ["remember_media", "get_media"])
def test_image_tools_are_host_only_and_denied_for_read_and_write_keys(tool) -> None:
    assert tool in policy.HOST_ONLY_TOOLS and tool in policy.MEDIA_TOOLS
    assert not policy.tool_allowed("read", tool) and not policy.tool_allowed("write", tool)
    assert tool not in policy.READ_TOOLS | policy.WRITE_ONLY_TOOLS


def test_remote_image_rights_stay_off_in_this_build() -> None:
    assert policy.REMOTE_MEDIA_TOOLS_ENABLED is False


def test_mesh_wait_is_host_only_while_remote_mesh_is_off() -> None:
    assert policy.REMOTE_MESH_TOOLS_ENABLED is False
    assert "mesh_wait" in policy.HOST_ONLY_TOOLS
    assert not policy.tool_allowed("read", "mesh_wait")
    assert not policy.tool_allowed("write", "mesh_wait")
