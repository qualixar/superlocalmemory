# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A remote key reaches one profile, for every tool, read and write (audit 4.1.20 L2 F2).

Before 4.1.20 a read key could recall another profile by passing
``profile_id``. These tests send what a hostile key holder would send.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from superlocalmemory.server import remote_profile_binding as binding
from superlocalmemory.server import remote_tool_policy as policy
from superlocalmemory.server.profile_runtime import ProfileRuntime
from superlocalmemory.server.remote_access import PRINCIPAL_SCOPE_KEY, RemotePrincipal

REPO = Path(__file__).resolve().parents[2]
READ_KEY = RemotePrincipal("remote-key", "rk_00000001", "viewer", "read", "work")
WRITE_KEY = RemotePrincipal("remote-key", "rk_00000002", "hermes", "write", "work")
LEGACY_KEY = RemotePrincipal("legacy-api-key", "api_key", "api_key", "write", None)


# -- the live registry: every argument of every remote tool is classified ----------------


@pytest.fixture(scope="module")
def registry() -> dict[str, list[str]]:
    code = (
        "import asyncio, json\n"
        "from superlocalmemory.mcp.server import server\n"
        "tools = asyncio.run(server.list_tools())\n"
        "print(json.dumps({t.name: sorted((t.input_schema or {}).get('properties', {}))"
        " for t in tools}))\n"
    )
    env = {**os.environ, "SLM_MCP_ALL_TOOLS": "1", "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=240, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_argument_of_every_remote_tool_is_classified(registry) -> None:
    callable_remotely = policy.WRITE_TOOLS | policy.MESH_TOOLS
    remote = {name: args for name, args in registry.items() if name in callable_remotely}
    assert remote, "the registry returned no remote-callable tools"
    assert policy.MESH_TOOLS <= set(remote)
    seen = {arg for args in remote.values() for arg in args}
    unclassified = {f"{t}.{a}" for t, args in remote.items() for a in args
                    if a not in binding.CLASSIFIED_ARGUMENTS}
    assert not unclassified, (
        "Classify these in server/remote_profile_binding.py: " + ", ".join(sorted(unclassified)))
    groups = (binding.PROFILE_ARGUMENTS, binding.READ_SCOPE_ARGUMENTS,
              binding.WRITE_SCOPE_ARGUMENTS, binding.NEUTRAL_ARGUMENTS)
    assert sum(len(g) for g in groups) == len(binding.CLASSIFIED_ARGUMENTS)
    assert binding.NEUTRAL_ARGUMENTS <= seen, sorted(binding.NEUTRAL_ARGUMENTS - seen)


def test_no_profile_or_scope_like_argument_is_filed_as_neutral(registry) -> None:
    looks_scoped = re.compile(r"profile|scope|shared|global|tenant|workspace|namespace")
    assert not {a for a in binding.NEUTRAL_ARGUMENTS if looks_scoped.search(a)}


def test_every_remote_tool_that_takes_scope_is_pinned_to_personal(registry) -> None:
    takes_scope = {t for t, args in registry.items()
                   if t in policy.WRITE_TOOLS and "scope" in args}
    assert takes_scope == binding.SCOPED_WRITE_TOOLS


def test_every_remote_tool_that_names_a_profile_is_told_the_keys_profile(registry) -> None:
    named = {t for t, args in registry.items()
             if t in policy.WRITE_TOOLS and set(args) & binding.PROFILE_ARGUMENTS}
    assert named == binding.PROFILE_ARGUMENT_TOOLS
    assert binding.ROUTED_TOOLS <= binding.PROFILE_ARGUMENT_TOOLS


def test_profile_free_tools_are_remote_tools_that_name_no_profile(registry) -> None:
    assert binding.PROFILE_FREE_TOOLS <= policy.WRITE_TOOLS
    assert not binding.PROFILE_FREE_TOOLS & binding.PROFILE_ARGUMENT_TOOLS
    for tool in binding.PROFILE_FREE_TOOLS:
        assert not set(registry[tool]) & binding.PROFILE_ARGUMENTS, tool


def test_every_remote_tool_is_routed_profile_free_or_active_only_with_a_reason(
        registry) -> None:
    """4.1.21: a key bound to one profile can use every remote tool while the
    computer is on another profile. A tool that cannot be routed must say why."""
    remote = set(policy.WRITE_TOOLS)
    routed, free = binding.ROUTED_TOOLS, binding.PROFILE_FREE_TOOLS
    active_only = set(binding.ACTIVE_ONLY_TOOLS)
    assert routed | free | active_only == remote, sorted(remote - routed - free - active_only)
    assert not routed & free and not routed & active_only and not free & active_only
    assert all(reason.strip() for reason in binding.ACTIVE_ONLY_TOOLS.values())
    assert active_only == set(), "every remote tool is routed in 4.1.21"
    assert {t for t in remote if t in registry} == remote, sorted(remote - set(registry))


# -- bind_arguments -----------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", "work", " work "])
@pytest.mark.parametrize("tool", sorted(binding.PROFILE_ARGUMENT_TOOLS))
def test_the_bound_profile_or_no_profile_is_accepted_and_made_explicit(tool, value) -> None:
    out = binding.bind_arguments(tool, {"query": "q", "profile_id": value},
                                 key_name="k", bound="work")
    assert out["profile_id"] == "work"


def test_prestage_context_without_a_profile_is_told_the_keys_profile() -> None:
    """Its own default is the profile named "default", not the key's."""
    out = binding.bind_arguments("prestage_context", {"query": "q"}, key_name="k",
                                 bound="work")
    assert out["profile_id"] == "work"


@pytest.mark.parametrize("value", ["clientx", "Work", "work2", 5, ["work"], {"id": "work"}])
def test_any_other_profile_is_refused(value) -> None:
    with pytest.raises(binding.BindingRefusal) as err:
        binding.bind_arguments("recall", {"profile_id": value}, key_name="k", bound="work")
    assert err.value.code == binding.PROFILE_DENIAL
    assert "bound to profile 'work'" in str(err.value)


@pytest.mark.parametrize("arguments", [
    {"payload": {"profile_id": "clientx"}},
    {"payload": {"nested": [{"profile_id": "clientx"}]}},
    {"items": [{"fact_id": "f", "profile_id": "clientx"}]},
    {"outcome": {"evidence": {"profile_id": "clientx"}}},
])
def test_a_profile_hidden_inside_a_structured_argument_is_refused(arguments) -> None:
    with pytest.raises(binding.BindingRefusal):
        binding.bind_arguments("record_agent_experience", arguments, key_name="k", bound="work")


def test_absurdly_deep_arguments_are_refused() -> None:
    deep: object = "x"
    for _ in range(40):
        deep = {"a": deep}
    with pytest.raises(binding.BindingRefusal):
        binding.bind_arguments("remember", {"payload": deep}, key_name="k", bound="work")


@pytest.mark.parametrize("arguments", [
    {"scope": "global"}, {"scope": "shared"}, {"scope": "GLOBAL"},
    {"shared_with": "clientx"}, {"shared_with": ["clientx"]},
    {"scope": "personal", "shared_with": "clientx"},
])
def test_a_remote_save_cannot_reach_other_profiles(arguments) -> None:
    with pytest.raises(binding.BindingRefusal) as err:
        binding.bind_arguments("remember", {"content": "c", **arguments}, key_name="k",
                               bound="work")
    assert err.value.code == binding.SCOPE_DENIAL


def test_a_remote_save_without_a_scope_is_made_personal() -> None:
    """A host whose default scope is global must not turn a remote save global."""
    for args in ({"content": "c"}, {"content": "c", "scope": ""}, {"content": "c", "scope": None}):
        assert binding.bind_arguments("remember", args, key_name="k",
                                      bound="work")["scope"] == "personal"


def test_recall_may_include_what_was_shared_with_the_bound_profile() -> None:
    out = binding.bind_arguments("recall", {"query": "q", "include_shared": True,
                                            "include_global": True}, key_name="k", bound="w")
    assert out["include_shared"] is True and out["include_global"] is True


def test_unclassified_arguments_and_non_objects_are_refused() -> None:
    with pytest.raises(binding.BindingRefusal):
        binding.bind_arguments("recall", {"target_profile": "x"}, key_name="k", bound="w")
    with pytest.raises(binding.BindingRefusal):
        binding.bind_arguments("recall", ["profile_id", "x"], key_name="k", bound="w")


# -- the ASGI enforcement -----------------------------------------------------------------


class _Stub:
    def __init__(self, runtime: ProfileRuntime) -> None:
        self.runtime = runtime
        self.reached: list[dict] = []
        self.leases_during_call: list[int] = []

    async def __call__(self, scope, receive, send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.reached.append(json.loads(body))
        self.leases_during_call.append(self.runtime._active_operations)
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [{"type": "text", "text": "ran"}], "isError": False}}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})


def _run(tool: str, arguments: dict, principal=READ_KEY, active: str = "work",
         runtime: ProfileRuntime | None = None):
    runtime = runtime or ProfileRuntime(active)
    stub = _Stub(runtime)
    app = policy.RemoteToolScopeASGI(stub, runtime_for=lambda _scope: runtime)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": arguments}}).encode()
    scope = {"type": "http", "method": "POST", "path": "/mcp/hermes", "root_path": "/mcp",
             "headers": [(b"content-length", str(len(body)).encode())],
             "client": ("peer", 1), "slm_remote_listener": True, PRINCIPAL_SCOPE_KEY: principal}
    queue = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    answer = json.loads(b"".join(m.get("body", b"") for m in sent
                                 if m["type"] == "http.response.body"))
    return status, answer, stub, runtime


@pytest.mark.parametrize("principal", [READ_KEY, WRITE_KEY])
@pytest.mark.parametrize("tool", ["recall", "list_corrections", "prestage_context"])
def test_a_read_or_write_key_cannot_read_another_profile(principal, tool) -> None:
    status, answer, stub, _ = _run(tool, {"query": "q", "profile_id": "clientx"}, principal)
    assert status == 200 and answer["result"]["isError"] is True
    assert answer["result"]["structuredContent"]["error"] == binding.PROFILE_DENIAL
    assert stub.reached == []


@pytest.mark.parametrize("arguments", [
    {"content": "c", "profile_id": "clientx"}, {"content": "c", "scope": "global"},
    {"content": "c", "scope": "shared", "shared_with": "clientx"},
])
def test_a_write_key_cannot_save_into_another_profile(arguments) -> None:
    status, answer, stub, _ = _run("remember", arguments, WRITE_KEY)
    assert answer["result"]["isError"] is True and stub.reached == []


def test_review_correction_cannot_reach_another_profile() -> None:
    _, answer, stub, _ = _run("review_correction",
                              {"case_id": "c1", "action": "accept", "profile_id": "clientx"},
                              WRITE_KEY)
    assert answer["result"]["isError"] is True and stub.reached == []


@pytest.mark.parametrize("active", ["work", "personal"])
def test_remember_and_recall_are_routed_to_the_keys_profile_whatever_is_active(active) -> None:
    """Host on 'personal', key bound to 'work': the call is served for 'work'."""
    _, answer, stub, runtime = _run("remember", {"content": "c"}, WRITE_KEY, active=active)
    assert answer["result"]["isError"] is False
    assert stub.reached[0]["params"]["arguments"] == {
        "content": "c", "scope": "personal", "profile_id": "work"}
    _, answer, stub, runtime = _run("recall", {"query": "q"}, READ_KEY, active=active)
    assert answer["result"]["isError"] is False
    assert stub.reached[0]["params"]["arguments"]["profile_id"] == "work"
    # Routed: no lease, and the host's active profile is left as it was.
    assert stub.leases_during_call == [0] and runtime.snapshot.profile_id == active


@pytest.mark.parametrize("active", ["work", "personal"])
@pytest.mark.parametrize("tool, args", [
    ("search", {"query": "q"}), ("fetch", {"fact_ids": "f1"}), ("list_recent", {}),
])
def test_read_back_tools_are_routed_to_the_keys_profile_whatever_is_active(
        tool, args, active) -> None:
    """4.1.21: a remote agent can check what it saved while the host is elsewhere."""
    _, answer, stub, runtime = _run(tool, args, READ_KEY, active=active)
    assert answer["result"]["isError"] is False, tool
    assert stub.reached[0]["params"]["arguments"]["profile_id"] == "work"
    assert stub.leases_during_call == [0] and runtime.snapshot.profile_id == active


@pytest.mark.parametrize("tool, args", [
    ("search", {"query": "q"}), ("fetch", {"fact_ids": "f1"}), ("list_recent", {}),
])
def test_read_back_tools_cannot_name_another_profile(tool, args) -> None:
    _, answer, stub, _ = _run(tool, {**args, "profile_id": "clientx"}, READ_KEY)
    assert answer["result"]["isError"] is True and stub.reached == []
    assert answer["result"]["structuredContent"]["error"] == binding.PROFILE_DENIAL


@pytest.mark.parametrize("tool", sorted(binding.PROFILE_FREE_TOOLS))
def test_profile_free_tools_run_whatever_is_active(tool) -> None:
    _, answer, stub, _ = _run(tool, {}, WRITE_KEY, active="personal")
    assert answer["result"]["isError"] is False and stub.reached


@pytest.fixture
def active_only(monkeypatch):
    """No tool is active-only in 4.1.21, so one is made so to test that path."""
    tool = "memory_kinds_status"
    monkeypatch.setattr(binding, "ROUTED_TOOLS", binding.ROUTED_TOOLS - {tool})
    return tool


def test_an_active_only_tool_holds_the_lease_for_the_whole_call(active_only) -> None:
    _, answer, stub, runtime = _run(active_only, {}, READ_KEY, active="work")
    assert answer["result"]["isError"] is False
    assert stub.leases_during_call == [1] and runtime._active_operations == 0


def test_an_active_only_tool_is_refused_while_another_profile_is_active(active_only) -> None:
    for tool, args in ((active_only, {}),):
        _, answer, stub, runtime = _run(tool, args, READ_KEY, active="secret-client")
        text = answer["result"]["content"][0]["text"]
        assert answer["result"]["isError"] is True and stub.reached == [], tool
        assert answer["result"]["structuredContent"]["error"] == binding.INACTIVE_DENIAL
        assert "another workspace right now" in text and "ask the host owner" in text
        assert "'work'" in text and "recall" in text and "secret-client" not in text
        assert runtime._active_operations == 0


_ROUTED = sorted(binding.ROUTED_TOOLS)


def _key_for(tool: str) -> RemotePrincipal:
    return WRITE_KEY if tool in policy.WRITE_ONLY_TOOLS else READ_KEY


@pytest.mark.parametrize("tool", _ROUTED)
def test_every_routed_tool_is_served_for_the_keys_profile_while_the_host_is_elsewhere(
        tool) -> None:
    """Host on 'personal', key bound to 'work': told 'work', no lease, host unmoved."""
    _, answer, stub, runtime = _run(tool, {}, _key_for(tool), active="personal")
    assert answer["result"]["isError"] is False, (tool, answer)
    assert stub.reached[0]["params"]["arguments"]["profile_id"] == "work", tool
    assert stub.leases_during_call == [0] and runtime.snapshot.profile_id == "personal"


@pytest.mark.parametrize("tool", _ROUTED)
def test_every_routed_tool_refuses_another_profile_in_its_arguments(tool) -> None:
    for arguments in ({"profile_id": "clientx"}, {"profile_id": "personal"},
                      {"payload": {"profile_id": "clientx"}}):
        _, answer, stub, _ = _run(tool, arguments, _key_for(tool), active="personal")
        assert answer["result"]["isError"] is True and stub.reached == [], (tool, arguments)
        assert answer["result"]["structuredContent"]["error"] == binding.PROFILE_DENIAL


@pytest.mark.parametrize("tool", sorted(policy.WRITE_ONLY_TOOLS))
def test_a_read_only_key_is_refused_every_write(tool) -> None:
    _, answer, stub, _ = _run(tool, {}, READ_KEY, active="personal")
    assert answer["result"]["isError"] is True and stub.reached == [], tool
    assert policy.READ_ONLY_TAG in answer["result"]["content"][0]["text"], tool


def test_a_routed_write_is_logged_with_the_remote_key_and_its_profile(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="superlocalmemory.remote.audit"):
        _, answer, stub, _ = _run("delete_memory", {"fact_id": "f1"}, WRITE_KEY,
                                  active="personal")
    assert answer["result"]["isError"] is False and stub.reached
    [line] = [r.getMessage() for r in caplog.records if "tools/call" in r.getMessage()]
    assert "key_id=rk_00000002" in line and "profile=work" in line, line
    assert "tool=delete_memory" in line and "decision=allow" in line, line


def test_the_older_api_key_reaches_only_the_active_profile() -> None:
    _, answer, stub, _ = _run("recall", {"query": "q", "profile_id": "clientx"}, LEGACY_KEY,
                              active="work")
    assert answer["result"]["isError"] is True and stub.reached == []
    _, answer, stub, _ = _run("recall", {"query": "q", "profile_id": "work"}, LEGACY_KEY,
                              active="work")
    assert answer["result"]["isError"] is False and stub.reached


def test_a_profile_switch_cannot_land_inside_a_remote_call(active_only) -> None:
    """The lease holds the switch until the call has finished."""
    runtime = ProfileRuntime("work")
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class _Slow(_Stub):
        async def __call__(self, scope, receive, send) -> None:
            entered.set()
            await asyncio.to_thread(release.wait, 10)
            order.append(f"call-ran-as-{runtime.snapshot.profile_id}")
            await super().__call__(scope, receive, send)

    def _switch() -> None:
        entered.wait(10)
        runtime.transition("clientx", lambda _prev, _target: order.append("switched"))

    switcher = threading.Thread(target=_switch)
    switcher.start()
    app = policy.RemoteToolScopeASGI(_Slow(runtime), runtime_for=lambda _s: runtime)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": active_only, "arguments": {}}}).encode()
    scope = {"type": "http", "method": "POST", "path": "/mcp/h", "root_path": "/mcp",
             "headers": [], "client": ("peer", 1), PRINCIPAL_SCOPE_KEY: READ_KEY}
    queue = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(_message):
        return None

    async def main():
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.sleep(0.3)  # the switch is now waiting on the lease
        release.set()
        await task

    asyncio.run(main())
    switcher.join(10)
    assert order == ["call-ran-as-work", "switched"]


def test_no_profile_runtime_means_no_remote_tool_call() -> None:
    stub = _Stub(ProfileRuntime("work"))
    app = policy.RemoteToolScopeASGI(stub, runtime_for=lambda _scope: None)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "recall", "arguments": {}}}).encode()
    scope = {"type": "http", "method": "POST", "path": "/mcp/h", "root_path": "/mcp",
             "headers": [], "client": ("peer", 1), PRINCIPAL_SCOPE_KEY: READ_KEY}
    sent: list[dict] = []
    queue = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 503 and stub.reached == []
