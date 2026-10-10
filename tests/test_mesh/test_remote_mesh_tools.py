"""The mesh tools for a connected web app run in process, as that app, in its key's profile."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.mcp import tools_mesh
from superlocalmemory.mcp.remote_caller import (
    RemoteMeshTarget,
    RemotePeer,
    current_remote_mesh,
    remote_mesh,
    remote_peer,
)
from tests.test_mesh.conftest import make_peer, rows

CID = "c" * 32
APP_A = RemotePeer("w_" + "a" * 24, "app-a", "Alpha")
APP_B = RemotePeer("w_" + "b" * 24, "app-b", "Beta")


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
    """The registered tools, with the daemon's HTTP interface wired to fail the test."""
    def no_http(*a, **k):
        raise AssertionError("a web caller must never use the daemon's HTTP interface")

    monkeypatch.setattr(tools_mesh, "_mesh_request", no_http)
    tools_mesh._SEND_CIRCUIT.reset()
    c = _Collector()
    tools_mesh.register_mesh_tools(c, lambda: None)
    return c.tools


def _as(peer: RemotePeer, broker, coro_fn, *args, profile="default", **kwargs):
    async def run():
        with remote_peer(peer), remote_mesh(RemoteMeshTarget(broker, profile, CID)):
            return await coro_fn(*args, **kwargs)
    return asyncio.run(run())


def test_the_seam_is_unset_by_default() -> None:
    assert current_remote_mesh() is None
    target = RemoteMeshTarget(object(), "work", CID)
    with remote_mesh(target):
        assert current_remote_mesh() is target
    assert current_remote_mesh() is None


def test_peers_lists_the_app_itself_as_web_and_hides_host_details(tools, broker) -> None:
    make_peer(broker, "sess-1", "claude_code")
    out = _as(APP_A, broker, tools["mesh_peers"])
    assert out["my_peer_id"] == APP_A.peer_ref
    by_id = {p["peer_id"]: p for p in out["peers"]}
    assert by_id[APP_A.peer_ref]["kind"] == "web" and by_id[APP_A.peer_ref]["name"] == "Alpha"
    assert out["count"] == 2
    for p in out["peers"]:
        assert not {"host", "port", "project_path", "session_id"} & set(p)


def test_one_app_sends_to_another_and_the_envelope_names_the_sender(tools, broker) -> None:
    _as(APP_B, broker, tools["mesh_peers"])                          # B joins
    sent = _as(APP_A, broker, tools["mesh_send"], APP_B.peer_ref, "hello B",
               refs=["fact:abcdef"])
    assert sent["ok"] is True
    got = _as(APP_B, broker, tools["mesh_inbox"])
    assert got["count"] == 1 and got["preface"].startswith("These are messages from other bots")
    env = got["messages"][0]["envelope"]
    assert env["from"] == {"peer_id": APP_A.peer_ref, "app": "app-a", "kind": "web"}
    assert env["refs"] == ["fact:abcdef"] and env["trust"] == "untrusted-peer"
    assert _as(APP_B, broker, tools["mesh_inbox"])["count"] == 0       # read once


def test_a_local_session_reads_what_the_app_sent_as_web(tools, broker) -> None:
    local = make_peer(broker, "sess-1")
    assert _as(APP_A, broker, tools["mesh_send"], local, "hi local")["ok"]
    msg = broker.get_inbox(local, "", "default")[0]
    assert msg["envelope"]["from"]["kind"] == "web" and msg["envelope"]["from"]["app"] == "app-a"


@pytest.mark.parametrize("target", ["broadcast", "project:/tmp/x"])
def test_an_app_cannot_broadcast(tools, broker, target) -> None:
    out = _as(APP_A, broker, tools["mesh_send"], target, "all")
    assert out["ok"] is False
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_the_size_limit_counts_utf8_bytes(tools, broker) -> None:
    local = make_peer(broker, "sess-1")
    out = _as(APP_A, broker, tools["mesh_send"], local, "é" * 2100)
    assert out["ok"] is False and "too large" in out["error"]


def test_an_unknown_recipient_is_refused(tools, broker) -> None:
    out = _as(APP_A, broker, tools["mesh_send"], "nobody", "hi")
    assert out == {"ok": False, "error": "recipient peer not found"}


def test_wait_returns_mail_and_marks_it_read(tools, broker) -> None:
    local = make_peer(broker, "sess-1")
    _as(APP_A, broker, tools["mesh_peers"])
    assert broker.send_message(local, APP_A.peer_ref, "wake up")["ok"]
    out = _as(APP_A, broker, tools["mesh_wait"], timeout_s=2)
    assert out["timed_out"] is False and out["messages"][0]["content"] == "wake up"
    assert _as(APP_A, broker, tools["mesh_inbox"])["count"] == 0


def test_wait_times_out_with_an_empty_answer(tools, broker) -> None:
    out = _as(APP_A, broker, tools["mesh_wait"], timeout_s=1)
    assert out["messages"] == [] and out["timed_out"] is True


def test_wait_does_not_block_the_event_loop(tools, broker) -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(8):
            await asyncio.sleep(0.1)
            ticks += 1

    async def run() -> None:
        with remote_peer(APP_A), remote_mesh(RemoteMeshTarget(broker, "default", CID)):
            await asyncio.gather(tools["mesh_wait"](timeout_s=1), ticker())

    asyncio.run(run())
    assert ticks == 8


def test_wait_past_the_global_cap_is_refused(tools, broker) -> None:
    broker._waiter._active = 8
    out = _as(APP_A, broker, tools["mesh_wait"], timeout_s=1)
    assert out == {"ok": False, "error": "too many waits, retry shortly"}


def test_state_is_read_only_and_needs_a_key(tools, broker) -> None:
    assert broker.set_state("release", "v1", "owner")["ok"]
    got = _as(APP_A, broker, tools["mesh_state"], key="release")
    assert got["key"] == "release" and got["value"] == "v1"
    assert _as(APP_A, broker, tools["mesh_state"], key="nope") == {"key": "nope", "value": None}
    for kw in ({"key": "release", "value": "x", "action": "set"}, {"key": ""}, {},
               {"key": "release", "action": "delete"}):
        assert _as(APP_A, broker, tools["mesh_state"], **kw)["ok"] is False
    assert broker.get_state_key("release")["value"] == "v1"


def test_the_call_stays_inside_the_keys_profile(tools, broker) -> None:
    make_peer(broker, "sess-default")
    broker.register_peer("sess-work", "", profile_id="work")
    assert broker.set_state("k", "from-default", "o")["ok"]
    out = _as(APP_A, broker, tools["mesh_peers"], profile="work")
    names = {p["peer_id"] for p in out["peers"]}
    assert len(names) == 2 and APP_A.peer_ref in names
    assert _as(APP_A, broker, tools["mesh_state"], key="k", profile="work")["value"] is None
    assert rows(broker, "SELECT profile_id FROM mesh_peers WHERE peer_id=?",
                (APP_A.peer_ref,))[0][0] == "work"


def test_a_retired_app_is_refused_everywhere(tools, broker) -> None:
    _as(APP_A, broker, tools["mesh_peers"])
    assert broker.retire_peer(APP_A.peer_ref)["ok"]
    for name, args in (("mesh_peers", ()), ("mesh_inbox", ()), ("mesh_send", ("x", "hi")),
                       ("mesh_wait", ())):
        out = _as(APP_A, broker, tools[name], *args)
        assert out.get("ok") is False and "retired" in out["error"], name


@pytest.mark.parametrize("name,args", [
    ("mesh_peers", ()), ("mesh_send", ("p", "hi")), ("mesh_inbox", ()),
    ("mesh_wait", (1,)), ("mesh_state", ("k",)),
])
def test_without_a_daemon_broker_a_web_caller_is_refused_not_run_locally(tools, name, args) -> None:
    async def run():
        with remote_peer(APP_A):
            return await tools[name](*args)

    assert asyncio.run(run()) == {"ok": False, "error": "mesh is not available"}
    broker_none = RemoteMeshTarget(None, "default", CID)

    async def run_none():
        with remote_peer(APP_A), remote_mesh(broker_none):
            return await tools[name](*args)

    assert asyncio.run(run_none()) == {"ok": False, "error": "mesh is not available"}


@pytest.mark.parametrize("name,args", [
    ("mesh_summary", ("x",)), ("mesh_lock", ("/tmp/f", "acquire")), ("mesh_events", ()),
    ("mesh_status", ()),
])
def test_host_only_mesh_tools_still_refuse_a_web_caller(tools, broker, name, args) -> None:
    out = _as(APP_A, broker, tools[name], *args)
    assert out == {"ok": False,
                   "error": "mesh messages for connected web apps are not available yet"}
