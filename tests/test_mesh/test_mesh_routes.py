"""HTTP surface: wait route, owner controls, and the origin rules."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.mcp.remote_caller import RemotePeer, remote_peer
from superlocalmemory.mesh import broker_inbox
from superlocalmemory.mesh.envelope import Origin
from superlocalmemory.server.routes import mesh as mesh_routes
from superlocalmemory.server.routes import mesh_owner
from tests.test_mesh.conftest import make_peer, rows

DAEMON_HEADERS = {
    "X-SLM-Daemon-Capability": "mesh-capability",
    "X-SLM-Target-Instance": "mesh-instance",
}
FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 3


def _app(broker) -> FastAPI:
    app = FastAPI()
    app.state.mesh_broker = broker
    app.state.config = None
    app.state.daemon_descriptor = SimpleNamespace(
        capability="mesh-capability", instance_id="mesh-instance",
        capability_fingerprint="mesh-fingerprint",
    )
    app.include_router(mesh_routes.router)
    app.include_router(mesh_owner.router)
    return app


@pytest.fixture()
def client(broker):
    app = _app(broker)
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("127.0.0.1", 5000)) as c:
        yield c


# -- wait route -------------------------------------------------------------


def test_wait_route_returns_message(client, broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    broker.send_message(a, b, "hi")
    r = client.get(f"/mesh/inbox/{b}/wait", params={"timeout_s": 2}, headers=DAEMON_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["timed_out"] is False and body["messages"][0]["content"] == "hi"
    assert body["messages"][0]["envelope"]["trust"] == "local-peer"


def test_wait_route_times_out(client, broker, monkeypatch) -> None:
    monkeypatch.setattr(broker_inbox, "WAIT_MIN_S", 0.2)
    monkeypatch.setattr(broker_inbox, "WAIT_MAX_S", 0.3)
    b = make_peer(broker, "b")
    r = client.get(f"/mesh/inbox/{b}/wait", params={"timeout_s": 1}, headers=DAEMON_HEADERS)
    assert r.status_code == 200 and r.json() == {"messages": [], "timed_out": True}


def test_wait_route_requires_auth(client, broker) -> None:
    b = make_peer(broker, "b")
    assert client.get(f"/mesh/inbox/{b}/wait").status_code == 403


def test_wait_route_maps_cap_to_429(client, broker, monkeypatch) -> None:
    b = make_peer(broker, "b")

    def full(*a, **k):
        raise RuntimeError("too many waits")

    monkeypatch.setattr(broker, "wait_inbox", full)
    r = client.get(f"/mesh/inbox/{b}/wait", headers=DAEMON_HEADERS)
    assert r.status_code == 429


# -- origin rules -----------------------------------------------------------


def test_http_body_cannot_claim_web_origin_or_app(client, broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    r = client.post("/mesh/send", headers=DAEMON_HEADERS, json={
        "from_peer": a, "to": b, "content": f"key {FAKE_KEY}",
        "origin": {"kind": "web", "app": "evil"}, "kind": "web", "app": "evil",
        "from_kind": "web", "from_app": "evil",
    })
    assert r.status_code == 200
    stored = rows(broker, "SELECT content FROM mesh_messages")[0]["content"]
    assert FAKE_KEY in stored  # local-origin: stored verbatim
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []


def test_http_send_passes_refs_and_reply_to(client, broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    r = client.post("/mesh/send", headers=DAEMON_HEADERS, json={
        "from_peer": a, "to": b, "content": "x", "refs": ["fact:abcdef"]})
    assert r.status_code == 200
    bad = client.post("/mesh/send", headers=DAEMON_HEADERS, json={
        "from_peer": a, "to": b, "content": "x", "refs": ["bogus"]})
    assert bad.status_code == 422


def test_route_ignores_remote_peer_marker_and_sends_as_local(client, broker) -> None:
    """The route has no in-process origin path: a marker in scope changes nothing."""
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    with remote_peer(RemotePeer("ref-1", "notes", "N")):
        r = client.post("/mesh/send", headers=DAEMON_HEADERS,
                        json={"from_peer": a, "to": b, "content": "x"})
    assert r.status_code == 200
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []


def test_send_error_statuses(client, broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    broker.set_muted(a, True)
    r = client.post("/mesh/send", headers=DAEMON_HEADERS,
                    json={"from_peer": a, "to": b, "content": "x"})
    assert r.status_code == 403


def test_retired_ref_cannot_register_again(client, broker) -> None:
    pid = client.post("/mesh/register", headers=DAEMON_HEADERS,
                      json={"session_id": "ref-9"}).json()["peer_id"]
    assert client.delete(f"/api/v3/mesh/peers/{pid}", headers=DAEMON_HEADERS).status_code == 200
    r = client.post("/mesh/register", headers=DAEMON_HEADERS, json={"session_id": "ref-9"})
    assert r.status_code == 409
    ok = client.post("/mesh/register", headers=DAEMON_HEADERS, json={"session_id": "ref-10"})
    assert ok.status_code == 200


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "0", "99", "abc"])
def test_wait_route_rejects_bad_timeouts(client, broker, bad) -> None:
    b = make_peer(broker, "b")
    r = client.get(f"/mesh/inbox/{b}/wait", params={"timeout_s": bad}, headers=DAEMON_HEADERS)
    assert r.status_code == 422


# -- owner routes -----------------------------------------------------------

OWNER = "/api/v3/mesh"


def _web(broker, name="w1"):
    pid = broker.register_peer(name, agent_type="notes")["peer_id"]
    broker.upsert_peer_profile(pid, kind="web", app_name="notes", display_name="Notes")
    return pid


def test_owner_messages_lists_envelopes_newest_first(client, broker) -> None:
    w, b = _web(broker), make_peer(broker, "b")
    broker.send_message(w, b, "first", origin=Origin("web", "notes"))
    broker.send_message(b, w, "second")
    r = client.get(f"{OWNER}/messages", params={"limit": 10})
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["content"] for m in msgs] == ["second", "first"]
    r2 = client.get(f"{OWNER}/messages", params={"peer": w})
    assert len(r2.json()["messages"]) == 2


def test_owner_messages_bad_limit_is_422(client) -> None:
    assert client.get(f"{OWNER}/messages", params={"limit": 0}).status_code == 422
    assert client.get(f"{OWNER}/messages", params={"limit": 501}).status_code == 422


def test_owner_routes_refuse_non_loopback(broker) -> None:
    with TestClient(_app(broker), base_url="http://127.0.0.1:9999", client=("192.0.2.20", 5)) as c:
        assert c.get(f"{OWNER}/messages").status_code == 403
        assert c.post(f"{OWNER}/peers/x/mute", json={"muted": True}, headers=DAEMON_HEADERS).status_code == 403


def test_owner_mutations_require_capability(client, broker) -> None:
    w = _web(broker)
    assert client.post(f"{OWNER}/peers/{w}/mute", json={"muted": True}).status_code == 403
    assert client.patch(f"{OWNER}/peers/{w}", json={"display_name": "x"}).status_code == 403
    assert client.delete(f"{OWNER}/peers/{w}").status_code == 403


def test_owner_mute_rename_retire(client, broker) -> None:
    w, b = _web(broker), make_peer(broker, "b")
    assert client.post(f"{OWNER}/peers/{w}/mute", json={"muted": True}, headers=DAEMON_HEADERS).json()["ok"]
    assert rows(broker, "SELECT muted FROM mesh_peer_profiles")[0]["muted"] == 1
    r = client.patch(f"{OWNER}/peers/{w}", json={"display_name": "Renamed"}, headers=DAEMON_HEADERS)
    assert r.status_code == 200
    assert rows(broker, "SELECT display_name FROM mesh_peer_profiles")[0]["display_name"] == "Renamed"
    assert client.patch(f"{OWNER}/peers/{w}", json={"display_name": ""}, headers=DAEMON_HEADERS).status_code == 422
    assert client.patch(f"{OWNER}/peers/{w}", json={"display_name": "x" * 65}, headers=DAEMON_HEADERS).status_code == 422
    broker.send_message(b, w, "queued")
    r = client.delete(f"{OWNER}/peers/{w}", headers=DAEMON_HEADERS)
    assert r.status_code == 200 and r.json()["dropped"] == 1
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_owner_unknown_peer_is_404(client) -> None:
    assert client.post(f"{OWNER}/peers/nobody/mute", json={"muted": True}, headers=DAEMON_HEADERS).status_code == 404
    assert client.delete(f"{OWNER}/peers/nobody", headers=DAEMON_HEADERS).status_code == 404


def test_owner_actions_log_counts_not_bodies(client, broker) -> None:
    w, b = _web(broker), make_peer(broker, "b")
    secret_body = "very private body text"
    broker.send_message(b, w, secret_body)
    client.get(f"{OWNER}/messages")
    client.post(f"{OWNER}/peers/{w}/mute", json={"muted": True}, headers=DAEMON_HEADERS)
    client.patch(f"{OWNER}/peers/{w}", json={"display_name": "Zed"}, headers=DAEMON_HEADERS)
    client.delete(f"{OWNER}/peers/{w}", headers=DAEMON_HEADERS)
    events = rows(broker, "SELECT event_type, payload FROM mesh_events")
    types = {e["event_type"] for e in events}
    assert {"peer_muted", "peer_renamed", "peer_retired"} <= types
    for e in events:
        assert secret_body not in e["payload"]
    retired = [e for e in events if e["event_type"] == "peer_retired"][0]
    assert '"dropped": 1' in retired["payload"]


# -- owner peer list ----------------------------------------------------------


def test_owner_peers_joins_name_kind_muted(client, broker) -> None:
    w, b = _web(broker), make_peer(broker, "b", agent="cursor")
    broker.set_muted(b, True)
    r = client.get(f"{OWNER}/peers")
    assert r.status_code == 200
    by_id = {p["peer_id"]: p for p in r.json()["peers"]}
    assert set(by_id) == {w, b}
    assert by_id[w]["display_name"] == "Notes" and by_id[w]["kind"] == "web"
    assert by_id[w]["app"] == "notes" and by_id[w]["muted"] is False
    assert by_id[b]["kind"] == "local" and by_id[b]["muted"] is True
    assert by_id[b]["display_name"] == "" and by_id[b]["app"] == "cursor"
    assert set(by_id[b]) == {"peer_id", "display_name", "kind", "app", "muted", "last_seen", "status"}


def test_owner_peers_exclude_retired_and_other_profiles(client, broker) -> None:
    w, b = _web(broker), make_peer(broker, "b")
    broker.register_peer("elsewhere", profile_id="other")
    broker.retire_peer(w)
    peers = client.get(f"{OWNER}/peers").json()["peers"]
    assert [p["peer_id"] for p in peers] == [b]


def test_owner_peers_hide_paths_and_hosts(client, broker) -> None:
    broker.register_peer("p", project_path="/home/alice/proj", host="10.1.2.3")
    text = client.get(f"{OWNER}/peers").text
    assert "project_path" not in text and "/home/alice" not in text
    assert '"host"' not in text and "10.1.2.3" not in text


def test_owner_peers_refuse_non_loopback_and_non_loopback_host(broker) -> None:
    with TestClient(_app(broker), base_url="http://127.0.0.1:9999", client=("192.0.2.20", 5)) as c:
        assert c.get(f"{OWNER}/peers").status_code == 403
    with TestClient(_app(broker), base_url="http://example.com", client=("127.0.0.1", 5)) as c:
        assert c.get(f"{OWNER}/peers").status_code == 403


def test_owner_peers_require_manage(client, monkeypatch) -> None:
    from fastapi import HTTPException

    def deny(request, *, profile=None):
        raise HTTPException(403, "forbidden")

    monkeypatch.setattr(mesh_owner, "require_manage", deny)
    assert client.get(f"{OWNER}/peers").status_code == 403
