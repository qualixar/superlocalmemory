"""A web app's peer id is reachable only in process, never over the daemon's HTTP routes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import mesh as mesh_routes
from tests.test_mesh.conftest import make_peer, rows

HEADERS = {
    "X-SLM-Daemon-Capability": "mesh-capability",
    "X-SLM-Target-Instance": "mesh-instance",
}
WEB_REF = "w_" + "ab" * 12


@pytest.fixture()
def client(broker):
    app = FastAPI()
    app.state.mesh_broker = broker
    app.state.config = None
    app.state.daemon_descriptor = SimpleNamespace(
        capability="mesh-capability", instance_id="mesh-instance",
        capability_fingerprint="mesh-fingerprint",
    )
    app.include_router(mesh_routes.router)
    with TestClient(app, base_url="http://127.0.0.1:9999", client=("127.0.0.1", 5000)) as c:
        yield c


@pytest.fixture()
def web_ref(broker) -> str:
    assert broker.ensure_web_peer(WEB_REF, app="notes", display_name="Notes",
                                  connection_id="c" * 32)["ok"]
    return WEB_REF


def test_http_send_naming_a_web_peer_as_the_sender_is_refused(client, broker, web_ref) -> None:
    local = make_peer(broker, "sess-1")
    r = client.post("/mesh/send", headers=HEADERS,
                    json={"from_peer": web_ref, "to": local, "content": "forged"})
    assert r.status_code == 403
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_http_send_from_a_local_peer_to_a_web_peer_still_works(client, broker, web_ref) -> None:
    local = make_peer(broker, "sess-1")
    r = client.post("/mesh/send", headers=HEADERS,
                    json={"from_peer": local, "to": web_ref, "content": "hello app"})
    assert r.status_code == 200


def test_http_reads_of_a_web_peers_mail_are_refused(client, broker, web_ref) -> None:
    local = make_peer(broker, "sess-1")
    broker.send_message(local, web_ref, "private to the app")
    assert client.get(f"/mesh/inbox/{web_ref}", headers=HEADERS).status_code == 403
    assert client.get(f"/mesh/inbox/{web_ref}/wait", params={"timeout_s": 1},
                      headers=HEADERS).status_code == 403
    r = client.post(f"/mesh/inbox/{web_ref}/read", json={"message_ids": [1]}, headers=HEADERS)
    assert r.status_code == 403
    assert rows(broker, "SELECT read FROM mesh_messages")[0][0] == 0


def test_http_reads_of_a_local_peers_mail_are_unchanged(client, broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    broker.send_message(a, b, "hi")
    r = client.get(f"/mesh/inbox/{b}", headers=HEADERS)
    assert r.status_code == 200 and r.json()["messages"][0]["content"] == "hi"


def test_http_pending_for_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.get(f"/mesh/pending/{web_ref}", headers=HEADERS)
    assert r.status_code == 403


def test_http_pending_for_a_local_peer_is_unchanged(client, broker) -> None:
    local = make_peer(broker, "sess-2")
    assert client.get(f"/mesh/pending/{local}", headers=HEADERS).status_code == 200


def test_http_deregister_of_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.post("/mesh/deregister", headers=HEADERS, json={"peer_id": web_ref})
    assert r.status_code == 403
    assert broker.is_web_peer(web_ref)


def test_http_heartbeat_of_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.post("/mesh/heartbeat", headers=HEADERS, json={"peer_id": web_ref})
    assert r.status_code == 403


def test_http_summary_of_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.post("/mesh/summary", headers=HEADERS,
                    json={"peer_id": web_ref, "summary": "forged"})
    assert r.status_code == 403


def test_http_lock_naming_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.post("/mesh/lock", headers=HEADERS,
                    json={"file_path": "a.py", "locked_by": web_ref, "action": "acquire"})
    assert r.status_code == 403


def test_http_state_write_naming_a_web_peer_is_refused(client, broker, web_ref) -> None:
    r = client.post("/mesh/state", headers=HEADERS,
                    json={"key": "k", "value": "v", "set_by": web_ref})
    assert r.status_code == 403
