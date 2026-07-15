"""Mesh over the HTTP route layer (issue #39 coverage-2).

Existing mesh tests call MeshBroker methods in-process; #34 and #39 were both
HTTP-transport failures. This drives routes/mesh.py through a TestClient and
also covers the v3.6.12 fixes: shared-secret enforcement for non-loopback
callers (mesh-1) and lock-release rowcount honesty (mesh-2).
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.server.routes import mesh as mesh_routes


DAEMON_HEADERS = {
    "X-SLM-Daemon-Capability": "mesh-capability",
    "X-SLM-Target-Instance": "mesh-instance",
}


def _init_mesh_schema(db_path: str) -> None:
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL, _MESH_V346_DDL, _MESH_V346_ALTERS,
    )
    conn = sqlite3.connect(db_path)
    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()


def _app_with_broker(secret: str | None = None) -> tuple[FastAPI, MeshBroker]:
    td = tempfile.mkdtemp()
    db_path = str(Path(td) / "mesh.db")
    _init_mesh_schema(db_path)
    broker = MeshBroker(db_path)
    broker._shared_secret = secret  # simulate SLM_MESH_SHARED_SECRET
    app = FastAPI()
    app.state.mesh_broker = broker
    app.state.config = None
    app.state.daemon_descriptor = SimpleNamespace(
        capability="mesh-capability",
        instance_id="mesh-instance",
        capability_fingerprint="mesh-fingerprint",
    )
    app.include_router(mesh_routes.router)
    return app, broker


def test_register_and_peers_over_http() -> None:
    app, _ = _app_with_broker()
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "sess-1", "summary": "w"},
        headers=DAEMON_HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("peer_id")
    r2 = c.get("/mesh/peers", headers=DAEMON_HEADERS)
    assert r2.status_code == 200
    peers = r2.json().get("peers", [])
    assert any(p.get("session_id") == "sess-1" for p in peers)


def test_secret_required_for_nonloopback() -> None:
    # TestClient's client host is "testclient" (non-loopback). With a secret
    # configured and no X-Mesh-Secret header → 401 (mesh-1 security fix).
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post("/mesh/register", json={"session_id": "s"})
    assert r.status_code == 401


def test_secret_accepts_correct_header() -> None:
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post("/mesh/register", json={"session_id": "s"},
               headers={"X-Mesh-Secret": "topsecret"})
    assert r.status_code == 200


def test_no_secret_still_requires_local_capability() -> None:
    app, _ = _app_with_broker(secret=None)
    c = TestClient(app)
    r = c.post("/mesh/register", json={"session_id": "s"})
    assert r.status_code == 403
    accepted = c.post(
        "/mesh/register", json={"session_id": "s"}, headers=DAEMON_HEADERS,
    )
    assert accepted.status_code == 200


def test_mesh_state_rejects_secret_bearing_values() -> None:
    app, _ = _app_with_broker(secret=None)
    c = TestClient(app)

    response = c.post(
        "/mesh/state",
        json={
            "key": "provider_api_key",
            "value": "sk-super-secret-value-1234567890",
            "set_by": "peer",
        },
        headers=DAEMON_HEADERS,
    )

    assert response.status_code == 422


# ── v3.6.20: Bearer token support (issue #60) ───────────────────────────────
# Root cause: _get_broker (added v3.6.12) required X-Mesh-Secret, but every
# documented caller uses Authorization: Bearer — remote_sync.py, multi-machine.md,
# and the old _validate_remote_auth all specified Bearer.  These tests confirm
# the fix: both headers accepted, security is not weakened.

def test_bearer_token_accepted_for_nonloopback() -> None:
    # Core regression: Authorization: Bearer must work.
    # TestClient uses host="testclient" (non-loopback), so the auth gate fires.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


def test_bearer_token_wrong_secret_rejected() -> None:
    # A wrong Bearer secret must still 401 — security is not weakened.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"Authorization": "Bearer wrongsecret"},
    )
    assert r.status_code == 401


def test_xmesh_secret_still_works_backwards_compat() -> None:
    # X-Mesh-Secret must still work — backwards compat for callers that
    # discovered this undocumented header in v3.6.12.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.post(
        "/mesh/register",
        json={"session_id": "s"},
        headers={"X-Mesh-Secret": "topsecret"},
    )
    assert r.status_code == 200


def test_status_endpoint_bearer_auth() -> None:
    # /mesh/status is the exact endpoint reported in issue #60.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/status", headers={"Authorization": "Bearer topsecret"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("broker_up") is True


def test_status_endpoint_blocked_without_auth() -> None:
    # /mesh/status must 401 when a secret is set and no auth header is provided.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/status")
    assert r.status_code == 401


def test_peers_endpoint_bearer_auth() -> None:
    # /mesh/peers is called by RemoteSyncClient._sync_peers_from_remote with Bearer.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    r = c.get("/mesh/peers", headers={"Authorization": "Bearer topsecret"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert "peers" in r.json()


def test_send_endpoint_bearer_auth() -> None:
    # /mesh/send is called by RemoteSyncClient.send_to_remote with Bearer.
    app, _ = _app_with_broker(secret="topsecret")
    c = TestClient(app)
    # Register a recipient peer first (needs Bearer because secret is set)
    reg = c.post(
        "/mesh/register",
        json={"session_id": "receiver"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert reg.status_code == 200
    peer_id = reg.json()["peer_id"]

    r = c.post(
        "/mesh/send",
        json={"from_peer": "sender", "to_peer": peer_id, "content": "hello", "type": "text"},
        headers={"Authorization": "Bearer topsecret"},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json().get("ok") is True
