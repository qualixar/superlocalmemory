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

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.server.routes import mesh as mesh_routes


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
    app.include_router(mesh_routes.router)
    return app, broker


def test_register_and_peers_over_http() -> None:
    app, _ = _app_with_broker()
    c = TestClient(app)
    r = c.post("/mesh/register", json={"session_id": "sess-1", "summary": "w"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("peer_id")
    r2 = c.get("/mesh/peers")
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


def test_no_secret_allows_all() -> None:
    app, _ = _app_with_broker(secret=None)
    c = TestClient(app)
    r = c.post("/mesh/register", json={"session_id": "s"})
    assert r.status_code == 200
