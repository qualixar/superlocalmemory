"""Regression contracts for dashboard browsing server truth."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient


def _app(engine):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app)


def test_entity_list_filters_before_count_and_pagination(engine_with_mock_deps):
    """A filtered entity beyond the unfiltered first page remains discoverable."""
    engine = engine_with_mock_deps
    for index in range(55):
        engine._db.execute(
            "INSERT INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"concept-{index}", "default", f"Concept {index:02d}", "concept", 100 - index),
        )
    engine._db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("person-needle", "default", "Needle Person", "person", 1),
    )
    response = _app(engine).get(
        "/api/entity/list?limit=50&offset=0&type=person&search=needle",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["has_more"] is False
    assert [entity["name"] for entity in body["entities"]] == ["Needle Person"]


def test_entity_default_paging_uses_fact_count_index(engine_with_mock_deps):
    """Large entity sets must not sort the whole profile for every page."""
    plan = engine_with_mock_deps._db.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT entity_id FROM canonical_entities "
        "WHERE profile_id = ? ORDER BY fact_count DESC LIMIT 50",
        ("default",),
    )
    assert any(
        "idx_entities_profile_fact_count" in str(dict(row).get("detail", ""))
        for row in plan
    )


def test_mesh_read_model_separates_remote_peers_from_local_sessions_and_expires_stale():
    """The dashboard must not report loopback sessions or dead peers as mesh peers."""
    from fastapi import FastAPI
    from superlocalmemory.mesh.broker import MeshBroker
    from superlocalmemory.server.routes import mesh as mesh_routes
    from superlocalmemory.storage.schema_v343 import _MESH_DDL, _MESH_V346_ALTERS, _MESH_V346_DDL

    import sqlite3
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "mesh.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(_MESH_DDL)
    for statement in _MESH_V346_ALTERS:
        try:
            connection.execute(statement)
        except sqlite3.OperationalError:
            pass
    connection.executescript(_MESH_V346_DDL)
    connection.commit()
    connection.close()

    broker = MeshBroker(db_path)
    local = broker.register_peer("local-agent", host="127.0.0.1")
    live_remote = broker.register_peer("remote-agent", host="10.0.0.8", profile_id="default")
    stale_remote = broker.register_peer("stale-agent", host="10.0.0.9", profile_id="default")
    other_profile = broker.register_peer("other-profile", host="10.0.0.10", profile_id="work")
    assert local["ok"] and live_remote["ok"] and stale_remote["ok"] and other_profile["ok"]

    expired_at = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE mesh_peers SET last_heartbeat=?, status='active' WHERE peer_id=?",
        (expired_at, stale_remote["peer_id"]),
    )
    connection.commit()
    connection.close()

    app = FastAPI()
    app.state.mesh_broker = broker
    app.state.config = None
    app.state.daemon_descriptor = SimpleNamespace(
        capability="test-capability", instance_id="test-instance",
        capability_fingerprint="test-fingerprint",
    )
    app.include_router(mesh_routes.router)
    headers = {
        "X-SLM-Daemon-Capability": "test-capability",
        "X-SLM-Target-Instance": "test-instance",
    }
    client = TestClient(app)

    peers = client.get("/mesh/peers?view=remote", headers=headers)
    status = client.get("/mesh/status", headers=headers)

    assert peers.status_code == 200, peers.text
    assert status.status_code == 200, status.text
    peer_body = peers.json()
    assert [peer["session_id"] for peer in peer_body["peers"]] == ["remote-agent"]
    assert [session["session_id"] for session in peer_body["local_sessions"]] == ["local-agent"]
    assert peer_body["peer_count"] == 2
    assert peer_body["remote_peer_count"] == 1
    assert peer_body["local_session_count"] == 1
    assert status.json()["peer_count"] == 2
    assert status.json()["remote_peer_count"] == 1
    assert status.json()["local_session_count"] == 1
