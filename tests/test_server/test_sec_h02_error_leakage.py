# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# SECURITY REGRESSION — SEC-H-02 Error-Message Leakage.
#
# Asserts that forced 500 errors on server/routes/memories.py routes return
# a GENERIC body (no DB schema, no column/constraint names, no filesystem
# paths) after the fix. Leakage previously occurred via
# ``detail=f"<Label>: {str(e)}"``.
#
# Run: SLM_TEST_ISOLATION=1 pytest tests/test_server/test_sec_h02_error_leakage.py -v

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {"X-SLM-Daemon-Capability": d.capability,
            "X-SLM-Target-Instance": d.instance_id}


@pytest.fixture
def client(engine_with_mock_deps):
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    for pid in ("default",):
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    app.state.rbac = RbacEngine(str(engine._config.db_path))
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app, raise_server_exceptions=False), _daemon_headers(app)


# ---------------------------------------------------------------------------
# SEC-H-02: forced DB error on /api/memories must not leak exception text
# ---------------------------------------------------------------------------

def test_memories_500_generic_body(client):
    """SEC-H-02: a forced DB error on GET /api/memories returns a generic detail,
    not the raw exception string (which can contain column names, constraint
    names, or the data-directory filesystem path)."""
    tc, h = client

    DB_SENTINEL = "UNIQUE constraint failed: atomic_facts.fact_id"

    with patch(
        "superlocalmemory.server.routes.memories.get_db_connection",
        side_effect=RuntimeError(DB_SENTINEL),
    ):
        resp = tc.get("/api/memories", headers=h)

    assert resp.status_code == 500, f"expected 500, got {resp.status_code}"

    body_text = resp.text
    # The sentinel DB error text MUST NOT appear in the client response.
    assert DB_SENTINEL not in body_text, (
        f"SEC-H-02: exception text leaked to HTTP client: {body_text!r}"
    )
    # The response MUST contain a generic message.
    assert "Internal server error" in body_text or "Database error" in body_text, (
        f"Expected generic error message in body, got: {body_text!r}"
    )


def test_graph_500_generic_body(client):
    """SEC-H-02: forced DB error on GET /api/graph returns no exception detail."""
    tc, h = client

    FS_SENTINEL = "/Users/varun/.superlocalmemory/memory.db: table does not exist"

    with patch(
        "superlocalmemory.server.routes.memories.get_db_connection",
        side_effect=RuntimeError(FS_SENTINEL),
    ):
        resp = tc.get("/api/graph", headers=h)

    assert resp.status_code == 500
    assert FS_SENTINEL not in resp.text, (
        f"SEC-H-02: filesystem path leaked to HTTP client: {resp.text!r}"
    )


def test_search_500_generic_body(client):
    """SEC-H-02: forced engine error on POST /api/search returns no exception detail."""
    tc, h = client

    SCHEMA_SENTINEL = "no such column: af.canonical_entities_json"

    # Patch _get_engine to return None so the fallback DB path runs, then patch that.
    with patch(
        "superlocalmemory.server.routes.memories.get_db_connection",
        side_effect=RuntimeError(SCHEMA_SENTINEL),
    ):
        # get_engine_lazy returns the real engine; also patch it to None so
        # the fallback DB path is exercised and blows up.
        with patch(
            "superlocalmemory.server.routes.memories.get_engine_lazy",
            return_value=None,
        ):
            resp = tc.post("/api/search", json={"query": "test", "limit": 5}, headers=h)

    assert resp.status_code == 500
    assert SCHEMA_SENTINEL not in resp.text, (
        f"SEC-H-02: DB schema leaked to HTTP client: {resp.text!r}"
    )


def test_clusters_500_generic_body(client):
    """SEC-H-02: forced DB error on GET /api/clusters returns no exception detail."""
    tc, h = client

    SENTINEL = "no such table: memory_scenes — CHECK constraint failed"

    with patch(
        "superlocalmemory.server.routes.memories.get_db_connection",
        side_effect=RuntimeError(SENTINEL),
    ):
        resp = tc.get("/api/clusters", headers=h)

    assert resp.status_code == 500
    assert SENTINEL not in resp.text, (
        f"SEC-H-02: exception text leaked: {resp.text!r}"
    )
