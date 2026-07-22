# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""PATCH /api/memories/{fact_id}/scope — dashboard-driven multi-scope (C2).

Lets a memory be re-scoped after creation (personal → shared/global) so a team
can share memory from the dashboard. shared_with is stored as a JSON array so
the _scope_where LIKE match resolves it. Profile-scoped: a caller cannot
re-scope a fact that belongs to another profile.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


def _seed_fact(engine, profile_id: str, fid: str) -> None:
    mid = f"mem-{fid}"
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, profile_id, "m", "s1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        " lifecycle, created_at, scope) VALUES (?,?,?,?,?,?,?)",
        (fid, mid, profile_id, "f", "active", "2026-01-01T00:00:00Z", "personal"),
    )


@pytest.fixture
def client(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other','other')"
    )
    _seed_fact(engine, "default", "f_mine")
    _seed_fact(engine, "other", "f_theirs")

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app), _daemon_headers(app)


def test_set_shared_stores_json_array(client):
    tc, h = client
    r = tc.patch("/api/memories/f_mine/scope",
                 json={"scope": "shared", "shared_with": "alice,bob"}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["shared_with"] == ["alice", "bob"]
    # Verify via a fresh read of the engine DB.
    eng = tc.app.state.engine
    got = eng._db.execute(
        "SELECT scope, shared_with FROM atomic_facts WHERE fact_id='f_mine'")
    scope, shared = got[0][0], got[0][1]
    assert scope == "shared"
    assert json.loads(shared) == ["alice", "bob"]


def test_invalid_scope_rejected(client):
    tc, h = client
    r = tc.patch("/api/memories/f_mine/scope", json={"scope": "bogus"}, headers=h)
    assert r.status_code == 400


def test_shared_requires_targets(client):
    tc, h = client
    r = tc.patch("/api/memories/f_mine/scope",
                 json={"scope": "shared", "shared_with": ""}, headers=h)
    assert r.status_code == 400


def test_global_clears_shared_with(client):
    tc, h = client
    tc.patch("/api/memories/f_mine/scope",
             json={"scope": "shared", "shared_with": "x"}, headers=h)
    r = tc.patch("/api/memories/f_mine/scope", json={"scope": "global"}, headers=h)
    assert r.status_code == 200
    assert r.json()["shared_with"] == []
    eng = tc.app.state.engine
    got = eng._db.execute(
        "SELECT scope, shared_with FROM atomic_facts WHERE fact_id='f_mine'")
    assert got[0][0] == "global"
    assert json.loads(got[0][1]) == []


def test_cannot_rescope_another_profiles_fact(client):
    tc, h = client
    # Active profile is 'default'; f_theirs belongs to 'other'.
    r = tc.patch("/api/memories/f_theirs/scope",
                 json={"scope": "global"}, headers=h)
    assert r.status_code == 404, "must not re-scope another profile's fact"


def test_scope_view_filters(client):
    """GET /api/memories?scope= widens the view; default stays profile-only."""
    tc, h = client
    # Baseline: default profile sees only its own fact (f_mine).
    base = tc.get("/api/memories").json()
    assert base["total"] == 1

    # Make f_mine global, then check each view.
    tc.patch("/api/memories/f_mine/scope", json={"scope": "global"}, headers=h)
    assert tc.get("/api/memories?scope=global").json()["total"] == 1
    assert tc.get("/api/memories?scope=shared").json()["total"] == 0
    # 'all' = this profile + global + shared-with-me.
    assert tc.get("/api/memories?scope=all").json()["total"] >= 1

    # Share the OTHER profile's fact with 'default', then shared view sees it.
    tc.app.state.engine._db.execute(
        "UPDATE atomic_facts SET scope='shared', shared_with=? WHERE fact_id='f_theirs'",
        ('["default"]',),
    )
    assert tc.get("/api/memories?scope=shared").json()["total"] == 1
