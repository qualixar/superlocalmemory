# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""RBAC HTTP surface: login/whoami + user & role admin + write enforcement.

Proves the RBAC layer is actually wired into the mutation path (the research
warning: a defined-but-uncalled access layer is worse than none):

  * owner (no users) bypasses RBAC — personal use has no friction.
  * a viewer is rejected (403) from deleting a memory; an admin passes the RBAC
    gate (404 for a missing fact, i.e. it reached the engine).
  * require_login mode blocks the owner's data mutations (401) but never locks
    the owner out of administration (MANAGE still allowed).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


@pytest.fixture
def client(engine_with_mock_deps):
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    app.state.rbac = RbacEngine(str(engine._config.db_path))
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app), _daemon_headers(app)


def test_owner_is_root_when_no_users(client):
    tc, h = client
    r = tc.get("/api/rbac/whoami", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "owner"
    assert body["rbac_active"] is False
    assert "delete" in body["permissions"]


def test_create_user_login_whoami(client):
    tc, h = client
    # Owner creates an admin user on the active profile.
    r = tc.post("/api/rbac/users",
                json={"username": "alice", "password": "password-1234",
                      "display_name": "Alice", "role": "admin"}, headers=h)
    assert r.status_code == 200, r.text

    st = tc.get("/api/rbac/status", headers=h).json()
    assert st["rbac_active"] is True and st["user_count"] == 1

    login = tc.post("/api/rbac/login",
                    json={"username": "alice", "password": "password-1234"},
                    headers=h)
    assert login.status_code == 200, login.text
    token = login.json()["token"]

    who = tc.get("/api/rbac/whoami",
                 headers={**h, "X-SLM-User-Session": token}).json()
    assert who["kind"] == "user" and who["role"] == "admin"


def test_bad_login_rejected(client):
    tc, h = client
    tc.post("/api/rbac/users",
            json={"username": "bob", "password": "password-1234"}, headers=h)
    r = tc.post("/api/rbac/login",
                json={"username": "bob", "password": "wrong-password"}, headers=h)
    assert r.status_code == 401


def test_viewer_cannot_delete_but_admin_passes_gate(client):
    tc, h = client
    # Create an admin and a viewer, both members of 'default'.
    tc.post("/api/rbac/users",
            json={"username": "adm", "password": "password-1234", "role": "admin"},
            headers=h)
    tc.post("/api/rbac/users",
            json={"username": "vwr", "password": "password-1234", "role": "viewer"},
            headers=h)
    # Mint sessions via the engine (the login body omits the token for
    # browser-like clients; TestClient persists cookies so a 2nd login looks
    # browser-like — engine-minted tokens keep the two users independent).
    rbac = tc.app.state.rbac
    users = {u["username"]: u["user_id"] for u in rbac.list_users()}
    adm_tok = rbac.create_session(users["adm"])
    vwr_tok = rbac.create_session(users["vwr"])

    # Viewer: blocked by RBAC (403).
    rv = tc.delete("/api/memories/nonexistent",
                   headers={**h, "X-SLM-User-Session": vwr_tok})
    assert rv.status_code == 403, rv.text

    # Admin: passes the RBAC gate → reaches the engine → 404 for a missing fact.
    ra = tc.delete("/api/memories/nonexistent",
                   headers={**h, "X-SLM-User-Session": adm_tok})
    assert ra.status_code == 404, ra.text


def test_manage_requires_permission(client):
    tc, h = client
    # A viewer must not be able to list users (MANAGE).
    tc.post("/api/rbac/users",
            json={"username": "v2", "password": "password-1234", "role": "viewer"},
            headers=h)
    rbac = tc.app.state.rbac
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["v2"]
    tok = rbac.create_session(uid)
    r = tc.get("/api/rbac/users", headers={**h, "X-SLM-User-Session": tok})
    assert r.status_code == 403


def test_require_login_blocks_owner_data_but_not_manage(client):
    tc, h = client
    # Turn on company mode.
    tc.post("/api/rbac/users",
            json={"username": "admin9", "password": "password-1234", "role": "admin"},
            headers=h)
    r = tc.post("/api/rbac/policy", json={"require_login": True}, headers=h)
    assert r.status_code == 200

    # Owner (no session) now blocked from data mutation (401)...
    rd = tc.delete("/api/memories/whatever", headers=h)
    assert rd.status_code == 401, rd.text

    # ...but owner can STILL administer (MANAGE) — no dashboard lockout.
    ru = tc.get("/api/rbac/users", headers=h)
    assert ru.status_code == 200, ru.text


@pytest.mark.parametrize(
    "path",
    (
        "/api/v3/brain",
        "/api/v3/brain/evolution-timeseries",
        "/api/v3/learning/stats",
        "/api/v3/patterns",
        "/api/v3/behavioral",
    ),
)
def test_brain_routes_authorize_the_requested_profile(client, path):
    """A valid install token cannot bypass profile-scoped READ authorization."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    tc, h = client
    engine = tc.app.state.engine
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other', 'other')"
    )
    tc.post(
        "/api/rbac/users",
        json={"username": "brain-viewer", "password": "password-1234",
              "role": "viewer"},
        headers=h,
    )
    rbac = tc.app.state.rbac
    user_id = {user["username"]: user["user_id"] for user in rbac.list_users()}["brain-viewer"]
    session = rbac.create_session(user_id)
    headers = {
        **h,
        "X-Install-Token": ensure_install_token(),
        "X-SLM-User-Session": session,
    }

    allowed = tc.get(path, params={"profile_id": "default"}, headers=headers)
    assert allowed.status_code == 200, allowed.text

    denied = tc.get(path, params={"profile_id": "other"}, headers=headers)
    assert denied.status_code == 403, denied.text


def test_brain_route_requires_session_in_company_mode(client):
    """Install-token ownership does not bypass company-mode data READ rules."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    tc, h = client
    tc.post(
        "/api/rbac/users",
        json={"username": "company-brain-admin", "password": "password-1234",
              "role": "admin"},
        headers=h,
    )
    tc.app.state.rbac.set_require_login(True)

    response = tc.get(
        "/api/v3/brain",
        headers={**h, "X-Install-Token": ensure_install_token()},
    )
    assert response.status_code == 401, response.text


@pytest.mark.parametrize(
    "path",
    (
        "/api/v3/associations",
        "/api/v3/associations/stats",
        "/api/v3/consolidation/status",
        "/api/v3/core-memory",
        "/api/v3/vector-store/status",
        "/api/v3/forgetting/stats",
        "/api/v3/quantization/stats",
        "/api/v3/ccq/blocks",
        "/api/v3/soft-prompts",
        "/api/v3/graph/communities",
        "/api/v3/v33/overview",
    ),
)
def test_v3_profile_readers_authorize_the_requested_profile(client, path):
    """A profile query may never bypass the role on that requested profile."""
    tc, h = client
    engine = tc.app.state.engine
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other-v3', 'other-v3')"
    )
    tc.post(
        "/api/rbac/users",
        json={"username": "v3-viewer", "password": "password-1234", "role": "viewer"},
        headers=h,
    )
    rbac = tc.app.state.rbac
    user_id = {user["username"]: user["user_id"] for user in rbac.list_users()}["v3-viewer"]
    headers = {**h, "X-SLM-User-Session": rbac.create_session(user_id)}

    denied = tc.get(path, params={"profile": "other-v3"}, headers=headers)
    assert denied.status_code == 403, denied.text


@pytest.mark.parametrize(
    "path",
    ("/api/v3/health/processes", "/api/v3/forgetting/stats"),
)
def test_v3_sensitive_readers_require_a_session_in_company_mode(client, path):
    tc, h = client
    tc.post(
        "/api/rbac/users",
        json={"username": "company-v3-admin", "password": "password-1234", "role": "admin"},
        headers=h,
    )
    tc.app.state.rbac.set_require_login(True)

    response = tc.get(path, headers=h)
    assert response.status_code == 401, response.text


@pytest.mark.parametrize(
    "path,body",
    (
        ("/api/v3/consolidation/trigger", {"profile": "other-manage"}),
        ("/api/v3/forgetting/run", {"profile": "other-manage"}),
    ),
)
def test_v3_profile_mutations_require_manage_on_target_profile(client, path, body):
    tc, h = client
    engine = tc.app.state.engine
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other-manage', 'other-manage')"
    )
    tc.post(
        "/api/rbac/users",
        json={"username": "target-viewer", "password": "password-1234", "role": "viewer"},
        headers=h,
    )
    rbac = tc.app.state.rbac
    user_id = {user["username"]: user["user_id"] for user in rbac.list_users()}["target-viewer"]
    response = tc.post(
        path,
        json=body,
        headers={**h, "X-SLM-User-Session": rbac.create_session(user_id)},
    )
    assert response.status_code == 403, response.text


def test_embedding_probe_requires_manage_before_outbound_network_access(client):
    tc, h = client
    tc.post(
        "/api/rbac/users",
        json={"username": "embedding-viewer", "password": "password-1234", "role": "viewer"},
        headers=h,
    )
    rbac = tc.app.state.rbac
    user_id = {user["username"]: user["user_id"] for user in rbac.list_users()}["embedding-viewer"]

    response = tc.post(
        "/api/v3/embedding/test",
        json={"api_endpoint": "http://127.0.0.1:9"},
        headers={**h, "X-SLM-User-Session": rbac.create_session(user_id)},
    )
    assert response.status_code == 403, response.text


def test_daemon_rejects_uncredentialed_browser_writes_from_another_local_port(client):
    """The production middleware must not trust localhost without port identity."""
    tc, _headers = client
    response = tc.post(
        "/api/v3/embedding/test",
        json={"api_endpoint": "http://127.0.0.1:9"},
        headers={"Origin": "http://localhost:8417"},
    )
    assert response.status_code == 403, response.text
    assert "cross-origin" in response.json()["error"]


def test_ratelimit_read_requires_session_in_company_mode(client):
    """Governance limits follow the same READ policy as other dashboard data."""
    tc, headers = client
    tc.post(
        "/api/rbac/users",
        json={"username": "rate-viewer", "password": "password-1234",
              "role": "viewer"},
        headers=headers,
    )
    rbac = tc.app.state.rbac
    user_id = {
        user["username"]: user["user_id"] for user in rbac.list_users()
    }["rate-viewer"]
    session = rbac.create_session(user_id)
    rbac.set_require_login(True)

    unauthenticated = tc.get("/api/v3/ratelimit", headers=headers)
    assert unauthenticated.status_code == 401, unauthenticated.text

    authenticated = tc.get(
        "/api/v3/ratelimit",
        headers={**headers, "X-SLM-User-Session": session},
    )
    assert authenticated.status_code == 200, authenticated.text
