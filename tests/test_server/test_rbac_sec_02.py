# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# SECURITY REGRESSION — RBAC (from audit-02). Originally PoCs that proved the
# gaps; now assert the FIXED behaviour so the fixes can never silently regress.
#
# Run: SLM_TEST_ISOLATION=1 pytest tests/test_server/test_rbac_sec_02.py -v

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


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
    for pid in ("default", "personal"):
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    app.state.rbac = RbacEngine(str(engine._config.db_path))
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app, raise_server_exceptions=False), _daemon_headers(app)


def _mk_user(tc, h, username, role=None, profile_id=None):
    body = {"username": username, "password": "password-1234"}
    if role:
        body["role"] = role
        if profile_id:
            body["profile_id"] = profile_id
    tc.post("/api/rbac/users", json=body, headers=h)
    uid = {u["username"]: u["user_id"] for u in tc.app.state.rbac.list_users()}[username]
    return uid


def _session(tc, uid):
    return tc.app.state.rbac.create_session(uid)


# -- SEC-C-01 read gate ------------------------------------------------------

def test_reads_open_in_single_operator_mode(client):
    """With zero users (personal mode) reads stay open — no friction."""
    tc, h = client
    assert tc.get("/api/memories", headers=h).status_code == 200


def test_non_member_user_cannot_read(client):
    """Once RBAC is active, a logged-in user with no membership is denied read."""
    tc, h = client
    orphan = _mk_user(tc, h, "orphan")  # no role → no membership
    tok = _session(tc, orphan)
    r = tc.get("/api/memories", headers={**h, "X-SLM-User-Session": tok})
    assert r.status_code == 403, r.text


def test_member_can_read(client):
    """A viewer member of the active profile can read."""
    tc, h = client
    uid = _mk_user(tc, h, "viewer1", role="viewer", profile_id="default")
    tok = _session(tc, uid)
    assert tc.get("/api/memories", headers={**h, "X-SLM-User-Session": tok}).status_code == 200


# -- M-API-5 config-metadata GETs respect the read gate ----------------------

_CONFIG_GETS = (
    "/api/v3/dashboard",
    "/api/v3/mode",
    "/api/v3/embedding/config",
    "/api/learning/status",
    "/api/learning/ranker_phase",
    "/api/behavioral/status",
    "/api/behavioral/assertions",
    "/api/behavioral/tool-events",
    "/api/behavioral/soft-prompts",
    "/api/patterns",
    "/api/feedback/stats",
    "/api/stats",
    "/api/timeline",
)


def test_config_gets_open_in_single_operator_mode(client):
    """Personal mode (no users): config GETs stay open — no login friction."""
    tc, _h = client
    for p in _CONFIG_GETS:
        assert tc.get(p).status_code != 401, p


def test_config_gets_require_login_in_company_mode(client):
    """M-API-5: config GETs expose base_dir + LLM stack. Once company mode
    (require_login) is on, an unauthenticated caller must be denied (401), not
    served the install path/provider/model."""
    tc, h = client
    _mk_user(tc, h, "admin1", role="admin", profile_id="default")  # RBAC now active
    tc.app.state.rbac.set_require_login(True)
    for p in _CONFIG_GETS:
        r = tc.get(p)  # no session, no daemon capability
        assert r.status_code == 401, f"{p} -> {r.status_code}: {r.text}"


_WRITE_MUTATIONS = (
    ("post", "/api/feedback", {}),
    ("post", "/api/feedback/dwell", {}),
    ("post", "/api/behavioral/report-outcome", {}),
    ("post", "/api/v3/tool-event", {}),
)
_DELETE_MUTATIONS = (
    ("delete", "/api/patterns/delete", {}),
    ("post", "/api/learning/reset", {}),
)
_MANAGE_MUTATIONS = (
    ("post", "/api/learning/backup", {}),
    ("post", "/api/learning/retrain", {}),
    ("post", "/api/learning/migrate-legacy", {}),
)


@pytest.mark.parametrize("method,path,body", _WRITE_MUTATIONS + _DELETE_MUTATIONS)
def test_learning_data_mutations_require_login_in_company_mode(
    client, method, path, body,
):
    tc, h = client
    _mk_user(tc, h, "company_admin", role="admin", profile_id="default")
    tc.app.state.rbac.set_require_login(True)

    response = tc.request(method.upper(), path, json=body, headers=h)

    assert response.status_code == 401, (
        f"{method.upper()} {path} bypassed company-mode login: "
        f"{response.status_code} {response.text}"
    )


@pytest.mark.parametrize(
    "method,path,body",
    _WRITE_MUTATIONS + _DELETE_MUTATIONS + _MANAGE_MUTATIONS,
)
def test_viewer_cannot_mutate_learning_or_behavioral_state(
    client, method, path, body,
):
    tc, h = client
    viewer = _mk_user(
        tc, h, f"viewer_{path.replace('/', '_')}",
        role="viewer", profile_id="default",
    )
    token = _session(tc, viewer)

    response = tc.request(
        method.upper(),
        path,
        json=body,
        headers={**h, "X-SLM-User-Session": token},
    )

    assert response.status_code == 403, (
        f"{method.upper()} {path} allowed viewer mutation: "
        f"{response.status_code} {response.text}"
    )


# -- SEC-H-01 config mutations are admin-only (privilege-escalation guard) ----

_CONFIG_MUTATIONS = [
    ("put", "/api/v3/mode", {"mode": "a"}),
    ("post", "/api/v3/mode/set", {"mode": "a", "provider": "none"}),
    ("put", "/api/v3/embedding/config", {"provider": "ollama"}),
    ("put", "/api/v3/scope/config", {"default_scope": "personal"}),
    ("put", "/api/v3/storage/config", {"graph_backend": "sqlite"}),
    ("put", "/api/v3/daemon/config", {"port": 9999}),
    ("put", "/api/v3/mesh/config", {"enabled": False}),
    ("put", "/api/v3/trust/config", {}),
    ("put", "/api/v3/forgetting/config", {}),
]


def test_config_mutations_blocked_for_viewer(client):
    """SEC-H-01: a logged-in VIEWER must NOT be able to change system config
    (swap LLM key, change daemon port, zero-out the forgetting curve). Each of
    the 9 config mutation endpoints must return 403 before mutating."""
    tc, h = client
    uid = _mk_user(tc, h, "viewer_cfg", role="viewer", profile_id="default")
    tok = _session(tc, uid)
    hs = {**h, "X-SLM-User-Session": tok}
    for method, path, body in _CONFIG_MUTATIONS:
        r = getattr(tc, method)(path, json=body, headers=hs)
        assert r.status_code == 403, f"{method.upper()} {path} -> {r.status_code}: {r.text}"


# -- SEC-C-02 cross-profile membership escalation ---------------------------

def test_cross_profile_set_membership_blocked(client):
    tc, h = client
    attacker = _mk_user(tc, h, "attacker", role="admin", profile_id="default")
    tok = _session(tc, attacker)
    r = tc.post("/api/rbac/members",
                json={"user_id": attacker, "role": "admin", "profile_id": "personal"},
                headers={**h, "X-SLM-User-Session": tok})
    assert r.status_code == 403, r.text
    assert tc.app.state.rbac.get_role(attacker, "personal") is None


def test_cross_profile_list_members_blocked(client):
    tc, h = client
    admin = _mk_user(tc, h, "nosy", role="admin", profile_id="default")
    tok = _session(tc, admin)
    r = tc.get("/api/rbac/members?profile_id=personal",
               headers={**h, "X-SLM-User-Session": tok})
    assert r.status_code == 403, r.text


# -- SEC-H-01 stale membership on profile delete ----------------------------

def test_membership_purged_on_profile_delete(client):
    tc, h = client
    rbac = tc.app.state.rbac
    alice = _mk_user(tc, h, "alice")
    from superlocalmemory.server.routes.helpers import (
        delete_profile_from_db, ensure_profile_in_db,
    )
    ensure_profile_in_db("corp", "corp")
    rbac.set_membership("corp", alice, "admin")
    assert rbac.get_role(alice, "corp") is not None
    delete_profile_from_db("corp")
    assert rbac.get_role(alice, "corp") is None, "stale membership survived delete"
    ensure_profile_in_db("corp", "new corp")
    assert rbac.get_role(alice, "corp") is None, "recreated profile re-granted access"


# -- SEC-H-02 install token only from loopback ------------------------------

def test_install_token_rejected_from_non_loopback():
    from unittest.mock import MagicMock, patch
    from fastapi import HTTPException
    from superlocalmemory.server.write_identity import require_write_actor

    req = MagicMock()
    req.client.host = "10.0.0.99"  # LAN, not loopback
    req.headers = {"X-Install-Token": "valid"}

    with (
        patch("superlocalmemory.server.write_identity._header",
              lambda r, n: r.headers.get(n, "")),
        patch("superlocalmemory.core.security_primitives.verify_install_token",
              return_value=True),
        patch("superlocalmemory.infra.auth_middleware.verify_api_key",
              return_value=False),
    ):
        with pytest.raises(HTTPException) as ei:
            require_write_actor(req, descriptor=None, actor_kind="test")
        assert ei.value.status_code == 403


# -- SEC-H-03 IDOR on user modification -------------------------------------

def test_idor_patch_user_blocked(client):
    tc, h = client
    rbac = tc.app.state.rbac
    alice = _mk_user(tc, h, "alice2", role="admin", profile_id="default")
    bob = _mk_user(tc, h, "bob2")
    rbac.set_membership("personal", bob, "viewer")  # bob NOT on default
    tok = _session(tc, alice)
    r = tc.patch(f"/api/rbac/users/{bob}", json={"password": "hacked-password!"},
                 headers={**h, "X-SLM-User-Session": tok})
    assert r.status_code == 403, r.text
    assert rbac.verify_credentials("bob2", "hacked-password!") is None


# -- SEC-M-01 no username enumeration (status parity) -----------------------

def test_unknown_and_wrong_password_both_401(client):
    tc, h = client
    _mk_user(tc, h, "real_user")
    r_unknown = tc.post("/api/rbac/login",
                        json={"username": "ghost_xyz", "password": "x-wrong-pass"}, headers=h)
    r_wrong = tc.post("/api/rbac/login",
                      json={"username": "real_user", "password": "x-wrong-pass"}, headers=h)
    assert r_unknown.status_code == 401 and r_wrong.status_code == 401


# -- SEC-M-02 cookie Secure under HTTPS -------------------------------------

def test_session_cookie_secure_under_https(client):
    tc, h = client
    _mk_user(tc, h, "cookietest")
    resp = tc.post("/api/rbac/login",
                   json={"username": "cookietest", "password": "password-1234"},
                   headers={**h, "X-Forwarded-Proto": "https"})
    assert resp.status_code == 200
    assert "secure" in resp.headers.get("set-cookie", "").lower()


# -- SEC-L-02 token not echoed to browser clients ---------------------------

def test_token_omitted_for_browser_clients(client):
    tc, h = client
    _mk_user(tc, h, "logtest")
    resp = tc.post("/api/rbac/login",
                   json={"username": "logtest", "password": "password-1234"},
                   headers={**h, "Referer": "http://127.0.0.1:8765/"})
    assert "token" not in resp.json(), "raw session token leaked to a browser client"
