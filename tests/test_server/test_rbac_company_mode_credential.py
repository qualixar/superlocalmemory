# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Company mode: administration needs the machine credential (audit F1).

A caller with no credential and no session used to resolve to the machine
owner, and the owner keeps MANAGE even when ``require_login`` is on. Any local
process could therefore POST /api/rbac/policy ``{require_login: false}`` and
read everyone's data. In company mode the owner's MANAGE (and every RBAC
route) now requires the install token / daemon capability / API key, and a
session token that does not resolve is refused instead of becoming the owner.
Personal mode is unchanged.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

LOOPBACK = ("127.0.0.1", 50123)


@pytest.fixture
def app_and_engine(engine_with_mock_deps):
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
    return app


def _loopback(app) -> TestClient:
    # A credential-less client on the loopback interface: any local process.
    return TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK)


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


def _company_mode(app) -> dict[str, str]:
    """Create one admin user and turn require_login on. Returns credentials."""
    creds = _daemon_headers(app)
    rbac = app.state.rbac
    rbac.create_user("admin1", "password-1234", display_name="Admin")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["admin1"]
    rbac.set_membership("default", uid, "admin", added_by="test")
    rbac.set_require_login(True)
    return creds


# -- the reproduction ----------------------------------------------------------

def test_credentialless_loopback_cannot_switch_company_mode_off(app_and_engine):
    app = app_and_engine
    _company_mode(app)

    r = _loopback(app).post("/api/rbac/policy", json={"require_login": False})

    assert r.status_code in (401, 403), r.text
    assert app.state.rbac.require_login() is True


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/api/rbac/users"),
        ("GET", "/api/rbac/members"),
        ("GET", "/api/rbac/status"),
        ("GET", "/api/rbac/whoami"),
        ("POST", "/api/rbac/logout"),
    ),
)
def test_every_rbac_route_needs_the_credential_in_company_mode(
    app_and_engine, method, path,
):
    app = app_and_engine
    _company_mode(app)

    r = _loopback(app).request(method, path)

    assert r.status_code in (401, 403), (path, r.text)


def test_valid_machine_credential_still_administers(app_and_engine):
    app = app_and_engine
    creds = _company_mode(app)
    tc = _loopback(app)

    users = tc.get("/api/rbac/users", headers=creds)
    assert users.status_code == 200, users.text

    off = tc.post("/api/rbac/policy", json={"require_login": False}, headers=creds)
    assert off.status_code == 200, off.text
    assert app.state.rbac.require_login() is False


def test_install_token_alone_does_not_administer_in_company_mode(app_and_engine):
    """The install token is served to any loopback caller, so it is not authority to administer."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)
    headers = {"X-Install-Token": ensure_install_token()}
    tc = _loopback(app)

    for path in ("/api/rbac/users", "/api/rbac/members"):
        r = tc.get(path, headers=headers)
        assert r.status_code in (401, 403), (path, r.text)
    # ...yet it is still accepted as the dashboard's identity bootstrap
    assert tc.get("/api/rbac/status", headers=headers).status_code == 200


def test_the_token_fetched_from_internal_token_cannot_switch_company_mode_off(app_and_engine):
    """The reviewer's attack: read the token over loopback, then POST the policy."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    ensure_install_token()   # the daemon writes the token file at start-up
    app = app_and_engine
    _company_mode(app)
    tc = _loopback(app)

    fetched = tc.get("/internal/token")
    assert fetched.status_code == 200, fetched.text
    token = fetched.json()["token"]

    r = tc.post("/api/rbac/policy", json={"require_login": False},
                headers={"X-Install-Token": token})

    assert r.status_code in (401, 403), r.text
    assert app.state.rbac.require_login() is True


def test_an_api_key_alone_cannot_administer_in_company_mode(app_and_engine, monkeypatch):
    from superlocalmemory.infra import auth_middleware

    app = app_and_engine
    _company_mode(app)
    monkeypatch.setattr(auth_middleware, "verify_api_key", lambda presented: presented == "lan-key")

    r = TestClient(app, base_url="http://10.0.0.5:8765", client=("10.0.0.5", 4000)).post(
        "/api/rbac/policy", json={"require_login": False}, headers={"X-SLM-API-Key": "lan-key"})

    assert r.status_code in (401, 403), r.text
    assert app.state.rbac.require_login() is True


def test_an_admin_session_with_the_install_token_administers_from_the_dashboard(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)
    rbac = app.state.rbac
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["admin1"]
    headers = {"X-Install-Token": ensure_install_token(),
               "X-SLM-User-Session": rbac.create_session(uid)}
    tc = _loopback(app)

    assert tc.get("/api/rbac/users", headers=headers).status_code == 200
    off = tc.post("/api/rbac/policy", json={"require_login": False}, headers=headers)
    assert off.status_code == 200, off.text


def test_a_viewer_session_cannot_administer_even_with_every_credential(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    creds = _company_mode(app)
    rbac = app.state.rbac
    rbac.create_user("viewer1", "password-1234")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["viewer1"]
    rbac.set_membership("default", uid, "viewer", added_by="test")
    headers = {**creds, "X-Install-Token": ensure_install_token(),
               "X-SLM-User-Session": rbac.create_session(uid)}

    r = _loopback(app).post("/api/rbac/policy", json={"require_login": False}, headers=headers)

    assert r.status_code == 403, r.text


def test_the_install_token_still_carries_a_signed_in_users_ordinary_writes(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)
    rbac = app.state.rbac
    rbac.create_user("member1", "password-1234")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["member1"]
    rbac.set_membership("default", uid, "admin", added_by="test")
    headers = {"X-Install-Token": ensure_install_token(),
               "X-SLM-User-Session": rbac.create_session(uid)}

    r = _loopback(app).delete("/api/memories/nonexistent", headers=headers)

    assert r.status_code == 404, r.text   # past the RBAC gate; the fact simply is not there


def test_with_no_users_the_install_token_can_still_create_the_first_admin(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    app.state.rbac.set_require_login(True)      # company mode chosen, nobody enrolled yet
    headers = {"X-Install-Token": ensure_install_token()}

    r = _loopback(app).post(
        "/api/rbac/users", json={"username": "first", "password": "password-1234", "role": "admin"},
        headers=headers)

    assert r.status_code == 200, r.text


def test_whoami_for_a_logged_out_owner_lists_no_permissions_in_company_mode(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)

    body = _loopback(app).get(
        "/api/rbac/whoami", headers={"X-Install-Token": ensure_install_token()}).json()

    assert body["kind"] == "owner" and body["permissions"] == []


def test_wrong_install_token_is_refused(app_and_engine):
    app = app_and_engine
    _company_mode(app)

    r = _loopback(app).post(
        "/api/rbac/policy", json={"require_login": False},
        headers={"X-Install-Token": "not-the-token"},
    )

    assert r.status_code in (401, 403), r.text
    assert app.state.rbac.require_login() is True


# -- invalid session ------------------------------------------------------------

def test_invalid_session_is_401_in_company_mode(app_and_engine):
    app = app_and_engine
    creds = _company_mode(app)

    r = _loopback(app).get(
        "/api/rbac/users", headers={**creds, "X-SLM-User-Session": "forged"},
    )

    assert r.status_code == 401, r.text


def test_invalid_session_cookie_cannot_delete_in_company_mode(app_and_engine):
    app = app_and_engine
    creds = _company_mode(app)

    r = _loopback(app).post(
        "/api/rbac/policy", json={"require_login": False},
        headers={**creds, "Cookie": "slm_session=expired-or-forged"},
    )

    assert r.status_code == 401, r.text
    assert app.state.rbac.require_login() is True


def test_valid_admin_session_with_credential_works(app_and_engine):
    app = app_and_engine
    creds = _company_mode(app)
    rbac = app.state.rbac
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["admin1"]
    token = rbac.create_session(uid)

    r = _loopback(app).get(
        "/api/rbac/users", headers={**creds, "X-SLM-User-Session": token},
    )

    assert r.status_code == 200, r.text


def test_whoami_with_expired_session_does_not_raise(app_and_engine):
    """The dashboard asks who it is before it shows the login form."""
    app = app_and_engine
    creds = _company_mode(app)

    r = _loopback(app).get(
        "/api/rbac/whoami", headers={**creds, "X-SLM-User-Session": "expired"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["kind"] != "user"


# -- personal mode is unchanged --------------------------------------------------

def test_personal_mode_credentialless_loopback_still_works(app_and_engine):
    app = app_and_engine
    tc = _loopback(app)

    assert tc.get("/api/rbac/status").status_code == 200
    created = tc.post(
        "/api/rbac/users", json={"username": "pm", "password": "password-1234"},
    )
    assert created.status_code == 200, created.text


def test_personal_mode_invalid_session_is_still_owner(app_and_engine):
    app = app_and_engine

    r = _loopback(app).get(
        "/api/rbac/whoami", headers={"X-SLM-User-Session": "stale-cookie"},
    )

    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "owner"
