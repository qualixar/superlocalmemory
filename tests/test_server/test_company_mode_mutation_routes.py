# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Company mode: every route that changes state asks for a signed-in user with the right role.

``test_company_mode_get_routes`` walks the GET side. This walks the other half:
every POST, PUT, PATCH and DELETE the daemon registers. A workspace that turned
on "Require login" believes each change is made by a named person with a role
that allows it. Before this walk existed, about fifteen routes (observe, profile
switch, the mesh, Optimize settings, compaction, tiering, heal and others) only
proved a machine credential, so any local process holding the install token
(which ``GET /internal/token`` hands to every loopback caller) could change the
shared workspace, and a read-only viewer could lock files and change settings.

Three callers are walked over every route:

* a loopback process with only the install token and no session,
* a loopback process with no credential at all,
* a signed-in ``viewer`` (read-only).

All three must be refused with 401 or 403 and a message that says why, before
any body validation runs (a 422 would mean the handler was reached first). Only
the routes in ``OPEN_ROUTES`` and ``VIEWER_MAY`` are exempt, each with its
reason written next to it. A new route that forgets its gate fails here.
"""

# ruff: noqa: F811  (imported pytest fixtures are re-declared as test parameters)

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from tests.test_server.test_rbac_company_mode_credential import (
    LOOPBACK,
    _company_mode,
    app_and_engine,  # noqa: F401  (pytest fixture)
)

MUTATING = ("POST", "PUT", "PATCH", "DELETE")

#: Routes that answer a caller with no session on purpose, with why.
OPEN_ROUTES = {
    ("POST", "/api/rbac/login"): "signing in is how a person gets a session; it cannot require one",
    ("POST", "/internal/prewarm"): (
        "hook hot path with its own four gates (loopback peer, no browser Origin, install "
        "token, bounded body); it fills a derived context cache and stores no memory"
    ),
    ("POST", "/api/rbac/logout"): "ends the caller's own session; none means nothing to end",
    ("POST", "/api/v3/embed"): (
        "computes vectors for text the caller sends and stores nothing; the MCP embedder "
        "proxy has no session. The machine credential check still applies"
    ),
}

#: Routes only the daemon capability opens (a file only this computer's user can read).
#: They refuse the install token, which is all the walk presents.
CAPABILITY_ONLY = {
    ("POST", "/stop"): "stopping the daemon is break-glass for the person at the keyboard",
    ("POST", "/internal/hook-events"): (
        "hooks of this computer's user hand tool events and reward signals to the daemon's "
        "single writer (4.2.8); before that they wrote memory.db directly"
    ),
}

#: Routes a signed-in read-only viewer may call, with why (they read, they do not write).
VIEWER_MAY = {
    ("POST", "/api/search"): "dashboard search; READ is enforced by the read gate",
    ("POST", "/api/v3/chat/stream"): "memory chat answers from memory; the read gate enforces READ",
    ("POST", "/api/v3/recall/trace"): "explains a recall; READ is enforced by the route",
    ("POST", "/session/open"): "warms recall context and changes no memory; READ",
}


def _walk(routes, prefix=""):
    for r in routes:
        if type(r).__name__ == "_IncludedRouter":
            yield from _walk(r.original_router.routes, prefix + (r.include_context.prefix or ""))
        elif hasattr(r, "path"):
            yield r, prefix + r.path


def _mutating(app) -> list[tuple[str, str]]:
    found = set()
    for route, path in _walk(app.routes):
        for method in getattr(route, "methods", None) or ():
            if method in MUTATING:
                found.add((method, path))
    return sorted(found)


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "x", path)


@pytest.fixture(autouse=True)
def _nothing_leaves_the_sandbox(monkeypatch):
    """A route the gate failed to stop must not spawn a process or stop the daemon."""
    import subprocess

    def _refuse(*_a, **_k):
        raise RuntimeError("test: a subprocess must not start")

    monkeypatch.setattr(subprocess, "Popen", _refuse)
    monkeypatch.setattr(
        "superlocalmemory.server.unified_daemon._request_graceful_shutdown", lambda *a, **k: None,
    )
    yield
    from superlocalmemory.server.manual_heal import join_manual_heals
    assert join_manual_heals(timeout_s=None), (
        "manual repair must finish before subprocess patches are restored")


@pytest.fixture
def company(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token
    from superlocalmemory.server.routes.optimize import router as optimize_router

    app = app_and_engine
    # The Optimize router is mounted when the daemon starts, not when the app is built.
    if not any(p.startswith("/api/optimize") for _, p in _walk(app.routes)):
        app.include_router(optimize_router)
    _company_mode(app)
    rbac = app.state.rbac
    rbac.create_user("viewer1", "password-1234", display_name="Viewer")
    rbac.create_user("member1", "password-1234", display_name="Member")
    ids = {u["username"]: u["user_id"] for u in rbac.list_users()}
    rbac.set_membership("default", ids["viewer1"], "viewer", added_by="test")
    rbac.set_membership("default", ids["member1"], "member", added_by="test")
    token = ensure_install_token()
    sessions = {
        name: {"X-SLM-User-Session": rbac.create_session(ids[name]), "X-Install-Token": token}
        for name in ("admin1", "viewer1", "member1")
    }
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    return app, client, {"token": {"X-Install-Token": token}, "none": {}, **sessions}


def _message(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text
    if isinstance(body, dict):
        return str(body.get("error") or body.get("detail") or "")
    return str(body)


def _call(client, method, path, headers):
    return client.request(method, _concrete(path), headers=headers)


def _unrefused(client, app, headers, *, allowed=()):
    out = []
    for method, path in _mutating(app):
        if (method, path) in allowed:
            continue
        r = _call(client, method, path, headers)
        if r.status_code not in (401, 403):
            out.append((method, path, r.status_code))
    return out


def test_the_walk_sees_the_routes(company):
    app, _, _ = company
    routes = _mutating(app)
    assert len(routes) > 150
    for needle in (("POST", "/observe"), ("POST", "/api/profiles/{name}/switch"),
                   ("POST", "/mesh/lock"), ("PUT", "/api/optimize/config"),
                   ("POST", "/api/lifecycle/compact"), ("POST", "/api/tiers/evaluate"),
                   ("POST", "/api/v3/components/heal"), ("POST", "/api/daemon/restart")):
        assert needle in routes, needle
    assert set(OPEN_ROUTES) | set(CAPABILITY_ONLY) | set(VIEWER_MAY) <= set(routes)


def test_every_mutating_route_refuses_the_install_token_without_a_session(company):
    app, client, who = company
    answered = _unrefused(client, app, who["token"], allowed=OPEN_ROUTES)
    assert not answered, f"changed state for a caller with no session in company mode: {answered}"


def test_every_mutating_route_refuses_a_caller_with_no_credential(company):
    app, client, who = company
    answered = _unrefused(client, app, who["none"], allowed=OPEN_ROUTES)
    assert not answered, f"changed state for an anonymous caller in company mode: {answered}"


def test_every_mutating_route_refuses_a_read_only_viewer(company):
    app, client, who = company
    allowed = {**OPEN_ROUTES, **VIEWER_MAY}
    answered = _unrefused(client, app, who["viewer1"], allowed=allowed)
    assert not answered, f"a viewer was not refused: {answered}"


def test_a_refusal_says_why_and_is_never_a_server_error(company):
    app, client, who = company
    for method, path in (("POST", "/observe"), ("POST", "/mesh/lock"),
                         ("PUT", "/api/optimize/config"), ("POST", "/api/lifecycle/compact")):
        no_session = _call(client, method, path, who["token"])
        viewer = _call(client, method, path, who["viewer1"])
        # 401 for a write (sign in), 403 for administration (a signed-in administrator is needed).
        assert no_session.status_code in (401, 403), (path, no_session.status_code)
        assert "login" in _message(no_session).lower(), (path, _message(no_session))
        assert viewer.status_code == 403, (path, viewer.status_code)
        assert "role" in _message(viewer).lower(), (path, _message(viewer))


def test_the_routes_a_viewer_may_call_stay_open_to_a_viewer(company):
    app, client, who = company
    for method, path in VIEWER_MAY:
        r = _call(client, method, path, who["viewer1"])
        assert r.status_code not in (401, 403), (path, r.status_code, _message(r))


def test_a_viewer_enters_a_workspace_they_belong_to_but_no_other(company):
    """The workspace picker is a read; the owner with only a token may not switch at all."""
    app, client, who = company
    other = client.post("/api/profiles/teamb/switch", headers=who["viewer1"])
    assert other.status_code == 403, (other.status_code, _message(other))
    own = client.post("/api/profiles/default/switch", headers=who["viewer1"])
    assert own.status_code not in (401, 403), (own.status_code, _message(own))
    token_only = client.post("/api/profiles/default/switch", headers=who["token"])
    assert token_only.status_code in (401, 403), (token_only.status_code, _message(token_only))


def test_a_member_still_writes_and_an_admin_still_administers(company):
    app, client, who = company
    member = client.post("/observe", json={"content": "the team agreed to ship on friday"},
                         headers=who["member1"])
    assert member.status_code not in (401, 403), (member.status_code, _message(member))
    for method, path in (("PUT", "/api/optimize/config"), ("POST", "/api/lifecycle/compact"),
                         ("POST", "/api/tiers/evaluate")):
        admin = _call(client, method, path, who["admin1"])
        assert admin.status_code not in (401, 403), (path, admin.status_code, _message(admin))
    # A member writes, but changing the product's settings is administration.
    assert _call(client, "PUT", "/api/optimize/config", who["member1"]).status_code == 403
    assert _call(client, "POST", "/api/v3/components/heal", who["member1"]).status_code == 403
    assert _call(client, "POST", "/mesh/lock", who["member1"]).status_code not in (401, 403)


def test_open_routes_carry_a_reason():
    for table in (OPEN_ROUTES, CAPABILITY_ONLY, VIEWER_MAY):
        assert all(reason.strip() for reason in table.values())


def test_capability_only_routes_refuse_the_install_token(company):
    _, client, who = company
    for method, path in CAPABILITY_ONLY:
        assert _call(client, method, path, who["token"]).status_code in (401, 403)
        assert _call(client, method, path, who["viewer1"]).status_code in (401, 403)
