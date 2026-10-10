# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Company mode: every GET route that carries data needs a signed-in user.

A workspace with ``require_login`` on and users enrolled must not hand its data
to a process that merely sits on the loopback interface. This walks every GET
route the daemon registers, so a new route that forgets its gate fails here
rather than in a later audit. Only the routes in ``OPEN_ROUTES`` may answer
without a session, and each one has a reason written next to it.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from tests.test_server.test_rbac_company_mode_credential import (
    LOOPBACK,
    _company_mode,
    app_and_engine,  # noqa: F401  (pytest fixture)
)

# Routes that answer without a session on purpose, with why.
OPEN_ROUTES = {
    "/": "the dashboard page itself; it has to load to show the sign-in form",
    "/favicon.ico": "a static icon",
    "/health": "liveness probe used by the CLI and the service manager; no memory content",
    "/status": "daemon discovery probe the CLI uses (`slm status`): counts and file paths, no memory content",
    "/api/version": "version string only",
    "/openapi.json": "the shape of the API, no data",
    "/docs": "API documentation page, no data",
    "/docs/oauth2-redirect": "API documentation helper page, no data",
    "/redoc": "API documentation page, no data",
    "/internal/token": "serves the install token to loopback only; the token alone cannot administer or read in company mode",
    "/api/v3/embed/ping": "readiness probe for the MCP embedder proxy (which has no session); names the model only",
    "/api/backup/oauth/github/callback": "OAuth redirect target; refuses anything without a matching state value",
    "/api/backup/oauth/google/callback": "OAuth redirect target; refuses anything without a matching state value",
    "/api/v3/connections/callback": "OAuth redirect target; refuses anything without a matching state value",
}

# Long-lived streams: only the refusal is checked, a signed-in call would not return.
STREAMING_ROUTES = {"/events/stream", "/mesh/inbox/{peer_id}/wait"}

# Routes that must answer 200 for an admin (the ones an audit named, plus core reads).
MUST_BE_200 = (
    "/api/export", "/api/v3/learning/signals", "/api/v3/trust/dashboard",
    "/api/v3/features", "/api/memories", "/api/profiles", "/api/lifecycle/status",
    "/api/tiers/stats", "/api/v3/components", "/api/v3/provider",
)


def _walk(routes, prefix=""):
    for r in routes:
        if type(r).__name__ == "_IncludedRouter":
            yield from _walk(r.original_router.routes, prefix + (r.include_context.prefix or ""))
        elif hasattr(r, "path"):
            yield r, prefix + r.path


def _get_paths(app) -> list[str]:
    return sorted({p for r, p in _walk(app.routes) if "GET" in (getattr(r, "methods", None) or ())})


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "x", path)


@pytest.fixture
def company(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)
    rbac = app.state.rbac
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["admin1"]
    # What the dashboard sends once someone has signed in.
    admin = {"X-SLM-User-Session": rbac.create_session(uid),
             "X-Install-Token": ensure_install_token()}
    # A route that errors for want of a runtime in this test app answers 500,
    # which is not a refusal; do not let it abort the walk.
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    return app, client, admin


def test_the_walk_sees_the_routes(company):
    app, _, _ = company
    paths = _get_paths(app)
    assert len(paths) > 100
    assert "/api/export" in paths and "/api/memories" in paths
    assert set(OPEN_ROUTES) <= set(paths), set(OPEN_ROUTES) - set(paths)


def test_every_data_route_refuses_a_caller_with_no_session(company):
    app, client, _ = company
    answered = []
    for path in _get_paths(app):
        if path in OPEN_ROUTES:
            continue
        status = client.get(_concrete(path)).status_code
        if status not in (401, 403):
            answered.append((path, status))
    assert not answered, f"answered without a session in company mode: {answered}"


def test_every_data_route_still_serves_an_admin_session(company):
    app, client, admin = company
    refused = []
    for path in _get_paths(app):
        if path in OPEN_ROUTES or path in STREAMING_ROUTES:
            continue
        status = client.get(_concrete(path), headers=admin).status_code
        if path in MUST_BE_200 and status != 200:
            refused.append((path, status))
        elif status in (401, 403):
            refused.append((path, status))
    assert not refused, f"an admin session was refused: {refused}"


def test_export_is_refused_to_a_signed_in_user_without_read(company):
    app, client, admin = company
    rbac = app.state.rbac
    rbac.create_user("nobody1", "password-1234")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["nobody1"]
    # A member of some other workspace only: no role on the active one.
    headers = {**admin, "X-SLM-User-Session": rbac.create_session(uid)}

    r = client.get("/api/export?format=json", headers=headers)

    assert r.status_code in (401, 403), r.text


def test_open_routes_are_listed_with_a_reason():
    assert all(reason.strip() for reason in OPEN_ROUTES.values())


def test_the_command_line_can_read_feature_status_with_the_daemon_capability(company) -> None:
    """`slm features` has no session; the capability (a file only this user reads) is enough for status."""
    app, client, _ = company
    d = app.state.daemon_descriptor
    capability = {"X-SLM-Daemon-Capability": d.capability, "X-SLM-Target-Instance": d.instance_id}

    assert client.get("/api/v3/features", headers=capability).status_code == 200
    assert client.get("/api/v3/features").status_code in (401, 403)
    # ...and the capability alone still opens no memory content
    assert client.get("/api/memories", headers=capability).status_code in (401, 403)
    assert client.get("/api/export", headers=capability).status_code in (401, 403)
    assert client.get("/api/v3/learning/signals", headers=capability).status_code in (401, 403)
