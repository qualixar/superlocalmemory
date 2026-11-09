# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Role x surface matrix, REST and dashboard reads (4.1.25 WP-9).

Who: anonymous (no credential), the machine owner without a session (install
token), a viewer, a member and an admin of workspace ``default`` (A), an admin
of workspace ``sales`` only (B), and an account that is a viewer of A but an
admin of B. The other half (HTTP MCP, stdio MCP) is in
``test_role_surface_matrix_mcp.py``.

The walks below go over every GET route the daemon registers, so a new route
that forgets to check the workspace it reads fails here. Each finding the
matrix closes is named next to its test:

* OPS-25: a route that honours ``?profile=`` / ``?profile_id=`` was checked
  against the active workspace only.
* OPS-09: the workspace list named every workspace and counted its memories.
* OPS-24: a signed-out (or expired, revoked) viewer session became the owner.
* G-M6: Optimize settings answered a logged-out caller.
* G-M5: a viewer could not list correction cases and was told it tried to write.
"""

# ruff: noqa: F811  (imported pytest fixtures are re-declared as test parameters)

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.server.read_gates import USER_READ_OPEN, USER_READ_SELF_GATED_PREFIXES
from tests.test_server.test_company_mode_get_routes import STREAMING_ROUTES, _concrete, _get_paths
from tests.test_server.test_rbac_company_mode_credential import (
    LOOPBACK,
    app_and_engine,  # noqa: F401  (pytest fixture)
)

#: Reads every role that belongs to a workspace may make there (positive control).
CORE_READS = ("/api/memories", "/api/v3/timeline", "/api/tiers/stats", "/api/profiles")


def _users(rbac):
    made = {}
    for name in ("viewer_a", "member_a", "admin_a", "admin_b", "dual"):
        made[name] = rbac.create_user(name, "password-1234", display_name=name)["user_id"]
    rbac.set_membership("default", made["viewer_a"], "viewer")
    rbac.set_membership("default", made["member_a"], "member")
    rbac.set_membership("default", made["admin_a"], "admin")
    rbac.set_membership("sales", made["admin_b"], "admin")
    rbac.set_membership("default", made["dual"], "viewer")
    rbac.set_membership("sales", made["dual"], "admin")
    return made


@pytest.fixture
def matrix(app_and_engine):
    """Two workspaces with content, five accounts, sessions for each."""
    from superlocalmemory.core.security_primitives import ensure_install_token
    from superlocalmemory.server.routes.optimize import router as optimize_router

    app = app_and_engine
    if not any(p.startswith("/api/optimize") for p in _get_paths(app)):
        app.include_router(optimize_router)
    db = app.state.engine._db
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('sales','sales')")
    rbac = app.state.rbac
    ids = _users(rbac)
    token = ensure_install_token()
    who = {name: {"X-SLM-User-Session": rbac.create_session(uid), "X-Install-Token": token}
           for name, uid in ids.items()}
    who["owner"] = {"X-Install-Token": token}
    who["anonymous"] = {}
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    return app, client, who, ids


def _company(app):
    app.state.rbac.set_require_login(True)


def _user_open(path: str) -> bool:
    return path in USER_READ_OPEN or path.startswith(USER_READ_SELF_GATED_PREFIXES)


# -- OPS-25: naming another workspace -----------------------------------------------


@pytest.mark.parametrize("login_required", [True, False], ids=["company", "login-optional"])
@pytest.mark.parametrize("who_name", ["viewer_a", "member_a", "admin_a"])
def test_no_get_route_serves_a_workspace_the_user_cannot_read(matrix, login_required, who_name):
    app, client, who, _ = matrix
    if login_required:
        _company(app)
    answered = []
    for path in _get_paths(app):
        if _user_open(path) or path in STREAMING_ROUTES:
            continue
        for key in ("profile", "profile_id"):
            r = client.get(_concrete(path), params={key: "sales"}, headers=who[who_name])
            if r.status_code not in (401, 403, 404, 405, 422):
                answered.append((path, key, r.status_code))
    assert not answered, f"{who_name} read another workspace: {answered}"


def test_a_default_viewer_cannot_read_sales_through_timeline_insights_or_tiers(matrix):
    """The audit's exact reproduction (OPS-25)."""
    app, client, who, _ = matrix
    _company(app)
    for path, params in (
        ("/api/v3/timeline", {"profile": "sales"}),
        ("/api/v3/insights/changed_this_week", {"profile": "sales"}),
        ("/api/v3/insights/opinions", {"profile": "sales"}),
        ("/api/tiers/stats", {"profile_id": "sales"}),
    ):
        r = client.get(path, params=params, headers=who["viewer_a"])
        assert r.status_code == 403, (path, r.status_code, r.text[:200])


def test_the_owner_of_sales_still_reads_sales(matrix):
    app, client, who, _ = matrix
    _company(app)
    r = client.get("/api/tiers/stats", params={"profile_id": "sales"}, headers=who["admin_b"])
    assert r.status_code == 200, r.text[:300]


@pytest.mark.parametrize("who_name", ["viewer_a", "member_a", "admin_a"])
def test_members_of_a_workspace_still_read_it(matrix, who_name):
    app, client, who, _ = matrix
    _company(app)
    for path in CORE_READS:
        r = client.get(path, headers=who[who_name])
        assert r.status_code == 200, (who_name, path, r.status_code, r.text[:200])


def test_an_admin_of_sales_only_cannot_read_default(matrix):
    app, client, who, _ = matrix
    _company(app)
    for path in ("/api/memories", "/api/v3/timeline", "/api/tiers/stats"):
        for key in ("profile", "profile_id"):
            r = client.get(path, params={key: "default"}, headers=who["admin_b"])
            assert r.status_code == 403, (path, key, r.status_code)
        # Naming nothing reads their own workspace (finding 2), never default.
        assert client.get(path, headers=who["admin_b"]).status_code == 200, path


# -- OPS-09: the workspace list ------------------------------------------------------


def test_the_workspace_list_names_only_the_callers_workspaces(matrix):
    app, client, who, _ = matrix
    _company(app)
    def names(caller):
        listed = client.get("/api/profiles", headers=who[caller]).json()["profiles"]
        return {p["name"] for p in listed}

    seen = {name: names(name) for name in ("viewer_a", "admin_b", "dual")}
    assert seen == {"viewer_a": {"default"}, "admin_b": {"sales"}, "dual": {"default", "sales"}}
    body = client.get("/api/profiles", headers=who["admin_b"]).json()
    # Their own active workspace (bound to their session), never the owner's.
    assert body["active_profile"] == "sales"


def test_the_owner_still_sees_every_workspace_in_personal_mode(matrix):
    app, client, who, _ = matrix
    listed = client.get("/api/profiles", headers=who["owner"]).json()["profiles"]
    names = {p["name"] for p in listed}
    assert {"default", "sales"} <= names


# -- OPS-24: an ended session is never the owner -------------------------------------


def test_a_signed_out_viewer_session_is_refused_not_promoted(matrix):
    """Login optional, accounts enrolled: the audit's reproduction, end to end."""
    app, client, who, _ = matrix
    viewer = who["viewer_a"]
    body = {"content": "a note the viewer must not write"}
    assert client.post("/remember", json=body, headers=viewer).status_code == 403
    assert client.post("/api/rbac/logout", headers=viewer).status_code == 200

    after = client.post("/remember", json=body, headers=viewer)
    assert after.status_code == 401, after.text[:200]
    assert client.get("/api/memories", headers=viewer).status_code == 401
    assert client.post("/api/tiers/evaluate", headers=viewer).status_code == 401
    whoami = client.get("/api/rbac/whoami", headers=viewer).json()
    assert whoami["session_invalid"] is True


def test_an_expired_or_forged_session_is_refused_in_personal_mode_too(matrix):
    app, client, who, _ = matrix
    forged = {**who["owner"], "X-SLM-User-Session": "not-a-session"}
    assert client.get("/api/memories", headers=forged).status_code == 401
    assert client.post("/remember", json={"content": "x"}, headers=forged).status_code == 401


def test_without_a_session_the_owner_is_unchanged_when_login_is_optional(matrix):
    app, client, who, _ = matrix
    assert client.get("/api/memories", headers=who["owner"]).status_code == 200
    assert client.get("/api/memories").status_code == 200


# -- G-M6 / G-M5 ---------------------------------------------------------------------


def test_optimize_settings_need_a_session_in_company_mode(matrix):
    app, client, who, _ = matrix
    _company(app)
    for path in ("/api/optimize/config", "/api/optimize/stats", "/api/optimize/savings"):
        assert client.get(path, headers=who["owner"]).status_code == 401, path
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers=who["admin_a"]).status_code == 200, path


def test_a_viewer_lists_corrections_but_cannot_act_on_them(matrix):
    app, client, who, _ = matrix
    _company(app)
    listed = client.get("/api/corrections", headers=who["viewer_a"])
    assert listed.status_code == 200, listed.text[:200]
    assert client.get("/api/overtaken-corrections", headers=who["viewer_a"]).status_code == 200
    foreign = client.get("/api/corrections", params={"profile_id": "sales"},
                         headers=who["viewer_a"])
    assert foreign.status_code == 403
    acted = client.post("/api/overtaken-corrections/c1/restore", json={},
                        headers=who["viewer_a"])
    assert acted.status_code == 403
    assert "write" in acted.json().get("error", acted.json().get("detail", ""))


def _pending_machine_case(engine, case_id: str, profile: str = "default") -> None:
    from tests.test_server.test_dismiss_corrections_route import _case

    _case(engine, case_id, profile=profile)


def test_a_viewer_cannot_dry_run_or_apply_a_bulk_dismissal(matrix):
    """Every POST passes the mutation gate (server/mutation_gate.py), which asks for
    ``write`` unless the route is listed there as a read; the dismissal is not, so a
    viewer is refused the dry run too, exactly like the restore route. Refused means
    nothing is moved and no backup is taken."""
    from pathlib import Path

    app, client, who, _ = matrix
    _company(app)
    engine = app.state.engine
    _pending_machine_case(engine, "m1")
    folder = Path(engine._db.db_path).parent
    url = "/api/overtaken-corrections/dismiss"

    for body in ({"profile_id": "default"}, {"profile_id": "default", "apply": True},
                 {"profile_id": "sales"}, {"profile_id": "sales", "apply": True}):
        refused = client.post(url, json=body, headers=who["viewer_a"])
        assert refused.status_code == 403, (body, refused.text[:200])
        assert "write" in refused.json().get("error", refused.json().get("detail", ""))
    assert list(folder.glob("*.before-*")) == []
    left = engine._db.execute("SELECT COUNT(*) AS n FROM correction_cases")[0]
    assert dict(left)["n"] == 1
    try:
        moved = engine._db.execute("SELECT COUNT(*) AS n FROM correction_cases_overtaken")[0]
        assert dict(moved)["n"] == 0
    except Exception as exc:   # the table is only created by a first overtake
        assert "no such table" in str(exc).lower()


def test_an_admin_applies_a_bulk_dismissal_in_their_own_workspace_only(matrix):
    """The role that may act on corrections (same ``update`` policy as restore) can
    dismiss in its own workspace; another workspace stays 403."""
    app, client, who, _ = matrix
    _company(app)
    engine = app.state.engine
    _pending_machine_case(engine, "m1")
    url = "/api/overtaken-corrections/dismiss"

    other = client.post(url, json={"profile_id": "sales", "apply": True}, headers=who["admin_a"])
    assert other.status_code == 403
    own = client.post(url, json={"profile_id": "default", "apply": True}, headers=who["admin_a"])
    assert own.status_code == 200, own.text[:300]
    assert own.json()["dismissed"] == 1 and own.json()["backup"]
    assert dict(engine._db.execute("SELECT COUNT(*) AS n FROM correction_cases")[0])["n"] == 0


def test_a_member_is_refused_a_bulk_dismissal_as_it_is_refused_a_restore(matrix):
    app, client, who, _ = matrix
    _company(app)
    engine = app.state.engine
    _pending_machine_case(engine, "m1")
    restore = client.post("/api/overtaken-corrections/c1/restore", json={}, headers=who["member_a"])
    dismiss = client.post("/api/overtaken-corrections/dismiss",
                          json={"profile_id": "default", "apply": True}, headers=who["member_a"])
    assert restore.status_code == dismiss.status_code
    if dismiss.status_code == 403:
        assert dict(engine._db.execute("SELECT COUNT(*) AS n FROM correction_cases")[0])["n"] == 1


# -- PR 248 review, finding 1: installation-wide settings ----------------------------

#: Every route that changes state shared by all workspaces (one value per install).
INSTALLATION_WIDE = (
    ("POST", "/api/rbac/policy"),
    ("POST", "/api/daemon/restart"),
    ("POST", "/api/v3/components/heal"),
    ("POST", "/api/v3/ide/connect"),
    ("POST", "/api/v3/ollama/validate"),
    ("PUT", "/api/v3/mode"),
    ("POST", "/api/v3/mode/set"),
    ("PUT", "/api/v3/provider"),
    ("POST", "/api/v3/provider/test"),
    ("PUT", "/api/v3/embedding/config"),
    ("POST", "/api/v3/embedding/test"),
    ("POST", "/api/v3/embedding/reindex"),
    ("POST", "/api/v3/embedding/reindex/upgrade"),
    ("PUT", "/api/v3/scope/config"),
    ("PUT", "/api/v3/runtime/config"),
    ("PUT", "/api/v3/auto-capture/config"),
    ("PUT", "/api/v3/auto-recall/config"),
    ("PUT", "/api/v3/auto-invoke/config"),
    ("PUT", "/api/v3/daemon/config"),
    ("PUT", "/api/v3/forgetting/config"),
    ("PUT", "/api/v3/graph/config"),
    ("PUT", "/api/v3/mesh/config"),
    ("PUT", "/api/v3/storage/config"),
    ("PUT", "/api/v3/trust/config"),
    ("PUT", "/api/v3/ratelimit"),
    ("POST", "/api/v3/features/media/enable"),
    ("POST", "/api/v3/features/media/disable"),
    ("PUT", "/api/optimize/config"),
    ("DELETE", "/api/optimize/cache/clear"),
    ("POST", "/api/v3/answer-check/jev/consent"),
    ("POST", "/api/v3/answer-check/jev/key"),
    ("DELETE", "/api/v3/answer-check/jev/key"),
    ("POST", "/api/v3/answer-check/jev/rerank"),
    ("POST", "/api/v3/answer-check/jev/test"),
    ("POST", "/api/v3/answer-check/laya/setup"),
    ("POST", "/api/v3/answer-check/laya/adopt"),
    ("POST", "/api/v3/answer-check/laya/remove"),
    ("POST", "/api/v3/answer-check/mode"),
    ("POST", "/api/v3/answer-check/save"),
    ("POST", "/api/backup/configure"),
    ("POST", "/api/backup/create"),
    ("POST", "/api/learning/backup"),
    ("POST", "/api/learning/migrate-legacy"),
    ("POST", "/api/backup/recovery-key"),
    ("POST", "/api/upgrade/restore"),
    ("POST", "/api/adapters/enable"),
    ("POST", "/api/adapters/start"),
    ("POST", "/api/evolution/enable"),
    ("POST", "/api/evolution/config"),
)


@pytest.fixture
def no_side_effects(monkeypatch):
    """A refusal must come before any handler runs; nothing may spawn regardless."""
    import subprocess

    def _refuse(*_a, **_k):
        raise AssertionError("a refused request reached a handler that spawns a process")

    monkeypatch.setattr(subprocess, "Popen", _refuse)


def _global_admin(app, who, ids):
    """An account that administers every workspace on the installation."""
    rbac = app.state.rbac
    uid = rbac.create_user("root_admin", "password-1234")["user_id"]
    for space in ("default", "sales"):
        rbac.set_membership(space, uid, "admin")
    ids["root_admin"] = uid
    who["root_admin"] = {**who["owner"], "X-SLM-User-Session": rbac.create_session(uid)}


def _gate_allows(app, method, path, headers) -> bool:
    from starlette.requests import Request

    from superlocalmemory.server.mutation_gate import mutation_gate

    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "app": app, "client": LOOPBACK, "server": ("127.0.0.1", 8765)}
    return mutation_gate(Request(scope), app.state) is None


@pytest.mark.parametrize("scoped", ["admin_a", "admin_b"])
def test_a_workspace_admin_cannot_change_installation_wide_state(matrix, scoped, no_side_effects):
    app, client, who, _ = matrix
    _company(app)
    home = "default" if scoped == "admin_a" else "sales"
    client.post(f"/api/profiles/{home}/switch", headers=who[scoped])
    answered = []
    for method, path in INSTALLATION_WIDE:
        r = client.request(method, path, json={}, headers=who[scoped])
        if r.status_code != 403:
            answered.append((method, path, r.status_code))
    assert not answered, f"{scoped} changed installation-wide state: {answered}"
    assert app.state.rbac.require_login() is True


def test_the_reviewers_reproduction_company_mode_stays_on(matrix):
    """P1 + P1b: a scoped admin turning login off, then reading with no session."""
    app, client, who, _ = matrix
    _company(app)
    client.post("/api/profiles/sales/switch", headers=who["admin_b"])
    pol = client.post("/api/rbac/policy", json={"require_login": False}, headers=who["admin_b"])
    assert pol.status_code == 403, pol.text[:200]
    pol = client.post("/api/rbac/policy", json={"require_login": False}, headers=who["admin_a"])
    assert pol.status_code == 403, pol.text[:200]
    assert app.state.rbac.require_login() is True
    assert client.get("/api/memories").status_code == 401
    for space in ("default", "sales"):
        anon = client.get("/api/tiers/stats", params={"profile_id": space})
        assert anon.status_code == 401, (space, anon.status_code)


@pytest.mark.parametrize("login_required", [True, False], ids=["company", "login-optional"])
def test_an_admin_of_every_workspace_passes_the_gate(matrix, login_required):
    app, _, who, ids = matrix
    _global_admin(app, who, ids)
    if login_required:
        _company(app)
    refused = [(m, p) for m, p in INSTALLATION_WIDE
               if not _gate_allows(app, m, p, who["root_admin"])]
    assert not refused, refused


def test_the_owner_passes_the_gate_in_personal_mode_and_with_the_capability(matrix):
    from tests.test_server.test_rbac_company_mode_credential import _daemon_headers

    app, _, who, _ = matrix
    assert all(_gate_allows(app, m, p, who["owner"]) for m, p in INSTALLATION_WIDE)
    _company(app)
    assert not any(_gate_allows(app, m, p, who["owner"]) for m, p in INSTALLATION_WIDE)
    cap = _daemon_headers(app)
    refused = [(m, p) for m, p in INSTALLATION_WIDE if not _gate_allows(app, m, p, cap)]
    assert not refused, refused


def test_an_admin_of_every_workspace_changes_the_login_policy(matrix):
    app, client, who, ids = matrix
    _global_admin(app, who, ids)
    _company(app)
    r = client.post("/api/rbac/policy", json={"require_login": False}, headers=who["root_admin"])
    assert r.status_code == 200, r.text[:200]
    assert app.state.rbac.require_login() is False
