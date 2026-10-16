# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The mutation gate's policy, refusals that stay refusals, and personal mode untouched.

The route walk (``test_company_mode_mutation_routes``) proves every state-changing
route refuses the callers who may not change state. This file pins the rest of
4.1.25 governance findings G-C1, G-H3, G-H4 and G-L3:

* a refusal (401/403) from erase, compaction and ``/ingest`` reaches the client as
  itself, never as a 500 or a 200 that says "Internal server error";
* a user is checked against the workspace the request is about, not just any;
* restarting the daemon is administration;
* with no team accounts, or with login optional, nothing the operator did before
  is refused.
"""

# ruff: noqa: F811  (imported pytest fixtures are re-declared as test parameters)

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.server import mutation_gate as gate_mod
from tests.test_server.test_company_mode_mutation_routes import (
    MUTATING,
    _call,
    _concrete,
    _message,
    _mutating,
    _nothing_leaves_the_sandbox,  # noqa: F401  (autouse fixture)
    company,  # noqa: F401  (pytest fixture)
)
from tests.test_server.test_rbac_company_mode_credential import (
    LOOPBACK,
    app_and_engine,  # noqa: F401  (pytest fixture)
)

ROUTES_DIR = Path(gate_mod.__file__).parent / "routes"


# -- G-H4: a refusal stays a refusal -------------------------------------------

def _split_user(app, who):
    """A user who writes to ``teamb`` but only reads the active workspace ``default``."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    rbac = app.state.rbac
    rbac.create_user("split1", "password-1234", display_name="Split")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["split1"]
    rbac.set_membership("default", uid, "viewer", added_by="test")
    rbac.set_membership("teamb", uid, "member", added_by="test")
    return {"X-SLM-User-Session": rbac.create_session(uid),
            "X-Install-Token": ensure_install_token()}


def test_ingest_refusal_is_a_403_not_a_500(company):
    app, client, who = company
    split = _split_user(app, who)
    body = {"content": "standup notes", "source_type": "slack", "dedup_key": "k1"}

    r = client.post("/ingest", json=body, headers=split)

    assert r.status_code == 403, (r.status_code, r.text)
    assert "role does not allow" in r.text
    assert "Internal server error" not in r.text


def test_ingest_without_a_session_is_a_401_that_says_login(company):
    _, client, who = company
    body = {"content": "standup notes", "source_type": "slack", "dedup_key": "k2"}

    r = client.post("/ingest", json=body, headers=who["token"])

    assert r.status_code == 401, (r.status_code, r.text)
    assert "login" in _message(r).lower()


def test_erasing_an_entity_without_manage_is_a_403_not_an_internal_error(company):
    app, client, who = company
    r = client.post("/api/compliance/gdpr/erase-entity",
                    json={"entity_name": "Acme", "confirm": "Acme"}, headers=who["member1"])

    assert r.status_code == 403, (r.status_code, r.text)
    assert "Internal server error" not in r.text
    assert "manage" in _message(r)


def test_erasing_a_profile_without_manage_is_a_403_not_an_internal_error(company, monkeypatch):
    app, client, who = company
    monkeypatch.setattr(
        "superlocalmemory.server.routes.compliance.get_active_profile", lambda: "teamb",
    )
    r = client.post("/api/compliance/gdpr/erase", json={"confirm": "teamb"},
                    headers=who["member1"])

    assert r.status_code == 403, (r.status_code, r.text)
    assert "Internal server error" not in r.text


def test_an_admin_whose_erase_really_fails_still_sees_the_failure(company, monkeypatch):
    """The honest message stays for a real failure; only refusals are passed through."""
    app, client, who = company

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("superlocalmemory.compliance.gdpr.GDPRCompliance.forget_entity", _boom)
    r = client.post("/api/compliance/gdpr/erase-entity",
                    json={"entity_name": "Acme", "confirm": "Acme"}, headers=who["admin1"])

    assert r.status_code == 200
    assert r.json()["success"] is False
    assert r.json()["error"] == "Internal server error"


def test_compaction_refusal_from_the_trust_hook_is_a_403(company, monkeypatch):
    """A 403 raised inside compaction used to come back as 'internal error'."""
    app, client, who = company
    from fastapi import HTTPException

    def _deny(*_a, **_k):
        raise HTTPException(403, detail="Write authorization rejected")

    monkeypatch.setattr("superlocalmemory.server.routes.lifecycle.authorize_route_mutation", _deny)
    monkeypatch.setattr(
        "superlocalmemory.server.routes.lifecycle.LifecycleManager",
        lambda db: type("M", (), {"get_lifecycle_state": lambda s, f: "archived"})(),
    )
    monkeypatch.setattr(
        "superlocalmemory.server.routes.lifecycle.get_engine_lazy",
        lambda state: type("E", (), {"_db": type("D", (), {
            "get_all_facts": lambda s, p: [
                type("F", (), {"fact_id": "f1", "lifecycle": "active"})()],
        })()})(),
    )
    r = client.post("/api/lifecycle/compact", json={"dry_run": False}, headers=who["admin1"])

    assert r.status_code == 403, (r.status_code, r.text)


def _handlers_that_swallow_a_refusal() -> list[str]:
    """Route functions whose ``except Exception`` would turn a refusal into a 500."""
    gates = {"require_permission", "require_manage", "_require_manage", "authorize_route_mutation",
             "_authorize_memory_mutation", "require_http_mutation_actor",
             "authenticated_request_actor", "require_write_actor"}

    def callee(node):
        f = node.func
        return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")

    found = []
    for path in sorted(ROUTES_DIR.glob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Try):
                    continue
                guarded = any(isinstance(n, ast.Call) and callee(n) in gates
                              for stmt in node.body for n in ast.walk(stmt))
                if not guarded:
                    continue
                caught_http = False
                for h in node.handlers:
                    kinds = {getattr(e, "id", getattr(e, "attr", "")) for e in
                             (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])}
                    caught_http = caught_http or "HTTPException" in kinds
                    broad = h.type is None or kinds & {"Exception", "BaseException"}
                    if broad and not caught_http:
                        found.append(f"{path.name}:{fn.name}:{h.lineno}")
    return found


def test_no_route_swallows_a_refusal_in_a_blanket_except():
    assert _handlers_that_swallow_a_refusal() == []


# -- G-C1/G-H3: the workspace the request is about --------------------------------

def test_a_user_is_checked_against_the_active_workspace_for_workspace_wide_writes(company):
    """Write access to another workspace does not open the active one's mesh or compaction."""
    app, client, who = company
    split = _split_user(app, who)
    for method, path in (("POST", "/mesh/lock"), ("POST", "/observe"),
                         ("POST", "/api/lifecycle/compact"), ("POST", "/api/tiers/evaluate"),
                         ("POST", "/session/close")):
        r = _call(client, method, path, split)
        assert r.status_code == 403, (path, r.status_code, _message(r))


def test_a_routed_write_to_a_workspace_the_user_may_write_is_not_stopped_by_the_gate(company):
    """Per-request profile routing keeps working: the route picks the exact workspace."""
    app, client, who = company
    split = _split_user(app, who)

    routed = client.post("/remember", json={"content": "x", "profile_id": "teamb"}, headers=split)
    active = client.post("/remember", json={"content": "x"}, headers=split)

    # Past the gate and past the route's own check on `teamb` (a member there).
    assert routed.status_code not in (401, 403), (routed.status_code, _message(routed))
    # No profile named: the active workspace, where this user only reads.
    assert active.status_code == 403, (active.status_code, _message(active))


def test_a_query_the_route_does_not_read_cannot_move_the_check(company):
    """`?profile_id=` on a route that works on the active workspace must not borrow another one."""
    app, client, who = company
    split = _split_user(app, who)
    for path in ("/api/lifecycle/compact", "/mesh/lock", "/observe"):
        r = client.post(f"{path}?profile_id=teamb&profile=teamb", headers=split)
        assert r.status_code == 403, (path, r.status_code, _message(r))


def test_tiers_are_checked_against_the_workspace_they_act_on(company):
    """Re-tiering is the operation policy's "correct" (owner/admin), checked on
    the workspace named, with the caller's real role (4.1.25 audit OPS-26: the
    route evaluated every caller as the machine owner)."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    app, client, who = company
    split = _split_user(app, who)  # viewer of default, member of teamb
    rbac = app.state.rbac
    rbac.create_user("teamb-admin", "password-1234")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["teamb-admin"]
    rbac.set_membership("default", uid, "viewer", added_by="test")
    rbac.set_membership("teamb", uid, "admin", added_by="test")
    admin_b = {"X-SLM-User-Session": rbac.create_session(uid),
               "X-Install-Token": ensure_install_token()}

    own = client.post("/api/tiers/evaluate?profile_id=teamb", headers=admin_b)
    other = client.post("/api/tiers/evaluate?profile_id=default", headers=admin_b)
    member = client.post("/api/tiers/evaluate?profile_id=teamb", headers=split)
    assert own.status_code != 403, (own.status_code, _message(own))
    assert other.status_code == 403
    assert member.status_code == 403 and "role" in _message(member)


def test_a_mesh_node_with_the_daemon_capability_still_reaches_the_mesh(company):
    """Like the mesh read gate, a computer's own agent is a machine principal, not a person."""
    app, client, _ = company
    d = app.state.daemon_descriptor
    cap = {"X-SLM-Daemon-Capability": d.capability, "X-SLM-Target-Instance": d.instance_id}

    r = client.post("/mesh/lock", headers=cap)

    assert r.status_code == 422, (r.status_code, _message(r))  # past the gate; no body


def test_a_mesh_node_with_the_fleet_secret_reaches_the_mesh(company):
    app, client, _ = company
    app.state.mesh_broker = type("B", (), {"_shared_secret": "fleet-secret-value"})()

    ok = client.post("/mesh/lock", headers={"X-Mesh-Secret": "fleet-secret-value"})
    wrong = client.post("/mesh/lock", headers={"X-Mesh-Secret": "not-the-secret"})

    assert ok.status_code == 422, (ok.status_code, _message(ok))
    assert wrong.status_code in (401, 403), wrong.status_code


def test_the_capability_does_not_open_a_workspace_write(company):
    """Only the mesh is machine-principal; a data write still needs a person."""
    app, client, _ = company
    d = app.state.daemon_descriptor
    cap = {"X-SLM-Daemon-Capability": d.capability, "X-SLM-Target-Instance": d.instance_id}
    for path in ("/observe", "/api/lifecycle/compact", "/api/tiers/evaluate"):
        assert client.post(path, headers=cap).status_code == 401, path


def test_a_bad_session_is_a_401_not_the_owner(company):
    _, client, who = company
    r = client.post("/observe", json={"content": "x"},
                    headers={**who["token"], "X-SLM-User-Session": "forged"})
    assert r.status_code == 401, (r.status_code, _message(r))


# -- G-L3: restarting is administration -----------------------------------------

def test_restart_needs_an_administrator_in_company_mode(company, monkeypatch):
    app, client, who = company
    spawned = []

    class _Spawn:
        def __init__(self, *args, **kwargs):
            spawned.append(args)

    monkeypatch.setattr("subprocess.Popen", _Spawn)
    for name in ("token", "none", "viewer1", "member1"):
        r = client.post("/api/daemon/restart", headers=who[name])
        assert r.status_code in (401, 403), (name, r.status_code, _message(r))
    assert spawned == []

    ok = client.post("/api/daemon/restart", headers=who["admin1"])
    assert ok.status_code == 200 and ok.json()["success"] is True
    assert len(spawned) == 1


def test_the_restart_route_asks_for_manage_itself(company, monkeypatch):
    """Not only the gate: past the gate, the handler still refuses a member (defence in depth)."""
    app, client, who = company
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(gate_mod, "mutation_gate", lambda request, state: None)

    r = client.post("/api/daemon/restart", headers=who["member1"])

    assert r.status_code == 403, (r.status_code, _message(r))
    assert spawned == []


# -- personal mode and optional login are unchanged ----------------------------------

@pytest.fixture
def personal(app_and_engine, monkeypatch):
    """No team accounts at all: the single-operator install."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    from superlocalmemory.server.routes.optimize import router as optimize_router

    if not any(p.startswith("/api/optimize") for _, p in
               ((r, r.path) for r in app.routes if hasattr(r, "path"))):
        app.include_router(optimize_router)
    assert app.state.rbac.user_count() == 0
    refused: list[tuple[str, int]] = []
    real = gate_mod.mutation_gate

    def _recording(request, state):
        out = real(request, state)
        if out is not None:
            refused.append((request.url.path, out.status_code))
        return out

    monkeypatch.setattr(gate_mod, "mutation_gate", _recording)
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    return app, client, {"token": {"X-Install-Token": ensure_install_token()}, "none": {}}, refused


def test_personal_mode_the_gate_refuses_nothing_on_any_route(personal):
    app, client, who, refused = personal
    for method, path in _mutating(app):
        for headers in (who["token"], who["none"]):
            _call(client, method, path, headers)
    assert refused == []


def test_personal_mode_the_operator_still_changes_settings_and_restarts(personal, monkeypatch):
    app, client, who, refused = personal
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.append(a))

    restart = client.post("/api/daemon/restart", headers=who["token"])
    optimize = client.put("/api/optimize/config", json={"cache_enabled": True},
                          headers=who["token"])
    switch = client.post("/api/profiles/default/switch", headers=who["token"])

    # Count restart spawns only: the optimize store may also run `ioreg` once to
    # read the machine id when its cache is cold, depending on test order.
    restarts = [a for a in spawned if "restart" in (a[0] if a else [])]
    assert restart.status_code == 200 and len(restarts) == 1, spawned
    assert optimize.status_code == 200, optimize.text
    assert switch.status_code == 200, switch.text
    assert refused == []


def test_login_that_is_optional_still_lets_the_operator_work(app_and_engine, monkeypatch):
    """Users exist but 'Require login' is off: a caller with no session is the operator."""
    from superlocalmemory.core.security_primitives import ensure_install_token
    from tests.test_server.test_rbac_company_mode_credential import _company_mode

    app = app_and_engine
    _company_mode(app)
    app.state.rbac.set_require_login(False)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: None)
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    headers = {"X-Install-Token": ensure_install_token()}

    for method, path in (("POST", "/api/lifecycle/compact"), ("POST", "/api/tiers/evaluate"),
                         ("POST", "/api/daemon/restart"), ("PUT", "/api/optimize/config")):
        r = client.request(method, path, headers=headers)
        assert r.status_code not in (401, 403), (path, r.status_code, _message(r))


def test_login_that_is_optional_still_binds_a_signed_in_viewer(app_and_engine):
    """A person who did sign in is held to their role even when login is optional."""
    from superlocalmemory.core.security_primitives import ensure_install_token
    from tests.test_server.test_rbac_company_mode_credential import _company_mode

    app = app_and_engine
    _company_mode(app)
    rbac = app.state.rbac
    rbac.set_require_login(False)
    rbac.create_user("viewer1", "password-1234")
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}["viewer1"]
    rbac.set_membership("default", uid, "viewer", added_by="test")
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                        raise_server_exceptions=False)
    headers = {"X-Install-Token": ensure_install_token(),
               "X-SLM-User-Session": rbac.create_session(uid)}

    assert client.post("/api/lifecycle/compact", headers=headers).status_code == 403


# -- the policy table -----------------------------------------------------------------

def test_reads_sent_as_post_need_only_read():
    for path in ("/api/search", "/api/v3/chat/stream", "/api/v3/recall/trace", "/session/open"):
        assert gate_mod.rule_for("POST", path).permission.value == "read", path


def test_get_and_head_are_not_this_gates_business():
    for method in ("GET", "HEAD", "OPTIONS"):
        assert gate_mod.rule_for(method, "/api/optimize/config") is None


def test_every_open_entry_names_why():
    assert all(reason.strip() for reason in gate_mod.OPEN_EXACT.values())


def test_a_route_nobody_listed_needs_write(company):
    """Deny by default: an invented mutating path is refused to a viewer."""
    _, client, who = company
    assert gate_mod.rule_for("POST", "/api/brand-new/thing").permission.value == "write"
    assert client.post("/api/brand-new/thing", headers=who["viewer1"]).status_code == 403
    assert client.post("/api/brand-new/thing", headers=who["token"]).status_code == 401


def test_every_walked_route_has_a_rule_or_a_stated_reason(company):
    app, _, _ = company
    for method, path in _mutating(app):
        concrete = _concrete(path)
        assert method in MUTATING
        if gate_mod.rule_for(method, concrete) is None:
            assert concrete in gate_mod.OPEN_EXACT or concrete.startswith(gate_mod.OPEN_PREFIXES), (
                method, path)
