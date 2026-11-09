# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""PR 248 re-check, finding N10 (routing half): the daemon acts on the caller's workspace.

``/consolidate/cognitive`` and ``/maintenance/run`` checked and ran against
``engine.profile_id``, the shared workspace, whatever workspace the caller was
bound to. The same was true of the other POST handlers that take a caller's
binding: ``/maintenance/compact`` (its audit workspace), ``/session/open``,
``/session/close`` and ``/observe``. Each now resolves the workspace once
(``server/acting_profile.py``), checks the role on THAT workspace, runs against
it and, where it answers with one, returns the workspace it acted on.
"""

# ruff: noqa: F811  (imported pytest fixtures are re-declared as test parameters)

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.server.caller_workspace import session_workspaces
from tests.test_governance.test_role_surface_matrix_rest import matrix  # noqa: F401
from tests.test_server.test_rbac_company_mode_credential import (  # noqa: F401
    _daemon_headers,
    app_and_engine,
)

BINDING = "b" * 32


@pytest.fixture()
def seen(monkeypatch):
    """Every workspace the maintenance work was asked to act on."""
    from superlocalmemory.core import maintenance, maintenance_scheduler
    from superlocalmemory.encoding import cognitive_consolidator as cc
    from superlocalmemory.learning import consolidation_worker, forgetting_scheduler

    calls: list[tuple[str, str]] = []
    result = SimpleNamespace(clusters_processed=1, blocks_created=2)

    def _record(name, value):
        def inner(*args, **kwargs):
            pid = next((a for a in args if isinstance(a, str)), kwargs.get("pid"))
            calls.append((name, pid))
            return value
        return inner

    monkeypatch.setattr(cc.CognitiveConsolidator, "run_pipeline",
                        lambda self, pid: calls.append(("consolidate", pid)) or result)
    monkeypatch.setattr(maintenance, "run_maintenance",
                        lambda db, config, pid: calls.append(("langevin", pid)) or {"updated": 0})
    monkeypatch.setattr(forgetting_scheduler.ForgettingScheduler, "run_decay_cycle",
                        lambda self, pid, force=False: calls.append(("forgetting", pid)) or {})
    monkeypatch.setattr(consolidation_worker.ConsolidationWorker, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(consolidation_worker.ConsolidationWorker, "_generate_patterns",
                        lambda self, pid, force: calls.append(("behavioral", pid)) or 0)
    monkeypatch.setattr(maintenance_scheduler, "compact_vector_store", lambda: {"ok": True})
    return calls


def _bind_owner(app, workspace="sales"):
    session_workspaces(app.state).bind("owner:" + BINDING, workspace)
    return {**_daemon_headers(app), "X-SLM-Workspace-Binding": BINDING}


def _user(app, name, memberships):
    rbac = app.state.rbac
    uid = rbac.create_user(name, "password-1234", display_name=name)["user_id"]
    for workspace, role in memberships.items():
        rbac.set_membership(workspace, uid, role)
    return {"X-SLM-User-Session": rbac.create_session(uid)}


def _enter_sales(client, headers):
    assert client.post("/api/profiles/sales/switch", headers=headers).status_code == 200


# -- /consolidate/cognitive ---------------------------------------------------


def test_consolidate_runs_against_the_pinned_workspace(matrix, seen):
    app, client, _, _ = matrix
    r = client.post("/consolidate/cognitive", json={}, headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert seen == [("consolidate", "sales")]
    assert r.json()["profile_id"] == "sales"


def test_consolidate_ignores_the_callers_guess_of_the_shared_workspace_when_unbound(matrix, seen):
    app, client, who, _ = matrix
    r = client.post("/consolidate/cognitive", json={}, headers=who["owner"])
    assert r.status_code == 200, r.text
    assert seen == [("consolidate", "default")]
    assert r.json()["profile_id"] == "default"


def test_consolidate_checks_the_role_on_the_bound_workspace_not_the_shared_one(matrix, seen):
    app, client, who, _ = matrix
    app.state.rbac.set_require_login(True)
    # admin of sales, viewer of default: refused on the shared workspace before.
    _enter_sales(client, who["dual"])
    r = client.post("/consolidate/cognitive", json={}, headers=who["dual"])
    assert r.status_code == 200, r.text
    assert seen == [("consolidate", "sales")]


def test_consolidate_refuses_a_bound_role_that_cannot_write(matrix, seen):
    app, client, _, _ = matrix
    app.state.rbac.set_require_login(True)
    reader = _user(app, "sales_reader", {"default": "member", "sales": "viewer"})
    _enter_sales(client, reader)
    r = client.post("/consolidate/cognitive", json={}, headers=reader)
    assert r.status_code == 403, r.text
    assert seen == []


def test_consolidate_naming_a_workspace_is_checked_on_that_workspace(matrix, seen):
    app, client, who, _ = matrix
    app.state.rbac.set_require_login(True)
    r = client.post("/consolidate/cognitive", json={"profile_id": "sales"}, headers=who["viewer_a"])
    assert r.status_code == 403, r.text
    assert seen == []


def test_consolidate_naming_an_unknown_workspace_is_a_404(matrix, seen):
    _, client, who, _ = matrix
    r = client.post("/consolidate/cognitive", json={"profile_id": "nowhere"}, headers=who["owner"])
    assert r.status_code == 404, r.text
    assert seen == []


# -- /maintenance/run ----------------------------------------------------------


def test_maintenance_runs_every_step_against_the_pinned_workspace(matrix, seen):
    app, client, _, _ = matrix
    r = client.post("/maintenance/run", json={}, headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert {pid for _, pid in seen} == {"sales"}
    assert {name for name, _ in seen} == {"langevin", "forgetting", "behavioral"}
    assert r.json()["profile"] == "sales"


def test_maintenance_by_a_signed_in_administrator_acts_on_their_workspace(matrix, seen):
    app, client, _, _ = matrix
    app.state.rbac.set_require_login(True)
    root = _user(app, "root", {"default": "admin", "sales": "admin"})
    _enter_sales(client, root)
    r = client.post("/maintenance/run", json={}, headers=root)
    assert r.status_code == 200, r.text
    assert {pid for _, pid in seen} == {"sales"}
    assert r.json()["profile"] == "sales"


def test_maintenance_refuses_a_bound_role_without_manage(matrix, seen):
    app, client, _, _ = matrix
    app.state.rbac.set_require_login(True)
    member = _user(app, "sales_member", {"default": "admin", "sales": "member"})
    _enter_sales(client, member)
    r = client.post("/maintenance/run", json={}, headers=member)
    assert r.status_code == 403, r.text
    assert seen == []


def test_maintenance_naming_a_workspace_the_caller_cannot_manage_is_refused(matrix, seen):
    app, client, who, _ = matrix
    app.state.rbac.set_require_login(True)
    r = client.post("/maintenance/run", json={"profile_id": "sales"}, headers=who["admin_a"])
    assert r.status_code == 403, r.text
    assert seen == []


# -- the other POST handlers that take a caller's binding ----------------------


def test_compact_records_the_pinned_workspace(matrix, seen, monkeypatch):
    app, client, _, _ = matrix
    from superlocalmemory.server import route_mutations

    profiles = []
    real = route_mutations.authorize_route_mutation
    monkeypatch.setattr(route_mutations, "authorize_route_mutation",
                        lambda request, **kw: profiles.append(kw.get("profile_id")) or real(request, **kw))
    r = client.post("/maintenance/compact", headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert profiles == ["sales"]


def test_session_close_closes_the_pinned_workspaces_session(matrix, monkeypatch):
    app, client, _, _ = matrix
    closed = []
    monkeypatch.setattr(type(app.state.engine), "close_session",
                        lambda self, sid, profile_id=None: closed.append((sid, profile_id)) or 1)
    r = client.post("/session/close", json={"session_id": "s-1"}, headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert closed == [("s-1", "sales")]


def test_session_close_refuses_a_bound_role_that_cannot_write(matrix, monkeypatch):
    app, client, _, _ = matrix
    app.state.rbac.set_require_login(True)
    closed = []
    monkeypatch.setattr(type(app.state.engine), "close_session",
                        lambda self, sid, profile_id=None: closed.append(sid) or 1)
    reader = _user(app, "sales_reader", {"default": "member", "sales": "viewer"})
    _enter_sales(client, reader)
    r = client.post("/session/close", json={"session_id": "s-1"}, headers=reader)
    assert r.status_code == 403, r.text
    assert closed == []


def test_session_open_warms_the_pinned_workspace(matrix, monkeypatch):
    app, client, _, _ = matrix
    warmed = []
    monkeypatch.setattr(type(app.state.engine), "recall",
                        lambda self, query, profile_id=None, **kw: warmed.append(profile_id)
                        or SimpleNamespace(results=[]))
    r = client.post("/session/open", json={"query": "q"}, headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert warmed == ["sales"]


def test_observe_files_the_observation_under_the_pinned_workspace(matrix, monkeypatch):
    app, client, _, _ = matrix
    from superlocalmemory.server import unified_daemon

    filed = []
    monkeypatch.setattr(unified_daemon._observe_buffer, "enqueue",
                        lambda content, **kw: filed.append(kw.get("profile_id")) or {"captured": True})
    r = client.post("/observe", json={"content": "x"}, headers=_bind_owner(app))
    assert r.status_code == 200, r.text
    assert filed == ["sales"]


def test_observe_refuses_a_bound_role_that_cannot_write(matrix, monkeypatch):
    app, client, _, _ = matrix
    app.state.rbac.set_require_login(True)
    from superlocalmemory.server import unified_daemon

    filed = []
    monkeypatch.setattr(unified_daemon._observe_buffer, "enqueue",
                        lambda content, **kw: filed.append(content) or {"captured": True})
    reader = _user(app, "sales_reader", {"default": "member", "sales": "viewer"})
    _enter_sales(client, reader)
    r = client.post("/observe", json={"content": "x"}, headers=reader)
    assert r.status_code in (401, 403), r.text
    assert filed == []


def test_observe_buffer_ingests_into_the_routed_workspace(monkeypatch):
    from superlocalmemory.core import engine_ingestion
    from superlocalmemory.server.unified_daemon import ObserveBuffer

    submitted = []
    command = SimpleNamespace(submit=lambda request: submitted.append(request.profile_id)
                              or SimpleNamespace(operation_id="op", fact_ids=(),
                                                 state=SimpleNamespace(value="queued")))
    monkeypatch.setattr(engine_ingestion, "build_engine_ingestion_command", lambda engine: command)
    buffer = ObserveBuffer(debounce_sec=0.01)
    buffer.set_engine(SimpleNamespace(_profile_id="default", _config=None))
    text = "We decided to use PostgreSQL for billing because it supports row level security."
    assert buffer.enqueue(text, profile_id="sales")["captured"] is True
    # the same words in the shared workspace are not a duplicate of the routed ones
    assert buffer.enqueue(text)["captured"] is True
    assert submitted == ["sales", "default"]
    buffer._timer.cancel()


# -- the MCP side: report what the daemon did, never the caller's guess ---------


def _v33_tools(profile="default"):
    from unittest.mock import MagicMock

    from superlocalmemory.mcp.tools_v33 import register_v33_tools

    class Server:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    server = Server()
    engine = MagicMock()
    engine.profile_id = profile
    register_v33_tools(server, MagicMock(return_value=engine))
    return server.tools


@pytest.mark.parametrize("tool, path, reply", [
    ("consolidate_cognitive", "/consolidate/cognitive",
     {"ok": True, "profile_id": "sales", "clusters_processed": 1, "blocks_created": 2}),
    ("run_maintenance", "/maintenance/run", {"ok": True, "profile": "sales"}),
])
def test_mcp_tools_send_no_workspace_and_report_the_one_the_daemon_acted_on(
        monkeypatch, tool, path, reply):
    from superlocalmemory.mcp import tools_v33

    posted = []
    monkeypatch.setattr(tools_v33, "_try_daemon_post",
                        lambda p, body, timeout_s=60.0: posted.append((p, body)) or reply)
    result = asyncio.run(_v33_tools("default")[tool]())
    assert posted == [(path, {})]
    assert result["success"] is True
    assert result.get("profile_id", result.get("profile")) == "sales"


def test_proxy_recall_reports_a_refusal_as_not_retryable(monkeypatch):
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    def refuse(*a, **k):
        raise daemon.DaemonRefused(403, "/recall")

    monkeypatch.setattr(daemon, "daemon_request", refuse)
    answer = DaemonPoolProxy(port=1).recall("anything")
    assert answer["ok"] is False
    assert answer["code"] == "NOT_AUTHORIZED"
    assert answer["retryable"] is False


# -- PR 248 re-check 2: an empty body must reach the daemon, and a daemon that
# answers with an error is never replaced by a local run on the shared workspace.


@pytest.mark.parametrize("path", ["/consolidate/cognitive", "/maintenance/run"])
def test_a_post_without_a_body_is_accepted(matrix, seen, path):
    app, client, _, _ = matrix
    r = client.post(path, headers=_bind_owner(app))
    assert r.status_code == 200, r.text


def test_daemon_request_sends_an_empty_json_body(monkeypatch):
    from superlocalmemory.cli import daemon
    from superlocalmemory.core import outbound_http

    sent = {}

    class _Resp:
        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        sent["data"] = req.data
        sent["type"] = req.headers.get("Content-type")
        return _Resp()

    monkeypatch.setattr(daemon, "_verified_legacy_health",
                        lambda: {"_legacy_port": 1, "status": "ok"})
    monkeypatch.setattr(daemon, "descriptor_path",
                        lambda: SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(daemon, "read_descriptor", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(outbound_http, "urlopen", fake_urlopen)
    daemon.daemon_request("POST", "/maintenance/run", {})
    assert sent["data"] == b"{}"
    assert sent["type"] == "application/json"


@pytest.mark.parametrize("tool", ["consolidate_cognitive", "run_maintenance"])
def test_a_running_daemon_that_does_not_complete_is_not_replaced_by_a_local_run(
        monkeypatch, tool):
    from superlocalmemory.cli import daemon

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: None)
    tools = _v33_tools("default")
    result = asyncio.run(tools[tool]())
    assert result["success"] is False
    assert result.get("via") != "local"
    assert "default" not in str(result.get("profile_id", result.get("profile", "")))
