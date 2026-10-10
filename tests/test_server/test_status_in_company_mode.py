# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""/status in company mode: a caller with no session learns only what discovery needs.

The command line finds the daemon through /status. It used to answer anyone on
this computer with the data-folder and database paths and the fact, entity and
edge counts, even where every person must sign in. Without a session (and
without the daemon's private key file) the answer is now the status, version,
port, process id and instance id. The command line and MCP send the key file's
capability, a signed-in dashboard sends its session, so both keep the full answer.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_server.test_rbac_company_mode_credential import (
    LOOPBACK,
    _company_mode,
    _daemon_headers,
    app_and_engine,  # noqa: F401  (pytest fixture)
)

DISCOVERY_KEYS = {"status", "version", "port", "pid", "instance_id", "details_hidden"}
HIDDEN_KEYS = (
    "base_dir", "db_path", "db_size_mb", "fact_count", "entity_count", "edge_count",
    "profile", "mode", "provider", "projection_queue_depth",
)


def _client(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                      raise_server_exceptions=False)


def _session(app, name="admin1") -> dict[str, str]:
    rbac = app.state.rbac
    uid = {u["username"]: u["user_id"] for u in rbac.list_users()}[name]
    return {"X-SLM-User-Session": rbac.create_session(uid)}


def test_personal_mode_is_unchanged(app_and_engine):
    body = _client(app_and_engine).get("/status").json()

    for key in ("base_dir", "db_path", "fact_count", "entity_count", "edge_count", "profile"):
        assert key in body, key


def test_company_mode_without_a_session_answers_discovery_only(app_and_engine):
    app = app_and_engine
    _company_mode(app)

    r = _client(app).get("/status")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == DISCOVERY_KEYS
    assert body["status"] == "running" and body["details_hidden"] is True
    assert body["pid"] and body["port"] and body["instance_id"] and body["version"]
    text = json.dumps(body)
    for key in HIDDEN_KEYS:
        assert key not in body, key
    assert str(app.state.config.base_dir) not in text


def test_the_install_token_alone_does_not_open_the_details(app_and_engine):
    from superlocalmemory.core.security_primitives import ensure_install_token

    app = app_and_engine
    _company_mode(app)

    body = _client(app).get("/status", headers={"X-Install-Token": ensure_install_token()}).json()

    assert set(body) == DISCOVERY_KEYS


def test_the_daemon_capability_keeps_the_full_answer_for_the_command_line(app_and_engine):
    app = app_and_engine
    creds = _company_mode(app)

    body = _client(app).get("/status", headers=creds).json()

    assert "details_hidden" not in body
    for key in ("base_dir", "db_path", "fact_count", "entity_count", "edge_count", "profile"):
        assert key in body, key


def test_a_signed_in_user_with_read_keeps_the_full_answer(app_and_engine):
    app = app_and_engine
    _company_mode(app)

    body = _client(app).get("/status", headers=_session(app)).json()

    assert "details_hidden" not in body and "db_path" in body


def test_a_signed_in_user_without_read_gets_discovery_only(app_and_engine):
    app = app_and_engine
    _company_mode(app)
    app.state.rbac.create_user("nobody1", "password-1234")

    body = _client(app).get("/status", headers=_session(app, "nobody1")).json()

    assert set(body) == DISCOVERY_KEYS


def test_a_forged_session_gets_discovery_only(app_and_engine):
    app = app_and_engine
    _company_mode(app)

    body = _client(app).get("/status", headers={"X-SLM-User-Session": "forged"}).json()

    assert set(body) == DISCOVERY_KEYS


# -- the command-line callers --------------------------------------------------------------

@pytest.fixture
def as_daemon_request(app_and_engine, monkeypatch):
    """Route cli.daemon.daemon_request through the test app, sending the capability like the real one."""
    app = app_and_engine
    client = _client(app)

    def fake(method, path, body=None, **kw):
        r = client.request(method, path, headers=_daemon_headers(app))
        return r.json() if r.status_code == 200 else None

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", fake)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
    return app, client


def test_slm_status_still_reports_the_daemon_in_company_mode(as_daemon_request, capsys):
    app, _ = as_daemon_request
    _company_mode(app)
    from argparse import Namespace

    from superlocalmemory.cli.commands import cmd_serve

    cmd_serve(Namespace(action="status"))

    out = capsys.readouterr().out
    assert "RUNNING" in out and "could not get status" not in out and "facts=" in out


def test_slm_serve_status_survives_a_discovery_only_answer(monkeypatch, capsys):
    """If a daemon ever answers the short form, the command says so rather than crashing."""
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", lambda *a, **k: {
        "status": "running", "pid": 7, "port": 8765, "version": "x", "instance_id": "i",
        "details_hidden": True})
    from argparse import Namespace

    from superlocalmemory.cli.commands import cmd_serve

    cmd_serve(Namespace(action="status"))

    out = capsys.readouterr().out
    assert "RUNNING" in out and "PID 7" in out and "sign in" in out.lower()


def test_ops_status_never_reports_healthy_from_a_hidden_answer(monkeypatch, capsys):
    from argparse import Namespace

    from superlocalmemory.cli import ops_cmd

    hidden = {"status": "running", "pid": 7, "port": 8765, "version": "x",
              "instance_id": "i", "details_hidden": True}
    full = {"status": "running", "dead_letter_count": 2, "degraded_operations": 0,
            "exhausted_obligations": 0, "writer_stalled": False, "unreadable_saves": 0}
    monkeypatch.setattr(ops_cmd, "_daemon_get", lambda path, *a, **k: hidden)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", lambda *a, **k: full)

    ops_cmd._cmd_ops_status(Namespace(json=False))

    out = capsys.readouterr().out
    assert "DEGRADED" in out and "HEALTHY" not in out


def test_ops_status_says_sign_in_when_even_the_capability_gets_the_short_answer(monkeypatch, capsys):
    from argparse import Namespace

    from superlocalmemory.cli import ops_cmd

    hidden = {"status": "running", "pid": 7, "details_hidden": True}
    monkeypatch.setattr(ops_cmd, "_daemon_get", lambda path, *a, **k: hidden)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", lambda *a, **k: hidden)

    with pytest.raises(SystemExit):
        ops_cmd._cmd_ops_status(Namespace(json=False))

    captured = capsys.readouterr()
    assert "HEALTHY" not in captured.out
    assert "sign in" in captured.err.lower()
