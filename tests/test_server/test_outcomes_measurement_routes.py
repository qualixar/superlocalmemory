# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-28 over HTTP: read the notice and state; change it only as the machine's owner."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.outcomes import disclosure
from superlocalmemory.server.routes import outcomes

URL = "/api/outcomes/measurement"
TOKEN = "install-token-for-this-test-0123456789abcdef"
CHANGES = (("POST", URL, {"on": False}), ("POST", URL + "/acknowledge", None))


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    for name in ("SLM_FEATURE_OUTCOMES_HOLDOUT", "SLM_FEATURE_OUTCOMES_SWAPS", "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _client(monkeypatch, *, credential: bool, manage: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(outcomes.router)
    if credential:  # otherwise the real credential check stays in place
        monkeypatch.setattr(outcomes, "_require_credential", lambda request: None)
    if manage:
        monkeypatch.setattr(outcomes, "_require_manage", lambda request: None)
    return TestClient(app)


@pytest.fixture()
def client(home, monkeypatch):
    return _client(monkeypatch, credential=True)


def test_get_returns_the_notice_the_state_and_that_it_is_pending(client, home) -> None:
    body = client.get(URL).json()
    assert body["success"] is True and body["pending"] is True and body["measuring"] is True
    assert "slm measurement off" in body["notice"]
    assert {f["id"]: f["state"] for f in body["flags"]}["outcomes.holdout"] == "on"


def test_get_does_not_acknowledge(client, home) -> None:
    client.get(URL)
    assert disclosure.pending(home) is True


def test_acknowledge_ends_pending_and_names_the_dashboard(client, home) -> None:
    body = client.post(URL + "/acknowledge").json()
    assert body["pending"] is False and disclosure.state(home)["channel"] == "dashboard"
    assert client.get(URL).json()["pending"] is False


def test_off_turns_the_two_interventions_off_and_nothing_else(client, home) -> None:
    body = client.post(URL, json={"on": False}).json()
    assert body["measuring"] is False
    flags = FeatureFlags(home, environ={})
    assert not flags.enabled("outcomes.holdout") and not flags.enabled("outcomes.swaps")
    assert flags.enabled("outcomes.receipts")
    assert client.post(URL, json={"on": True}).json()["measuring"] is True


@pytest.mark.parametrize("body", [{}, {"on": "yes"}, {"on": 1}, {"on": False, "receipts": True},
                                  {"enabled": False}])
def test_a_malformed_change_is_refused_and_nothing_changes(client, home, body) -> None:
    assert client.post(URL, json=body).status_code == 422
    assert FeatureFlags(home, environ={}).enabled("outcomes.holdout")


@pytest.mark.parametrize("method,path,body", CHANGES)
def test_a_change_without_a_credential_is_refused(home, monkeypatch, tmp_path, method, path, body) -> None:
    client = _client(monkeypatch, credential=False)
    token_file = tmp_path / ".install_token"
    token_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: token_file)
    assert client.request(method, path, json=body).status_code == 403
    assert client.request(method, path, json=body,
                          headers={"X-Install-Token": "wrong"}).status_code == 403
    assert FeatureFlags(home, environ={}).enabled("outcomes.holdout")
    assert disclosure.pending(home) is True


def test_a_remote_caller_with_the_install_token_is_refused(home, monkeypatch, tmp_path) -> None:
    """The token is readable from the dashboard's JS, so it is honoured from loopback only."""
    from superlocalmemory.server import write_identity

    client = _client(monkeypatch, credential=False)
    token_file = tmp_path / ".install_token"
    token_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: token_file)
    monkeypatch.setattr(write_identity, "_TEST_ISOLATION_ALLOWED", False)
    monkeypatch.setattr(write_identity, "_is_loopback_host", lambda host: False)
    response = client.post(URL, json={"on": False}, headers={"X-Install-Token": TOKEN})
    assert response.status_code == 403
    assert FeatureFlags(home, environ={}).enabled("outcomes.holdout")


def test_without_the_manage_role_a_change_is_refused(home, monkeypatch) -> None:
    from fastapi import HTTPException

    def deny(request):
        raise HTTPException(403, detail="not an owner or admin")

    client = _client(monkeypatch, credential=True, manage=False)
    monkeypatch.setattr(outcomes, "_require_manage", deny)
    assert client.post(URL, json={"on": False}).status_code == 403
    assert FeatureFlags(home, environ={}).enabled("outcomes.holdout")
    assert client.get(URL).status_code == 200  # reading is not gated by the role


def test_the_kill_switch_shows_as_not_measuring(client, home) -> None:
    from superlocalmemory.core.feature_flags import set_kill_switch

    set_kill_switch(home, True)
    body = client.get(URL).json()
    assert body["kill_switch"] is True and body["measuring"] is False


def test_reading_it_needs_a_signed_in_user_where_company_mode_asks_for_one() -> None:
    from superlocalmemory.server import read_gates

    assert read_gates.is_sensitive_dashboard_read("GET", URL)
    assert not read_gates.is_sensitive_dashboard_read("POST", URL)  # the POSTs have their own gate
