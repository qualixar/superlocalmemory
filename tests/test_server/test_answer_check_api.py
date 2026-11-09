# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The answer-check settings API: on-device Laya, hosted Jev, or off.

Over 75% of SLM's users are non-technical, so everything here must degrade
cleanly: a finished background install that nobody was watching still gets
applied, a key is never echoed back or logged, and a failed switch never
leaves two options looking active at once. These tests pin that behaviour
against fakes for the runtime contracts (``laya_runtime``, ``judge_keys``,
``jev_judge``, ``engine_wiring``) that other agents are implementing in
parallel — today they raise ``NotImplementedError`` when called for real.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import engine_wiring, judge_keys, laya_runtime
from superlocalmemory.retrieval import jev_judge, sufficiency
from superlocalmemory.server.routes import answer_check

FAKE_KEY = "sk-test-not-a-real-key-abcd"


# ---------------------------------------------------------------------------
# Fakes for the parallel-landing contracts
# ---------------------------------------------------------------------------


class _FakeKeyStore:
    """In-memory stand-in for ``JudgeKeyStore`` — shared dict per test."""

    _by_test: dict = {}

    def __init__(self, slm_home=None) -> None:
        self._store = _FakeKeyStore._by_test

    def set_key(self, provider: str, key: str) -> None:
        if provider not in judge_keys.PROVIDERS:
            raise ValueError("Unknown provider.")
        if not key or not key.strip():
            raise ValueError("A key is required.")
        self._store[provider] = key

    def has_key(self, provider: str) -> bool:
        return provider in self._store

    def masked(self, provider: str) -> str:
        key = self._store.get(provider)
        return f"****{key[-4:]}" if key else ""

    def clear(self, provider: str) -> None:
        self._store.pop(provider, None)

    def load(self, provider: str):
        return self._store.get(provider)


class _FakeJob:
    """Stand-in for ``LayaInstallJob.instance()``."""

    def __init__(self, status: "laya_runtime.LayaRuntimeStatus", *, running: bool = False) -> None:
        self._status = status
        self.running = running
        self.start_calls = 0

    def start(self, *args, **kwargs) -> bool:
        self.start_calls += 1
        self.start_kwargs = kwargs
        if self.running:
            return False
        self.running = True
        return True

    def status(self):
        return self._status


@pytest.fixture(autouse=True)
def _reset_fake_key_store():
    _FakeKeyStore._by_test = {}
    yield
    _FakeKeyStore._by_test = {}


@pytest.fixture()
def call_order():
    """Shared list so tests can assert ordering across monkeypatched calls."""
    return []


@pytest.fixture()
def client(monkeypatch, tmp_path, call_order):
    answer_check._last_jev_test.clear()
    monkeypatch.setattr(answer_check, "_require_manage", lambda request: None)
    # The credential gate has its own suite (test_answer_check_auth.py).
    monkeypatch.setattr(answer_check, "_require_credential", lambda request: None,
                        raising=False)
    monkeypatch.setattr(judge_keys, "JudgeKeyStore", _FakeKeyStore)
    monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)

    def _default_attach(retrieval_engine, retrieval_config):
        call_order.append(("attach", retrieval_config.sufficiency_judge))
        return retrieval_config.sufficiency_judge

    monkeypatch.setattr(
        engine_wiring, "attach_sufficiency_judge", _default_attach, raising=False,
    )
    monkeypatch.setattr(engine_wiring, "resolve_judge_mode", None, raising=False)

    not_installed = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_NOT_INSTALLED)
    monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: not_installed)
    monkeypatch.setattr(
        laya_runtime, "remove",
        lambda: (call_order.append(("remove",)) or not_installed),
    )
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance",
                         staticmethod(lambda: _FakeJob(not_installed)))

    (tmp_path / "config.json").write_text(json.dumps({
        "retrieval": {
            "sufficiency_judge": "off",
            "sufficiency_jev_provider": "typesafe",
            "sufficiency_jev_consent": False,
        },
    }), encoding="utf-8")

    class _FakeEngine:
        _retrieval_engine = SimpleNamespace(_sufficiency_judge=None)

    app = FastAPI()
    app.state.engine = _FakeEngine()
    app.include_router(answer_check.router)
    return TestClient(app)


_STATE_FILE = "answer_check.json"


def _set_retrieval(tmp_path, **fields) -> None:
    """Set answer-check settings as a test precondition, wherever they live:
    config.json before the routes first save them, their own file after."""
    path = tmp_path / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("retrieval", {}).update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")
    state = tmp_path / _STATE_FILE
    if state.exists():
        stored = json.loads(state.read_text(encoding="utf-8"))
        stored.update(fields)
        state.write_text(json.dumps(stored), encoding="utf-8")


def _retrieval(tmp_path) -> dict:
    """The answer-check settings in effect: their own file over config.json."""
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    merged = dict(data.get("retrieval", {}))
    state = tmp_path / _STATE_FILE
    if state.exists():
        merged.update(json.loads(state.read_text(encoding="utf-8")))
    return merged


# ---------------------------------------------------------------------------
# GET status
# ---------------------------------------------------------------------------


def test_get_status_shape(client):
    body = client.get("/api/v3/answer-check").json()
    assert body["mode"] == "off"
    assert body["active"] == "off"
    assert "laya" in body and "jev" in body
    assert body["jev"]["provider"] == "typesafe"
    assert body["jev"]["has_key"] is False
    assert body["apple_silicon"] is True
    assert "calibration_status" in body


# ---------------------------------------------------------------------------
# Mode validation matrix
# ---------------------------------------------------------------------------


def test_mode_laya_rejected_when_not_ready(client, tmp_path):
    response = client.post("/api/v3/answer-check/mode", json={"mode": "laya"})
    assert response.status_code in (400, 409), response.text
    assert "set up" in response.json()["error"].lower()
    assert _retrieval(tmp_path)["sufficiency_judge"] == "off"


def test_mode_laya_accepted_when_ready(client, tmp_path, monkeypatch, call_order):
    ready = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_READY, python="/usr/bin/python3")
    monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: ready)

    response = client.post("/api/v3/answer-check/mode", json={"mode": "laya"})
    assert response.status_code == 200, response.text
    assert _retrieval(tmp_path)["sufficiency_judge"] == "laya"
    assert call_order.count(("attach", "laya")) == 1


def test_mode_jev_rejected_without_consent(client):
    response = client.post("/api/v3/answer-check/mode", json={"mode": "jev"})
    assert response.status_code in (400, 409), response.text
    assert "consent" in response.json()["error"].lower() or "accept" in response.json()["error"].lower()


def test_mode_jev_rejected_without_key(client, tmp_path):
    _set_retrieval(tmp_path, sufficiency_jev_consent=True)
    response = client.post("/api/v3/answer-check/mode", json={"mode": "jev"})
    assert response.status_code in (400, 409), response.text
    assert "key" in response.json()["error"].lower()


def test_mode_jev_accepted_with_consent_and_key(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_jev_consent=True)
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY

    response = client.post("/api/v3/answer-check/mode", json={"mode": "jev"})
    assert response.status_code == 200, response.text
    assert _retrieval(tmp_path)["sufficiency_judge"] == "jev"
    assert call_order.count(("attach", "jev")) == 1


def test_mode_off_always_allowed(client):
    response = client.post("/api/v3/answer-check/mode", json={"mode": "off"})
    assert response.status_code == 200, response.text


def test_mode_rejects_unknown_value(client):
    response = client.post("/api/v3/answer-check/mode", json={"mode": "auto"})
    assert response.status_code == 422


def test_mode_rejects_unknown_field(client):
    response = client.post(
        "/api/v3/answer-check/mode", json={"mode": "off", "extra": "nope"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Exclusivity
# ---------------------------------------------------------------------------


def test_switching_mode_is_exclusive(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_jev_consent=True)
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    client.post("/api/v3/answer-check/mode", json={"mode": "jev"})
    assert _retrieval(tmp_path)["sufficiency_judge"] == "jev"

    ready = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_READY)
    import superlocalmemory.core.laya_runtime as laya_runtime_mod
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(laya_runtime_mod, "detect", lambda cfg=None: ready)
        resp = client.post("/api/v3/answer-check/mode", json={"mode": "laya"})
        assert resp.status_code == 200, resp.text

    retrieval = _retrieval(tmp_path)
    assert retrieval["sufficiency_judge"] == "laya"
    assert call_order.count(("attach", "laya")) == 1


# ---------------------------------------------------------------------------
# Laya setup / adopt / remove
# ---------------------------------------------------------------------------


def test_laya_setup_rejects_off_apple_silicon(client, monkeypatch):
    monkeypatch.setattr(sufficiency, "laya_supported", lambda: False)
    response = client.post("/api/v3/answer-check/laya/setup")
    assert response.status_code == 400
    assert "apple silicon" in response.json()["error"].lower()


def test_laya_setup_starts_job(client, monkeypatch):
    installing = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_INSTALLING, progress=0.1)
    job = _FakeJob(installing)
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance", staticmethod(lambda: job))

    response = client.post("/api/v3/answer-check/laya/setup")
    assert response.status_code == 200, response.text
    assert job.start_calls == 1
    assert response.json()["state"] == laya_runtime.STATE_INSTALLING


def test_laya_setup_conflicts_when_already_running(client, monkeypatch):
    installing = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_INSTALLING)
    job = _FakeJob(installing, running=True)
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance", staticmethod(lambda: job))

    response = client.post("/api/v3/answer-check/laya/setup")
    assert response.status_code == 409


def _adopt_and_wait(client, monkeypatch, result, body):
    """Adopting is a background job (it loads a model); run it to the end."""
    monkeypatch.setattr(laya_runtime, "adopt", lambda python, hf_home, model_path="": result)
    monkeypatch.setattr(laya_runtime, "check_interpreter", lambda python, **k: "")
    laya_runtime.LayaAdoptJob._instance = None
    response = client.post("/api/v3/answer-check/laya/adopt", json=body)
    job = laya_runtime.LayaAdoptJob._instance
    if job is not None and job._thread is not None:
        job._thread.join(timeout=5)
    laya_runtime.LayaAdoptJob._instance = None
    return response


def test_laya_adopt_persists_paths_when_ready(client, tmp_path, monkeypatch, call_order):
    ready = laya_runtime.LayaRuntimeStatus(
        state=laya_runtime.STATE_READY, python="/opt/laya/bin/python3",
        hf_home="/opt/laya/hf", model_path="/opt/laya/models/laya-mlx",
    )
    response = _adopt_and_wait(client, monkeypatch, ready, {
        "python": "/opt/laya/bin/python3", "hf_home": "/opt/laya/hf",
        "model_path": "/opt/laya/models/laya-mlx",
    })
    assert response.status_code == 202, response.text
    assert response.json()["running"] is True
    retrieval = _retrieval(tmp_path)
    assert retrieval["sufficiency_python"] == "/opt/laya/bin/python3"
    assert retrieval["sufficiency_hf_home"] == "/opt/laya/hf"
    assert retrieval["sufficiency_model"] == "/opt/laya/models/laya-mlx"


def test_laya_adopt_does_not_persist_when_verification_fails(client, tmp_path, monkeypatch):
    failed = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_FAILED, error="could not load")
    response = _adopt_and_wait(client, monkeypatch, failed, {"python": "/usr/bin/python3"})
    assert response.status_code == 202, response.text
    assert _retrieval(tmp_path).get("sufficiency_python", "") == ""


def test_laya_remove_switches_off_first_then_removes(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_judge="laya")
    running = SimpleNamespace(backend="laya",
                              shutdown=lambda: call_order.append(("stopped",)))
    client.app.state.engine._retrieval_engine._sufficiency_judge = running

    response = client.post("/api/v3/answer-check/laya/remove")
    assert response.status_code == 200, response.text
    assert _retrieval(tmp_path)["sufficiency_judge"] == "off"
    assert call_order == [("stopped",), ("remove",)]
    assert client.app.state.engine._retrieval_engine._sufficiency_judge is None


def test_laya_remove_when_not_active_skips_attach(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_judge="off")

    response = client.post("/api/v3/answer-check/laya/remove")
    assert response.status_code == 200, response.text
    assert call_order == [("remove",)]


# ---------------------------------------------------------------------------
# Jev keys
# ---------------------------------------------------------------------------


def test_jev_key_saved_and_masked(client):
    response = client.post(
        "/api/v3/answer-check/jev/key",
        json={"provider": "typesafe", "key": FAKE_KEY},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["has_key"] is True
    assert body["key_hint"] == f"****{FAKE_KEY[-4:]}"
    assert FAKE_KEY not in response.text


def test_jev_key_rejects_whitespace_only_key(client):
    """Passes the schema's min_length but is empty once stripped — the
    store's own ValueError must surface as a plain 400, not a 500."""
    response = client.post(
        "/api/v3/answer-check/jev/key",
        json={"provider": "typesafe", "key": "   "},
    )
    assert response.status_code == 400
    assert "required" in response.json()["error"].lower()


def test_jev_key_save_failure_never_logs_the_key(client, monkeypatch, caplog):
    """An unexpected store failure is reported generically — the key itself
    must never reach the log line that records the failure."""
    caplog.set_level(logging.DEBUG)

    def _boom(self, provider, key):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_FakeKeyStore, "set_key", _boom)
    response = client.post(
        "/api/v3/answer-check/jev/key",
        json={"provider": "typesafe", "key": FAKE_KEY},
    )
    assert response.status_code == 500
    assert response.json() == {"error": "Internal server error"}
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in response.text


def test_jev_key_rejects_invalid_provider(client):
    response = client.post(
        "/api/v3/answer-check/jev/key",
        json={"provider": "bogus", "key": FAKE_KEY},
    )
    assert response.status_code == 422


def test_jev_key_never_in_response_or_log(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/api/v3/answer-check/jev/key", json={"provider": "typesafe", "key": FAKE_KEY})
    client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    client.request(
        "DELETE", "/api/v3/answer-check/jev/key", json={"provider": "typesafe"},
    )
    assert FAKE_KEY not in caplog.text


def test_jev_key_delete_while_active_switches_off(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_judge="jev", sufficiency_jev_consent=True)
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY

    response = client.request(
        "DELETE", "/api/v3/answer-check/jev/key", json={"provider": "typesafe"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "off"
    assert _retrieval(tmp_path)["sufficiency_judge"] == "off"
    assert ("attach", "off") in call_order


def test_jev_key_delete_while_inactive_keeps_mode(client, tmp_path):
    _set_retrieval(tmp_path, sufficiency_judge="off")
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY

    response = client.request(
        "DELETE", "/api/v3/answer-check/jev/key", json={"provider": "typesafe"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "off"
    assert _retrieval(tmp_path)["sufficiency_judge"] == "off"


# ---------------------------------------------------------------------------
# Jev connection test + rate limit
# ---------------------------------------------------------------------------


def test_jev_test_without_key_does_not_call_check_connection(client, monkeypatch):
    called = []
    monkeypatch.setattr(
        jev_judge, "check_connection",
        lambda provider, key, timeout_s=10.0: called.append(1) or (True, "ok"),
    )
    response = client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is False
    assert called == []


def test_jev_test_with_key_calls_check_connection(client, monkeypatch):
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    monkeypatch.setattr(
        jev_judge, "check_connection",
        lambda provider, key, timeout_s=10.0: (True, "Reached typesafe."),
    )
    response = client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert {k: body[k] for k in ("ok", "message")} == {"ok": True, "message": "Reached typesafe."}
    # The result is kept, with when and how long, so a reload still shows it.
    assert body["result"]["ok"] is True and body["result"]["at"]
    assert client.get("/api/v3/answer-check").json()["tests"]["jev"]["message"] == \
        "Reached typesafe."


def test_jev_test_is_rate_limited(client, monkeypatch):
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    monkeypatch.setattr(
        jev_judge, "check_connection",
        lambda provider, key, timeout_s=10.0: (True, "ok"),
    )
    first = client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    assert first.status_code == 200
    second = client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    assert second.status_code == 429

    answer_check._last_jev_test["typesafe"] = time.monotonic() - 100
    third = client.post("/api/v3/answer-check/jev/test", json={"provider": "typesafe"})
    assert third.status_code == 200


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_consent_accept_persists(client, tmp_path):
    response = client.post(
        "/api/v3/answer-check/jev/consent",
        json={"accepted": True, "provider": "typesafe"},
    )
    assert response.status_code == 200, response.text
    retrieval = _retrieval(tmp_path)
    assert retrieval["sufficiency_jev_consent"] is True
    assert retrieval["sufficiency_jev_provider"] == "typesafe"


def test_consent_withdraw_while_active_switches_off(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_judge="jev", sufficiency_jev_consent=True)

    response = client.post(
        "/api/v3/answer-check/jev/consent",
        json={"accepted": False, "provider": "typesafe"},
    )
    assert response.status_code == 200, response.text
    assert _retrieval(tmp_path)["sufficiency_judge"] == "off"
    assert ("attach", "off") in call_order


def test_consent_withdraw_while_inactive_does_not_attach(client, tmp_path, call_order):
    _set_retrieval(tmp_path, sufficiency_judge="off", sufficiency_jev_consent=True)

    response = client.post(
        "/api/v3/answer-check/jev/consent",
        json={"accepted": False, "provider": "typesafe"},
    )
    assert response.status_code == 200, response.text
    assert call_order == []


# ---------------------------------------------------------------------------
# apply-if-ready
# ---------------------------------------------------------------------------


def test_a_finished_setup_applies_itself_once(client, tmp_path, monkeypatch, call_order):
    """The job saves and switches on a finished install itself — once — and a
    status read never does (a GET must not write or start anything)."""
    _set_retrieval(tmp_path, sufficiency_judge="auto")
    ready = laya_runtime.LayaRuntimeStatus(
        state=laya_runtime.STATE_READY, python="/opt/laya/bin/python3",
        hf_home="/opt/laya/hf", model_path="/opt/laya/models/laya-mlx",
    )
    job = _FakeJob(laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_INSTALLING))
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance", staticmethod(lambda: job))

    assert client.post("/api/v3/answer-check/laya/setup").status_code == 200
    assert call_order == []
    job._status = ready
    job.start_kwargs["on_done"](ready)          # what the job thread does at the end
    retrieval = _retrieval(tmp_path)
    assert retrieval["sufficiency_python"] == "/opt/laya/bin/python3"
    assert retrieval["sufficiency_model"] == "/opt/laya/models/laya-mlx"
    assert call_order.count(("attach", "auto")) == 1

    for _ in range(2):
        assert client.get("/api/v3/answer-check").status_code == 200
    assert call_order.count(("attach", "auto")) == 1  # never re-applied by a read


# ---------------------------------------------------------------------------
# Auth — real daemon middleware + real RBAC (no business-logic fakes: every
# mutating handler calls ``require_manage`` before touching anything else, so
# a denial never reaches the stub runtime contracts).
# ---------------------------------------------------------------------------


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {"X-SLM-Daemon-Capability": d.capability, "X-SLM-Target-Instance": d.instance_id}


@pytest.fixture
def daemon_client(engine_with_mock_deps):
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


@pytest.mark.parametrize(
    "method,path,body",
    (
        ("POST", "/api/v3/answer-check/calibration/cancel", None),
        ("POST", "/api/v3/answer-check/mode", {"mode": "off"}),
        ("POST", "/api/v3/answer-check/laya/setup", None),
        ("POST", "/api/v3/answer-check/laya/adopt", {"python": "/usr/bin/python3"}),
        ("POST", "/api/v3/answer-check/laya/remove", None),
        ("POST", "/api/v3/answer-check/jev/key",
         {"provider": "typesafe", "key": FAKE_KEY}),
        ("DELETE", "/api/v3/answer-check/jev/key", {"provider": "typesafe"}),
        ("POST", "/api/v3/answer-check/jev/test", {"provider": "typesafe"}),
        ("POST", "/api/v3/answer-check/jev/consent",
         {"accepted": True, "provider": "typesafe"}),
    ),
)
def test_mutations_require_manage(daemon_client, method, path, body):
    """A viewer session is rejected by RBAC before any handler body runs."""
    tc, h = daemon_client
    tc.post(
        "/api/rbac/users",
        json={"username": "ac-viewer", "password": "password-1234", "role": "viewer"},
        headers=h,
    )
    rbac = tc.app.state.rbac
    user_id = {u["username"]: u["user_id"] for u in rbac.list_users()}["ac-viewer"]
    token = rbac.create_session(user_id)

    response = tc.request(
        method, path, json=body, headers={**h, "X-SLM-User-Session": token},
    )
    assert response.status_code == 403, response.text


def test_mutation_requires_write_credential(daemon_client):
    """The install-token / daemon-capability gate is the same for every v3
    route — proved the same way the existing suite proves it for others:
    an uncredentialed cross-origin write is rejected before RBAC even runs."""
    tc, _h = daemon_client
    response = tc.post(
        "/api/v3/answer-check/mode",
        json={"mode": "off"},
        headers={"Origin": "http://localhost:8417"},
    )
    assert response.status_code == 403, response.text
    assert "cross-origin" in response.json()["error"]


def test_owner_passes_manage_with_daemon_credential(daemon_client, monkeypatch):
    """Sanity check: the 403s above are about role, not about the route
    being unreachable — the owner (daemon capability, no session) clears both
    gates. The runtime contracts this route calls (``laya_runtime`` etc.) are
    still real stubs outside this test's fakes, so a 500 past the gates is
    expected here; only 401/403 would mean the gate itself is wrong."""
    tc, h = daemon_client
    response = tc.post("/api/v3/answer-check/mode", json={"mode": "off"}, headers=h)
    assert response.status_code not in (401, 403), response.text
