# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Slow answer-check work never holds the daemon's event loop, and reading the
status never changes anything.

The daemon is the one process serving MCP-over-HTTP, the CLI, remember and the
dashboard. Answer-check routes used to run their slow parts — loading a model
to check an install (up to minutes), a billed connection test, deleting a
1.1 GB install, switching the running check, detecting an interpreter — right
on that loop, so everything else froze while they ran. Checking an existing
install is now a background job polled like Set up, and the rest runs on worker
threads.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, judge_selection, laya_runtime
from superlocalmemory.retrieval import jev_judge, sufficiency
from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    FAKE_KEY,
    _FakeKeyStore,
    _reset_fake_key_store,
    _retrieval,
    _set_retrieval,
    call_order,
    client,
)

#: /ping must answer well inside the slow call; generous for a loaded laptop.


@pytest.fixture(autouse=True)
def _fresh_jobs():
    laya_runtime.LayaInstallJob._instance = None
    laya_runtime.LayaAdoptJob._instance = None
    yield
    for job_cls in (laya_runtime.LayaInstallJob, laya_runtime.LayaAdoptJob):
        job = job_cls._instance
        thread = getattr(job, "_thread", None) if job is not None else None
        if thread is not None:
            thread.join(timeout=10)
        job_cls._instance = None


@pytest.fixture()
def pinged(client):  # noqa: F811
    """One event loop for every request, as in the daemon. (A TestClient used
    outside ``with`` starts a fresh loop per request, so a handler blocking
    its own loop would never delay another request and these tests would
    pass against the very defect they exist to catch.)"""
    from fastapi.testclient import TestClient

    app = client.app

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    with TestClient(app) as shared_loop:
        yield shared_loop


class _Hold:
    """Stands in for a slow step: says when it is entered, then holds until
    released — so a test knows, without guessing with sleeps, that a request
    is inside its handler for as long as the test needs."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, result=None):
        self.entered.set()
        self.release.wait()
        return result


#: Only ever reached when the code under test is broken (the step is never
#: entered, or /ping is stuck behind it): then the test fails instead of
#: hanging. A passing run never waits on it.
_HANG_GUARD_S = 60.0


def _ping_answers_while_held(client, hold, method, path, body=None) -> bool:  # noqa: F811
    """Whether /ping answered while the slow request was still held inside
    its handler. Deterministic: no latency threshold, no sleeps."""
    worker = threading.Thread(target=lambda: client.request(method, path, json=body),
                              daemon=True)
    worker.start()
    try:
        assert hold.entered.wait(_HANG_GUARD_S), "the slow step was never reached"
        guard = threading.Timer(_HANG_GUARD_S, hold.release.set)
        guard.daemon = True
        guard.start()
        try:
            assert client.get("/ping").status_code == 200
            return not hold.release.is_set()
        finally:
            guard.cancel()
    finally:
        hold.release.set()
        worker.join(_HANG_GUARD_S)


def _ready(**fields):
    base = dict(state=laya_runtime.STATE_READY, python="/opt/laya/venv/bin/python",
                hf_home="/opt/laya/hf", model_path="/opt/laya/model")
    base.update(fields)
    return laya_runtime.LayaRuntimeStatus(**base)


@pytest.fixture()
def real_jobs(monkeypatch):
    """The real background jobs (the suite's fixture replaces the install job)."""
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance",
                        classmethod(lambda cls: _singleton(cls)))
    monkeypatch.setattr(laya_runtime, "check_interpreter", lambda python, **k: "")


def _singleton(cls):
    if cls._instance is None:
        cls._instance = cls()
    return cls._instance


def _wait_for_job(job_cls, timeout=60.0) -> None:
    """Join the job's thread; ``timeout`` is a hang guard, never a threshold."""
    job = job_cls._instance
    thread = getattr(job, "_thread", None) if job is not None else None
    assert thread is not None, "the background job never started"
    thread.join(timeout)
    assert not thread.is_alive(), "the background job never finished"


# ---------------------------------------------------------------------------
# H-1: nothing slow on the loop
# ---------------------------------------------------------------------------


class TestOtherRequestsAreNotHeldUp:
    """/ping must answer while each slow step is provably still running."""

    def test_while_an_existing_install_is_being_checked(self, pinged, monkeypatch, real_jobs):
        hold = _Hold()
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: hold(_ready()))
        assert _ping_answers_while_held(pinged, hold, "POST", "/api/v3/answer-check/laya/adopt",
                                        {"python": "/opt/laya/venv/bin/python"})

    def test_while_a_key_is_being_tested(self, pinged, monkeypatch):
        _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
        hold = _Hold()
        monkeypatch.setattr(jev_judge, "check_connection",
                            lambda provider, key, timeout_s=10.0: hold((True, "ok")))
        assert _ping_answers_while_held(pinged, hold, "POST", "/api/v3/answer-check/jev/test",
                                        {"provider": "typesafe"})

    def test_while_the_install_is_being_deleted(self, pinged, monkeypatch):
        removed = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_NOT_INSTALLED)
        hold = _Hold()
        monkeypatch.setattr(laya_runtime, "remove", lambda: hold(removed))
        assert _ping_answers_while_held(pinged, hold, "POST", "/api/v3/answer-check/laya/remove")

    def test_while_the_running_check_is_being_switched(self, pinged, monkeypatch):
        hold = _Hold()
        monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge",
                            lambda engine, cfg: hold("off"), raising=False)
        assert _ping_answers_while_held(pinged, hold, "POST", "/api/v3/answer-check/mode",
                                        {"mode": "off"})

    def test_while_the_status_is_being_worked_out(self, pinged, monkeypatch):
        hold = _Hold()
        monkeypatch.setattr(engine_wiring, "resolve_judge_mode",
                            lambda cfg: hold("off"), raising=False)
        pinged.app.state.engine = None
        assert _ping_answers_while_held(pinged, hold, "GET", "/api/v3/answer-check")


# ---------------------------------------------------------------------------
# adopt: a background job, polled; F8: refused at once where it cannot run
# ---------------------------------------------------------------------------


class TestCheckingAnExistingInstall:
    def test_it_answers_at_once_and_reports_progress(self, client, monkeypatch, real_jobs):  # noqa: F811
        gate = threading.Event()
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: gate.wait() and _ready())
        try:
            response = client.post("/api/v3/answer-check/laya/adopt",
                                   json={"python": "/opt/laya/venv/bin/python"})
        except BaseException:
            gate.set()
            raise
        # It answered while the check was still held — no clock involved.
        assert not gate.is_set()
        assert response.status_code == 202, response.text
        assert response.json()["running"] is True
        assert client.get("/api/v3/answer-check").json()["adopt"]["running"] is True
        gate.set()
        _wait_for_job(laya_runtime.LayaAdoptJob)
        after = client.get("/api/v3/answer-check").json()["adopt"]
        assert after["running"] is False and after["state"] == laya_runtime.STATE_READY

    def test_a_passing_check_is_saved_without_anyone_watching(
            self, client, monkeypatch, tmp_path, real_jobs):  # noqa: F811
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: _ready())
        client.post("/api/v3/answer-check/laya/adopt",
                    json={"python": "/opt/laya/venv/bin/python"})
        _wait_for_job(laya_runtime.LayaAdoptJob)
        retrieval = _retrieval(tmp_path)
        assert retrieval["sufficiency_python"] == "/opt/laya/venv/bin/python"
        assert retrieval["sufficiency_model"] == "/opt/laya/model"

    def test_a_failing_check_saves_nothing_and_says_why(
            self, client, monkeypatch, tmp_path, real_jobs):  # noqa: F811
        failed = laya_runtime.LayaRuntimeStatus(state=laya_runtime.STATE_FAILED,
                                                error="The check did not pass.")
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: failed)
        client.post("/api/v3/answer-check/laya/adopt",
                    json={"python": "/opt/laya/venv/bin/python"})
        _wait_for_job(laya_runtime.LayaAdoptJob)
        assert _retrieval(tmp_path).get("sufficiency_python", "") == ""
        assert client.get("/api/v3/answer-check").json()["adopt"]["error"] == \
            "The check did not pass."

    def test_off_apple_silicon_it_is_refused_at_once(self, client, monkeypatch):  # noqa: F811
        monkeypatch.setattr(sufficiency, "laya_supported", lambda: False)
        called = []
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: called.append(1))
        response = client.post("/api/v3/answer-check/laya/adopt",
                               json={"python": "/opt/laya/venv/bin/python"})
        assert response.status_code == 400
        assert "apple silicon" in response.json()["error"].lower()
        assert called == []

    def test_an_unsafe_interpreter_is_refused_at_once(self, client, monkeypatch):  # noqa: F811
        called = []
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: called.append(1))
        response = client.post("/api/v3/answer-check/laya/adopt", json={"python": "/bin/sh"})
        assert response.status_code == 400
        assert response.json()["error"]
        assert called == []

    def test_only_one_slow_laya_job_at_a_time(self, client, monkeypatch, real_jobs):  # noqa: F811
        gate = threading.Event()
        monkeypatch.setattr(laya_runtime, "adopt", lambda *a, **k: gate.wait() and _ready())
        try:
            assert client.post("/api/v3/answer-check/laya/adopt",
                               json={"python": "/opt/laya/venv/bin/python"}).status_code == 202
            assert client.post("/api/v3/answer-check/laya/setup").status_code == 409
        finally:
            gate.set()


# ---------------------------------------------------------------------------
# F13: a status read changes nothing; a finished install applies itself
# ---------------------------------------------------------------------------


class TestReadingTheStatusChangesNothing:
    def test_a_read_writes_no_settings_and_starts_no_check(
            self, client, monkeypatch, tmp_path, call_order):  # noqa: F811
        _set_retrieval(tmp_path, sufficiency_judge="auto")
        finished = SimpleNamespace(status=lambda: _ready(), start=lambda **k: True,
                                   running=False)
        monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance",
                            staticmethod(lambda: finished))
        before = (tmp_path / "config.json").read_text(encoding="utf-8")
        for _ in range(3):
            assert client.get("/api/v3/answer-check").status_code == 200
        assert (tmp_path / "config.json").read_text(encoding="utf-8") == before
        assert not (tmp_path / "answer_check.json").exists()
        assert call_order == []

    def test_a_finished_setup_is_applied_with_nobody_watching(
            self, client, monkeypatch, tmp_path, call_order, real_jobs):  # noqa: F811
        _set_retrieval(tmp_path, sufficiency_judge="auto")
        monkeypatch.setattr(laya_runtime, "install", lambda **k: _ready())
        assert client.post("/api/v3/answer-check/laya/setup").status_code == 200
        _wait_for_job(laya_runtime.LayaInstallJob)
        assert _retrieval(tmp_path)["sufficiency_python"] == "/opt/laya/venv/bin/python"
        assert ("attach", "auto") in call_order


# ---------------------------------------------------------------------------
# L-4: the status poll does not start a process each time
# ---------------------------------------------------------------------------


def test_repeated_status_reads_detect_the_interpreter_once(client, monkeypatch, tmp_path):  # noqa: F811
    _set_retrieval(tmp_path, sufficiency_judge="laya")
    monkeypatch.setattr(engine_wiring, "resolve_judge_mode",
                        judge_selection.resolve_judge_mode, raising=False)
    client.app.state.engine = None
    spawned = []
    monkeypatch.setattr(judge_selection, "interpreter_has_module",
                        lambda python, module: spawned.append(python) or False)
    for _ in range(5):
        assert client.get("/api/v3/answer-check").status_code == 200
    assert len(spawned) <= 1, f"{len(spawned)} interpreter probes for 5 reads"
