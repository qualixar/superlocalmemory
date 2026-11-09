# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Remove really stops the on-device check; a re-check never runs two models;
the status says what runs, on what calibration, and what is wrong with a key.

* M-7: in the default "auto" mode, Remove deleted the install but left the
  running model in memory, judging recalls, and left its paths in the config,
  so the dashboard then called the removed install "Needs a check".
* M-8: Set up again / Use an existing install loaded a second copy of the model
  next to the running one for its check.
* F11: the status named Laya's calibration even while Jev was the one running.
* F14: a key file changed outside SLM turned the whole status read into a 500.
* F21: "auto" is the starting state only; the dashboard offers the three
  explicit choices and shows "auto" as what it actually does.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, judge_keys, judge_selection, laya_runtime
from superlocalmemory.retrieval import judge_recipe
from superlocalmemory.server.routes import answer_check
from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    FAKE_KEY,
    _FakeKeyStore,
    _reset_fake_key_store,
    _retrieval,
    _set_retrieval,
    call_order,
    client,
)

# Writes Laya runtime state: pin the data root to tmp_path even without the
# root conftest (--noconftest), see tests/isolation_guard.py.
from ..isolation_guard import explicit_slm_root  # noqa: F401

#: The real functions, captured before the suite's fixture replaces them.
REAL_DETECT = laya_runtime.detect
REAL_REMOVE = laya_runtime.remove


class _Judge:
    def __init__(self, backend: str, rerank_k: int = 0) -> None:
        self.backend = backend
        self.rerank_k = rerank_k
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def _fresh_jobs():
    laya_runtime.LayaInstallJob._instance = None
    laya_runtime.LayaAdoptJob._instance = None
    yield
    for job_cls in (laya_runtime.LayaInstallJob, laya_runtime.LayaAdoptJob):
        job = job_cls._instance
        if job is not None and getattr(job, "_thread", None) is not None:
            job._thread.join(timeout=10)
        job_cls._instance = None


@pytest.fixture()
def engine(client):  # noqa: F811
    retrieval_engine = SimpleNamespace(_sufficiency_judge=None)
    client.app.state.engine = SimpleNamespace(_retrieval_engine=retrieval_engine)
    return retrieval_engine


@pytest.fixture()
def rebuilds(monkeypatch, engine):
    """attach builds what the config asks for (Laya when laya/auto and a path)."""
    seen: list[str] = []

    def _attach(retrieval_engine, cfg):
        mode = cfg.sufficiency_judge
        seen.append(mode)
        old = retrieval_engine._sufficiency_judge
        if old is not None:
            old.shutdown()
        wants_laya = mode in ("laya", "auto") and bool(cfg.sufficiency_python)
        retrieval_engine._sufficiency_judge = _Judge("laya") if wants_laya else None
        return judge_selection.backend_of(retrieval_engine._sufficiency_judge)

    monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge", _attach, raising=False)
    return seen


def _managed_install(tmp_path: Path) -> Path:
    import sys

    run_dir = laya_runtime.runtime_dir()
    python = run_dir / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    model = run_dir / "hf-cache" / "model"
    model.mkdir(parents=True)
    (run_dir / ".slm-managed").write_text(json.dumps(
        {"python": str(python), "model_path": str(model), "verified": True}), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# M-7
# ---------------------------------------------------------------------------


class TestRemoveReallyStopsIt:
    @pytest.fixture()
    def installed(self, client, monkeypatch, tmp_path, engine, rebuilds):  # noqa: F811
        monkeypatch.setattr(laya_runtime, "detect", REAL_DETECT)
        monkeypatch.setattr(laya_runtime, "remove", REAL_REMOVE)
        monkeypatch.setattr(laya_runtime, "_apple_silicon", lambda: True)
        run_dir = _managed_install(tmp_path)
        python = str(run_dir / "venv" / "bin" / "python")
        _set_retrieval(tmp_path, sufficiency_judge="auto", sufficiency_python=python,
                       sufficiency_model=str(run_dir / "hf-cache" / "model"),
                       sufficiency_hf_home=str(run_dir / "hf-cache"))
        running = _Judge("laya")
        engine._sufficiency_judge = running
        return running

    def test_in_auto_mode_the_running_model_is_stopped(self, client, engine, installed):  # noqa: F811
        response = client.post("/api/v3/answer-check/laya/remove")
        assert response.status_code == 200, response.text
        assert installed.stopped is True
        assert engine._sufficiency_judge is None

    def test_its_paths_are_cleared_so_it_reads_not_installed(
            self, client, tmp_path, installed):  # noqa: F811
        client.post("/api/v3/answer-check/laya/remove")
        retrieval = _retrieval(tmp_path)
        assert retrieval["sufficiency_python"] == ""
        assert retrieval["sufficiency_hf_home"] == ""
        status = client.get("/api/v3/answer-check").json()
        assert status["laya"]["state"] == laya_runtime.STATE_NOT_INSTALLED
        assert status["active"] == "off"

    def test_the_model_stops_before_its_files_are_deleted(
            self, client, monkeypatch, installed):  # noqa: F811
        order = []
        real = laya_runtime.remove
        monkeypatch.setattr(installed, "shutdown",
                            lambda: order.append("stopped") or setattr(installed, "stopped", True))
        monkeypatch.setattr(laya_runtime, "remove",
                            lambda: order.append("deleted") or real())
        client.post("/api/v3/answer-check/laya/remove")
        assert order == ["stopped", "deleted"]

    def test_a_second_engine_holding_the_model_is_stopped_too(
            self, client, monkeypatch, installed):  # noqa: F811
        """A hot reconfigure can leave a second engine holding the one on-device
        model; Remove must stop it everywhere, before the files go."""
        from superlocalmemory.core import judge_selection

        order = []
        real = laya_runtime.remove
        monkeypatch.setattr(judge_selection, "stop_on_device_check",
                            lambda: order.append("stopped everywhere") or True)
        monkeypatch.setattr(laya_runtime, "remove",
                            lambda: order.append("deleted") or real())
        response = client.post("/api/v3/answer-check/laya/remove")
        assert response.status_code == 200, response.text
        assert order == ["stopped everywhere", "deleted"]

    def test_a_removed_install_is_never_brought_back(self, client, tmp_path, installed):  # noqa: F811
        client.post("/api/v3/answer-check/laya/remove")
        for _ in range(2):
            client.get("/api/v3/answer-check")
        assert _retrieval(tmp_path)["sufficiency_python"] == ""


# ---------------------------------------------------------------------------
# M-8 (route side): the running copy is stopped for the check, then restored
# ---------------------------------------------------------------------------


def test_set_up_again_stops_the_running_model_for_its_check(
        client, monkeypatch, tmp_path, engine, rebuilds):  # noqa: F811
    monkeypatch.setattr(laya_runtime.LayaInstallJob, "instance",
                        classmethod(lambda cls: cls._instance or _new(cls)))
    _set_retrieval(tmp_path, sufficiency_judge="auto", sufficiency_python="/opt/v/bin/python")
    running = _Judge("laya")
    engine._sufficiency_judge = running
    seen_at_check: list = []

    def _install(*, progress=None, before_verify=None, **_):
        before_verify()
        seen_at_check.append((running.stopped, engine._sufficiency_judge))
        return laya_runtime.LayaRuntimeStatus(
            state=laya_runtime.STATE_READY, python="/opt/v/bin/python",
            hf_home="/opt/v/hf", model_path="/opt/v/model")

    monkeypatch.setattr(laya_runtime, "install", _install)
    assert client.post("/api/v3/answer-check/laya/setup").status_code == 200
    laya_runtime.LayaInstallJob._instance._thread.join(timeout=5)
    assert seen_at_check == [(True, None)], "a second model would have loaded"
    assert isinstance(engine._sufficiency_judge, _Judge), "the check was not restored"


def _new(cls):
    cls._instance = cls()
    return cls._instance


# ---------------------------------------------------------------------------
# F11 / F14 / F21: the status tells the truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend,key", [("laya", "laya"), ("jev", "jev")])
def test_calibration_status_names_the_check_that_runs(client, engine, backend, key):  # noqa: F811
    engine._sufficiency_judge = _Judge(backend)
    body = client.get("/api/v3/answer-check").json()
    assert body["active"] == backend
    assert body["calibration_status"] == judge_recipe.calibration_or_unmeasured(key).status


def test_with_reordering_the_calibration_is_the_reordered_one(client, engine, tmp_path):  # noqa: F811
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    _set_retrieval(tmp_path, sufficiency_judge="jev", sufficiency_jev_consent=True,
                   sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True)
    engine._sufficiency_judge = _Judge("jev", rerank_k=20)
    body = client.get("/api/v3/answer-check").json()
    assert body["calibration_status"] == \
        judge_recipe.calibration_or_unmeasured("jev-listwise").status


def test_no_calibration_is_named_when_nothing_runs(client, engine):  # noqa: F811
    assert client.get("/api/v3/answer-check").json()["calibration_status"] is None


class _TamperedStore(_FakeKeyStore):
    def has_key(self, provider):
        raise judge_keys.JudgeKeyStoreError("jev-typesafe.key is a symlink; refusing to use it.")

    masked = has_key
    load = has_key


@pytest.fixture()
def tampered(monkeypatch):
    monkeypatch.setattr(judge_keys, "JudgeKeyStore", _TamperedStore)


def test_a_tampered_key_file_is_a_clear_state_not_a_crash(client, tampered):  # noqa: F811
    response = client.get("/api/v3/answer-check")
    assert response.status_code == 200, response.text
    jev = response.json()["jev"]
    assert jev["has_key"] is False
    assert "outside SLM" in jev["key_problem"]
    assert "remove" in jev["key_problem"].lower()


def test_turning_jev_on_with_a_tampered_key_says_what_to_do(client, tampered, tmp_path):  # noqa: F811
    _set_retrieval(tmp_path, sufficiency_jev_consent=True)
    response = client.post("/api/v3/answer-check/mode", json={"mode": "jev"})
    assert response.status_code == 400
    assert "outside SLM" in response.json()["error"]


def test_a_tampered_key_can_still_be_removed(client, tampered):  # noqa: F811
    response = client.request("DELETE", "/api/v3/answer-check/jev/key",
                              json={"provider": "typesafe"})
    assert response.status_code == 200, response.text


def test_auto_is_shown_as_what_it_does_and_is_not_a_choice(client, engine, tmp_path):  # noqa: F811
    _set_retrieval(tmp_path, sufficiency_judge="auto")
    body = client.get("/api/v3/answer-check").json()
    assert body["mode"] == "auto" and body["active"] == "off"
    assert client.post("/api/v3/answer-check/mode", json={"mode": "auto"}).status_code == 422
    doc = (Path(answer_check.__file__).resolve().parents[4] / "docs" / "answer-check.md").read_text(encoding="utf-8")
    assert "starting state" in doc and "never offered" in doc.lower()
