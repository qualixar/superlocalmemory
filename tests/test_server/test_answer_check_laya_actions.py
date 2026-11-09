# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The on-device panel's buttons, against the real install detection.

Recreates the state of a real 4.1.19 machine — an install made outside SLM
that was never checked, plus a stale, half-made SLM install folder — and
checks that each button the dashboard offers does what it says: Check this
install (no download), Test (persistent result), Cancel, Remove (clears a
half-made setup, keeps someone else's install), Forget (drops only SLM's
record). Every worker is a local fake; nothing loads a model or downloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import laya_process
from superlocalmemory.core import laya_runtime as lr
from superlocalmemory.retrieval import sufficiency
from superlocalmemory.server.routes import answer_check, answer_check_actions
from tests.helpers.owned_python import owned_environment

API = "/api/v3/answer-check"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_check, "_require_manage", lambda request: None)
    monkeypatch.setattr(answer_check, "_require_credential", lambda request: None)
    monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_is_in_temp_dir", lambda path: "pytest-of-ghost" in str(path))
    for job in (lr.LayaInstallJob, lr.LayaAdoptJob, lr.LayaTestJob):
        job._instance = None
    laya_process.CANCEL.clear()
    app = FastAPI()
    app.state.engine = None
    app.include_router(answer_check.router)
    app.include_router(answer_check_actions.router)
    yield TestClient(app)
    laya_process.CANCEL.clear()
    for job in (lr.LayaInstallJob, lr.LayaAdoptJob, lr.LayaTestJob):
        job._instance = None


def _external(tmp_path: Path) -> tuple[str, str]:
    python = owned_environment(tmp_path / "laya-venv")
    model = tmp_path / "model"
    model.mkdir()
    return str(python), str(model)


def _varuns_machine(tmp_path: Path) -> tuple[str, str]:
    python, model = _external(tmp_path)
    run_dir = lr.runtime_dir()
    (run_dir / "venv" / "bin").mkdir(parents=True)
    (run_dir / "hf-cache" / "model").mkdir(parents=True)
    (run_dir / ".install-steps.json").write_text('{"venv": true, "pip": true, "weights": true}', encoding="utf-8")
    (run_dir / ".slm-managed").write_text(json.dumps({
        "verified": False, "python": str(run_dir / "venv/bin/python"),
        "model_path": str(run_dir / "hf-cache/hub/x")}), encoding="utf-8")
    (run_dir / "adopted.json").write_text(json.dumps({
        "python": "/private/var/folders/pytest-of-ghost/venv/bin/python",
        "model_path": "/private/var/folders/pytest-of-ghost/model", "verified": True}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"retrieval": {
        "sufficiency_judge": "auto", "sufficiency_python": python,
        "sufficiency_model": model, "sufficiency_jev_provider": "typesafe"}}), encoding="utf-8")
    return python, model


def _wait(client, key="adopt"):
    """Join every background job's thread, then read the status: no polling,
    no deadline a loaded machine could miss. (60 s is a hang guard only.)"""
    for job_cls in (lr.LayaInstallJob, lr.LayaAdoptJob, lr.LayaTestJob):
        thread = getattr(job_cls._instance, "_thread", None)
        if thread is not None:
            thread.join(60)
            assert not thread.is_alive(), "job never finished"
    return client.get(API).json()


def test_varuns_state_is_offered_a_check_not_a_download(client, tmp_path):
    _varuns_machine(tmp_path)
    laya = client.get(API).json()["laya"]
    assert laya["state"] == "failed" and laya["action"] == "check"
    assert "Check this install" in laya["step"]


def test_check_this_install_turns_the_on_device_check_on(client, tmp_path, monkeypatch):
    python, model = _varuns_machine(tmp_path)
    monkeypatch.setattr(lr, "verify", lambda *a, **k: (True, "Looks good."))
    response = client.post(f"{API}/laya/adopt", json={"python": python, "model_path": model,
                                                      "hf_home": ""})
    assert response.status_code == 202, response.text
    body = _wait(client)
    assert body["laya"]["state"] == "ready"
    assert body["mode"] == "auto"             # auto: on by itself once checked


def test_a_failed_check_says_why_and_stays_checkable(client, tmp_path, monkeypatch):
    python, model = _varuns_machine(tmp_path)
    monkeypatch.setattr(lr, "verify", lambda *a, **k: (False, "Couldn't load the local model."))
    client.post(f"{API}/laya/adopt", json={"python": python, "model_path": model, "hf_home": ""})
    body = _wait(client)
    assert body["laya"]["action"] == "check"
    assert body["laya"]["error"] == "Couldn't load the local model."


def test_test_keeps_its_result(client, tmp_path, monkeypatch):
    python, model = _external(tmp_path)
    monkeypatch.setattr(lr, "verify", lambda *a, **k: (True, "Looks good."))
    client.post(f"{API}/laya/adopt", json={"python": python, "model_path": model, "hf_home": ""})
    _wait(client)
    assert client.post(f"{API}/laya/test").status_code == 202
    body = _wait(client)
    result = body["tests"]["laya"]
    assert result["ok"] is True and result["at"] and result["seconds"] is not None
    assert client.get(API).json()["tests"]["laya"] == result      # survives a reload


def test_test_needs_a_ready_install(client):
    response = client.post(f"{API}/laya/test")
    assert response.status_code == 400
    assert "Set up" in response.json()["error"]


def test_remove_clears_a_half_made_setup_but_keeps_someone_elses(client, tmp_path):
    python, model = _external(tmp_path)
    run_dir = lr.runtime_dir()
    run_dir.mkdir(parents=True)
    (run_dir / ".install-steps.json").write_text('{"venv": true}', encoding="utf-8")
    (run_dir / "install.lock").write_text("", encoding="utf-8")
    (run_dir / "venv").mkdir()
    (run_dir / "adopted.json").write_text(json.dumps(
        {"python": python, "model_path": model, "verified": True}), encoding="utf-8")
    assert client.get(API).json()["laya"]["state"] == "ready"   # the adopted one wins
    # The half-made setup is still SLM's to clear:
    assert lr.remove().state == lr.STATE_NOT_INSTALLED
    assert sorted(p.name for p in run_dir.iterdir()) == ["adopted.json"]


def test_remove_on_a_half_made_setup_via_the_dashboard(client, tmp_path):
    run_dir = lr.runtime_dir()
    run_dir.mkdir(parents=True)
    (run_dir / ".install-steps.json").write_text('{"venv": true, "pip": true}', encoding="utf-8")
    (run_dir / "install.lock").write_text("", encoding="utf-8")
    (run_dir / "hf-cache").mkdir()
    laya = client.get(API).json()["laya"]
    assert (laya["state"], laya["action"]) == ("failed", "setup")
    response = client.post(f"{API}/laya/remove")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "not_installed"
    assert not run_dir.exists()


def test_forget_an_install_made_elsewhere_leaves_its_files(client, tmp_path):
    python, model = _varuns_machine(tmp_path)
    response = client.post(f"{API}/laya/remove")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "not_installed"
    assert Path(python).exists() and Path(model).is_dir()
    status = client.get(API).json()
    assert status["laya"]["state"] == "not_installed"


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the stand-in interpreter is a shell script; Laya runs only on Apple silicon")
def test_cancel_stops_a_running_setup(client, tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
    monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
    exe = tmp_path / "fakepy"
    exe.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
    exe.chmod(0o755)
    real = lr._download_weights
    monkeypatch.setattr(lr, "_download_weights", lambda python, *a, **k: real(exe, *a, **k))
    assert client.post(f"{API}/laya/setup").status_code == 200
    # Cancel at once — the moment a person changes their mind — must still land.
    assert client.post(f"{API}/laya/cancel").status_code == 200
    body = _wait(client)
    assert body["laya"]["state"] == "failed" and body["laya"]["action"] == "setup"
    assert "cancelled" in body["laya"]["error"]
    assert client.post(f"{API}/laya/cancel").status_code == 409
