"""The job service: idle and spawning nothing when images and documents are off."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.daemon.services import ServiceRegistry
from superlocalmemory.documents import runner as runner_mod
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


def names():
    return {t.name for t in threading.enumerate()}


def test_feature_off_is_idle_and_creates_nothing(root, tmp_path):
    service = runner_mod.default_service(SimpleNamespace(state=SimpleNamespace()))
    service.start()
    assert "document-jobs" not in names()
    assert service.health()["state"] == "stopped"
    service.wake()
    assert "document-jobs" not in names() and service.stop(1.0) is True
    assert not (root / "media.db").exists() and not (root / "media").exists()


def test_enabled_service_runs_a_job_in_its_thread_and_stops_cleanly(root, tmp_path):
    store = open_media_store(create=True, data_root=root)
    receipt = submit_document(pdf_input(("x",)), profile_id="p1", actor_id="a",
                              config=SimpleNamespace(pii_redaction=False), store=store)
    runtime = Runtime()
    service = make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40]))
    service.start()
    try:
        assert service.health()["state"] == "running"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and store.get_job(receipt.job_id)["state"] != "done":
            time.sleep(0.05)
        assert store.get_job(receipt.job_id)["state"] == "done"
    finally:
        assert service.stop(5.0) is True
    assert service.health()["state"] == "stopped" and "document-jobs" not in names()
    store.close()


def test_registration_helpers_register_once_and_stop(root, monkeypatch):
    registry = ServiceRegistry()
    state = SimpleNamespace()
    runner_mod.start_document_jobs(SimpleNamespace(state=state), registry)
    runner_mod.start_document_jobs(SimpleNamespace(state=state), registry)
    assert registry.get("document-jobs") is not None and registry.snapshot()["document-jobs"]["state"] == "stopped"
    runner_mod.stop_document_jobs(registry)
    assert registry.get("document-jobs") is not None
