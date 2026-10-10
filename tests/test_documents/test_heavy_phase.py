"""A document job takes the shared RAM reservation for each parse step and yields to a model swap."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from superlocalmemory.core import ram_lock
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input

CFG = SimpleNamespace(pii_redaction=False)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    monkeypatch.setattr(ram_lock, "RAM_LOCK_PATH", tmp_path / "ram.sem")
    s = open_media_store(create=True, data_root=tmp_path / "slm")
    yield s
    s.close()


def _submit(store):
    return submit_document(pdf_input(("a", "b")), profile_id="p1", actor_id="a", config=CFG, store=store)


def _service(store, tmp_path, **deps):
    return make_service(store, Runtime(), FakeClient(), fake_script(tmp_path, ["x" * 40, "y" * 40]), deps=deps)


def test_the_parse_waits_for_another_heavy_job_then_runs(store, tmp_path):
    receipt = _submit(store)
    held, release = threading.Event(), threading.Event()

    def other_heavy_job():
        with ram_lock.ram_reservation("fake-heavy-job", required_mb=0):
            held.set()
            release.wait(30)

    holder = threading.Thread(target=other_heavy_job)
    holder.start()
    assert held.wait(5)
    service = _service(store, tmp_path)
    worker = threading.Thread(target=service.process_next)
    worker.start()
    try:
        time.sleep(0.6)
        assert store.get_document(receipt.document_id)["state"] != "ready"
        assert store.get_pages(receipt.document_id) == []
        release.set()
        holder.join(5)
        worker.join(30)
        assert store.get_document(receipt.document_id)["state"] == "ready"
    finally:
        release.set()


def test_a_refused_reservation_puts_the_job_back_and_does_not_fail_it(store, tmp_path):
    receipt = _submit(store)

    @contextmanager
    def refused():
        raise RuntimeError("not enough free memory")
        yield

    service = _service(store, tmp_path, heavy=refused)
    assert service.process_next() is False
    job = store.job_for_document(receipt.document_id)
    assert job["state"] == "queued" and not job.get("error")
    assert store.get_document(receipt.document_id)["state"] != "failed"
    service2 = _service(store, tmp_path)
    assert service2.process_next() is True
    assert store.get_document(receipt.document_id)["state"] == "ready"


def test_nothing_is_claimed_while_background_work_is_paused(store, tmp_path):
    receipt = _submit(store)
    paused = {"on": True}
    service = _service(store, tmp_path, background_paused=lambda: paused["on"])
    assert service.process_next() is False
    assert store.job_for_document(receipt.document_id)["state"] == "queued"
    paused["on"] = False
    assert service.process_next() is True
    assert store.get_document(receipt.document_id)["state"] == "ready"


def test_a_job_in_flight_steps_aside_when_a_swap_pauses_background_work(store, tmp_path):
    receipt = _submit(store)
    calls = {"n": 0}

    def paused_after_first_page():
        calls["n"] += 1
        return calls["n"] > 1

    service = _service(store, tmp_path, background_paused=paused_after_first_page)
    assert service.process_next() is False
    assert store.job_for_document(receipt.document_id)["state"] == "queued"
    done_before = len(store.get_pages(receipt.document_id))
    assert done_before < 2
    calls["n"] = -1000
    assert service.process_next() is True
    assert len(store.get_pages(receipt.document_id)) == 2
    assert store.get_document(receipt.document_id)["state"] == "ready"
