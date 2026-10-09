"""Limits and restarts: a stuck or greedy parse is killed, finished pages stay, a restart resumes."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input

CFG = SimpleNamespace(pii_redaction=False)
PAGES = ["page one text " * 4, "page two text " * 4, "page three text " * 4]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    s = open_media_store(create=True, data_root=tmp_path / "slm")
    yield s
    s.close()


def submit(store):
    r = submit_document(pdf_input(("a", "b", "c")), profile_id="p1", actor_id="a", config=CFG, store=store)
    assert r.status == "processing"
    return r


def check_failed(store, receipt, reason, kept):
    job = store.get_job(receipt.job_id)
    assert job["state"] == "failed" and job["error"] == reason
    doc = store.get_document(receipt.document_id)
    assert doc["state"] == "failed" and len(store.get_pages(receipt.document_id)) == kept
    assert doc["pages_text_layer"] == kept


def test_a_page_that_takes_too_long_fails_the_job_and_keeps_earlier_pages(store, tmp_path):
    receipt, runtime = submit(store), Runtime()
    service = make_service(store, runtime, FakeClient(), fake_script(tmp_path, PAGES, stall_on=2), page_timeout_s=1.0)
    started = time.monotonic()
    service.process_next()
    assert time.monotonic() - started < 20
    check_failed(store, receipt, "page_timeout", 1)
    assert len(runtime.requests) == 1


def test_the_whole_job_has_a_wall_clock_limit(store, tmp_path):
    receipt = submit(store)
    service = make_service(store, Runtime(), FakeClient(), fake_script(tmp_path, PAGES, stall_on=2),
                           page_timeout_s=30.0, job_timeout_s=1.0)
    service.process_next()
    check_failed(store, receipt, "time_limit", 1)


def test_memory_use_is_polled_and_the_parse_is_killed(store, tmp_path):
    pytest.importorskip("psutil")
    receipt = submit(store)
    service = make_service(store, Runtime(), FakeClient(), fake_script(tmp_path, PAGES, bloat_on=2), rss_limit_mb=120)
    service.process_next()
    check_failed(store, receipt, "memory_limit", 1)


def test_parse_errors_fail_the_job_with_a_short_reason(store, tmp_path):
    receipt = submit(store)
    service = make_service(store, Runtime(), FakeClient(), fake_script(tmp_path, [], error="encrypted"))
    service.process_next()
    job = store.get_job(receipt.job_id)
    assert job["state"] == "failed" and job["error"] == "encrypted"
    assert store.get_document(receipt.document_id)["state"] == "failed"


def test_a_page_limit_below_the_page_count_is_refused_by_the_script_contract(store, tmp_path):
    receipt = submit(store)
    service = make_service(store, Runtime(), FakeClient(), fake_script(tmp_path, [], error="too_many_pages"))
    service.process_next()
    assert store.get_job(receipt.job_id)["error"] == "too_many_pages"


def test_stop_mid_job_then_restart_resumes_without_duplicates(store, tmp_path):
    receipt = submit(store)
    holder = {}
    runtime = Runtime(on_remember=lambda adm: holder["service"].request_stop() if adm.idempotency_key.endswith(":1:1") else None)
    service = holder["service"] = make_service(store, runtime, FakeClient(), fake_script(tmp_path, PAGES))
    service.process_next()
    job = store.get_job(receipt.job_id)
    assert job["state"] == "queued" and job["lease_owner"] is None and job["done"] == 1
    assert store.get_document(receipt.document_id)["state"] == "processing"
    assert [p["page_no"] for p in store.get_pages(receipt.document_id)] == [1]
    service.clear_stop()
    service.process_next()
    assert store.get_document(receipt.document_id)["state"] == "ready"
    assert store.get_job(receipt.job_id)["state"] == "done"
    assert runtime.keys == [f"doc:{receipt.document_id}:{n}:1" for n in (1, 2, 3)]
    assert len(store.list_items("p1", kind="page")) == 3


def test_a_page_saved_but_not_recorded_is_not_saved_twice(store, tmp_path):
    receipt = submit(store)
    runtime, client = Runtime(), FakeClient()
    service = make_service(store, runtime, client, fake_script(tmp_path, PAGES))
    service.process_next()
    with store._write() as conn:           # as if the process died before the page row was written
        conn.execute("DELETE FROM doc_pages WHERE document_id = ? AND page_no = 2", (receipt.document_id,))
        conn.execute("UPDATE jobs SET state = 'queued', lease_owner = NULL WHERE job_id = ?", (receipt.job_id,))
    service.process_next()
    assert len(set(runtime.keys)) == 3 and len(store.list_items("p1", kind="page")) == 3
    pages = {p["page_no"]: p for p in store.get_pages(receipt.document_id)}
    assert json.loads(pages[2]["memory_ids_json"]) == ["mem2"]


def test_a_lost_lease_stops_the_run_without_finishing(store, tmp_path):
    receipt = submit(store)
    def steal(adm):
        with store._write() as conn:
            conn.execute("UPDATE jobs SET lease_owner = 'someone-else' WHERE job_id = ?", (receipt.job_id,))
    service = make_service(store, Runtime(on_remember=steal), FakeClient(), fake_script(tmp_path, PAGES))
    service.process_next()
    job = store.get_job(receipt.job_id)
    assert job["state"] == "running" and job["lease_owner"] == "someone-else"
    assert store.get_document(receipt.document_id)["state"] == "processing"
