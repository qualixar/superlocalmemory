"""Job status and soft removal."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from superlocalmemory.documents.status import job_status, remove_document
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input

CFG = SimpleNamespace(pii_redaction=False)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    s = open_media_store(create=True, data_root=tmp_path / "slm")
    yield s
    s.close()


def test_job_status_reports_progress_and_hides_other_profiles(store, tmp_path):
    r = submit_document(pdf_input(("a",)), profile_id="p1", actor_id="a", config=CFG, store=store)
    s = job_status(r.job_id, "p1", store=store)
    assert s["state"] == "queued" and s["done"] == 0 and s["document_id"] == r.document_id
    assert s["document"]["state"] == "processing" and "error" in s
    assert job_status(r.job_id, "p2", store=store) is None and job_status("0" * 32, "p1", store=store) is None
    assert "payload_json" not in s and "user_words" not in json.dumps(s)


def test_soft_remove_hides_pages_and_archives_their_memories(store, tmp_path):
    r = submit_document(pdf_input(("a", "b")), content="note", profile_id="p1", actor_id="a", config=CFG, store=store)
    runtime = Runtime()
    make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40, "b" * 40], title="T")).process_next()
    assert len(store.list_items("p1", kind="page")) == 2
    assert remove_document(r.document_id, "p2", runtime=runtime, store=store) is False
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is True
    assert store.get_document(r.document_id)["state"] == "tombstoned"
    assert store.list_items("p1", kind="page") == [] and len(store.list_items("p1", kind="page", state="tombstoned")) == 2
    assert sorted(f for _, f in runtime.archived) == ["fact1", "fact2", "fact3"]
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is False   # already gone


def test_hard_removal_is_not_available_yet(store):
    r = submit_document(pdf_input(("a",)), profile_id="p1", actor_id="a", config=CFG, store=store)
    with pytest.raises(NotImplementedError):
        remove_document(r.document_id, "p1", hard=True, runtime=Runtime(), store=store)
