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


class _BusyWriter(Runtime):
    """The real writer under contention: the first ``busy`` sends of each key are only accepted."""

    def __init__(self, busy):
        super().__init__()
        self.busy, self.sends = busy, {}

    def remember(self, admission, actor, *, deadline_ms, accept_after_ms):
        key = admission.idempotency_key
        self.sends[key] = self.sends.get(key, 0) + 1
        if self.sends[key] <= self.busy:
            self.requests.append(admission)
            return SimpleNamespace(payload={"status": "accepted", "operation_id": None, "fact_ids": []})
        return super().remember(admission, actor, deadline_ms=deadline_ms, accept_after_ms=accept_after_ms)


def test_a_queued_save_never_leaves_a_page_without_ids_and_removal_hides_everything(store, tmp_path, monkeypatch):
    """Audit round 2 (CX1): an accepted receipt has no ids; the page must not be recorded empty."""
    from superlocalmemory.memory_core import submit as submit_mod

    monkeypatch.setattr(submit_mod, "SETTLE_WAIT_S", 0.0)
    r = submit_document(pdf_input(("a", "b")), profile_id="p1", actor_id="a", config=CFG, store=store)
    runtime = _BusyWriter(busy=1)
    service = make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40, "b" * 40], title="T"))
    for _ in range(10):
        service.process_next()
        if store.get_document(r.document_id)["state"] == "ready":
            break
    assert store.get_document(r.document_id)["state"] == "ready"
    for page in store.get_pages(r.document_id):
        assert json.loads(page["fact_ids_json"] or "[]"), "a page was recorded without its fact ids"
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is True
    saved = {f for _, f in runtime._by_key.values()}
    assert {f for _, f in runtime.archived} == saved


def test_removing_a_document_while_its_job_finishes_keeps_it_removed_and_hidden(store, tmp_path):
    """Audit round 2 (CX2): removal between the last page and the ready step must win."""
    r = submit_document(pdf_input(("a", "b")), profile_id="p1", actor_id="a", config=CFG, store=store)
    runtime = Runtime()

    def remove_at_finish(admission):
        if admission.idempotency_key.endswith(":doc"):
            assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is True

    runtime.on_remember = remove_at_finish
    make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40, "b" * 40], title="T")).process_next()
    assert store.get_document(r.document_id)["state"] == "tombstoned"
    assert store.job_for_document(r.document_id)["state"] == "cancelled"
    saved = {f for _, f in runtime._by_key.values()}
    assert {f for _, f in runtime.archived} == saved   # the document's own memory too


class _FlakyArchive(Runtime):
    def __init__(self):
        super().__init__()
        self.fail_next = True

    def archive_fact(self, profile_id, fact_id, *, idempotency_key=None):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("writer busy")
        return super().archive_fact(profile_id, fact_id, idempotency_key=idempotency_key)


def test_a_removal_that_could_not_hide_a_memory_says_so_and_can_be_retried(store, tmp_path):
    """Audit round 2 (CX4): no success report while a page is still recallable, and no dead end."""
    r = submit_document(pdf_input(("a",)), profile_id="p1", actor_id="a", config=CFG, store=store)
    runtime = _FlakyArchive()
    make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40], title="T")).process_next()
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is False
    assert store.get_document(r.document_id)["state"] == "ready"
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is True
    assert store.get_document(r.document_id)["state"] == "tombstoned"
    assert {f for _, f in runtime.archived} == {f for _, f in runtime._by_key.values()}
