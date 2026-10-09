"""Handing a PDF over: checks, dedup, quota, where the original goes, what the job carries."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

from superlocalmemory.documents import submit as submit_mod
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from superlocalmemory.media.ingest import MediaInput
from tests.test_documents.pdfs import make_pdf
from tests.test_documents.support import KEY, pdf_input

CFG = SimpleNamespace(pii_redaction=False)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


@pytest.fixture()
def store(root):
    s = open_media_store(create=True, data_root=root)
    yield s
    s.close()


def go(store, inp=None, **kw):
    kw.setdefault("profile_id", "p1")
    return submit_document(inp or pdf_input(("hello",)), actor_id="actor-1", config=CFG, store=store, **kw)


def test_receipt_row_job_and_original(store, root):
    data = make_pdf(["hello"])
    r = go(store, pdf_input(("hello",), file_name="Report.pdf"), content="my words", tags="a,b",
           session_date="2024-02-01")
    sha = hashlib.sha256(data).hexdigest()
    assert r.status == "processing" and r.document_id and r.job_id
    doc = store.get_document(r.document_id)
    assert doc["state"] == "processing" and doc["sha256"] == sha and doc["profile_id"] == "p1"
    assert doc["mime"] == "application/pdf" and doc["bytes"] == len(data) and doc["title"] == "Report"
    assert doc["source_relpath"] == f"{sha[:2]}/{sha}.pdf"
    orig = root / "media" / doc["source_relpath"]
    assert orig.read_bytes() == data and stat.S_IMODE(orig.stat().st_mode) == 0o600
    job = store.get_job(r.job_id)
    assert job["kind"] == "document" and job["state"] == "queued" and job["profile_id"] == "p1"
    payload = json.loads(job["payload_json"])
    assert payload["document_id"] == r.document_id and payload["user_words"] == "my words"
    assert payload["tags"] == "a,b" and payload["session_date"] == "2024-02-01" and payload["actor_id"] == "actor-1"
    assert not list((root / "media" / "tmp").iterdir())


def test_words_are_stored_prepared_never_raw(store):
    r = submit_document(pdf_input(("x",)), content="mail bob@example.com", profile_id="p1", actor_id="a",
                        config=SimpleNamespace(pii_redaction=True), store=store)
    payload = json.loads(store.get_job(r.job_id)["payload_json"])
    assert "@" not in payload["user_words"]
    r2 = go(store, pdf_input(("y",)), content=f"key {KEY}")
    assert KEY in json.loads(store.get_job(r2.job_id)["payload_json"])["user_words"]


def test_same_pdf_twice_is_a_duplicate_until_removed(store):
    first = go(store)
    again = go(store)
    assert again.status == "duplicate" and again.document_id == first.document_id and again.job_id == first.job_id
    assert len(store.list_jobs("p1")) == 1
    assert go(store, profile_id="p2").status == "processing"
    store.tombstone_document(first.document_id)
    third = go(store)
    assert third.status == "processing" and third.document_id != first.document_id


def test_a_failed_document_is_retried_under_the_same_id(store):
    first = go(store)
    store.update_document(first.document_id, state="failed")
    again = go(store)
    assert again.status == "processing" and again.document_id == first.document_id and again.job_id != first.job_id
    assert store.get_document(first.document_id)["state"] == "processing"


def test_same_key_is_the_same_document(store):
    a = go(store, pdf_input(("one",)), idempotency_key="k1")
    b = go(store, pdf_input(("one",)), idempotency_key="k1")
    assert b.status == "duplicate" and b.document_id == a.document_id


@pytest.mark.parametrize("data,reason", [(b"\x89PNG\r\n\x1a\n" + b"x" * 30, "image"), (b"", "empty"),
                                         (b"hello there, not a pdf", "PDF")])
def test_only_pdfs_are_accepted(store, data, reason):
    r = go(store, MediaInput(base64=base64.b64encode(data).decode()))
    assert r.status == "refused" and reason in r.reason and store.list_jobs("p1") == []


def test_input_problems_are_refused(store, tmp_path, monkeypatch):
    assert go(store, MediaInput()).status == "refused"
    assert go(store, MediaInput(base64="not base64 !!")).status == "refused"
    assert go(store, MediaInput(path=tmp_path / "missing.pdf")).status == "refused"
    assert go(store, MediaInput(path=tmp_path)).status == "refused"
    monkeypatch.setattr(submit_mod, "MAX_BASE64_BYTES", 10)
    assert "too large" in go(store).reason
    monkeypatch.setenv("SLM_DOC_MAX_MB", "0.0001")
    big = tmp_path / "big.pdf"
    big.write_bytes(make_pdf(["x" * 500]))
    assert "too large" in go(store, MediaInput(path=big)).reason


def test_a_path_is_read_and_filed_by_name_only(store, tmp_path):
    f = tmp_path / "My Notes.pdf"
    f.write_bytes(make_pdf(["x"]))
    r = go(store, MediaInput(path=f))
    assert r.status == "processing" and store.get_document(r.document_id)["title"] == "My Notes"


def test_the_library_quota_is_enforced(store, monkeypatch):
    monkeypatch.setattr(submit_mod, "QUOTA_BYTES", 100)
    r = go(store)
    assert r.status == "refused" and "full" in r.reason and store.list_jobs("p1") == []


def test_feature_off_refuses_and_creates_nothing(root):
    r = submit_document(pdf_input(("x",)), profile_id="p1", actor_id="a", config=CFG)
    assert r.status == "refused" and "turned off" in r.reason
    assert not (root / "media.db").exists() and not (root / "media").exists()


def test_a_failed_insert_removes_a_new_original(store, root, monkeypatch):
    monkeypatch.setattr(store, "insert_document", lambda **kw: (_ for _ in ()).throw(RuntimeError("disk")))
    r = go(store)
    assert r.status == "refused"
    assert not list((root / "media").glob("*/*.pdf"))
