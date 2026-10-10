"""The daemon route that saves a finished upload: local callers with credentials, the link's own profile."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.documents.submit import DocumentReceipt
from superlocalmemory.media.ingest import MediaReceipt
from superlocalmemory.media.upload_links import UploadLinks
from superlocalmemory.server import write_governance
from superlocalmemory.server.routes import media_upload as routes
from tests.test_media.test_media_routes import Db, Hooks, Registry

CID, LOCAL, REMOTE = "a" * 32, ("127.0.0.1", 50000), ("10.1.2.3", 50000)
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100
PDF = b"%PDF-1.7\n" + b"0" * 100
NONCE = "n" * 22


def make(monkeypatch, tmp_path, *, kind="image", data=PNG, state="finishing", actor="authenticated:test",
         client=LOCAL, profile="p2", receipt=None, runtime=True, allowed=True, hooks=None, note="a whiteboard"):
    links = UploadLinks(tmp_path)
    minted = links.mint(CID, "key1", profile, kind, note)
    links.accept_chunk(minted.token, CID, 0, len(data), data, NONCE)
    if state == "finishing":
        links.begin_finish(minted.token, CID, NONCE)
    calls = []

    def fake_media(inp, **kw):
        calls.append(("image", inp, kw))
        return receipt or MediaReceipt("stored", media_id="m" * 32, memory_id="mem1")

    def fake_doc(inp, **kw):
        calls.append(("document", inp, kw))
        return receipt or DocumentReceipt("processing", document_id="d" * 32, job_id="j" * 32)

    monkeypatch.setattr(routes, "remember_media", fake_media)
    monkeypatch.setattr(routes, "submit_document", fake_doc)
    monkeypatch.setattr(routes, "default_links", lambda: links)
    monkeypatch.setattr(write_governance, "_registry", Registry(allowed))
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=SimpleNamespace(pii_redaction=False),
                                       _db=Db(), _hooks=hooks or Hooks())
    if runtime:
        app.state.canonical_remember_runtime = object()

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(routes.router)
    row = links.find(minted.token, CID)
    return TestClient(app, client=client), calls, links, row


def url(row):
    return f"/api/v3/media/uploads/{row.upload_id}/finish"


def test_a_finished_image_is_saved_into_the_links_own_profile_with_remote_rules(monkeypatch, tmp_path):
    c, calls, _, row = make(monkeypatch, tmp_path)
    r = c.post(url(row))
    assert r.status_code == 200 and r.json()["status"] == "stored"
    kind, inp, kw = calls[0]
    assert kind == "image" and inp.data == PNG and inp.remote is True and inp.path is None
    assert kw["profile_id"] == "p2" and kw["content"] == "a whiteboard"
    assert kw["scope"] == "personal" and tuple(kw["shared_with"]) == ()
    assert kw["actor_id"] == "authenticated:test" and kw["idempotency_key"] == f"upload:{row.upload_id}"


def test_a_finished_document_goes_to_the_document_pipeline(monkeypatch, tmp_path):
    c, calls, _, row = make(monkeypatch, tmp_path, kind="document", data=PDF)
    r = c.post(url(row))
    assert r.status_code == 202 and r.json()["status"] == "processing"
    kind, inp, kw = calls[0]
    assert kind == "document" and inp.data == PDF and kw["scope"] == "personal"


def test_a_refusal_is_a_422_with_the_reason(monkeypatch, tmp_path):
    c, _, _, row = make(monkeypatch, tmp_path, receipt=MediaReceipt("refused", reason="The image is too blurry."))
    r = c.post(url(row))
    assert r.status_code == 422 and r.json()["detail"] == "The image is too blurry."


def test_only_local_callers_with_credentials_and_only_a_finishing_link(monkeypatch, tmp_path):
    c, calls, _, row = make(monkeypatch, tmp_path, actor="")
    assert c.post(url(row)).status_code == 403
    c, calls, _, row = make(monkeypatch, tmp_path, client=REMOTE)
    assert c.post(url(row)).status_code == 403
    c, calls, _, row = make(monkeypatch, tmp_path, state="receiving")
    assert c.post(url(row)).status_code == 404
    assert c.post("/api/v3/media/uploads/not-an-id/finish").status_code == 404
    assert c.post("/api/v3/media/uploads/" + "f" * 32 + "/finish").status_code == 404
    assert calls == []


def test_the_scratch_file_is_checked_again_before_saving(monkeypatch, tmp_path):
    c, calls, links, row = make(monkeypatch, tmp_path)
    links.temp_path(row.upload_id).write_bytes(PDF)  # swapped after the first chunk was checked
    r = c.post(url(row))
    assert r.status_code == 422 and calls == []
    c, calls, links, row = make(monkeypatch, tmp_path / "b")
    links.temp_path(row.upload_id).write_bytes(PNG + b"extra")  # wrong size
    assert c.post(url(row)).status_code == 422 and calls == []
    c, calls, links, row = make(monkeypatch, tmp_path / "c")
    links.temp_path(row.upload_id).unlink()
    assert c.post(url(row)).status_code == 422 and calls == []


def test_governance_and_profile_checks_apply(monkeypatch, tmp_path):
    c, calls, _, row = make(monkeypatch, tmp_path, allowed=False)
    assert c.post(url(row)).status_code == 403 and calls == []
    c, calls, _, row = make(monkeypatch, tmp_path / "b", hooks=Hooks(exc=RuntimeError("no")))
    assert c.post(url(row)).status_code == 403 and calls == []
    c, calls, _, row = make(monkeypatch, tmp_path / "c", profile="gone")
    assert c.post(url(row)).status_code == 404 and calls == []


def test_the_writer_must_be_ready_for_pictures(monkeypatch, tmp_path):
    c, calls, _, row = make(monkeypatch, tmp_path, runtime=False)
    assert c.post(url(row)).status_code == 503 and calls == []
