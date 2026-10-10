"""Document routes: local callers, credentials, the same governance as images, one profile's jobs stay its own."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.documents import status as status_mod
from superlocalmemory.documents.submit import DocumentReceipt, submit_document
from superlocalmemory.media import open_media_store
from superlocalmemory.server import write_governance
from superlocalmemory.server.routes import media as routes
from tests.test_documents.support import Runtime, pdf_input

LOCAL, REMOTE = ("127.0.0.1", 50000), ("10.1.2.3", 50000)
BODY = {"base64": base64.b64encode(b"%PDF-1.4 x").decode(), "content": "hello", "tags": "a"}


class Hooks:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def run_pre(self, name, payload):
        self.calls.append((name, payload))
        if self.exc:
            raise self.exc


class Registry:
    def __init__(self, allowed=True):
        self.allowed, self.calls = allowed, []

    def evaluate(self, kind, ctx, mode):
        self.calls.append((kind, ctx, mode))
        return SimpleNamespace(allowed=self.allowed, reason="no")


class Db:
    def execute(self, sql, params=()):
        return [{"one": 1}] if params and params[0] in ("default", "p2") else []


def make(monkeypatch, *, actor="authenticated:test", client=LOCAL, hooks=None, allowed=True, runtime=None):
    calls = []

    def fake(inp, **kw):
        calls.append((inp, kw))
        return DocumentReceipt("processing", document_id="d" * 32, job_id="j" * 32)

    monkeypatch.setattr(routes, "submit_document", fake)
    monkeypatch.setattr(write_governance, "_registry", Registry(allowed))
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=SimpleNamespace(pii_redaction=False),
                                       _db=Db(), _hooks=hooks or Hooks())
    app.state.canonical_remember_runtime = runtime

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(routes.router)
    c = TestClient(app, client=client)
    c.calls = calls
    return c


def test_post_passes_the_actor_profile_and_words_and_answers_202(monkeypatch):
    c = make(monkeypatch)
    r = c.post("/api/v3/documents", json=BODY)
    assert r.status_code == 202 and r.json()["status"] == "processing" and r.json()["job_id"] == "j" * 32
    inp, kw = c.calls[0]
    assert inp.base64 == BODY["base64"] and kw["content"] == "hello" and kw["tags"] == "a"
    assert kw["actor_id"] == "authenticated:test" and kw["profile_id"] == "default"


def test_post_codes_follow_the_receipt(monkeypatch):
    c = make(monkeypatch)
    for status, code in (("duplicate", 200), ("refused", 422), ("processing", 202)):
        monkeypatch.setattr(routes, "submit_document", lambda inp, _s=status, **kw: DocumentReceipt(_s, reason="r"))
        assert c.post("/api/v3/documents", json=BODY).status_code == code


def test_credentials_loopback_and_body_checks(monkeypatch):
    c = make(monkeypatch, actor="")
    assert c.post("/api/v3/documents", json=BODY).status_code == 403 and c.calls == []
    c = make(monkeypatch, client=REMOTE)
    assert c.post("/api/v3/documents", json=BODY).status_code == 403
    assert c.get("/api/v3/jobs/" + "a" * 32).status_code == 403
    assert c.delete("/api/v3/documents/" + "a" * 32).status_code == 403 and c.calls == []
    c = make(monkeypatch)
    assert c.post("/api/v3/documents", json={"content": "x"}).status_code == 422
    assert c.post("/api/v3/documents", json={**BODY, "profile_id": "nope"}).status_code == 404


def test_governance_runs_with_redacted_words_only(monkeypatch):
    hooks = Hooks()
    c = make(monkeypatch, hooks=hooks)
    c.app.state.engine._config = SimpleNamespace(pii_redaction=True)
    r = c.post("/api/v3/documents", json={**BODY, "content": "mail bob@example.com " + "x" * 200})
    assert r.status_code == 202
    name, payload = hooks.calls[0]
    assert name == "store" and payload["content_preview"].startswith("mail [PII:EMAIL]")
    assert write_governance._registry.calls[0][0].name == "REMEMBER"


def test_a_refusing_hook_or_policy_stops_the_submit(monkeypatch):
    c = make(monkeypatch, hooks=Hooks(PermissionError("x")))
    assert c.post("/api/v3/documents", json=BODY).status_code == 403 and c.calls == []
    c = make(monkeypatch, allowed=False)
    assert c.post("/api/v3/documents", json=BODY).status_code == 403 and c.calls == []


def test_job_status_is_profile_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    store = open_media_store(create=True, data_root=tmp_path)
    cfg = SimpleNamespace(pii_redaction=False)
    mine = submit_document(pdf_input(("a",)), profile_id="default", actor_id="a", config=cfg, store=store)
    theirs = submit_document(pdf_input(("b",)), profile_id="p2", actor_id="a", config=cfg, store=store)
    store.close()
    c = make(monkeypatch)
    ok = c.get(f"/api/v3/jobs/{mine.job_id}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["state"] == "queued" and body["document_id"] == mine.document_id and body["document"]["state"] == "processing"
    assert c.get(f"/api/v3/jobs/{theirs.job_id}").status_code == 404
    assert c.get(f"/api/v3/jobs/{theirs.job_id}?profile_id=p2").status_code == 200
    assert c.get("/api/v3/jobs/" + "d" * 32).status_code == 404
    assert c.get("/api/v3/jobs/not-an-id").status_code == 404


def test_delete_is_soft_and_profile_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    store = open_media_store(create=True, data_root=tmp_path)
    cfg = SimpleNamespace(pii_redaction=False)
    mine = submit_document(pdf_input(("a",)), profile_id="default", actor_id="a", config=cfg, store=store)
    theirs = submit_document(pdf_input(("b",)), profile_id="p2", actor_id="a", config=cfg, store=store)
    store.close()
    runtime = Runtime()
    runtime.ready = True
    c = make(monkeypatch, runtime=runtime)
    assert c.delete(f"/api/v3/documents/{theirs.document_id}").status_code == 404
    r = c.delete(f"/api/v3/documents/{mine.document_id}")
    assert r.status_code == 200 and r.json()["removed"] is True
    assert c.delete("/api/v3/documents/not-an-id").status_code == 404
    c2 = make(monkeypatch, runtime=None)
    assert c2.delete(f"/api/v3/documents/{mine.document_id}").status_code == 503


def _document_get_client(monkeypatch, **kw):
    c = make(monkeypatch, **kw)
    seen = []
    monkeypatch.setattr(routes, "document_index", lambda profile, limit, cursor, **k: seen.append(
        (profile, limit, cursor)) or {"documents": [], "next_cursor": None})
    monkeypatch.setattr(routes, "document_lint", lambda profile, **k: seen.append((profile,)) or {"empty_pages": []})
    c.seen = seen
    return c


def test_index_and_lint_routes_are_profile_scoped_reads(monkeypatch):
    c = _document_get_client(monkeypatch)
    r = c.get("/api/v3/documents?cursor=abc&limit=7&profile_id=p2")
    assert r.status_code == 200 and r.json() == {"documents": [], "next_cursor": None}
    assert c.get("/api/v3/documents/lint").json() == {"empty_pages": []}
    assert c.seen == [("p2", 7, "abc"), ("default",)]
    assert c.get("/api/v3/documents?limit=0").status_code == 422
    assert c.get("/api/v3/documents?profile_id=nope").status_code == 404


def test_index_and_lint_are_local_only(monkeypatch):
    c = _document_get_client(monkeypatch, client=REMOTE)
    assert c.get("/api/v3/documents").status_code == 403
    assert c.get("/api/v3/documents/lint").status_code == 403 and c.seen == []


def test_hard_delete_runs_delete_permission_and_forget_policy_then_erases(monkeypatch):
    c = make(monkeypatch)
    calls = []
    monkeypatch.setattr(routes, "remove_document", lambda *a, **k: calls.append((a, k)) or True)
    r = c.delete("/api/v3/documents/" + "a" * 32 + "?hard=true")
    assert r.status_code == 200 and r.json() == {"removed": True, "document_id": "a" * 32, "erased": True}
    (args, kw) = calls[0]
    assert args == ("a" * 32, "default") and kw["hard"] is True and callable(kw["eraser"])
    assert write_governance._registry.calls[0][0].name == "FORGET"


def test_hard_delete_is_refused_by_policy_credentials_and_loopback(monkeypatch):
    calls = []
    for kw in ({"allowed": False}, {"actor": ""}, {"client": REMOTE}):
        c = make(monkeypatch, **kw)
        monkeypatch.setattr(routes, "remove_document", lambda *a, **k: calls.append(1) or True)
        assert c.delete("/api/v3/documents/" + "a" * 32 + "?hard=true").status_code == 403
    assert calls == []


def test_hard_delete_of_an_unknown_or_unerased_document_is_404(monkeypatch):
    c = make(monkeypatch)
    monkeypatch.setattr(routes, "remove_document", lambda *a, **k: False)
    assert c.delete("/api/v3/documents/" + "a" * 32 + "?hard=true").status_code == 404


def test_a_refused_document_receipt_carries_its_reason_as_detail(monkeypatch):
    c = make(monkeypatch)
    monkeypatch.setattr(routes, "submit_document", lambda inp, **kw: DocumentReceipt("refused", reason="Not a PDF."))
    r = c.post("/api/v3/documents", json=BODY)
    assert r.status_code == 422 and r.json()["detail"] == "Not a PDF." and r.json()["reason"] == "Not a PDF."
    monkeypatch.setattr(routes, "submit_document", lambda inp, **kw: DocumentReceipt("processing", job_id="j" * 32))
    assert "detail" not in c.post("/api/v3/documents", json=BODY).json()


LINK = "https://files.example/doc.pdf?sig=SECRET"


def test_a_document_link_is_accepted_and_handed_to_the_submit(monkeypatch):
    c = make(monkeypatch)
    r = c.post("/api/v3/documents", json={"download_url": LINK, "file_name": "Scan.pdf"})
    assert r.status_code == 202
    inp, _ = c.calls[0]
    assert inp.download_url == LINK and inp.file_name == "Scan.pdf"
    assert inp.remote is False and inp.file_param is False and inp.path is None and inp.base64 is None


def test_a_remote_document_link_gets_the_remote_rules_and_the_file_flag(monkeypatch):
    c = make(monkeypatch)
    body = {"download_url": LINK, "origin": "remote", "from_file": True, "scope": "global"}
    assert c.post("/api/v3/documents", json=body).status_code == 202
    inp, kw = c.calls[0]
    assert inp.remote is True and inp.file_param is True
    assert kw["scope"] == "personal" and kw["shared_with"] == ()  # a remote save is pinned to personal


def test_a_document_link_with_another_source_is_refused_before_saving(monkeypatch):
    c = make(monkeypatch)
    assert c.post("/api/v3/documents", json={**BODY, "download_url": LINK}).status_code == 422
    assert c.calls == []


def test_a_remote_document_link_still_cannot_carry_a_path_or_a_big_paste(monkeypatch):
    c = make(monkeypatch)
    assert c.post("/api/v3/documents", json={"path": "/etc/hosts", "origin": "remote"}).status_code == 422
    assert c.calls == []
