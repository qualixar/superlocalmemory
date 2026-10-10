"""Trying a failed document again by its id: the saved original is reused, nothing is dropped again."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.documents import submit as submit_mod
from superlocalmemory.documents.submit import retry_document, submit_document
from superlocalmemory.media import open_media_store
from superlocalmemory.server import write_governance
from superlocalmemory.server.routes import media as routes
from tests.test_documents.support import pdf_input

CFG = SimpleNamespace(pii_redaction=False)
LOCAL, REMOTE = ("127.0.0.1", 50000), ("10.1.2.3", 50000)


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


def failed_document(store, profile="p1", **kw):
    first = submit_document(pdf_input(("hello",)), content="my words", tags="a,b", profile_id=profile,
                            actor_id="actor-1", config=CFG, store=store, **kw)
    store.update_document(first.document_id, state="failed")
    job = store.claim_job("t", kinds=("document",))
    assert store.finish_job(job["job_id"], "t", "failed", "page_failed")
    return first


def test_a_failed_document_is_queued_again_under_the_same_id_with_its_own_words(store):
    first = failed_document(store)

    again = retry_document(first.document_id, profile_id="p1", actor_id="someone", store=store)

    assert again.status == "processing" and again.document_id == first.document_id
    assert again.job_id and again.job_id != first.job_id
    assert store.get_document(first.document_id)["state"] == "processing"
    payload = json.loads(store.get_job(again.job_id)["payload_json"])
    assert payload["document_id"] == first.document_id
    assert payload["user_words"] == "my words" and payload["tags"] == "a,b"
    assert payload["actor_id"] == "actor-1"


def test_a_second_try_while_the_first_is_queued_does_not_queue_twice(store):
    first = failed_document(store)
    one = retry_document(first.document_id, profile_id="p1", actor_id="a", store=store)

    two = retry_document(first.document_id, profile_id="p1", actor_id="a", store=store)

    assert two.status == "processing" and two.job_id == one.job_id
    assert len([j for j in store.list_jobs("p1") if j["state"] == "queued"]) == 1


def test_another_profiles_document_or_an_unknown_one_is_not_found(store):
    first = failed_document(store)

    assert retry_document(first.document_id, profile_id="p2", actor_id="a", store=store) is None
    assert retry_document("f" * 32, profile_id="p1", actor_id="a", store=store) is None
    assert store.get_document(first.document_id)["state"] == "failed"


def test_a_finished_document_is_left_alone(store):
    first = failed_document(store)
    store.update_document(first.document_id, state="ready")

    again = retry_document(first.document_id, profile_id="p1", actor_id="a", store=store)

    assert again.status == "duplicate" and again.document_id == first.document_id
    assert not [j for j in store.list_jobs("p1") if j["state"] == "queued"]


def test_a_removed_document_is_not_found(store):
    first = failed_document(store)
    store.tombstone_document(first.document_id)

    assert retry_document(first.document_id, profile_id="p1", actor_id="a", store=store) is None


def test_a_missing_original_says_so_in_plain_words_and_stays_failed(store, root):
    first = failed_document(store)
    doc = store.get_document(first.document_id)
    (root / "media" / doc["source_relpath"]).unlink()

    again = retry_document(first.document_id, profile_id="p1", actor_id="a", store=store)

    assert again.status == "refused"
    assert "saved copy" in again.reason and "add the document again" in again.reason.lower()
    assert store.get_document(first.document_id)["state"] == "failed"


def test_with_the_old_job_gone_the_retry_still_works_and_uses_the_caller_as_actor(store):
    first = failed_document(store)
    with store._write() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (first.job_id,))

    again = retry_document(first.document_id, profile_id="p1", actor_id="owner-9", store=store)

    assert again.status == "processing"
    payload = json.loads(store.get_job(again.job_id)["payload_json"])
    assert payload["actor_id"] == "owner-9" and payload["document_id"] == first.document_id


def test_a_failed_queue_write_puts_the_document_back_to_failed(store, monkeypatch):
    first = failed_document(store)
    monkeypatch.setattr(submit_mod, "_queue", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))

    again = retry_document(first.document_id, profile_id="p1", actor_id="a", store=store)

    assert again.status == "refused"
    assert store.get_document(first.document_id)["state"] == "failed"


# -- the route ------------------------------------------------------------------------------

class Registry:
    def __init__(self, allowed=True):
        self.allowed = allowed

    def evaluate(self, kind, ctx, mode):
        return SimpleNamespace(allowed=self.allowed, reason="no")


class Db:
    def execute(self, sql, params=()):
        return [{"one": 1}] if params and params[0] in ("default", "p2") else []


class Hooks:
    def run_pre(self, name, payload):
        pass


def client(monkeypatch, *, actor="authenticated:test", host=LOCAL, allowed=True, found=None):
    calls = []

    def fake(document_id, **kw):
        calls.append((document_id, kw))
        return found

    monkeypatch.setattr(routes, "retry_document", fake)
    monkeypatch.setattr(write_governance, "_registry", Registry(allowed))
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=CFG, _db=Db(), _hooks=Hooks())

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(routes.router)
    c = TestClient(app, client=host)
    c.calls = calls
    return c


ID = "a" * 32


def test_the_route_retries_by_id_for_the_callers_profile(monkeypatch):
    receipt = submit_mod.DocumentReceipt("processing", document_id=ID, job_id="j" * 32)
    c = client(monkeypatch, found=receipt)

    r = c.post(f"/api/v3/documents/{ID}/retry")

    assert r.status_code == 202
    assert r.json()["status"] == "processing" and r.json()["document_id"] == ID
    assert c.calls[0][0] == ID
    assert c.calls[0][1]["profile_id"] == "default" and c.calls[0][1]["actor_id"] == "authenticated:test"


def test_the_route_answers_404_for_an_unknown_or_malformed_id(monkeypatch):
    c = client(monkeypatch, found=None)

    assert c.post(f"/api/v3/documents/{ID}/retry").status_code == 404
    assert c.post("/api/v3/documents/not-an-id/retry").status_code == 404


def test_the_route_needs_credentials_loopback_policy_and_a_known_profile(monkeypatch):
    receipt = submit_mod.DocumentReceipt("processing", document_id=ID, job_id="j" * 32)
    for kw in ({"actor": ""}, {"host": REMOTE}, {"allowed": False}):
        c = client(monkeypatch, found=receipt, **kw)
        assert c.post(f"/api/v3/documents/{ID}/retry").status_code == 403
        assert c.calls == []
    c = client(monkeypatch, found=receipt)
    assert c.post(f"/api/v3/documents/{ID}/retry?profile_id=nope").status_code == 404
    assert c.calls == []


def test_the_route_codes_follow_the_receipt(monkeypatch):
    for status, code in (("duplicate", 200), ("refused", 422), ("processing", 202)):
        c = client(monkeypatch, found=submit_mod.DocumentReceipt(status, document_id=ID, reason="because"))
        r = c.post(f"/api/v3/documents/{ID}/retry")
        assert r.status_code == code
    c = client(monkeypatch, found=submit_mod.DocumentReceipt("refused", reason="because"))
    assert c.post(f"/api/v3/documents/{ID}/retry").json()["detail"] == "because"
