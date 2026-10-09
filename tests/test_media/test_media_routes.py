"""The image routes: local callers only, credentials for writes, one profile's pictures stay its own."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.media import ingest, open_media_store
from superlocalmemory.media.ingest import MediaReceipt
from superlocalmemory.server.routes import media as routes

LOCAL, REMOTE = ("127.0.0.1", 50000), ("10.1.2.3", 50000)
BODY = {"base64": base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode(), "content": "hello", "tags": "a,b"}


class Db:
    def execute(self, sql, params=()):
        return [{"one": 1}] if params and params[0] in ("default", "p2") else []


def make(monkeypatch, *, actor="authenticated:test", client=LOCAL, runtime=True):
    calls = []

    def fake(inp, **kw):
        calls.append((inp, kw))
        return MediaReceipt("stored", media_id="m" * 32, memory_id="mem1")

    monkeypatch.setattr(routes, "remember_media", fake)
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=SimpleNamespace(pii_redaction=False), _db=Db())
    if runtime:
        app.state.canonical_remember_runtime = object()

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(routes.router)
    return TestClient(app, client=client), calls


def test_remember_passes_the_authenticated_actor_and_active_profile(monkeypatch):
    c, calls = make(monkeypatch)
    r = c.post("/api/v3/media/remember", json=BODY)
    assert r.status_code == 200 and r.json()["status"] == "stored" and r.json()["media_id"] == "m" * 32
    inp, kw = calls[0]
    assert inp.base64 == BODY["base64"] and kw["content"] == "hello" and kw["tags"] == "a,b"
    assert kw["actor_id"] == "authenticated:test" and kw["profile_id"] == "default"


def test_a_named_profile_must_exist(monkeypatch):
    c, calls = make(monkeypatch)
    assert c.post("/api/v3/media/remember", json={**BODY, "profile_id": "p2"}).status_code == 200
    assert calls[-1][1]["profile_id"] == "p2"
    assert c.post("/api/v3/media/remember", json={**BODY, "profile_id": "nope"}).status_code == 404


def test_writes_need_credentials(monkeypatch):
    c, calls = make(monkeypatch, actor="")
    assert c.post("/api/v3/media/remember", json=BODY).status_code == 403
    assert calls == []


def test_remote_callers_are_refused_even_with_credentials(monkeypatch):
    c, calls = make(monkeypatch, client=REMOTE)
    assert c.post("/api/v3/media/remember", json=BODY).status_code == 403
    assert c.get("/api/v3/media/" + "a" * 32 + "/thumb").status_code == 403
    assert calls == []


def test_writer_not_ready_is_503_and_status_codes_follow_the_receipt(monkeypatch):
    c, _ = make(monkeypatch, runtime=False)
    assert c.post("/api/v3/media/remember", json=BODY).status_code == 503
    for status, code in (("warming", 202), ("refused", 422), ("duplicate", 200)):
        monkeypatch.setattr(routes, "remember_media", lambda inp, _s=status, **kw: MediaReceipt(_s, reason="r"))
        c2, _ = make(monkeypatch)
        monkeypatch.setattr(routes, "remember_media", lambda inp, _s=status, **kw: MediaReceipt(_s, reason="r"))
        assert c2.post("/api/v3/media/remember", json=BODY).status_code == code


def test_a_body_with_neither_path_nor_data_is_422(monkeypatch):
    c, _ = make(monkeypatch)
    assert c.post("/api/v3/media/remember", json={"content": "x"}).status_code == 422


def test_thumbnail_is_served_only_to_the_owning_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    s = open_media_store(create=True, data_root=tmp_path)
    base = dict(kind="image", source_sha256="a" * 64, mime="image/png", bytes=1, origin="tool",
                thumb_webp=b"RIFFxxxxWEBP")
    mine = s.insert_item(profile_id="default", **base)
    theirs = s.insert_item(profile_id="other", **{**base, "source_sha256": "b" * 64})
    nothumb = s.insert_item(profile_id="default", **{**base, "source_sha256": "c" * 64, "thumb_webp": None})
    s.close()
    c, _ = make(monkeypatch)
    ok = c.get(f"/api/v3/media/{mine}/thumb")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/webp" and ok.content == b"RIFFxxxxWEBP"
    assert c.get(f"/api/v3/media/{theirs}/thumb").status_code == 404
    assert c.get(f"/api/v3/media/{nothumb}/thumb").status_code == 404
    assert c.get("/api/v3/media/" + "d" * 32 + "/thumb").status_code == 404
    assert c.get("/api/v3/media/not-an-id/thumb").status_code == 404


def test_thumbnail_without_a_media_database_is_404(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "empty"))
    c, _ = make(monkeypatch)
    assert c.get("/api/v3/media/" + "a" * 32 + "/thumb").status_code == 404
    assert not (tmp_path / "empty" / "media.db").exists()
