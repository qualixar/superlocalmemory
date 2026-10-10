"""A body marked ``origin: remote`` only ever makes the daemon stricter."""

from __future__ import annotations

import base64

import pytest

from tests.test_media.test_media_routes import BODY, make

SMALL = base64.b64encode(b"x" * (512 * 1024)).decode()
BIG = base64.b64encode(b"x" * (512 * 1024 + 1)).decode()


def test_origin_remote_fetches_with_the_remote_rules(monkeypatch):
    c, calls = make(monkeypatch)
    body = {"download_url": "https://files.example/a.png", "origin": "remote"}
    assert c.post("/api/v3/media/remember", json=body).status_code == 200
    assert calls[0][0].remote is True


def test_a_local_body_keeps_the_local_rules(monkeypatch):
    c, calls = make(monkeypatch)
    c.post("/api/v3/media/remember", json={"download_url": "https://files.example/a.png"})
    assert calls[0][0].remote is False
    c.post("/api/v3/media/remember", json={**BODY, "origin": "local"})
    assert calls[1][0].remote is False


def test_origin_remote_refuses_a_path_and_a_big_base64_before_saving(monkeypatch):
    c, calls = make(monkeypatch)
    r = c.post("/api/v3/media/remember", json={"path": "/etc/hosts", "origin": "remote"})
    assert r.status_code == 422 and calls == []
    r = c.post("/api/v3/media/remember", json={"base64": BIG, "origin": "remote"})
    assert r.status_code == 422 and calls == []
    assert c.post("/api/v3/media/remember", json={"base64": SMALL, "origin": "remote"}).status_code == 200
    assert c.post("/api/v3/media/remember", json={"base64": BIG}).status_code == 200


def test_documents_origin_remote_is_stricter_only(monkeypatch):
    from tests.test_documents.test_routes import make as make_doc

    c = make_doc(monkeypatch)
    r = c.post("/api/v3/documents", json={"path": "/etc/hosts", "origin": "remote"})
    assert r.status_code == 422 and c.calls == []
    r = c.post("/api/v3/documents", json={"base64": BIG, "origin": "remote"})
    assert r.status_code == 422 and c.calls == []
    assert c.post("/api/v3/documents", json={"base64": BIG}).status_code in (200, 202)


def test_a_remote_link_reaches_the_fetcher_with_the_remote_rules(monkeypatch):
    """Empty host list: the fixed sentence. A listed host: the fetcher runs with remote=True."""
    import httpx

    from superlocalmemory.core import media_fetch
    from superlocalmemory.media import ingest

    seen = {}
    real = media_fetch.fetch_media

    def spy(link, **kw):
        seen["remote"] = kw["remote"]
        return real(link, transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})),
            resolver=lambda host, port: ["93.184.216.34"], **kw)

    monkeypatch.setattr(media_fetch, "fetch_media", spy)
    link = "https://files.example/a.png"
    monkeypatch.delenv(media_fetch.HOSTS_ENV, raising=False)
    with pytest.raises(ingest._Stop) as refused:
        ingest._download(ingest.MediaInput(download_url=link, remote=True))
    assert "until a host list is set" in refused.value.receipt.reason and seen["remote"] is True
    monkeypatch.setenv(media_fetch.HOSTS_ENV, "files.example")
    assert ingest._download(ingest.MediaInput(download_url=link, remote=True)) == b"\x89PNG"


def test_a_file_object_reaches_the_fetcher_with_the_default_file_hosts(monkeypatch):
    """``file_param`` on the input is what makes the fetcher add DEFAULT_FILE_HOSTS; a plain link does not."""
    from superlocalmemory.core import media_fetch
    from superlocalmemory.media import ingest

    seen = []

    def spy(link, **kw):
        seen.append(kw)
        return media_fetch.FetchedMedia(b"\x89PNG", link, "image/png")

    monkeypatch.setattr(media_fetch, "fetch_media", spy)
    link = "https://files.example/a.png"
    ingest._download(ingest.MediaInput(download_url=link, remote=True, file_param=True))
    ingest._download(ingest.MediaInput(download_url=link, remote=True))
    assert seen[0]["file_param"] is True and seen[1]["file_param"] is False


def test_the_images_route_passes_the_file_flag_through(monkeypatch):
    c, calls = make(monkeypatch)
    body = {"download_url": "https://files.example/a.png", "origin": "remote", "from_file": True}
    assert c.post("/api/v3/media/remember", json=body).status_code == 200
    assert calls[0][0].file_param is True and calls[0][0].remote is True
    c.post("/api/v3/media/remember", json={"download_url": "https://files.example/a.png", "origin": "remote"})
    assert calls[1][0].file_param is False
