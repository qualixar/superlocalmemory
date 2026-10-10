"""The image tools for local AI apps: routes used, image blocks, and the remote refusal."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from superlocalmemory.cli import daemon
from superlocalmemory.mcp import tools_media
from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_caller

MID = "a" * 32
THUMB = b"RIFFxxxxWEBPpixels"
LINK = "https://img.example.com/a.png?token=SECRET123&sig=abc"


class Spy:
    def __init__(self, answer=None, exc=None):
        self.calls, self.answer, self.exc = [], answer, exc

    def __call__(self, method, path, body=None, **kw):
        self.calls.append((method, path, body, kw))
        if self.exc:
            raise self.exc
        return self.answer


@pytest.fixture
def srv():
    s = SLMFastMCP("t")
    tools_media.register_media_tools(s)
    tools_media.register_document_tools(s)
    tools_media.register_media_resources(s)
    return s


def run(coro):
    return asyncio.run(coro)


def call(srv, name, args):
    res = run(srv.call_tool(name, args))
    if name in ("remember_media", "remember_document", "media_status"):  # a dict answer arrives as JSON text
        return json.loads(text_of(res))
    return res


def text_of(res) -> str:
    return " ".join(c.text for c in res.content if isinstance(c, TextContent))


def thumb_answer(raw=THUMB):
    return {"mime": "image/webp", "base64": base64.b64encode(raw).decode()}


def test_remember_media_posts_to_the_local_route(srv, monkeypatch):
    spy = Spy({"status": "stored", "media_id": MID, "memory_id": "m1"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, "remember_media", {"download_url": LINK, "content": "a cat", "tags": "pets"})
    method, path, body, _ = spy.calls[0]
    assert (method, path) == ("POST", "/api/v3/media/remember")
    assert body["download_url"] == LINK and body["content"] == "a cat" and body["tags"] == "pets"
    assert "path" not in body and "base64" not in body
    assert res["media_id"] == MID and res["status"] == "stored"
    assert res["resource"] == f"slm://media/{MID}"


def test_remember_media_needs_exactly_one_source_before_any_call(srv, monkeypatch):
    spy = Spy({})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    for args in ({}, {"path": "/a.png", "download_url": LINK}):
        res = call(srv, "remember_media", args)
        assert res["status"] == "refused" and "exactly one" in res["error"]
    assert spy.calls == []


def test_remember_media_error_never_echoes_the_link_query(srv, monkeypatch):
    err = daemon.DaemonUnprocessable("", f"could not read {LINK} today")
    monkeypatch.setattr(daemon, "daemon_request", Spy(exc=err))
    res = call(srv, "remember_media", {"download_url": LINK})
    assert res["status"] == "refused"
    assert "SECRET123" not in str(res) and "token=" not in str(res)


def test_remember_media_daemon_down_is_retryable(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(None))
    res = call(srv, "remember_media", {"base64": "QUJD"})
    assert res["success"] is False and res["retryable"] is True


def test_get_media_gives_an_image_block_and_nothing_in_structured_content(srv, monkeypatch):
    spy = Spy(thumb_answer())
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, "get_media", {"media_id": MID})
    assert isinstance(res, CallToolResult) and not res.is_error
    assert spy.calls[0][0] == "GET"
    assert spy.calls[0][1].startswith(f"/api/v3/media/{MID}/thumb?format=json")
    images = [c for c in res.content if isinstance(c, ImageContent)]
    assert len(images) == 1 and images[0].mime_type == "image/webp"
    assert base64.b64decode(images[0].data) == THUMB
    assert res.structured_content is None
    assert images[0].data not in text_of(res)
    assert f"slm://media/{MID}" in text_of(res)


def test_get_media_refuses_a_bad_id_or_variant_without_a_call(srv, monkeypatch):
    spy = Spy(thumb_answer())
    monkeypatch.setattr(daemon, "daemon_request", spy)
    for args in ({"media_id": "../etc"}, {"media_id": MID, "variant": "original"}):
        res = call(srv, "get_media", args)
        assert res.is_error and not [c for c in res.content if isinstance(c, ImageContent)]
    assert spy.calls == []


def test_get_media_not_found_is_a_plain_error(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(exc=daemon.DaemonNotFound(404, "", "Not found.")))
    res = call(srv, "get_media", {"media_id": MID})
    assert res.is_error and "not found" in text_of(res).lower()


def test_get_media_oversized_thumbnail_is_refused(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(thumb_answer(b"x" * (32 * 1024 + 1))))
    res = call(srv, "get_media", {"media_id": MID})
    assert res.is_error and not [c for c in res.content if isinstance(c, ImageContent)]


@pytest.mark.parametrize("name,args", [
    ("remember_media", {"download_url": LINK}),
    ("remember_media", {"path": "/home/x/secret.png"}),
    ("remember_media", {"base64": "QUJD"}),
    ("get_media", {"media_id": MID}),
])
def test_a_remote_caller_is_refused_with_zero_daemon_calls(srv, monkeypatch, name, args):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with remote_caller("rk_00000001"):
        res = call(srv, name, args)
    assert spy.calls == []
    assert "not available to remote apps" in (res["error"] if isinstance(res, dict) else text_of(res))
    assert "secret.png" not in str(res)


def test_the_media_resources_serve_the_thumbnail_and_refuse_remote(srv, monkeypatch):
    spy = Spy(thumb_answer())
    monkeypatch.setattr(daemon, "daemon_request", spy)
    got = run(srv.read_resource(f"slm://media/{MID}/thumb"))
    assert got[0].content == THUMB and got[0].mime_type == "image/webp"
    card = run(srv.read_resource(f"slm://media/{MID}"))
    assert f"slm://media/{MID}/thumb" in card[0].content
    n = len(spy.calls)
    with remote_caller("rk_00000001"):
        with pytest.raises(Exception):
            run(srv.read_resource(f"slm://media/{MID}/thumb"))
    assert len(spy.calls) == n
    with pytest.raises(Exception):
        run(srv.read_resource("slm://media/not-an-id/thumb"))


def test_remember_media_shows_the_specific_reason_the_route_gave(srv, monkeypatch):
    import io
    import urllib.error

    reason = "That file is not a picture this build can read."
    body = json.dumps({"status": "refused", "reason": reason, "detail": reason}).encode()
    http_err = urllib.error.HTTPError("http://x", 422, "Unprocessable", {}, io.BytesIO(body))
    monkeypatch.setattr(daemon, "daemon_request", Spy(exc=daemon._unprocessable(http_err)))
    res = call(srv, "remember_media", {"base64": "QUJD"})
    assert res["status"] == "refused" and res["error"] == reason


JOB = "b" * 32
DOC = "c" * 32


def test_remember_document_posts_to_the_documents_route(srv, monkeypatch):
    spy = Spy({"status": "processing", "document_id": DOC, "job_id": JOB})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, "remember_document", {"base64": "QUJD", "file_name": "a.pdf", "content": "notes",
                                           "tags": "t", "profile_id": "p2", "idempotency_key": "k1"})
    method, path, body, kw = spy.calls[0]
    assert (method, path) == ("POST", "/api/v3/documents")
    assert body == {"base64": "QUJD", "file_name": "a.pdf", "content": "notes", "tags": "t",
                    "profile_id": "p2", "idempotency_key": "k1"}
    assert kw.get("preserve_unprocessable") is True
    assert (res["document_id"], res["job_id"], res["status"]) == (DOC, JOB, "processing")


def test_remember_document_needs_exactly_one_source_before_any_call(srv, monkeypatch):
    spy = Spy({})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    for args in ({}, {"path": "/a.pdf", "base64": "QUJD"}):
        res = call(srv, "remember_document", args)
        assert res["status"] == "refused" and "exactly one" in res["error"]
    assert spy.calls == []


def test_remember_document_has_no_link_argument(srv):
    names = {t.name: t for t in run(srv.list_tools())}
    assert "download_url" not in names["remember_document"].input_schema["properties"]


def test_remember_document_error_never_echoes_a_link(srv, monkeypatch):
    err = daemon.DaemonUnprocessable("", f"could not read {LINK} today")
    monkeypatch.setattr(daemon, "daemon_request", Spy(exc=err))
    res = call(srv, "remember_document", {"base64": "QUJD"})
    assert res["status"] == "refused"
    assert "SECRET123" not in str(res) and "token=" not in str(res)


def test_remember_document_daemon_down_is_retryable(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(None))
    res = call(srv, "remember_document", {"base64": "QUJD"})
    assert res["success"] is False and res["retryable"] is True


def test_media_status_gets_the_job_and_cleans_its_error(srv, monkeypatch):
    spy = Spy({"job_id": JOB, "state": "failed", "done": 1, "total": 3, "error": f"bad {LINK}"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, "media_status", {"job_id": JOB, "profile_id": "p 2"})
    assert spy.calls[0][0] == "GET" and spy.calls[0][1] == f"/api/v3/jobs/{JOB}?profile_id=p%202"
    assert res["state"] == "failed" and res["total"] == 3
    assert "SECRET123" not in str(res)


def test_media_status_refuses_a_bad_id_without_a_call(srv, monkeypatch):
    spy = Spy({})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    for bad in ("../etc", "A" * 32, "b" * 31, "", f"{JOB}/x"):
        res = call(srv, "media_status", {"job_id": bad})
        assert res["status"] == "refused" and res["code"] == "invalid_request"
    assert spy.calls == []


def test_media_status_unknown_job_is_not_found(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(exc=daemon.DaemonNotFound(404, "", "Not found.")))
    res = call(srv, "media_status", {"job_id": JOB})
    assert res["status"] == "refused" and res["code"] == "not_found"


@pytest.mark.parametrize("name,args", [
    ("remember_document", {"path": "/home/x/secret.pdf"}),
    ("remember_document", {"base64": "QUJD"}),
    ("media_status", {"job_id": JOB}),
    ("media_status", {"job_id": "not-an-id"}),
])
def test_remote_document_calls_are_refused_with_zero_daemon_calls(srv, monkeypatch, name, args):
    spy = Spy({"status": "processing"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with remote_caller("rk_00000001"):
        res = call(srv, name, args)
    assert spy.calls == []
    assert res["code"] == "not_for_remote" and "not available to remote apps" in res["error"]
    assert "secret.pdf" not in str(res)


@pytest.mark.parametrize("tool,url", [("remember_media", "/api/v3/media/remember"),
                                      ("remember_document", "/api/v3/documents")])
def test_save_tools_pass_scope_and_the_sharing_list(srv, monkeypatch, tool, url):
    spy = Spy({"status": "stored", "media_id": MID})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, tool, {"base64": "QUJD", "scope": "shared", "shared_with": "p2, p3"})
    _, path, body, _ = spy.calls[0]
    assert path == url and body["scope"] == "shared" and body["shared_with"] == ["p2", "p3"]
    call(srv, tool, {"base64": "QUJD"})
    assert "scope" not in spy.calls[1][2] and "shared_with" not in spy.calls[1][2]
