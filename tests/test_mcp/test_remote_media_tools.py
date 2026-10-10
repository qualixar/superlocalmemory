"""The image and document tools for an allowed remote app: what it may do and what stays refused."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from superlocalmemory.cli import daemon
from superlocalmemory.mcp import tools_media
from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_caller, remote_media_allowed

MID = "a" * 32
JOB = "b" * 32
THUMB = b"RIFFxxxxWEBPpixels"
BIG = base64.b64encode(b"x" * (512 * 1024 + 1)).decode()
SMALL = base64.b64encode(b"x" * (512 * 1024)).decode()


class Spy:
    def __init__(self, answer=None):
        self.calls, self.answer = [], answer

    def __call__(self, method, path, body=None, **kw):
        self.calls.append((method, path, body))
        return self.answer


@pytest.fixture
def srv():
    s = SLMFastMCP("t")
    tools_media.register_media_tools(s)
    tools_media.register_document_tools(s)
    tools_media.register_media_resources(s)
    return s


def call(srv, name, args):
    res = asyncio.run(srv.call_tool(name, args))
    if name == "get_media":
        return res
    return json.loads(" ".join(c.text for c in res.content if isinstance(c, TextContent)))


def allowed():
    class Both:
        def __enter__(self):
            self.a, self.b = remote_caller("rk_00000001"), remote_media_allowed(True)
            self.a.__enter__(), self.b.__enter__()

        def __exit__(self, *exc):
            self.b.__exit__(*exc), self.a.__exit__(*exc)

    return Both()


def thumb_answer():
    return {"mime": "image/webp", "base64": base64.b64encode(THUMB).decode()}


def test_a_remote_path_is_refused_with_no_daemon_call(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        for name in ("remember_media", "remember_document"):
            res = call(srv, name, {"path": "/home/me/secret.png"})
            assert res["code"] == "path_not_for_remote"
            assert "Remote apps cannot name a file on this computer" in res["error"]
            assert "secret" not in str(res)
    assert spy.calls == []


@pytest.mark.parametrize("name", ["remember_media", "remember_document"])
def test_remote_base64_over_512_kb_is_refused_and_at_the_limit_goes_through(srv, monkeypatch, name):
    spy = Spy({"status": "stored", "media_id": MID})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        res = call(srv, name, {"base64": BIG})
        assert res["code"] == "too_large_for_remote" and spy.calls == []
        call(srv, name, {"base64": SMALL})
    assert spy.calls[0][2]["origin"] == "remote"


def test_local_calls_have_no_origin_and_no_size_cap(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_media", {"base64": BIG})
    call(srv, "remember_document", {"base64": BIG})
    assert all("origin" not in c[2] for c in spy.calls) and len(spy.calls) == 2


def test_remote_download_url_goes_to_the_daemon_marked_remote(srv, monkeypatch):
    spy = Spy({"status": "stored", "media_id": MID})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        call(srv, "remember_media", {"download_url": "https://files.example/a.png"})
    assert spy.calls[0][2]["origin"] == "remote"
    assert spy.calls[0][2]["download_url"] == "https://files.example/a.png"


def test_remote_status_and_get_media_work_and_thumbnails_are_image_blocks_only(srv, monkeypatch):
    spy = Spy({"status": "done"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        assert call(srv, "media_status", {"job_id": JOB})["status"] == "done"
        monkeypatch.setattr(daemon, "daemon_request", Spy(thumb_answer()))
        res = call(srv, "get_media", {"media_id": MID})
    assert isinstance(res, CallToolResult) or hasattr(res, "content")
    assert any(isinstance(c, ImageContent) for c in res.content)
    structured = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
    assert not structured or "base64" not in json.dumps(structured)


@pytest.mark.parametrize("name,args", [
    ("remember_media", {"download_url": "https://files.example/a.png"}),
    ("remember_document", {"base64": "QUJD"}),
    ("media_status", {"job_id": JOB}),
    ("get_media", {"media_id": MID}),
])
def test_a_remote_caller_without_the_media_flag_is_still_refused(srv, monkeypatch, name, args):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with remote_caller("rk_00000001"):
        with remote_media_allowed(False):
            res = call(srv, name, args)
    assert spy.calls == []
    text = res["error"] if isinstance(res, dict) else " ".join(
        c.text for c in res.content if isinstance(c, TextContent))
    assert "not available to remote apps" in text


def test_the_resources_stay_local_only_even_for_an_allowed_remote_app(srv, monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(thumb_answer()))
    with allowed():
        with pytest.raises(Exception):
            asyncio.run(srv.read_resource(f"slm://media/{MID}/thumb"))


def _payload():
    return {"results": [{"fact_id": "f1", "content": "a cat", "media": {"media_id": MID}},
                        {"fact_id": "f2", "content": "text only"}]}


def test_recall_images_allowed_remote_gets_thumbnails_like_local(monkeypatch):
    monkeypatch.setattr(daemon, "daemon_request", Spy(thumb_answer()))
    with allowed():
        out = asyncio.run(tools_media.with_recall_images(_payload()))
    assert isinstance(out, CallToolResult)
    assert sum(isinstance(c, ImageContent) for c in out.content) == 1


def test_recall_images_other_remote_callers_get_the_media_free_payload(monkeypatch):
    spy = Spy(thumb_answer())
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with remote_caller("rk_00000001"):
        out = asyncio.run(tools_media.with_recall_images(_payload()))
    assert spy.calls == []
    assert [("media" in r) for r in out["results"]] == [False, False]
    assert out["results"][0]["content"] == "a cat"      # the picture-memory text stays
