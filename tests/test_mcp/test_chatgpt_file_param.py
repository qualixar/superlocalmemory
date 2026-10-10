# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""ChatGPT attachments: the ``file`` argument and the ``openai/fileParams`` marker on the two saving tools."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.types import TextContent

from superlocalmemory.cli import daemon
from superlocalmemory.mcp import tools_media
from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_caller, remote_media_allowed

FILE = {"download_url": "https://files.chat.example/file-1?sig=SECRET123", "file_id": "file_abc",
        "mime_type": "image/png", "file_name": "cat.png"}
PDF = {"download_url": "https://files.chat.example/file-2?sig=SECRET123", "file_id": "file_def",
       "mime_type": "application/pdf", "file_name": "Report.pdf"}
TOOLS = ("remember_media", "remember_document")


class Spy:
    def __init__(self, answer=None):
        self.calls, self.answer = [], answer

    def __call__(self, method, path, body=None, **kw):
        self.calls.append((method, path, body, kw))
        return self.answer


@pytest.fixture
def srv():
    s = SLMFastMCP("t")
    tools_media.register_media_tools(s)
    tools_media.register_document_tools(s)
    return s


def call(srv, name, args):
    res = asyncio.run(srv.call_tool(name, args))
    return json.loads(" ".join(c.text for c in res.content if isinstance(c, TextContent)))


def listed(srv):
    return {t.name: t for t in asyncio.run(srv.list_tools())}


def allowed():
    class Both:
        def __enter__(self):
            self.a, self.b = remote_caller("rk_00000001"), remote_media_allowed(True)
            self.a.__enter__(), self.b.__enter__()

        def __exit__(self, *exc):
            self.b.__exit__(*exc), self.a.__exit__(*exc)

    return Both()


# -- the descriptor ChatGPT reads ---------------------------------------------------------

@pytest.mark.parametrize("name", TOOLS)
def test_the_descriptor_declares_the_file_param(srv, name):
    dumped = listed(srv)[name].model_dump(by_alias=True, exclude_none=True)
    assert dumped["_meta"]["openai/fileParams"] == ["file"]


@pytest.mark.parametrize("name", TOOLS)
def test_the_file_schema_declares_all_four_properties_and_two_are_required(srv, name):
    schema = listed(srv)[name].input_schema
    file_schema = schema["properties"]["file"]
    assert "$ref" not in json.dumps(file_schema)  # inline: the host reads the property schema directly
    options = file_schema.get("anyOf")
    obj = next(o for o in options if o.get("type") == "object") if options else file_schema
    assert obj["type"] == "object"
    assert set(obj["properties"]) == {"download_url", "file_id", "mime_type", "file_name"}
    assert set(obj["required"]) == {"download_url", "file_id"}
    assert all(obj["properties"][k]["type"] == "string" for k in obj["properties"])
    assert "file" not in schema.get("required", [])


@pytest.mark.parametrize("name", TOOLS)
def test_the_other_arguments_are_still_there_for_other_hosts(srv, name):
    props = listed(srv)[name].input_schema["properties"]
    assert {"path", "base64", "download_url", "content", "tags", "scope"} <= set(props)


def test_other_tools_do_not_claim_file_params(srv):
    for name in ("get_media", "media_status"):
        assert "_meta" not in listed(srv)[name].model_dump(by_alias=True, exclude_none=True) or \
            "openai/fileParams" not in (listed(srv)[name].meta or {})


# -- using it ------------------------------------------------------------------------------

def test_an_attached_picture_goes_to_the_daemon_as_a_link_marked_from_a_file(srv, monkeypatch):
    spy = Spy({"status": "stored", "media_id": "a" * 32})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, "remember_media", {"file": FILE, "content": "my cat"})
    _, path, body, _ = spy.calls[0]
    assert path == "/api/v3/media/remember" and res["status"] == "stored"
    assert body["download_url"] == FILE["download_url"] and body["from_file"] is True
    assert "file" not in body and "file_id" not in str(body) and body["content"] == "my cat"


def test_an_attached_document_goes_to_the_daemon_with_its_file_name(srv, monkeypatch):
    spy = Spy({"status": "processing", "document_id": "d" * 32, "job_id": "j" * 32})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_document", {"file": PDF})
    _, path, body, _ = spy.calls[0]
    assert path == "/api/v3/documents"
    assert body["download_url"] == PDF["download_url"] and body["from_file"] is True
    assert body["file_name"] == "Report.pdf"


def test_a_given_file_name_wins_over_the_attachments(srv, monkeypatch):
    spy = Spy({"status": "processing"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_document", {"file": PDF, "file_name": "Mine.pdf"})
    assert spy.calls[0][2]["file_name"] == "Mine.pdf"


def test_a_typed_download_url_is_not_marked_from_a_file(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_media", {"download_url": FILE["download_url"]})
    call(srv, "remember_document", {"download_url": PDF["download_url"]})
    assert all("from_file" not in c[2] for c in spy.calls) and len(spy.calls) == 2


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("other", [{"path": "/a.png"}, {"base64": "QUJD"}, {"download_url": "https://x.example/a"}])
def test_a_file_plus_another_source_is_refused_with_no_call(srv, monkeypatch, name, other):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    res = call(srv, name, {"file": FILE, **other})
    assert res["status"] == "refused" and "exactly one" in res["error"] and "file" in res["error"]
    assert spy.calls == []


@pytest.mark.parametrize("bad", [
    {}, {"download_url": FILE["download_url"]}, {"file_id": "f"}, {"download_url": "", "file_id": "f"},
    {"download_url": FILE["download_url"], "file_id": ""},
])
def test_a_file_without_both_required_fields_is_refused_with_no_call(srv, monkeypatch, bad):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    for name in TOOLS:
        res = call(srv, name, {"file": bad})
        assert res["status"] == "refused"
        assert "SECRET123" not in str(res)
    assert spy.calls == []


def test_a_non_object_file_is_not_a_tool_crash(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with pytest.raises(Exception):
        asyncio.run(srv.call_tool("remember_media", {"file": "https://x.example/a.png"}))
    assert spy.calls == []


def test_a_file_with_extra_fields_is_accepted_and_the_extras_are_dropped(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_media", {"file": {**FILE, "size": 99, "other": {"x": 1}}})
    assert "other" not in str(spy.calls[0][2]) and spy.calls[0][2]["download_url"] == FILE["download_url"]


# -- remote rules stay ----------------------------------------------------------------------

@pytest.mark.parametrize("name,file", [("remember_media", FILE), ("remember_document", PDF)])
def test_an_allowed_remote_app_may_save_an_attachment_marked_remote(srv, monkeypatch, name, file):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        call(srv, name, {"file": file})
    body = spy.calls[0][2]
    assert body["origin"] == "remote" and body["from_file"] is True and body["download_url"] == file["download_url"]


@pytest.mark.parametrize("name,file", [("remember_media", FILE), ("remember_document", PDF)])
def test_a_remote_app_without_the_media_flag_is_refused_before_any_call(srv, monkeypatch, name, file):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with remote_caller("rk_00000001"):
        with remote_media_allowed(False):
            res = call(srv, name, {"file": file})
    assert spy.calls == [] and "not available to remote apps" in res["error"]


def test_a_remote_app_still_cannot_name_a_path_next_to_a_file(srv, monkeypatch):
    spy = Spy({"status": "stored"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    with allowed():
        res = call(srv, "remember_media", {"file": FILE, "path": "/home/me/x.png"})
    assert spy.calls == [] and res["status"] == "refused"


def test_a_document_link_waits_longer_for_the_daemon_than_a_short_call(srv, monkeypatch):
    spy = Spy({"status": "processing"})
    monkeypatch.setattr(daemon, "daemon_request", spy)
    call(srv, "remember_document", {"file": PDF})
    call(srv, "remember_document", {"base64": "QUJD"})
    assert spy.calls[0][3]["timeout_seconds"] > spy.calls[1][3]["timeout_seconds"]
