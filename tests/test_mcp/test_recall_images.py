"""Recall shows the pictures it found to local apps, and changes nothing else."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import ImageContent, TextContent

from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_caller
from superlocalmemory.mcp.tools_core import register_core_tools

IDS = [f"{n:032x}" for n in range(1, 6)]
THUMB = b"RIFFxxxxWEBPpixels"


def _hit(fid, media_id=None):
    row = {"fact_id": fid, "content": "c", "score": 0.5}
    if media_id:
        row["media"] = {"media_id": media_id, "kind": "image",
                        "thumbnail_uri": f"slm://media/{media_id}/thumb",
                        "page": None, "document_id": None, "citation": ""}
    return row


def _answer(*hits):
    return {"ok": True, "results": list(hits), "result_count": len(hits), "query_type": "q"}


def _pool(answer):
    pool = MagicMock()
    pool.recall.return_value = answer
    return pool


def _recall(answer):
    srv = SLMFastMCP("t")
    register_core_tools(srv, MagicMock())
    with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=_pool(answer)):
        return asyncio.run(srv.call_tool("recall", {"query": "cat"}))


def _baseline(payload):
    """What the framework makes of the same dict returned by a plain ``-> dict`` tool."""
    srv = SLMFastMCP("ref")

    @srv.tool()
    async def recall() -> dict:
        return payload

    return asyncio.run(srv.call_tool("recall", {}))


def _payload(answer):
    res = _recall(answer)
    return json.loads(res.content[0].text)


@pytest.fixture
def tools_media():
    """The module as the recall tool imports it now (other tests may have reloaded it)."""
    import importlib

    return importlib.import_module("superlocalmemory.mcp.tools_media")


@pytest.fixture
def thumbs(monkeypatch, tools_media):
    calls = []

    def fake(media_id, profile_id=""):
        calls.append(media_id)
        return THUMB, ""

    monkeypatch.setattr(tools_media, "thumb_via_daemon", fake)
    return calls


def test_no_media_block_means_the_output_is_exactly_what_it_was(thumbs):
    answer = _answer(_hit("f1"), _hit("f2"))
    got = _recall(answer)
    expected = _baseline(_payload(answer))
    assert got.model_dump_json(by_alias=True) == expected.model_dump_json(by_alias=True)
    assert thumbs == []


def test_a_remote_caller_gets_no_media_key_at_all_and_no_daemon_call(thumbs):
    answer = _answer(_hit("f1", IDS[0]), _hit("f2"), _hit("f3", IDS[1]))
    srv = SLMFastMCP("t")
    register_core_tools(srv, MagicMock())
    with patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=_pool(answer)):
        with remote_caller("key1"):
            got = asyncio.run(srv.call_tool("recall", {"query": "cat"}))
    assert [type(c) for c in got.content] == [TextContent]
    assert "media" not in got.model_dump_json(by_alias=True) and IDS[0] not in got.content[0].text
    pre_media = _answer(_hit("f1"), _hit("f2"), _hit("f3"))
    assert got.model_dump_json(by_alias=True) == _recall(pre_media).model_dump_json(by_alias=True)
    assert thumbs == []


def test_a_remote_caller_without_media_results_gets_the_same_object(thumbs, tools_media):
    import asyncio as aio

    payload = {"success": True, "results": [_hit("f1"), _hit("f2")]}
    with remote_caller("key1"):
        assert aio.run(tools_media.with_recall_images(payload)) is payload
        assert aio.run(tools_media.with_recall_images({"success": True})) is not None
    with remote_caller("key1"):
        out = aio.run(tools_media.with_recall_images({"results": [_hit("a", IDS[0]), _hit("b")]}))
    assert out["results"][1] == _hit("b") and "media" not in out["results"][0]


def test_a_local_caller_still_gets_the_media_key(thumbs):
    got = _recall(_answer(_hit("f1", IDS[0])))
    assert json.loads(got.content[0].text)["results"][0]["media"]["media_id"] == IDS[0]


def test_local_results_with_media_get_up_to_three_image_blocks_after_the_normal_output(thumbs):
    answer = _answer(*[_hit(f"f{n}", IDS[n]) for n in range(4)], _hit("f9", IDS[0]))
    plain = _baseline(_payload(answer))
    thumbs.clear()
    got = _recall(answer)
    assert got.content[0] == plain.content[0]
    images = got.content[1:]
    assert len(images) == 3 and all(isinstance(c, ImageContent) for c in images)
    assert images[0].data == base64.b64encode(THUMB).decode() and images[0].mime_type == "image/webp"
    assert thumbs == IDS[:3]
    assert got.structured_content is None and plain.structured_content is None


def test_thumbnail_data_never_appears_in_structured_content(thumbs):
    got = _recall(_answer(_hit("f1", IDS[0])))
    assert got.structured_content is None
    assert base64.b64encode(THUMB).decode() not in got.content[0].text


def test_a_failed_thumbnail_is_skipped_and_the_recall_still_works(monkeypatch, tools_media):
    def fake(media_id, profile_id=""):
        if media_id == IDS[0]:
            raise RuntimeError("daemon down")
        return (None, "Image not found.") if media_id == IDS[1] else (THUMB, "")

    monkeypatch.setattr(tools_media, "thumb_via_daemon", fake)
    answer = _answer(_hit("f1", IDS[0]), _hit("f2", IDS[1]), _hit("f3", IDS[2]))
    got = _recall(answer)
    assert json.loads(got.content[0].text)["success"] is True
    assert [type(c) for c in got.content] == [TextContent, ImageContent]


def test_every_thumbnail_failing_returns_the_plain_output(monkeypatch, tools_media):
    monkeypatch.setattr(tools_media, "thumb_via_daemon", lambda *a, **k: (None, "gone"))
    answer = _answer(_hit("f1", IDS[0]))
    got = _recall(answer)
    assert got.model_dump_json(by_alias=True) == \
        _baseline(json.loads(got.content[0].text)).model_dump_json(by_alias=True)
