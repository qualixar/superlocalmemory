"""``media_upload_link``: a remote app asks for a one-time link; a caller on this computer is pointed at a path."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from mcp.types import TextContent

from superlocalmemory.media.upload_links import UploadLinks
from superlocalmemory.mcp import tools_media_upload as tool_mod
from superlocalmemory.mcp.http_transport import SLMFastMCP
from superlocalmemory.mcp.remote_caller import remote_caller, remote_grant, remote_media_allowed
from superlocalmemory.remote_connections.grant import RemoteGrant
from superlocalmemory.server.remote_keys import RemoteKeyStore

CID = "a" * 32
URL = re.compile(r"https://mcp\.superlocalmemory\.com/u/" + CID + r"/[A-Za-z0-9_-]{43}")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    store = RemoteKeyStore(tmp_path / "remote_keys.json")
    key, _ = store.add("web-" + CID, "write", profile="p2")
    store.set_extras(key.name, ["media"])
    links = UploadLinks(tmp_path)
    monkeypatch.setattr(tool_mod, "default_links", lambda: links)
    monkeypatch.setattr(tool_mod, "_key_store", lambda: store)
    server = SLMFastMCP("t")
    tool_mod.register_upload_link_tool(server)
    return server, store, links, store.list()[0]


def grant(scopes=("slm:read", "slm:write", "slm:media"), cid=CID):
    return RemoteGrant(cid, "auth-1", 1, "chatgpt", frozenset(scopes), False, 1)


def call(server, args):
    res = asyncio.run(server.call_tool("media_upload_link", args))
    return json.loads(" ".join(c.text for c in res.content if isinstance(c, TextContent)))


class Remote:
    def __init__(self, key_id, *, media=True, g=None):
        self.parts = [remote_caller(key_id), remote_media_allowed(media), remote_grant(g or grant())]

    def __enter__(self):
        for part in self.parts:
            part.__enter__()

    def __exit__(self, *exc):
        for part in reversed(self.parts):
            part.__exit__(*exc)


def test_a_remote_app_gets_a_link_bound_to_its_connection_key_and_profile(world):
    server, _, links, key = world
    with Remote(key.key_id):
        out = call(server, {"kind": "image", "note": "the whiteboard"})
    assert out["success"] is True and out["kind"] == "image" and out["max_mb"] == 25
    assert URL.fullmatch(out["url"]) and out["expires_at"].endswith("Z")
    assert out["url"] in out["message"] and "10 minutes" in out["message"] and "once" in out["message"]
    row = links.find(out["url"].rsplit("/", 1)[1], CID)
    assert (row.key_id, row.profile_id, row.kind, row.note) == (key.key_id, "p2", "image", "the whiteboard")


def test_a_document_link_says_document_and_allows_more(world):
    server, _, _, key = world
    with Remote(key.key_id):
        out = call(server, {"kind": "document"})
    assert out["kind"] == "document" and out["max_mb"] == 100 and "document" in out["message"]


def test_the_key_profile_wins_over_an_argument(world):
    server, _, links, key = world
    with Remote(key.key_id):
        out = call(server, {"kind": "image", "profile_id": "someone-else"})
    assert links.find(out["url"].rsplit("/", 1)[1], CID).profile_id == "p2"


def test_a_connector_without_upload_support_is_told_to_update(world, monkeypatch):
    from superlocalmemory.remote_connections import companion

    server, _, links, key = world
    monkeypatch.setattr(companion, "CONNECTOR_FEATURES", ("grant-v1",))
    with Remote(key.key_id):
        out = call(server, {"kind": "image"})
    assert out["success"] is False and out["code"] == "update_required" and "update" in out["error"].lower()
    assert "url" not in out and links.cleanup() == 0


def test_a_local_caller_is_pointed_at_a_path(world):
    server, *_ = world
    out = call(server, {"kind": "image"})
    assert out["success"] is False and out["code"] == "not_for_local" and "path" in out["error"]


@pytest.mark.parametrize("case", ["no_media_flag", "no_write_scope", "no_media_scope", "no_grant",
                                  "other_connection", "read_key", "no_opt_in", "revoked"])
def test_a_remote_caller_that_lacks_any_part_of_the_double_opt_in_is_refused(world, case):
    server, store, links, key = world
    g, media = grant(), True
    if case == "no_media_flag":
        media = False
    elif case == "no_write_scope":
        g = grant(("slm:read", "slm:media"))
    elif case == "no_media_scope":
        g = grant(("slm:read", "slm:write"))
    elif case == "other_connection":
        g = grant(cid="c" * 32)
    elif case == "no_opt_in":
        store.set_extras(key.name, [])
    elif case == "revoked":
        store.revoke(key.name)
    elif case == "read_key":
        store.revoke(key.name)
        read, _ = store.add("web-" + CID, "read", profile="p2")
        store.set_extras(read.name, ["media"])
        key = read
    ctx = Remote(key.key_id, media=media, g=g)
    if case == "no_grant":
        ctx = Remote(key.key_id)
        ctx.parts[2] = remote_grant(None)
    with ctx:
        out = call(server, {"kind": "image"})
    assert out["success"] is False and out["code"] in ("not_for_remote", "not_allowed")
    assert "url" not in out and links.cleanup() == 0


def test_bad_arguments_and_the_open_link_limit_give_plain_refusals(world):
    server, _, _, key = world
    with Remote(key.key_id):
        assert call(server, {"kind": "video"})["code"] == "invalid_kind"
        assert call(server, {"kind": "image", "note": "x" * 2001})["code"] == "note_too_long"
        for _ in range(3):
            assert call(server, {"kind": "image"})["success"] is True
        out = call(server, {"kind": "image"})
    assert out["success"] is False and out["code"] == "too_many_open" and "three" in out["error"]


def test_the_link_survives_the_remote_redaction(world, monkeypatch):
    from superlocalmemory.server import remote_redaction

    monkeypatch.setattr(remote_redaction, "_host_strings", lambda: ("fN", "a", "x_"))
    server, _, _, key = world
    with Remote(key.key_id):
        out = call(server, {"kind": "image"})
    redacted = remote_redaction.redact_value(out)
    assert redacted["url"] == out["url"] and out["url"] in redacted["message"]
