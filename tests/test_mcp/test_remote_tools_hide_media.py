# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""search, fetch and list_recent do not show a remote app pictures, pages or folder files
it may not see, and show a caller on this computer everything."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.mcp.remote_caller import remote_caller, remote_media_allowed
from superlocalmemory.mcp.tools_core import register_core_tools
from superlocalmemory.media.store import MediaStore

GOOD, DIRTY = "1" * 32, "2" * 32
SOURCES = {
    "zebra picture good": {"type": "media", "media_id": GOOD, "origin": "tool"},
    "zebra picture dirty": {"type": "media", "media_id": DIRTY, "origin": "tool"},
    "zebra folder note": {"type": "folder", "source_id": "s1", "relpath": "a.md"},
    "zebra plain note": None,
}


class _Server:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def world(engine_with_mock_deps):
    engine = engine_with_mock_deps
    ids: dict[str, str] = {}
    for text, source in SOURCES.items():
        fact_id = engine.store(text)[0]
        ids[text] = fact_id
        if source:
            memory_id = engine._db.execute(
                "SELECT memory_id FROM atomic_facts WHERE fact_id = ?", (fact_id,))[0]["memory_id"]
            engine._db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
                               (json.dumps({"_slm_source": source}), memory_id))
    root = Path(engine._db.db_path).parent
    store = MediaStore(root / "media.db")
    for media_id, ok in ((GOOD, 1), (DIRTY, 0)):
        store.insert_item(media_id=media_id, profile_id=engine.profile_id, kind="image",
                          source_sha256="ab" * 32, mime="image/png", bytes=1, origin="tool",
                          remote_ok=ok)
    store.close()
    server = _Server()
    register_core_tools(server, lambda: engine)
    return server, ids


def _run(server, name, **args):
    return asyncio.run(server.tools[name](**args))


def _texts(out) -> set[str]:
    assert out["success"], out
    return {r["content"] for r in out["results"]}


def _as(key: bool, media: bool):
    from contextlib import ExitStack

    stack = ExitStack()
    if key:
        stack.enter_context(remote_caller("rk_00000001"))
        stack.enter_context(remote_media_allowed(media))
    return stack


def test_a_local_caller_sees_everything(world) -> None:
    server, ids = world
    with _as(False, False):
        assert _texts(_run(server, "search", query="zebra")) == set(SOURCES)
        assert _texts(_run(server, "fetch", fact_ids=list(ids.values()))) == set(SOURCES)


def test_a_remote_app_without_the_permission_sees_only_plain_notes(world) -> None:
    server, ids = world
    with _as(True, False):
        assert _texts(_run(server, "search", query="zebra")) == {"zebra plain note"}
        assert _texts(_run(server, "fetch", fact_ids=list(ids.values()))) == {"zebra plain note"}
        recent = _run(server, "list_recent")
        assert {r["content"] for r in recent["results"]} == {"zebra plain note"}


def test_a_remote_app_with_the_permission_sees_vetted_pictures_only(world) -> None:
    server, ids = world
    expect = {"zebra plain note", "zebra picture good"}
    with _as(True, True):
        assert _texts(_run(server, "search", query="zebra")) == expect
        assert _texts(_run(server, "fetch", fact_ids=list(ids.values()))) == expect
        recent = _run(server, "list_recent")
        assert {r["content"] for r in recent["results"]} == expect


def test_a_hidden_fact_is_reported_as_not_found(world) -> None:
    server, ids = world
    with _as(True, False):
        out = _run(server, "fetch", fact_ids=[ids["zebra picture good"], ids["zebra plain note"]])
    assert out["not_found"] == [ids["zebra picture good"]]
