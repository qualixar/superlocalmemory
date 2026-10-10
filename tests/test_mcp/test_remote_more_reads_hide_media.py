# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The remaining memory-reading tools show a remote app no picture, page or folder file it
may not see, and a caller on this computer everything: recall_trace, prestage_context,
run_view, get_lifecycle_status, review_memory_kinds, get_memory_summary, list_corrections."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.mcp.remote_caller import remote_caller, remote_media_allowed
from superlocalmemory.media.store import MediaStore

GOOD, DIRTY = "1" * 32, "2" * 32
SOURCES = {
    "zebra picture good": {"type": "media", "media_id": GOOD, "origin": "tool"},
    "zebra picture dirty": {"type": "media", "media_id": DIRTY, "origin": "tool"},
    "zebra folder note": {"type": "folder", "source_id": "s1", "relpath": "a.md"},
    "zebra plain note": None,
}
EVERYTHING = set(SOURCES)
PLAIN = {"zebra plain note"}
VETTED = {"zebra plain note", "zebra picture good"}


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
    store = MediaStore(Path(engine._db.db_path).parent / "media.db")
    for media_id, ok in ((GOOD, 1), (DIRTY, 0)):
        store.insert_item(media_id=media_id, profile_id=engine.profile_id, kind="image",
                          source_sha256="ab" * 32, mime="image/png", bytes=1, origin="tool",
                          remote_ok=ok)
    store.close()
    return engine, ids


def _as(key: bool, media: bool):
    stack = ExitStack()
    if key:
        stack.enter_context(remote_caller("rk_00000001"))
        stack.enter_context(remote_media_allowed(media))
    return stack


CALLERS = [("local", False, False, EVERYTHING), ("remote", True, False, PLAIN),
           ("remote_media", True, True, VETTED)]
CASES = pytest.mark.parametrize("who,key,media,expect", CALLERS, ids=[c[0] for c in CALLERS])


def _daemon(engine) -> TestClient:
    """The real daemon routes over this engine, reached the way the tools reach them."""
    from superlocalmemory.server.unified_daemon import create_app

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    return TestClient(app)


def _relay(monkeypatch, client: TestClient) -> list[str]:
    paths: list[str] = []

    def request(method, path, body=None, **kw):
        paths.append(path)
        res = client.request(method, path, json=body)
        return res.json() if res.status_code == 200 else None

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda *a, **k: True)
    return paths


# -- recall_trace ---------------------------------------------------------------------------


@CASES
def test_recall_trace(world, monkeypatch, who, key, media, expect) -> None:
    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.mcp.tools_v3 import register_v3_tools

    engine, _ = world
    _relay(monkeypatch, _daemon(engine))
    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _daemon_proxy.DaemonPoolProxy(port=1))
    server = _Server()
    register_v3_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["recall_trace"](query="zebra", limit=10))
    assert out["success"], out
    assert {r["content"] for r in out["results"]} == expect


# -- prestage_context -----------------------------------------------------------------------


@CASES
def test_prestage_context(world, monkeypatch, who, key, media, expect) -> None:
    from superlocalmemory.mcp import server as mcp_server
    from superlocalmemory.mcp.tools_context import register_prestage_tool

    engine, _ = world
    monkeypatch.setattr(mcp_server, "get_engine", lambda: engine)
    server = _Server()
    register_prestage_tool(server, mcp_server._prestage_recall)
    with _as(key, media):
        out = asyncio.run(server.tools["prestage_context"](query="zebra", limit=10))
    assert {m["text"] for m in out["memories"]} == expect, out


# -- get_lifecycle_status -------------------------------------------------------------------


@CASES
def test_get_lifecycle_status(world, who, key, media, expect) -> None:
    from superlocalmemory.mcp.tools_v28 import register_v28_tools

    engine, _ = world
    server = _Server()
    register_v28_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["get_lifecycle_status"](limit=50))
    assert out["success"], out
    shown = {s["content"] for state in out["samples"].values() for s in state}
    assert shown == expect


# -- review_memory_kinds --------------------------------------------------------------------


@CASES
def test_review_memory_kinds(world, monkeypatch, who, key, media, expect) -> None:
    from superlocalmemory.mcp.tools_kinds import register_kind_tools

    engine, ids = world
    for fact_id in ids.values():
        engine._db.execute(
            "UPDATE atomic_facts SET memory_kind = 'fact', memory_kind_source = 'rules',"
            " memory_kind_confidence = NULL WHERE fact_id = ?", (fact_id,))
    _relay(monkeypatch, _daemon(engine))
    server = _Server()
    register_kind_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["review_memory_kinds"](limit=50))
    wanted = {ids[t] for t in expect}
    assert {i["fact_id"] for i in out["items"]} == wanted, out
    assert {i["content_preview"] for i in out["items"]} == expect


# -- get_memory_summary ---------------------------------------------------------------------


@CASES
def test_get_memory_summary_day(world, monkeypatch, who, key, media, expect) -> None:
    from superlocalmemory.mcp import tools_summaries

    engine, ids = world
    monkeypatch.setattr(tools_summaries, "state_path", lambda name: Path(engine._db.db_path))
    day = engine._db.execute("SELECT created_at FROM atomic_facts LIMIT 1")[0]["created_at"][:10]
    server = _Server()
    tools_summaries.register_summary_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["get_memory_summary"](kind="day", target=day))
    assert out["success"], out
    assert set(out["source_fact_ids"]) == {ids[t] for t in expect}
    for text in EVERYTHING - expect:
        assert text not in out["summary"]


# -- list_corrections -----------------------------------------------------------------------


def _cases(ids: dict[str, str]) -> dict[str, tuple[str, str]]:
    plain, good = ids["zebra plain note"], ids["zebra picture good"]
    return {"c-plain": (plain, plain), "c-good": (plain, good),
            "c-dirty": (ids["zebra picture dirty"], plain),
            "c-folder": (plain, ids["zebra folder note"])}


CORRECTIONS = {"local": {"c-plain", "c-good", "c-dirty", "c-folder"},
               "remote": {"c-plain"}, "remote_media": {"c-plain", "c-good"}}


@CASES
def test_list_corrections(world, monkeypatch, who, key, media, expect) -> None:
    from types import SimpleNamespace

    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.server.routes import memories, overtaken

    engine, ids = world
    cases = _cases(ids)
    row = {k: None for k in ("profile_id", "scope", "reason_code", "status", "version", "created_at",
                             "updated_at", "reviewed_at", "applied_at", "system_effective_at",
                             "event_valid_from", "event_valid_until")}
    monkeypatch.setattr(memories, "_correction_store_for", lambda *a: SimpleNamespace(
        list_cases=lambda profile, limit: [
            SimpleNamespace(case_id=c, predecessor_fact_id=p, successor_fact_id=s, **row)
            for c, (p, s) in cases.items()]))
    monkeypatch.setattr(memories, "_authorize_memory_mutation",
                        lambda *a, **k: (engine, engine.profile_id, {}))
    monkeypatch.setattr(overtaken, "overtaken_for", lambda *a, **k: [
        {"case_id": "o-" + c, "predecessor_fact_id": p, "successor_fact_id": s}
        for c, (p, s) in cases.items()])
    _relay(monkeypatch, _daemon(engine))
    server = _Server()
    register_core_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["list_corrections"](limit=50))
    assert out["success"], out
    assert {c["case_id"] for c in out["corrections"]} == CORRECTIONS[who]
    assert {c["case_id"] for c in out["overtaken"]} == {"o-" + c for c in CORRECTIONS[who]}
    if who != "local":
        hidden = {ids[t] for t in EVERYTHING - expect}
        assert not hidden & {v for c in out["corrections"] + out["overtaken"]
                             for v in (c["predecessor_fact_id"], c["successor_fact_id"])}


# -- run_view -------------------------------------------------------------------------------


@CASES
def test_run_view(world, monkeypatch, tmp_path, who, key, media, expect) -> None:
    from superlocalmemory.mcp import tools_views
    from superlocalmemory.server.routes import views as routes
    from superlocalmemory.views import ViewStore

    from ..test_views._store import learning_db

    engine, _ = world
    data = tmp_path / "views-data"
    monkeypatch.setenv("SLM_DATA_DIR", str(data))
    learning_db(data)
    ViewStore(data / "learning.db").create(engine.profile_id, name="Z", query="zebra", limit=10)

    async def runtime_profile(_get_engine, explicit=""):
        return engine.profile_id
    monkeypatch.setattr("superlocalmemory.mcp.tools_core._runtime_profile", runtime_profile)
    monkeypatch.setattr(routes, "_profile", lambda: engine.profile_id)
    app = FastAPI()
    app.include_router(routes.router)
    app.state.engine = engine
    app.state.config = engine._config
    paths = _relay(monkeypatch, TestClient(app))
    server = _Server()
    tools_views.register_view_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["run_view"](name="Z"))
    assert out["success"], out
    assert {r["content"] for r in out["results"]} == expect, out
    if who == "local":  # byte-identical request: nothing was added to it
        assert paths == ["/api/v3/views/run?name=Z&via=mcp"]


# -- the other summary kinds ----------------------------------------------------------------


def _summary_world(world, monkeypatch):
    from superlocalmemory.mcp import tools_summaries

    engine, ids = world
    monkeypatch.setattr(tools_summaries, "state_path", lambda name: Path(engine._db.db_path))
    for fact_id in ids.values():
        engine._db.execute("UPDATE atomic_facts SET session_id = 's-1' WHERE fact_id = ?", (fact_id,))
        engine._db.execute(
            "UPDATE memories SET metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.project', 'zoo')"
            " WHERE memory_id = (SELECT memory_id FROM atomic_facts WHERE fact_id = ?)", (fact_id,))
    server = _Server()
    tools_summaries.register_summary_tools(server, lambda: engine)
    return server, ids


@CASES
def test_get_memory_summary_session_and_project(world, monkeypatch, who, key, media, expect) -> None:
    server, ids = _summary_world(world, monkeypatch)
    with _as(key, media):
        by_session = asyncio.run(server.tools["get_memory_summary"](kind="session", target="s-1"))
        by_project = asyncio.run(server.tools["get_memory_summary"](kind="project", target="zoo"))
    for out in (by_session, by_project):
        assert out["success"], out
        assert set(out["source_fact_ids"]) == {ids[t] for t in expect}
        for text in EVERYTHING - expect:
            assert text not in out["summary"]


@CASES
def test_a_session_list_is_not_offered_to_a_remote_app(world, monkeypatch, who, key, media, expect) -> None:
    server, _ = _summary_world(world, monkeypatch)
    with _as(key, media):
        out = asyncio.run(server.tools["get_memory_summary"](kind="session"))
    assert bool(out["recent_sessions"]) is (who == "local")


@CASES
def test_a_community_with_a_hidden_member_is_not_found(world, monkeypatch, who, key, media, expect) -> None:
    from superlocalmemory.core.community_summary import CommunitySummaryBuilder

    server, ids = _summary_world(world, monkeypatch)
    engine, _ = world
    monkeypatch.setattr(CommunitySummaryBuilder, "get_summary", lambda self, pid, cid: {
        "summary": "zebra picture dirty is here", "keywords": "", "fact_count": 2,
        "fact_ids_json": json.dumps([ids["zebra plain note"], ids["zebra picture dirty"]])})
    with _as(key, media):
        out = asyncio.run(server.tools["get_memory_summary"](kind="community", target="1"))
    assert out["success"] is (who == "local"), out
    if who != "local":
        assert "zebra picture dirty" not in json.dumps(out)
