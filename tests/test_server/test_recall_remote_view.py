# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A remote key's recall reaches the daemon marked as remote, and the daemon recalls
under the matching visibility rules. A local recall carries and sets nothing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy
from superlocalmemory.mcp.remote_caller import remote_caller, remote_media_allowed
from superlocalmemory.retrieval import visibility


def _sent_query(**kw) -> dict:
    seen: dict = {}

    def fake(method, path, *a, **k):
        seen["path"] = path
        return {"ok": True, "results": []}

    with patch("superlocalmemory.cli.daemon.daemon_request", fake):
        DaemonPoolProxy(port=1).recall("q", **kw)
    return parse_qs(urlparse(seen["path"]).query)


def test_a_local_recall_sends_no_caller_view() -> None:
    assert "caller_view" not in _sent_query()


def test_a_remote_recall_sends_its_view() -> None:
    with remote_caller("rk_00000001"):
        assert _sent_query()["caller_view"] == ["remote"]
    with remote_caller("rk_00000001"), remote_media_allowed(True):
        assert _sent_query()["caller_view"] == ["remote_media"]


def _client(engine, seen: list) -> TestClient:
    from superlocalmemory.server.unified_daemon import create_app

    def recall(*a, **k):
        seen.append(visibility.current())
        return SimpleNamespace(results=[], query="q", query_type="lookup", retrieval_time_ms=1.0,
                               channel_weights={}, total_candidates=0, no_confident_match=True)

    engine.recall = recall
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    return TestClient(app)


@pytest.mark.parametrize("view,hide_media,vetted", [("remote", True, False), ("remote_media", False, True)])
def test_the_daemon_recalls_under_the_remote_rules(engine_with_mock_deps, view, hide_media, vetted) -> None:
    seen: list = []
    reply = _client(engine_with_mock_deps, seen).get("/recall", params={"q": "x", "caller_view": view})
    assert reply.status_code == 200, reply.text
    (ctx,) = seen
    assert ctx.hide_media is hide_media and ctx.hide_sources is True
    assert (ctx.vetted_media is not None) is vetted


def test_a_local_recall_runs_with_nothing_hidden(engine_with_mock_deps) -> None:
    seen: list = []
    reply = _client(engine_with_mock_deps, seen).get("/recall", params={"q": "x"})
    assert reply.status_code == 200, reply.text
    assert seen == [visibility.VisibilityContext()]
    assert visibility.is_empty()


def test_an_unknown_view_gets_the_strictest_remote_rules(engine_with_mock_deps) -> None:
    seen: list = []
    _client(engine_with_mock_deps, seen).get("/recall", params={"q": "x", "caller_view": "bogus"})
    assert seen == [visibility.VisibilityContext(hide_media=True, hide_sources=True)]


def test_the_keyword_fallback_hides_what_the_context_hides(tmp_path) -> None:
    from superlocalmemory.server import recall_fallback
    from tests.test_retrieval._media_support import MemoryDb

    db = MemoryDb()
    db.add("m-pic", "f-pic", {"type": "media", "media_id": "a" * 32})
    db.add("m-txt", "f-txt")
    engine = SimpleNamespace(_db=db, profile_id="default", _config=SimpleNamespace())
    rows = [{"fact_id": "f-pic", "content": "picture text"}, {"fact_id": "f-txt", "content": "plain"}]
    with patch.object(recall_fallback, "_fetch_candidates", lambda *a: list(rows)):
        with visibility.use(visibility.VisibilityContext(hide_media=True)):
            out = recall_fallback.recall_keyword_fallback(engine, "q", 10)
        local = recall_fallback.recall_keyword_fallback(engine, "q", 10)
    assert [r["fact_id"] for r in out["results"]] == ["f-txt"]
    assert [r["fact_id"] for r in local["results"]] == ["f-pic", "f-txt"]
