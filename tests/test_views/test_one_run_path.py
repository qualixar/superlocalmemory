# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""One view, one answer, on every surface — and the history says who asked.

The dashboard, ``slm view run`` and the MCP ``run_view`` tool all reach
``GET /api/v3/views/run`` on the real daemon app, which hands the view to
``server.recall_core.run_recall`` — the function ``GET /recall`` calls. These
tests run one saved view through all three against a real engine and require
the same memory ids in the same order, equal to what ``/recall`` itself returns
for the view's question. Then they check the Answer Check history labels each
run by the surface it came from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

import pytest

from superlocalmemory.cli import view_cmd
from superlocalmemory.core import answer_check_history as history
from superlocalmemory.mcp import tools_views
from superlocalmemory.views import ViewStore

from ._store import learning_db


@pytest.fixture()
def daemon(engine_with_mock_deps, monkeypatch):
    """The real daemon app over a real engine, plus a daemon_request bound to it."""
    from tests.conftest import force_sync_enrichment
    from tests.test_server.test_canonical_remember_route import _client

    engine = force_sync_enrichment(engine_with_mock_deps)
    for text in ("The widget release shipped on Tuesday after both audits.",
                 "We decided the widget cache lives in learning.db.",
                 "Widget docs were updated for the release.",
                 "Widget pricing stays the same this quarter.",
                 "The cat sat on the mat."):
        engine.store(text)
    root = engine._config.base_dir
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    learning_db(root)
    ViewStore(root / "learning.db").create(
        engine.profile_id, name="Widget", query="widget release", limit=4)
    with _client(engine) as client:
        yield from _bound(client, engine, monkeypatch)


def _bound(client, engine, monkeypatch):

    def daemon_request(method, path, body=None, **_kw):
        res = client.request(method, path, json=body)
        assert res.status_code == 200, res.text
        return res.json()

    monkeypatch.setattr(view_cmd, "daemon_request", daemon_request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", daemon_request)
    yield SimpleNamespace(client=client, engine=engine)


def _dashboard(d) -> dict:
    res = d.client.get("/api/v3/views/run", params={"name": "Widget"})
    assert res.status_code == 200, res.text
    return res.json()


def _cli(capsys) -> dict:
    parser = argparse.ArgumentParser(prog="slm")
    sub = parser.add_subparsers(dest="command")
    view_cmd.register_view_parser(sub)
    view_cmd.cmd_view(parser.parse_args(["view", "run", "Widget", "--json"]))
    return json.loads(capsys.readouterr().out)["data"]


def _mcp(d) -> dict:
    tools: dict = {}

    class Collector:
        def tool(self, *a, **k):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register
    tools_views.register_view_tools(Collector(), lambda: d.engine)
    return asyncio.run(tools["run_view"](name="Widget"))


def test_one_view_gives_one_answer_on_every_surface(daemon, capsys) -> None:
    dashboard, cli, mcp = _dashboard(daemon), _cli(capsys), _mcp(daemon)
    assert dashboard["result_ids"], dashboard
    assert dashboard["result_ids"] == cli["result_ids"] == mcp["result_ids"]
    for run in (dashboard, cli, mcp):
        assert [r["rank"] for r in run["results"]] == list(range(1, run["count"] + 1))
        assert [r["fact_id"] for r in run["results"]] == run["result_ids"]
    # ... and it is recall's own answer to the view's question.
    direct = daemon.client.get("/recall", params={"q": "widget release", "limit": 4})
    assert [r["fact_id"] for r in direct.json()["results"]] == dashboard["result_ids"]


def test_the_over_budget_answer_is_shared_too(daemon, monkeypatch) -> None:
    """Past its budget an explicit question answers with keyword matches labelled
    unchecked (#304: a 503 "ask again" on an overrun never succeeded on retry);
    a view runs the same function, so it says the same thing."""
    import time

    from superlocalmemory.server import recall_core

    def slow(*a, **k):
        time.sleep(0.8)
        raise RuntimeError("never reached in time")
    monkeypatch.setattr(daemon.engine, "recall", slow)
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.5")
    monkeypatch.setattr(recall_core, "EXPLICIT_RUN_GRACE_S", 0.05)
    monkeypatch.setattr(recall_core, "EXPLICIT_CEILING_S", 0.5)
    view = daemon.client.get("/api/v3/views/run", params={"name": "Widget"})
    direct = daemon.client.get("/recall", params={"q": "widget release", "limit": 4})
    for res in (view, direct):
        assert res.status_code == 200, res.text
        assert "Retry-After" not in res.headers
        assert res.json().get("code") != "RECALL_BUSY"
        assert res.json()["retrieval_mode"] == "degraded_lexical"


def test_the_history_says_which_surface_ran_the_view(daemon, capsys, monkeypatch) -> None:
    history._reset_for_testing()
    history.enable(True)
    seen: list[str] = []
    real = history.record_recall_verdict

    def spy(response, *, profile_id):
        seen.append(history._ORIGIN.get())
        return real(response, profile_id=profile_id)
    monkeypatch.setattr(history, "record_recall_verdict", spy)
    try:
        _dashboard(daemon)
        _cli(capsys)
        _mcp(daemon)
        daemon.client.get("/recall", params={"q": "widget release"})
        events, _last = history.recent(daemon.engine.profile_id, after_seq=0, limit=50)
    finally:
        history._reset_for_testing()
    assert [ev.origin for _seq, ev in sorted(events, key=lambda e: e[0])] == [
        "view-dashboard", "view-cli", "view-mcp", ""], seen


def test_an_unknown_surface_label_is_refused(daemon) -> None:
    res = daemon.client.get("/api/v3/views/run", params={"name": "Widget", "via": "evil"})
    assert res.status_code == 422
