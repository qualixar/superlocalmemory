# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The project reaches recall from every door, and the answer says what it did
(GitHub #150): MCP ``recall`` / ``session_init``, the daemon proxy, HTTP
``GET /recall`` / ``POST /session/open`` / ``POST /remember``, the CLI, the
context-file builder and the Stop hook."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_FILTER_FELL_BACK = {"filter": {"project": "ghost", "key": "ghost", "applied": False,
                                "matched": 0, "reason": "no_match",
                                "note": "None of the memories found for this question were "
                                        "saved under project 'ghost', so these results are "
                                        "not narrowed to it."}}


class _Server:
    def __init__(self):
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        return lambda fn: self.tools.setdefault(fn.__name__, fn)

    def resource(self, *args, **kwargs):
        return lambda fn: fn

    def prompt(self, *args, **kwargs):
        return lambda fn: fn


# -- MCP ----------------------------------------------------------------------


def test_daemon_proxy_sends_project_parameters_only_when_set(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths: list[str] = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=48517)
    proxy.recall("q", prefer_project="/Users/x/acme billing", project="acme")
    assert "prefer_project=%2FUsers%2Fx%2Facme+billing" in paths[0]
    assert "project=acme" in paths[0]
    proxy.recall("q")
    assert "project" not in paths[1]


def test_mcp_recall_forwards_both_and_returns_the_report(monkeypatch) -> None:
    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.mcp.tools_core import register_core_tools

    seen: dict = {}

    class _Pool:
        def recall(self, *args, **kwargs):
            seen.update(kwargs)
            scope = _FILTER_FELL_BACK if "project" in kwargs else None
            return {"ok": True, "results": [{"fact_id": "f1", "content": "x"}],
                    "result_count": 1, "project_scope": scope}

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: _Pool())
    srv = _Server()
    register_core_tools(srv, lambda: SimpleNamespace(profile_id="default"))
    out = asyncio.run(srv.tools["recall"]("q", project=" ghost ", prefer_project=" /x/acme "))
    assert seen["project"] == "ghost" and seen["prefer_project"] == "/x/acme"
    assert out["success"] is True and out["count"] == 1
    assert out["project_scope"] == _FILTER_FELL_BACK

    seen.clear()
    plain = asyncio.run(srv.tools["recall"]("q"))
    assert "project" not in seen and "prefer_project" not in seen
    assert plain["project_scope"] is None


def test_pool_recall_forwards_project_parameters(monkeypatch) -> None:
    from superlocalmemory.mcp import _pool_adapter

    seen: dict = {}

    class _Pool:
        def recall(self, **kwargs):
            seen.update(kwargs)
            return {"ok": True, "results": []}

    monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Pool())
    _pool_adapter.pool_recall("q", prefer_project=" /x/acme ", project="")
    assert seen["prefer_project"] == "/x/acme" and "project" not in seen


def test_session_init_prefers_its_project_and_names_it_in_the_query() -> None:
    from superlocalmemory.mcp._pool_adapter import PoolRecallResponse
    from superlocalmemory.mcp.tools_active import register_active_tools

    seen: list = []

    def fake_pool_recall(query, limit=10, **kwargs):
        seen.append({"query": query, **kwargs})
        return PoolRecallResponse()

    engine = MagicMock()
    engine.profile_id = "default"
    engine.db.get_pinned.return_value = []
    engine._auto_invoker = None
    rules = MagicMock()
    rules.should_recall.return_value = True
    rules.get_recall_config.return_value = {"relevance_threshold": 0.3}
    srv = _Server()
    register_active_tools(srv, lambda: engine)
    with patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=rules), \
         patch("superlocalmemory.mcp._pool_adapter.pool_recall", fake_pool_recall), \
         patch("superlocalmemory.mcp.tools_active._emit_event", create=True):
        asyncio.run(srv.tools["session_init"](project_path="/Users/alice/work/acme-billing"))
        asyncio.run(srv.tools["session_init"](query="release plan"))
    assert seen[0] == {"query": "project context acme-billing", "fast": None,
                       "prefer_project": "/Users/alice/work/acme-billing", "via": "internal"}
    assert seen[1]["query"] == "release plan" and "prefer_project" not in seen[1]


def test_prefer_project_is_a_neutral_remote_argument() -> None:
    from superlocalmemory.server.remote_profile_binding import NEUTRAL_ARGUMENTS

    assert "prefer_project" in NEUTRAL_ARGUMENTS


# -- HTTP ---------------------------------------------------------------------


def _response(project_scope=None):
    from superlocalmemory.storage.models import (
        AtomicFact,
        FactType,
        RecallResponse,
        RetrievalResult,
    )

    fact = AtomicFact(fact_id="f1", content="x", fact_type=FactType.SEMANTIC)
    return RecallResponse(query="q", results=[RetrievalResult(fact=fact, score=0.5)],
                          project_scope=project_scope)


def test_http_recall_takes_prefer_project_and_returns_the_report(
        engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen: dict = {}

    def fake_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response(_FILTER_FELL_BACK)

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_recall)
    with _client(engine_with_mock_deps) as client:
        r = client.get("/recall", params={"q": "invoices", "project": "ghost",
                                          "prefer_project": "/x/acme"})
    assert r.status_code == 200, r.text
    assert seen["facets"].as_dict() == {"project": "ghost", "prefer_project": "/x/acme"}
    assert r.json()["project_scope"] == _FILTER_FELL_BACK


def test_http_session_open_prefers_the_project(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen: list = []

    def fake_recall(query, **kwargs):
        seen.append((query, kwargs))
        return _response()

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_recall)
    with _client(engine_with_mock_deps) as client:
        r = client.post("/session/open", json={"project_path": "/Users/alice/work/acme-billing"})
    assert r.status_code == 200, r.text
    query, kwargs = seen[0]
    assert query == "project context acme-billing"
    assert kwargs["facets"].as_dict() == {"prefer_project": "/Users/alice/work/acme-billing"}


@pytest.mark.parametrize("given,stored", [(" /Users/x/acme-billing/ ", "/Users/x/acme-billing/"),
                                          ("/", ""), ("acme", "acme")])
def test_http_remember_saves_a_clean_project(engine_with_mock_deps, given, stored) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    with _client(engine_with_mock_deps) as client:
        r = client.post("/remember", json={
            "content": f"Invoice rule number {stored or 'none'} for the billing service.",
            "metadata": {"project": given}})
    assert r.status_code == 200, r.text
    rows = engine_with_mock_deps._db.execute("SELECT metadata_json FROM memories")
    assert json.loads(dict(rows[0])["metadata_json"])["project"] == stored


# -- CLI ----------------------------------------------------------------------


def _recall_args(**overrides) -> Namespace:
    base = dict(query="invoices", limit=5, json=False, project="", saved_by="", about="",
                prefer_project="", fast=None, window="", as_of="", known_as_of="",
                valid_at="", include_unknown=False, include_global=None,
                include_shared=None, session_id="", profile_id="", kind="")
    base.update(overrides)
    return Namespace(**base)


def test_cli_recall_sends_prefer_project_and_says_when_a_filter_fell_back(
        monkeypatch, capsys) -> None:
    from superlocalmemory.cli import commands, daemon

    paths: list[str] = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda method, path, *a, **k: paths.append(
        path) or {"results": [{"content": "an invoice memory", "score": 0.5}],
                  "project_scope": _FILTER_FELL_BACK})
    commands.cmd_recall(_recall_args(project="ghost", prefer_project="/x/acme"))
    assert "&project=ghost" in paths[0] and "&prefer_project=/x/acme" in paths[0]
    out = capsys.readouterr().out
    assert "an invoice memory" in out and "not narrowed" in out


def test_cli_parser_accepts_prefer_project() -> None:
    import argparse
    import sys

    class _Captured(Exception):
        pass

    holder: dict = {}

    def _capture(self, *a, **k):
        holder["parser"] = self
        raise _Captured()

    with patch.object(argparse.ArgumentParser, "parse_args", _capture), \
            patch.object(sys, "argv", ["slm", "mcp"]):
        from superlocalmemory.cli.main import main
        with pytest.raises(_Captured):
            main()
    args = holder["parser"].parse_args(["recall", "q", "--prefer-project", "/x/acme"])
    assert args.prefer_project == "/x/acme"


# -- context file builder ------------------------------------------------------


def test_context_builder_asks_for_the_projects_memories() -> None:
    from superlocalmemory.hooks.context_payload import build_payload

    calls: list = []

    def recall(query, limit, profile_id, **kwargs):
        calls.append((query, kwargs))
        return [{"text": f"hit for {query}", "score": 0.5}]

    payload = build_payload("default", "project", None, recall_fn=recall,
                            project="/Users/alice/work/acme-billing")
    assert [q for q, _ in calls] == ["acme-billing topics", "acme-billing entities",
                                     "acme-billing recent decisions", "acme-billing memories"]
    assert all(kw == {"project": "acme-billing"} for _, kw in calls)
    assert payload.project_memories == ("hit for acme-billing memories",)


def test_context_builder_without_a_project_is_unchanged() -> None:
    from superlocalmemory.hooks.context_payload import build_payload

    calls: list = []
    build_payload("default", "project", None,
                  recall_fn=lambda q, limit, pid: calls.append(q) or [])
    build_payload("default", "global", None, project="/x/acme",
                  recall_fn=lambda q, limit, pid, **kw: calls.append((q, kw)) or [])
    assert calls[:4] == ["project topics", "project entities", "recent decisions",
                         "project memories"]
    assert calls[4:] == [("topics", {}), ("entities", {}), ("recent decisions", {}),
                         ("memories", {})]


def test_context_prestage_uses_the_working_directory_as_the_project(
        tmp_path, monkeypatch) -> None:
    from superlocalmemory.cli import context_commands

    project_dir = tmp_path / "acme-billing"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    seen: list = []
    monkeypatch.setattr(context_commands, "_get_recall_fn",
                        lambda: lambda q, limit, pid, **kw: seen.append((q, kw)) or [])
    context_commands.cmd_context(Namespace(subcommand="prestage", query="", limit=5,
                                           profile_id="default", json=True, tool=False))
    assert seen and all(kw == {"project": "acme-billing"} for _, kw in seen)


def test_context_recall_never_starts_a_daemon(monkeypatch) -> None:
    from superlocalmemory.cli import context_commands, daemon

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon, "ensure_daemon",
                        lambda *a, **k: pytest.fail("must not start a daemon"))
    assert context_commands._get_recall_fn()("q", 3, "default", project="acme") == []


def test_context_recall_reads_a_running_daemon(monkeypatch) -> None:
    from superlocalmemory.cli import context_commands, daemon

    paths: list[str] = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda method, path, *a, **k: paths.append(
        path) or {"results": [{"content": "an acme memory", "score": 0.7}]})
    rows = context_commands._get_recall_fn()("acme topics", 3, "default", project="acme")
    assert rows == [{"text": "an acme memory", "score": 0.7}]
    assert "project=acme" in paths[0] and "answer_check=context" in paths[0]


# -- Stop hook -----------------------------------------------------------------


@patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
@patch("superlocalmemory.hooks.hook_handlers._daemon_post")
@patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
def test_stop_hook_saves_the_summary_under_its_project(mock_run, mock_post, _consolidate,
                                                       monkeypatch) -> None:
    from superlocalmemory.hooks.hook_handlers import handle_hook

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/alice/work/acme-billing")
    mock_run.return_value = MagicMock(stdout="", returncode=0)
    mock_post.return_value = True
    with pytest.raises(SystemExit):
        handle_hook("stop")
    path, body = mock_post.call_args_list[0][0][:2]
    assert path == "/remember"
    assert body["metadata"] == {"project": "/Users/alice/work/acme-billing"}
    assert body["content"].startswith("[acme-billing] session ended")


# -- recall_trace explains the recall it claims to ------------------------------


def _tools_with_pool(monkeypatch, pool):
    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools

    monkeypatch.setattr(_daemon_proxy, "choose_pool", lambda: pool)
    srv = _Server()
    register_core_tools(srv, lambda: SimpleNamespace(profile_id="default"))
    register_v3_tools(srv, lambda: SimpleNamespace(profile_id="default"))
    return srv.tools


def test_recall_trace_takes_the_same_project_arguments_as_recall(monkeypatch) -> None:
    calls: list[dict] = []
    envelope = {"ok": True, "result_count": 1, "project_scope": _FILTER_FELL_BACK,
                "results": [{"fact_id": "f1", "content": "x", "score": 0.5,
                             "evidence_chain": ["bm25(rank=1)", "same_project"]}]}

    class _Pool:
        def recall(self, *args, **kwargs):
            calls.append(kwargs)
            return dict(envelope)

    tools = _tools_with_pool(monkeypatch, _Pool())
    args = {"project": " ghost ", "prefer_project": " /x/acme "}
    asyncio.run(tools["recall"]("q", **args))
    trace = asyncio.run(tools["recall_trace"]("q", **args))

    project_args = [{k: c.get(k) for k in ("project", "prefer_project")} for c in calls]
    assert project_args[0] == project_args[1] == {"project": "ghost", "prefer_project": "/x/acme"}
    assert trace["success"] is True
    assert trace["project_scope"] == _FILTER_FELL_BACK
    assert trace["results"][0]["evidence_chain"][-1] == "same_project"


def test_recall_trace_without_a_project_sends_none(monkeypatch) -> None:
    calls: list[dict] = []

    class _Pool:
        def recall(self, *args, **kwargs):
            calls.append(kwargs)
            return {"ok": True, "results": []}

    tools = _tools_with_pool(monkeypatch, _Pool())
    asyncio.run(tools["recall_trace"]("q", project="  "))
    assert "project" not in calls[0] and "prefer_project" not in calls[0]


def test_daemon_proxy_answers_recall_trace_keywords(monkeypatch) -> None:
    """The real proxy accepts every keyword recall_trace sends (a fake pool
    would accept anything)."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths: list[str] = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    tools = _tools_with_pool(monkeypatch, _daemon_proxy.DaemonPoolProxy(port=48517))
    out = asyncio.run(tools["recall_trace"]("q", project="acme", prefer_project="/x/acme"))
    assert out["success"] is True, out
    assert "project=acme" in paths[0] and "prefer_project=%2Fx%2Facme" in paths[0]
