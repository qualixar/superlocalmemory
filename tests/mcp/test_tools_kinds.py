# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The four memory-kind MCP tools (LLD/WP8 4.1.19) — thin clients of
``/api/memory-kinds``, bound to a real ``TestClient`` of those routes so the
MCP surface and the dashboard/CLI cannot disagree about a kind.
"""

from __future__ import annotations

import asyncio

from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound
from tests.test_server.test_memory_kinds_routes import _client, _fact


class _Server:
    def __init__(self) -> None:
        self.captured: dict = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.captured[fn.__name__] = fn
            return fn
        return deco


def _tools():
    from superlocalmemory.mcp import tools_kinds

    server = _Server()
    tools_kinds.register_kind_tools(server, lambda: None)
    return server.captured


def _bind(monkeypatch, tc) -> None:
    def fake_daemon_request(method, path, body=None, *, preserve_conflict=False,
                            preserve_not_found=False, preserve_unprocessable=False, **_kw):
        response = tc.request(method, path, json=body)
        if response.status_code == 409 and preserve_conflict:
            raise DaemonConflict(response.json().get("detail", ""))
        if response.status_code == 404 and preserve_not_found:
            raise DaemonNotFound(404, "not_found", "daemon returned 404", path)
        if response.status_code >= 400:
            return None
        return response.json()

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", fake_daemon_request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)


def test_memory_kinds_status(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.")
    _bind(monkeypatch, tc)
    tools = _tools()
    out = asyncio.run(tools["memory_kinds_status"]())
    assert out["success"] is True
    assert out["schema_ready"] is True


def test_set_memory_kind_returns_kind_fields(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    fid = _fact(db, "We decided to use SQLite.")
    _bind(monkeypatch, tc)
    tools = _tools()
    out = asyncio.run(tools["set_memory_kind"](fid, "decision"))
    assert out["success"] is True and out["ok"] is True
    assert out["memory_kind"] == "decision"
    assert out["memory_kind_state"] == "confirmed"


def test_set_memory_kind_refuses_an_unknown_kind_before_any_request(monkeypatch) -> None:
    called = []
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running",
                        lambda: called.append(1) or True)
    tools = _tools()
    out = asyncio.run(tools["set_memory_kind"]("some-fact-id", "not-a-real-kind"))
    assert out["success"] is False and out["code"] == "INVALID_KIND"
    assert not called, "the daemon must not be probed for a kind that never parsed"


def test_set_memory_kind_not_found(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _bind(monkeypatch, tc)
    tools = _tools()
    out = asyncio.run(tools["set_memory_kind"]("no-such-fact", "rule"))
    assert out["success"] is False


def test_review_memory_kinds_lists_suggestions(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _fact(db, "Never push to main.", kind="rule", source="rules")
    _bind(monkeypatch, tc)
    tools = _tools()
    out = asyncio.run(tools["review_memory_kinds"](kind="rule", limit=10))
    assert out["success"] is True
    assert "items" in out


def test_review_memory_kinds_refuses_an_unknown_kind(monkeypatch) -> None:
    tools = _tools()
    out = asyncio.run(tools["review_memory_kinds"](kind="not-a-real-kind"))
    assert out["success"] is False and out["code"] == "INVALID_KIND"


def test_confirm_memory_kinds_applies_items(tmp_path, monkeypatch) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    fid = _fact(db, "We decided to use SQLite.")
    _bind(monkeypatch, tc)
    tools = _tools()
    out = asyncio.run(tools["confirm_memory_kinds"]([{"fact_id": fid, "kind": "decision"}]))
    assert out["success"] is True
    assert out["items"][0]["ok"] is True
    assert out["items"][0]["memory_kind"] == "decision"


def test_confirm_memory_kinds_rejects_empty_items() -> None:
    out = asyncio.run(_tools()["confirm_memory_kinds"]([]))
    assert out["success"] is False and out["code"] == "INVALID_ITEMS"


def test_daemon_unavailable_is_reported_as_retryable(monkeypatch) -> None:
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: False)
    tools = _tools()
    out = asyncio.run(tools["memory_kinds_status"]())
    assert out["success"] is False and out["retryable"] is True
