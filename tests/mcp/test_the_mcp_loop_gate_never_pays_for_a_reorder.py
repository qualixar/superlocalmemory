# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Over MCP, the bounded-loop gate asks the answer check but never the reorder.

The gate runs one recall per lap. Through the MCP server that recall goes to
the daemon via the pool proxy, so the proxy must carry the gate's request on
the wire; otherwise every lap pays for a reorder nobody reads.
"""

from __future__ import annotations

import urllib.parse

import pytest

from superlocalmemory.core.answer_check_scope import skip_answer_check
from superlocalmemory.mcp import _daemon_proxy, tools_loops
from superlocalmemory.retrieval.answer_check_status import REQUEST_NO_REORDER


@pytest.fixture
def sent(monkeypatch):
    paths: list[str] = []

    def fake_request(method, path, **_kw):
        paths.append(path)
        return {"ok": True, "results": []}

    import superlocalmemory.cli.daemon as daemon_mod
    monkeypatch.setattr(daemon_mod, "daemon_request", fake_request)
    return paths


def _param(path: str) -> list[str]:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(path).query).get("answer_check", [])


def test_the_gate_asks_the_proxy_for_no_reorder_and_the_proxy_sends_it(sent):
    proxy = _daemon_proxy.DaemonPoolProxy(port=1)
    kwargs = tools_loops._gate_recall_kwargs(proxy.recall)
    assert kwargs == {"answer_check": REQUEST_NO_REORDER}
    proxy.recall("is the build green?", limit=3, fast=True, **kwargs)
    assert _param(sent[-1]) == [REQUEST_NO_REORDER]


def test_a_context_load_still_sends_skip(sent):
    proxy = _daemon_proxy.DaemonPoolProxy(port=1)
    proxy.recall("project context", answer_check=False)
    with skip_answer_check():
        proxy.recall("project context", answer_check=REQUEST_NO_REORDER)
    assert [_param(p) for p in sent] == [["skip"], ["skip"]]


def test_an_ordinary_recall_leaves_the_query_string_unchanged(sent):
    proxy = _daemon_proxy.DaemonPoolProxy(port=1)
    proxy.recall("what did we decide?")
    proxy.recall("what did we decide?", answer_check=True)
    assert [_param(p) for p in sent] == [[], []]
