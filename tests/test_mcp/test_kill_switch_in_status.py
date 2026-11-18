# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-23 over MCP: ``get_status`` and ``health`` say whether the kill switch is on (no new tool)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, set_kill_switch
from superlocalmemory.mcp import tools_core


class _Tools:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(KILL_ENV, raising=False)
    return tmp_path


def test_the_state_helper_reads_the_file_and_the_environment(home, monkeypatch) -> None:
    assert tools_core._kill_switch_state()["on"] is False
    set_kill_switch(home, True)
    assert tools_core._kill_switch_state()["source"] == "file"
    set_kill_switch(home, False)
    monkeypatch.setenv(KILL_ENV, "1")
    got = tools_core._kill_switch_state()
    assert got["on"] is True and got["source"] == "environment"


def test_get_status_carries_it_whichever_way_it_is_answered(home, monkeypatch) -> None:
    from superlocalmemory.cli import daemon

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda *a, **k: {
        "profile": "default", "mode": "a", "fact_count": 1})
    server = _Tools()
    tools_core.register_core_tools(server, lambda: SimpleNamespace())
    get_status = server.tools["get_status"]
    assert asyncio.run(get_status())["kill_switch"]["on"] is False
    set_kill_switch(home, True)
    shown = asyncio.run(get_status())["kill_switch"]
    assert shown["on"] is True and "off" in shown["words"]


def test_health_carries_it(home) -> None:
    from superlocalmemory.mcp import tools_v3

    set_kill_switch(home, True)
    engine = SimpleNamespace(
        profile_id="default", _db=SimpleNamespace(get_fact_count=lambda p: 0, execute=lambda *a: []),
        _config=SimpleNamespace(
            mode=SimpleNamespace(value="a"), embedding=SimpleNamespace(model_name="m"),
            llm=SimpleNamespace(provider="none"),
            math=SimpleNamespace(fisher_temperature=1.0, sheaf_contradiction_threshold=0.5)),
        _embedder=None, _llm=None, _sheaf_checker=None)
    server = _Tools()
    tools_v3.register_v3_tools(server, lambda: engine)
    got = asyncio.run(server.tools["health"]())
    assert got["success"] and got["kill_switch"]["on"] is True
