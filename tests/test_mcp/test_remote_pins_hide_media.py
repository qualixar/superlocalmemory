# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Pinned facts, standing rules and scheduled facts follow the same view as recall: a remote app
sees no picture, page or folder file it may not see; a caller on this computer sees all."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_mcp.test_remote_more_reads_hide_media import (  # noqa: F401  (fixture and helpers)
    CASES, _as, _daemon, _relay, _Server, world)


def _pin_all(engine, ids) -> None:
    for fact_id in ids.values():
        engine._db.set_pinned(fact_id, True)


@CASES
def test_core_memory_list(world, who, key, media, expect) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    _pin_all(engine, ids)
    server = _Server()
    register_active_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["core_memory"](action="list"))
    assert out["success"], out
    assert {p["content"] for p in out["pinned"]} == expect
    assert out["count"] == len(expect)


@CASES
def test_session_init_pins_standing_rules_and_schedule(world, monkeypatch, who, key, media, expect) -> None:
    import datetime

    from superlocalmemory.core import standing_rules
    from superlocalmemory.mcp import tools_active
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    _pin_all(engine, ids)
    calls: list[list[str]] = []
    from superlocalmemory.mcp import remote_visibility

    real = remote_visibility.hidden_fact_ids
    monkeypatch.setattr(remote_visibility, "hidden_fact_ids",
                        lambda db, pid, found: calls.append(list(found)) or real(db, pid, found))
    by_id = {v: k for k, v in ids.items()}
    monkeypatch.setattr(standing_rules, "standing_facts", lambda db, pid, exclude=frozenset(): [
        standing_rules.StandingFact(fact_id=i, content=c, kind="rule", importance=0.5, access_count=0)
        for c, i in ids.items() if i not in exclude])
    monkeypatch.setattr(tools_active, "_upcoming_scheduled_facts", lambda *a, **k: [
        {"fact_id": i, "content": c, "scheduled_at": "2030-01-01"} for c, i in ids.items()])
    server = _Server()
    register_active_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["session_init"](query="nothing-matches-qqq"))
    text = str(out)
    for content in set(ids) - expect:
        assert content not in text, content
    if key:
        assert len(calls) == 1, calls
