# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``fetch(why=true)``: the `slm why` report on each fetched memory, no new tool."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.mcp.tools_core import register_core_tools
from superlocalmemory.trust import why as why_mod
from tests.test_trust_signing.conftest import save


class _Server:
    def __init__(self):
        self.tools = {}

    def tool(self, *args, **kwargs):
        def wrap(fn):
            self.tools[fn.__name__] = fn
            return fn
        return wrap


class _Engine:
    profile_id = "default"

    def __init__(self, db):
        self._db = db


@pytest.fixture
def fetch(db):
    server = _Server()
    register_core_tools(server, lambda: _Engine(db))
    return lambda *a, **k: asyncio.run(server.tools["fetch"](*a, **k))


def test_fetch_without_why_is_unchanged(db, fetch) -> None:
    fact = save(db, "Plain fetch of a memory.")
    out = fetch(fact.fact_id)
    assert out["success"] and "why" not in out["results"][0]


def test_fetch_with_why_adds_the_report(db, fetch) -> None:
    fact = save(db, "Fetched with its explanation.")
    out = fetch(fact.fact_id, why=True)
    report = out["results"][0]["why"]
    assert out["success"] and report["fact_id"] == fact.fact_id
    assert {"origin", "firewall", "signature", "truth", "compiler", "deliveries"} <= set(report)


def test_at_most_ten_are_explained_per_call_and_none_fail_the_fetch(db, fetch) -> None:
    ids = [save(db, f"Memory number {i} about subject {i * 11}.").fact_id for i in range(12)]
    out = fetch(",".join(ids), why=True)
    assert out["success"] and out["count"] == 12
    explained = [r for r in out["results"] if "origin" in r["why"]]
    assert len(explained) == why_mod.MAX_ATTACHED
    assert [r["why"] for r in out["results"] if "origin" not in r["why"]] == [{"skipped": "limit"}] * 2


def test_an_unreadable_store_leaves_the_fetch_intact(db, fetch, monkeypatch) -> None:
    fact = save(db, "Still fetched.")
    import sqlite3

    from superlocalmemory.truth import invariants

    monkeypatch.setattr(invariants, "open_read_only",
                        lambda p: (_ for _ in ()).throw(sqlite3.OperationalError("x")))
    out = fetch(fact.fact_id, why=True)
    assert out["success"] and out["results"][0]["why"]["error"].startswith("store not readable")


def test_the_argument_is_classified_for_remote_profile_binding() -> None:
    from superlocalmemory.server.remote_profile_binding import CLASSIFIED_ARGUMENTS, NEUTRAL_ARGUMENTS

    assert "why" in NEUTRAL_ARGUMENTS and "why" in CLASSIFIED_ARGUMENTS


def test_no_new_tool_was_added(db) -> None:
    server = _Server()
    register_core_tools(server, lambda: _Engine(db))
    assert "why" not in server.tools
