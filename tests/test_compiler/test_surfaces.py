# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Where the compiler meets the rest: origin channel, flags, and the MCP door to pages."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from superlocalmemory.compiler import page_sources, proposals
from superlocalmemory.core.feature_flags import FLAGS, FlagState
from superlocalmemory.mcp import tools_summaries
from superlocalmemory.storage import fact_origin
from superlocalmemory.trust import origin_class as oc
from tests.test_compiler.conftest import ENTITY, PROFILE, raw

PERSON = proposals.Basis.person("alice", "cli")


def test_external_edit_is_a_channel_with_a_reasoned_class() -> None:
    assert "external_edit" in fact_origin.CHANNELS
    fact_origin.Origin("external_edit", "verbatim")  # accepted
    got = oc.classify(oc.Signals(channel="external_edit", transform="verbatim", role="user"))
    assert got == oc.Classified(oc.AGENT_INFERRED, "v3:edit.external")
    row = next(r for r in oc.ROWS if r.row_id == "edit.external")
    assert "close to user_direct" in row.why and "tool or an agent" in row.why
    assert "user_confirmed" in row.why  # the approval is what raises it
    assert oc.classify(oc.Signals(channel="external_edit", transform="model_rewritten")
                       ).origin_class == oc.MODEL_GENERATED  # the transform still caps it
    assert not oc.at_least(oc.AGENT_INFERRED, oc.USER_DIRECT)


def test_flags_are_registered_off_and_the_dream_cycle_has_no_shadow() -> None:
    assert FLAGS["compiler.dream"].default is FlagState.OFF and not FLAGS["compiler.dream"].shadowable
    assert FLAGS["compiler.wiki"].default is FlagState.OFF and FLAGS["compiler.wiki"].shadowable
    assert FLAGS["compiler.auto_apply"].default is FlagState.OFF


@pytest.fixture
def tool(store):
    db, ids = store
    tools: dict = {}

    class Collector:
        def tool(self, *a, **k):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register

    tools_summaries.register_summary_tools(
        Collector(), lambda: SimpleNamespace(profile_id=PROFILE, config=None, db=db, _db=db))
    conn = raw(db)
    try:
        drafts = [d for d in page_sources.build_drafts(conn, PROFILE, ("topic",))
                  if d["subject_key"] == f"entity:{ENTITY}"]
        set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[
            {"kind": "create_page", "payload": {"page": d}} for d in drafts])
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    finally:
        conn.close()
    return tools["get_memory_summary"], db, ids


def test_get_memory_summary_lists_pages_and_reads_one_with_its_citations(tool) -> None:
    fn, _db, ids = tool
    listed = asyncio.run(fn(kind="page"))
    assert listed["success"] and listed["kind"] == "page" and listed["generated_by"] == "extractive"
    (page,) = listed["metadata"]["pages"]
    assert page["kind"] == "topic" and page["page_id"] in listed["summary"]
    read = asyncio.run(fn(kind="page", target=page["page_id"]))
    assert read["success"] and read["coverage"] == "full" and read["metadata"]["origin"] == "derived"
    assert sorted(read["source_fact_ids"]) == sorted(ids)
    assert [s["facts"] for s in read["metadata"]["sentences"]] == [[i] for i in reversed(ids)]
    assert read["metadata"]["stale"] is False and read["metadata"]["withheld"]["count"] == 0
    assert "No model wrote any of it" in read["metadata"]["notice"]


def test_a_changed_memory_is_hidden_and_the_coverage_says_partial(tool) -> None:
    fn, db, ids = tool
    page = asyncio.run(fn(kind="page"))["metadata"]["pages"][0]
    db.execute("UPDATE atomic_facts SET content = 'Changed.' WHERE fact_id = ?", (ids[0],))
    read = asyncio.run(fn(kind="page", target=page["page_id"]))
    assert read["metadata"]["stale"] and read["coverage"] == "partial"
    assert "1 sentences withheld" in read["summary"] and ids[0] not in read["source_fact_ids"]


def test_an_unknown_page_is_refused_not_empty(tool) -> None:
    fn, _db, _ids = tool
    answer = asyncio.run(fn(kind="page", target="nope"))
    assert answer["success"] is False and "nope" in answer["error"]


def test_a_remote_caller_never_sees_a_page_that_cites_a_memory_it_may_not(tool, monkeypatch) -> None:
    from superlocalmemory.mcp import remote_visibility

    fn, _db, ids = tool
    page = asyncio.run(fn(kind="page"))["metadata"]["pages"][0]
    monkeypatch.setattr(remote_visibility, "current_view", lambda: object())
    monkeypatch.setattr(remote_visibility, "hidden_fact_ids", lambda db, pid, cited: {ids[0]})
    assert asyncio.run(fn(kind="page"))["metadata"]["pages"] == []
    refused = asyncio.run(fn(kind="page", target=page["page_id"]))
    assert refused["success"] is False


def test_a_store_without_pages_says_so(tmp_path) -> None:
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    tools: dict = {}

    class Collector:
        def tool(self, *a, **k):
            def register(fn):
                tools[fn.__name__] = fn
                return fn
            return register

    tools_summaries.register_summary_tools(
        Collector(), lambda: SimpleNamespace(profile_id=PROFILE, config=None, db=db, _db=db))
    got = asyncio.run(tools["get_memory_summary"](kind="page"))
    assert got["success"] and got["summary"] == "No compiled pages yet."
    assert json.dumps(got)
