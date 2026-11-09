# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""What a save calls "queryable" is found under what it was saved with, at once
and again once enriched.

The 4.1.21 failure: a decision saved through the tool interface, answered
"queryable", was then not found by a decision-filtered recall. Each save here
declares a kind, a project and a scope (one goes to a named profile, one is
global); every filter it declared must find it right after the save answers,
and must still find the MEMORY once enrichment has replaced the first fact
with derived ones.

Each save also carries tags (4.1.22): recall, search and list_recent must
find it by an exact tag filter the same way, although no tag label appears in
its words.

Composed path: a REAL daemon subprocess and a REAL ``slm mcp`` stdio child.
"""
# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import time
import uuid

import pytest

import tests.test_integration.test_mcp_declared_kind_transport as transport
from tests.test_integration.test_mcp_declared_kind_transport import (  # noqa: F401
    stub_embedder,  # the fixture, used by ``lane``
)

RUN = uuid.uuid4().hex[:6].upper()
PROJECT = f"synthetic-heron-{RUN.lower()}"
OTHER = transport.PROFILE  # a named profile the lane pre-creates

TAG = f"parity-label-{RUN.lower()}"
SAVES = {
    "decision": dict(kind="decision", project=PROJECT, tags=f"{TAG},decided"),
    "rule-global": dict(kind="rule", project=PROJECT, scope="global", tags=TAG),
    "procedure-named": dict(kind="procedure", project=PROJECT, profile_id=OTHER, tags=TAG),
}


def _content(label: str) -> str:
    return (f"Parity record PR{label[:3].upper()}{RUN}: the heron fixture uses "
            f"setting {len(label)} for synthetic checks.")


@pytest.fixture(scope="module")
def lane(tmp_path_factory, stub_embedder):
    # The hermetic stub embedder (same as the transport tests): enrichment
    # runs on every machine, with no model download and no real model cache.
    original = transport.MCP_TOOLS
    transport.MCP_TOOLS = original + ",list_recent,search"
    root = tmp_path_factory.mktemp("raw-parity")
    lane = transport._Lane(root, stub_embedder)
    try:
        lane.open_mcp()
        for label, declared in SAVES.items():
            out = lane.tool("remember", content=_content(label),
                            idempotency_key=f"parity-{label}-{RUN}", **declared)
            assert out["success"] is True and out["fact_ids"], out
            lane.saved[label] = out
        yield lane
    finally:
        transport.MCP_TOOLS = original
        lane.close_mcp()
        lane.daemon.stop(lane.foreign)


def _memory_of(lane, fact_id: str) -> str:
    return lane.rows("SELECT memory_id FROM atomic_facts WHERE fact_id=?", (fact_id,))[0][0]


def _found_memories(lane, tool: str, **arguments) -> set[str]:
    out = lane.tool(tool, **arguments)
    items = out.get("results") or out.get("memories") or out.get("items") or []
    found = set()
    for item in items:
        fid = item.get("fact_id") or item.get("id")
        if fid and lane.rows("SELECT 1 FROM atomic_facts WHERE fact_id=?", (fid,)):
            found.add(_memory_of(lane, fid))
    return found


def _queries(label: str) -> list[tuple[str, dict]]:
    declared = SAVES[label]
    query = _content(label)
    profile = {"profile_id": declared["profile_id"]} if "profile_id" in declared else {}
    asks = [
        ("recall", dict(query=query, kind=declared["kind"], **profile)),
        ("recall", dict(query=query, project=PROJECT, **profile)),
        ("recall", dict(query=query, kind=declared["kind"], project=PROJECT, **profile)),
        ("recall", dict(query=query, tags=TAG.upper(), **profile)),
        ("recall", dict(query=query, tags=[TAG, "absent-label"], tags_match="any", **profile)),
        ("search", dict(query=query, tags=TAG, kind=declared["kind"], **profile)),
        ("list_recent", dict(tags=TAG, **profile)),
    ]
    if declared.get("scope") == "global":
        asks.append(("recall", dict(query=query, kind=declared["kind"], profile_id=OTHER,
                                    include_global=True)))
    return asks


def _excluded(label: str) -> list[tuple[str, dict]]:
    """A tag the memory does not carry must hide it on every tool, or the
    positive tag checks above would pass on a store this small regardless."""
    query = _content(label)
    declared = SAVES[label]
    profile = {"profile_id": declared["profile_id"]} if "profile_id" in declared else {}
    return [("recall", dict(query=query, tags="absent-label", **profile)),
            ("search", dict(query=query, tags="absent-label", **profile)),
            ("list_recent", dict(tags="absent-label", **profile))]


def _misses(lane) -> list:
    misses = []
    for label in SAVES:
        want = _memory_of(lane, lane.saved[label]["fact_ids"][0])
        for tool, arguments in _queries(label):
            if want not in _found_memories(lane, tool, limit=10, **arguments):
                misses.append((label, tool, sorted(k for k in arguments if k != "query")))
        for tool, arguments in _excluded(label):
            if want in _found_memories(lane, tool, limit=10, **arguments):
                misses.append((label, tool, "found under a tag it does not carry"))
    return misses


def test_every_declared_filter_finds_the_save_as_soon_as_it_is_queryable(lane):
    assert _misses(lane) == []


def test_the_first_queryable_fact_carries_what_was_declared(lane):
    for label, declared in SAVES.items():
        for fid in lane.saved[label]["fact_ids"]:
            kind, source = lane.kind_of(fid)
            assert (kind, source) == (declared["kind"], "caller"), (label, fid)
            scope = lane.rows("SELECT scope, profile_id FROM atomic_facts WHERE fact_id=?",
                              (fid,))[0]
            assert scope[0] == declared.get("scope", "personal"), (label, scope)
            assert scope[1] == declared.get("profile_id", "default"), (label, scope)


def test_every_declared_filter_still_finds_the_memory_once_enriched(lane):
    # Enrichment waits for a warm embedding model. A daemon that cannot embed
    # here (seen with this test environment on 4.1.22 itself: the model loads
    # but returns no vectors) can never enrich, which is not what this checks.
    warm_by = time.monotonic() + 120
    while not lane.daemon.request("GET", "/health")[1].get("embedding_warm"):
        if time.monotonic() > warm_by:
            pytest.skip("this daemon reports embedding_warm=false: enrichment cannot run")
        time.sleep(2.0)
    deadline = time.monotonic() + 300
    pending = set(SAVES)
    while pending and time.monotonic() < deadline:
        for label in list(pending):
            state = lane.rows("SELECT state FROM ingestion_operations WHERE operation_id=?",
                              (lane.saved[label]["operation_id"],))
            if state and state[0][0] == "complete":
                pending.discard(label)
        time.sleep(1.0)
    if pending == set(SAVES):
        pytest.fail(f"enrichment finished for none of the saves within 300 s: {pending}")
    assert _misses(lane) == []
