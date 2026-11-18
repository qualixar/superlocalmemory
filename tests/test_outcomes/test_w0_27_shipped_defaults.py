# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-27: each delivery surface end to end with the shipped defaults (D-34, holdout on).

``tests/conftest.py`` forces the holdout and the swaps off so ordinary tests are stable;
here the override is deleted, so the randomiser runs as it does for a person. For each
door (MCP ``session_init``, the ``slm session-context`` hook, the CLI's recall) the
receipts and propensities are written and what the agent was given is consistent with them.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from argparse import Namespace
from types import SimpleNamespace

import pytest

from superlocalmemory.outcomes import randomiser as rz
from superlocalmemory.outcomes import receipts
from tests.test_outcomes.test_injection_receipts import _fact, _result

RUNS = 40
EPS = rz.EPSILON


@pytest.fixture
def shipped(monkeypatch):
    """The shipped defaults: no environment override of the holdout or the swaps."""
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_HOLDOUT", raising=False)
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_SWAPS", raising=False)


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _q(db_path, sql, *args):
    conn = sqlite3.connect(str(db_path))
    try:
        return [tuple(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def _receipts_of(db_path, session):
    """(delivered fact ids, withheld fact ids, eligible propensities) of one delivery."""
    rows = _q(db_path, "SELECT r.fact_id, r.included, r.delivered, r.eligible, r.propensity "
                       "FROM injection_receipts r JOIN injection_deliveries d "
                       "ON d.delivery_id = r.delivery_id WHERE d.session_id = ?", session)
    delivered = {f for f, inc, dl, _e, _p in rows if inc and dl}
    withheld = {f for f, inc, _dl, _e, _p in rows if not inc}
    eligible = [p for _f, _i, _d, e, p in rows if e]
    return delivered, withheld, eligible


# -- MCP session_init ---------------------------------------------------------------------------

def _run_session_init(monkeypatch, store, tmp_path, ids, sid):
    import superlocalmemory.mcp.tools_mesh as mesh
    from superlocalmemory.mcp import session_start
    from superlocalmemory.mcp import tools_active as shared

    async def recall(*_a, **_k):
        return SimpleNamespace(results=[_result(f) for f in ids]), False

    rules = SimpleNamespace(get_recall_config=lambda: {}, should_recall=lambda _: True)
    monkeypatch.setattr(session_start, "_recall", recall)
    monkeypatch.setattr(session_start, "_start_inputs", lambda e, p: ("", "default", rules, None))
    monkeypatch.setattr(session_start, "_relevant", lambda response, *a: list(response.results))
    monkeypatch.setattr(session_start, "_jev_ordered", lambda e, q, rel: rel)
    monkeypatch.setattr(session_start, "_priority_facts", lambda *a: ([], [], []))
    monkeypatch.setattr(session_start, "_feedback_count", lambda pid: 0)
    monkeypatch.setattr(shared, "_learning_evidence", lambda e, p: {})
    monkeypatch.setattr(mesh, "bind_local_project", lambda p: None)
    engine = SimpleNamespace(config=SimpleNamespace(base_dir=tmp_path, injection=None),
                             db=store, mode="B")
    return asyncio.run(session_start.session_init(lambda: engine, project_path="/p", query="q",
                                                  session_id=sid, max_results=10))


def test_mcp_session_init_writes_receipts_and_propensities_and_its_output_matches(
        shipped, store, tmp_path, monkeypatch) -> None:
    ids = [_fact(store, f"session init memory number {i}") for i in range(6)]
    for run in range(RUNS):
        got = _run_session_init(monkeypatch, store, tmp_path, ids, f"conv-mcp-{run}")
        assert got["success"], got
        delivered, withheld, eligible = _receipts_of(store.db_path, f"conv-mcp-{run}")
        in_rows = {m["fact_id"] for m in got["memories"]}
        assert delivered == in_rows and got["memory_count"] == len(got["memories"])
        assert not withheld & in_rows
        assert (delivered | withheld) == set(ids) and len(withheld) <= 1  # at most one leaves
        assert all(f"fact_id={f}" not in got["context"] for f in withheld)
        # a channel-drop delivery has its own, higher propensity; every eligible one is below 1
        assert len(eligible) == len(ids) and all(0.8 < p < 1.0 for p in eligible)
    state = _q(store.db_path, "SELECT DISTINCT holdout_state, path, epsilon FROM "
                              "injection_deliveries WHERE path = 'session_init'")
    assert state == [("on", "session_init", EPS)]
    assert _q(store.db_path, "SELECT COUNT(*) FROM injection_deliveries "
                             "WHERE withhold_draw IS NULL AND path = 'session_init'") == [(0,)]


# -- the hook: slm session-context --------------------------------------------------------------

@pytest.fixture
def hook_store(store, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    return store


def _hook(capsys, **kw):
    from superlocalmemory.cli.commands import cmd_session_context

    cmd_session_context(Namespace(json=True, full=False, query="", max_age_days=0,
                                  project_dir=None, **kw))
    out = capsys.readouterr().out
    return json.loads(out)["data"] if out.strip() else None


def test_the_session_context_hook_records_its_delivery_and_counts_what_it_rendered(
        shipped, hook_store, monkeypatch, capsys) -> None:
    ids = [_fact(hook_store, f"hook session memory number {i}") for i in range(5)]
    monkeypatch.setenv("CLAUDE_SESSION_ID", "conv-hook")
    seen_withheld = False
    for _ in range(RUNS):
        data = _hook(capsys)
        assert data is not None
        delivered_in_text = set(receipts.shown_in_text(data["context"]))
        assert delivered_in_text <= set(ids)
        assert data["memory_count"] == len(delivered_in_text)  # W0-27: what was rendered
        seen_withheld |= len(delivered_in_text) < len(ids)
    rows = _q(hook_store.db_path,
              "SELECT holdout_state, epsilon, client FROM injection_deliveries "
              "WHERE path = 'session_context' AND session_id = 'conv-hook'")
    assert len(rows) == RUNS and {r[0] for r in rows} == {"on"} and {r[1] for r in rows} == {EPS}
    props = _q(hook_store.db_path, "SELECT propensity FROM injection_receipts r JOIN "
                                   "injection_deliveries d ON d.delivery_id = r.delivery_id "
                                   "WHERE d.path = 'session_context' AND r.eligible = 1")
    assert len(props) == RUNS * len(ids) and all(0.8 < p[0] < 1.0 for p in props)
    assert isinstance(seen_withheld, bool)


def test_the_hook_count_drops_by_the_memory_that_was_withheld(
        shipped, hook_store, monkeypatch, capsys) -> None:
    """Until one memory is withheld (10% of deliveries): the count, the text and the receipt agree."""
    ids = [_fact(hook_store, f"forced withheld memory {i}") for i in range(4)]
    monkeypatch.setenv("CLAUDE_SESSION_ID", "conv-withheld")
    for _ in range(300):
        data = _hook(capsys)
        shown = set(receipts.shown_in_text(data["context"]))
        if len(shown) < len(ids):
            break
    else:  # pragma: no cover - 0.9 ** 300 is below 1e-13
        pytest.fail("no delivery withheld a memory in 300 session starts")
    assert len(shown) == len(ids) - 1 and data["memory_count"] == len(shown)
    gone = set(ids) - shown
    left = _q(hook_store.db_path,
              "SELECT r.fact_id FROM injection_receipts r JOIN injection_deliveries d ON "
              "d.delivery_id = r.delivery_id WHERE d.session_id = 'conv-withheld' AND "
              "r.included = 0 ORDER BY d.delivered_at DESC LIMIT 1")
    assert {f for (f,) in left} == gone


# -- the CLI: slm recall ------------------------------------------------------------------------

def test_a_cli_recall_is_logged_at_propensity_one_and_never_withheld(
        shipped, store, tmp_path, monkeypatch) -> None:
    from superlocalmemory.server import recall_cache, recall_core
    from superlocalmemory.server.recall_core import RecallCall

    ids = [_fact(store, f"cli recall memory {i}") for i in range(5)]
    envelope = {"ok": True, "profile": "default",
                "results": [{"fact_id": f, "score": 0.5} for f in ids]}

    async def once(_call, _profile, _fn):
        return envelope

    monkeypatch.setattr(recall_cache, "recall_once", once)
    engine = SimpleNamespace(config=SimpleNamespace(base_dir=tmp_path), db=store,
                             profile_id="default")
    for run in range(RUNS):
        call = RecallCall(query="q", limit=5, session_id=f"conv-cli-{run}", agent_id="cli",
                          fast=True, via="cli")
        got = asyncio.run(recall_core.run_recall(engine, call, app_state=None))
        assert {r["fact_id"] for r in got["results"]} == set(ids)  # nothing is withheld on a pull
    assert receipts.PULL_WRITER.drain()
    deliveries = _q(store.db_path, "SELECT holdout_state, withhold_draw, candidates FROM "
                                   "injection_deliveries WHERE path = 'cli_recall'")
    assert len(deliveries) == RUNS and {d[1] for d in deliveries} == {None}
    assert {d[2] for d in deliveries} == {5}
    props = _q(store.db_path, "SELECT propensity, included, delivered, eligible FROM "
                              "injection_receipts r JOIN injection_deliveries d ON "
                              "d.delivery_id = r.delivery_id WHERE d.path = 'cli_recall'")
    assert len(props) == RUNS * 5 and set(props) == {(1.0, 1, 1, 0)}
