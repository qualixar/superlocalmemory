# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-1: an injection receipt on every recall an agent asks for (D-22).

A pull is never randomised: every result is delivered and logged at
propensity 1. The receipt is written by a background writer, so a recall
never waits for it, and one delivery is recorded per caller even when
identical questions shared one run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import FlagState, write_flag
from superlocalmemory.outcomes import randomiser, receipts
from superlocalmemory.server import recall_core
from superlocalmemory.server.recall_core import RecallCall, receipt_path
from tests.test_outcomes.test_injection_receipts import _fact, _rows


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _pull(tmp_path, db_path, results, *, path="mcp_recall", session_id="conv-1", **kw):
    return receipts.Pull(tmp_path, db_path, "default", session_id, path, "where is the key",
                         tuple(results), kw.get("client"))


def _deliveries(db_path: Path) -> list[tuple]:
    return _rows(db_path, "SELECT path, session_id, holdout_state, epsilon, eligible_count, "
                          "withhold_draw, candidates, client FROM injection_deliveries")


def _receipts(db_path: Path) -> list[tuple]:
    return _rows(db_path, "SELECT fact_id, role, position, eligible, exclusion, propensity, "
                          "included, delivered FROM injection_receipts ORDER BY position")


# -- the path set ---------------------------------------------------------------------------------

def test_every_pull_path_writes_receipts_and_none_is_randomised() -> None:
    assert randomiser.PULL_PATHS <= randomiser.PATHS
    assert not randomiser.PULL_PATHS & randomiser.RANDOMISED_PATHS


def test_a_pull_path_never_withholds_even_with_the_holdout_on() -> None:
    cands = [randomiser.Candidate(f"f{i}", "recall", 0.5) for i in range(6)]
    for n in range(300):
        a = randomiser.choose(b"k" * 32, f"d{n}", "mcp_recall", cands, holdout_on=True)
        assert all(c.included and c.propensity == 1.0 for c in a.choices)
        assert {c.exclusion for c in a.choices} == {randomiser.NOT_RANDOMISED_PATH}
        assert a.withhold_draw is None and a.eligible_count == 0


# -- storing one pull -----------------------------------------------------------------------------

def test_a_pull_records_every_result_as_delivered_at_propensity_one(store, tmp_path,
                                                                    monkeypatch) -> None:
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "on")
    a, b = _fact(store, "the spare key is under the mat"), _fact(store, "the gate code is 41")
    assert receipts.record_pull(_pull(tmp_path, store.db_path, [(a, 0.9), (b, None)],
                                      client="mcp:claude"))
    assert _deliveries(store.db_path) == [
        ("mcp_recall", "conv-1", "on", randomiser.EPSILON, 0, None, 2, "mcp:claude")]
    assert _receipts(store.db_path) == [
        (a, "recall", 1, 0, randomiser.NOT_RANDOMISED_PATH, 1.0, 1, 1),
        (b, "recall", 2, 0, randomiser.NOT_RANDOMISED_PATH, 1.0, 1, 1)]


def test_a_recall_that_found_nothing_is_still_on_record(store, tmp_path) -> None:
    assert receipts.record_pull(_pull(tmp_path, store.db_path, []))
    assert [d[0] for d in _deliveries(store.db_path)] == ["mcp_recall"]
    assert _deliveries(store.db_path)[0][6] == 0 and _receipts(store.db_path) == []


def test_no_pull_receipt_when_receipts_are_off(store, tmp_path) -> None:
    write_flag(tmp_path, "outcomes.receipts", FlagState.OFF)
    assert not receipts.record_pull(_pull(tmp_path, store.db_path, [("x", 0.5)]))
    assert _deliveries(store.db_path) == []


def test_a_pull_never_raises_when_the_store_is_missing(tmp_path) -> None:
    assert not receipts.record_pull(_pull(tmp_path, tmp_path / "nope" / "memory.db",
                                          [("x", 0.5)]))


def test_a_plan_that_would_withhold_on_a_pull_is_refused(store, tmp_path, monkeypatch) -> None:
    real = receipts.plan

    def withholding(**kw):
        p = real(**kw)
        c = p.assignment.choices[0]
        forged = randomiser.Choice(c.fact_id, c.role, c.position, c.score, True, None, 0.9, False)
        return receipts.DeliveryPlan(**{**p.__dict__, "assignment": randomiser.Assignment(
            (forged,), 1, 0.1, 0.1)})

    monkeypatch.setattr(receipts, "plan", withholding)
    assert not receipts.record_pull(_pull(tmp_path, store.db_path, [("x", 0.5)]))
    assert _deliveries(store.db_path) == []


# -- from a serialized answer ---------------------------------------------------------------------

def _engine(store, tmp_path):
    return SimpleNamespace(config=SimpleNamespace(base_dir=tmp_path), db=store,
                           profile_id="default")


def test_the_receipt_is_built_from_the_answer_the_agent_received(store, tmp_path) -> None:
    results = [{"fact_id": "a", "score": 0.7}, {"fact_id": "", "score": 0.2},
               {"fact_id": "b", "score": "n/a"}, "not a row", {"score": 0.1}]
    pull = receipts.pull_from_results(_engine(store, tmp_path), path="cli_recall",
                                      profile_id="default", session_id="conv-9", query="q",
                                      results=results, client="")
    assert pull.results == (("a", 0.7), ("b", None))
    assert (pull.path, pull.session_id, pull.client) == ("cli_recall", "conv-9", None)


@pytest.mark.parametrize("sid", ["", "http:1700000000000", "view:abc", "mcp:x"])
def test_an_invented_session_id_is_stored_as_none(store, tmp_path, sid) -> None:
    pull = receipts.pull_from_results(_engine(store, tmp_path), path="http_recall",
                                      profile_id="default", session_id=sid, query="q",
                                      results=[])
    assert pull.session_id is None


def test_no_receipt_without_a_real_store_or_for_a_push_path(store, tmp_path) -> None:
    fake = SimpleNamespace(config=None, db=None)
    assert receipts.pull_from_results(fake, path="mcp_recall", profile_id="p", session_id="s",
                                      query="q", results=[]) is None
    assert receipts.pull_from_results(_engine(store, tmp_path), path="session_init",
                                      profile_id="p", session_id="s", query="q",
                                      results=[]) is None


# -- the background writer ------------------------------------------------------------------------

def test_the_background_writer_stores_what_it_was_given(store, tmp_path) -> None:
    a = _fact(store, "the boiler is serviced in March")
    assert receipts.submit_pull(_pull(tmp_path, store.db_path, [(a, 0.6)], path="hook_recall"))
    assert receipts.PULL_WRITER.drain()
    assert [d[0] for d in _deliveries(store.db_path)] == ["hook_recall"]


def test_the_writer_thread_ends_when_idle_and_restarts_on_the_next_receipt(
        store, tmp_path, monkeypatch) -> None:
    import time

    monkeypatch.setattr(receipts, "PULL_IDLE_S", 0.05)
    writer = receipts._PullWriter()
    assert writer.submit(_pull(tmp_path, store.db_path, []))
    assert writer.drain()
    deadline = time.monotonic() + 2
    while writer._thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer._thread is None
    assert writer.submit(_pull(tmp_path, store.db_path, [])) and writer.drain()
    assert writer.stored == 2 and len(_deliveries(store.db_path)) == 2


def test_a_full_queue_drops_and_counts_and_never_blocks(tmp_path, monkeypatch) -> None:
    writer = receipts._PullWriter()
    monkeypatch.setattr(writer, "_start", lambda: None)  # no thread: the queue only fills
    pull = _pull(tmp_path, tmp_path / "memory.db", [])
    for _ in range(receipts.PULL_QUEUE_MAX):
        assert writer.submit(pull)
    assert not writer.submit(pull) and writer.dropped == 1


# -- which surface it was -------------------------------------------------------------------------

def _call(**kw) -> RecallCall:
    return RecallCall(query="q", limit=5, session_id="conv-1", agent_id="mcp:claude",
                      fast=True, **kw)


@pytest.mark.parametrize(("kw", "path"), [
    ({"via": "mcp"}, "mcp_recall"), ({"via": "cli"}, "cli_recall"),
    ({"via": "hook"}, "hook_recall"), ({}, "http_recall"), ({"via": "evil"}, "http_recall"),
    ({"origin": "view-cli"}, "view_recall"), ({"origin": "view-mcp"}, "view_recall"),
    ({"origin": "dashboard"}, None), ({"origin": "view-dashboard"}, None),
])
def test_each_surface_names_its_path_and_a_person_reading_is_not_a_delivery(kw, path) -> None:
    assert receipt_path(_call(**kw)) == path


def _run(engine, call, envelope, monkeypatch, sent):
    from superlocalmemory.server import recall_cache

    async def once(_call, _profile, _fn):
        return envelope

    monkeypatch.setattr(recall_cache, "recall_once", once)
    monkeypatch.setattr(receipts, "submit_pull", sent.append)
    return asyncio.run(recall_core.run_recall(engine, call, app_state=None))


def test_each_caller_gets_its_own_receipt_even_when_the_run_was_shared(store, tmp_path,
                                                                      monkeypatch) -> None:
    sent: list = []
    envelope = {"ok": True, "profile": "default", "results": [{"fact_id": "a", "score": 0.5}]}
    engine = _engine(store, tmp_path)
    for via in ("mcp", "cli"):
        assert _run(engine, _call(via=via), envelope, monkeypatch, sent) is envelope
    assert [(p.path, p.results) for p in sent] == [("mcp_recall", (("a", 0.5),)),
                                                  ("cli_recall", (("a", 0.5),))]


@pytest.mark.parametrize("envelope", [
    {"ok": False, "code": "RECALL_BUSY"}, {"code": "RECALL_BUSY"}, None, "text"])
def test_a_refused_or_busy_answer_is_not_a_delivery(store, tmp_path, monkeypatch,
                                                    envelope) -> None:
    sent: list = []
    _run(_engine(store, tmp_path), _call(via="mcp"), envelope, monkeypatch, sent)
    assert sent == []


def test_a_dashboard_trace_is_not_a_delivery(store, tmp_path, monkeypatch) -> None:
    sent: list = []
    _run(_engine(store, tmp_path), _call(origin="dashboard"),
         {"ok": True, "results": [{"fact_id": "a"}]}, monkeypatch, sent)
    assert sent == []


def test_a_receipt_failure_never_fails_the_recall(store, tmp_path, monkeypatch) -> None:
    def boom(_pull):
        raise RuntimeError("disk on fire")

    envelope = {"ok": True, "results": [{"fact_id": "a"}]}
    from superlocalmemory.server import recall_cache

    async def once(_c, _p, _f):
        return envelope

    monkeypatch.setattr(recall_cache, "recall_once", once)
    monkeypatch.setattr(receipts, "submit_pull", boom)
    got = asyncio.run(recall_core.run_recall(_engine(store, tmp_path), _call(via="mcp"),
                                             app_state=None))
    assert got is envelope


def test_via_never_splits_a_shared_run() -> None:
    from superlocalmemory.server import recall_cache

    assert "via" in recall_cache._NOT_IN_KEY


# -- the clients name themselves ------------------------------------------------------------------

def test_the_mcp_proxy_says_mcp_and_a_cli_caller_can_say_cli(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    seen: list[str] = []

    def fake(method, path, **_kw):
        seen.append(path)
        return {"ok": True, "results": []}

    monkeypatch.setattr(daemon, "daemon_request", fake)
    proxy = DaemonPoolProxy(port=1)
    proxy.recall("q")
    proxy.recall("q", via="cli")
    assert "via=mcp" in seen[0] and "via=cli" in seen[1]


def test_the_auto_recall_fallback_says_hook(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.hooks import auto_recall_hook

    seen: list[str] = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda m, p, **k: seen.append(p) or {"results": []})
    auto_recall_hook._fallback_response("where is the key", 3, "conv-1")
    assert seen and "via=hook" in seen[0]


def test_the_hook_queue_answers_are_hook_deliveries(store, tmp_path, monkeypatch) -> None:
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    sent: list = []
    monkeypatch.setattr(receipts, "submit_pull", sent.append)
    adapter = EngineRecallAdapter(_engine(store, tmp_path))
    monkeypatch.setattr(adapter, "_recall", lambda q, limit, session_id: {
        "ok": True, "results": [{"fact_id": "a", "score": 0.4}]})
    assert adapter.recall("q", 3, "conv-2")["ok"]
    (pull,) = sent
    assert (pull.path, pull.session_id, pull.results, pull.client) == (
        "hook_recall", "conv-2", (("a", 0.4),), "hook")


def test_a_hook_receipt_failure_never_fails_the_recall(store, tmp_path, monkeypatch) -> None:
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    monkeypatch.setattr(receipts, "submit_pull", lambda _p: 1 / 0)
    adapter = EngineRecallAdapter(_engine(store, tmp_path))
    monkeypatch.setattr(adapter, "_recall", lambda q, limit, session_id: {"ok": True,
                                                                          "results": []})
    assert adapter.recall("q")["ok"]


def test_the_cli_says_cli(capsys) -> None:
    from argparse import Namespace
    from unittest.mock import patch

    from superlocalmemory.cli.commands import cmd_recall

    paths: list[str] = []
    args = Namespace(query="where is the key", limit=5, json=False, fast=False,
                     include_global=None, include_shared=None, window="", as_of="",
                     known_as_of="", valid_at="", include_unknown=False)
    with (patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
          patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
          patch("superlocalmemory.cli.daemon.daemon_request",
                side_effect=lambda m, p, *a, **k: paths.append(p) or {"results": []})):
        cmd_recall(args)
    assert paths and "&via=cli" in paths[0]


# -- a real engine, not a stand-in ----------------------------------------------------------------

from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_engine_exposes_its_store_to_the_receipts(mode_a_engine) -> None:  # noqa: F811
    home, db_path = receipts.engine_paths(mode_a_engine)
    assert home == mode_a_engine._config.base_dir and db_path == mode_a_engine.db.db_path


def test_a_real_hook_recall_is_on_record(mode_a_engine) -> None:  # noqa: F811
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    mode_a_engine.store("The invoice run happens on the last working day of the month.")
    out = EngineRecallAdapter(mode_a_engine).recall("invoice run", 5, "conv-real")
    shown = [r["fact_id"] for r in out["results"]]
    assert shown and receipts.PULL_WRITER.drain()
    db_path = mode_a_engine.db.db_path
    assert _rows(db_path, "SELECT path, session_id, candidates FROM injection_deliveries") == [
        ("hook_recall", "conv-real", len(shown))]
    assert [r[0] for r in _rows(db_path, "SELECT fact_id FROM injection_receipts "
                                         "WHERE delivered = 1 ORDER BY position")] == shown


# -- a fetch whose caller records its own delivery is not a second delivery -----------------------

def test_session_inits_own_fetch_writes_no_daemon_receipt() -> None:
    assert receipt_path(_call(via=recall_core.VIA_INTERNAL)) is None


def test_session_init_fetches_as_internal_and_the_adapter_forwards_it(monkeypatch) -> None:
    import asyncio as _asyncio

    from superlocalmemory.mcp import _pool_adapter, session_start

    seen: dict = {}
    real_pool_recall = _pool_adapter.pool_recall

    def fake_pool_recall(query, limit=10, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(_pool_adapter, "pool_recall", fake_pool_recall)
    _asyncio.run(session_start._recall("q", "", "", "default", 5, 30))
    assert seen.get("via") == recall_core.VIA_INTERNAL

    sent: dict = {}

    class Proxy:
        def recall(self, **kw):
            sent.update(kw)
            return {"ok": True, "results": []}

    monkeypatch.setattr(_pool_adapter, "_pool", lambda: Proxy())
    try:
        real_pool_recall("q", limit=3, via="internal")
    except Exception:  # noqa: BLE001 - the reshape of an empty answer is not under test
        pass
    assert sent.get("via") == "internal"
