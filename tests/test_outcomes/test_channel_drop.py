# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-3 (D-44, design ``one-out-v2``): one retrieval channel dropped at random, on record.

A treated session start (10%, D-28) gets one change: a memory withheld or a
channel switched off for its fetch, never both. Only ``session_init`` runs
retrieval, so only it has the channel arm (50/50); the hook's start keeps the
memory arm at the full 10%. Every probability is logged; a drop is honoured only
on an internal fetch; a drop whose receipt could not be stored is undone.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from types import SimpleNamespace

import pytest

from superlocalmemory.outcomes import randomiser as r
from superlocalmemory.outcomes import receipts
from superlocalmemory.retrieval import channel_ablation as ab
from tests.test_outcomes.test_injection_receipts import KEY, _fact, _rows

EPS = r.EPSILON


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


# -- the design -----------------------------------------------------------------------------------

def test_one_change_per_treated_delivery_and_the_shares_match_the_design() -> None:
    n = 20_000
    arms, channels = Counter(), Counter()
    for i in range(n):
        t = r.treat(KEY, f"d{i}", "session_init", holdout_on=True)
        assert (t.arm is not None) is (t.withhold_draw < EPS)
        assert (t.channel is not None) is (t.arm == r.CHANNEL_ARM)
        arms[t.arm] += 1
        channels[t.channel] += t.channel is not None
    for arm in (r.MEMORY_ARM, r.CHANNEL_ARM):  # D-28: together at most EPS of deliveries
        e = EPS / 2
        assert abs(arms[arm] - n * e) < 5 * math.sqrt(n * e * (1 - e))
    q = EPS / 2 / len(r.DROPPABLE_CHANNELS)
    assert set(channels) - {None} <= set(r.DROPPABLE_CHANNELS)
    assert all(abs(channels[c] - n * q) < 5 * math.sqrt(n * q * (1 - q))
               for c in r.DROPPABLE_CHANNELS)
    assert r.channel_drop_probability(r.treat(KEY, "x", "session_init", holdout_on=True)) == \
        pytest.approx(q)


def test_a_start_without_retrieval_never_drops_a_channel() -> None:
    for i in range(3000):
        t = r.treat(KEY, f"d{i}", "session_context", holdout_on=True)
        assert t.channel is None and t.arm in (None, r.MEMORY_ARM) and t.memory_share == 1.0


@pytest.mark.parametrize("kw", [{"path": "mcp_recall", "holdout_on": True},
                                {"path": "session_init", "holdout_on": False}])
def test_a_pull_or_the_holdout_off_is_never_treated(kw) -> None:
    for i in range(500):
        assert r.treat(KEY, f"d{i}", **kw) == r.UNTREATED


def test_after_the_fetch_only_the_memory_arm_is_possible() -> None:
    for i in range(3000):
        t = r.treat(KEY, f"d{i}", "session_init", holdout_on=True, channel_arm=False)
        assert t.channel is None and t.memory_share == 1.0


def test_media_is_never_dropped() -> None:
    assert "media" not in r.DROPPABLE_CHANNELS and not ab.droppable("media")


# -- the drop reaches retrieval, only where it may ------------------------------------------------

def test_the_drop_holds_inside_its_context_only() -> None:
    assert ab.current() == frozenset()
    with ab.dropping("bm25"):
        assert ab.current() == {"bm25"}
    with ab.dropping("media"), ab.dropping(None):
        assert ab.current() == frozenset()
    assert ab.current() == frozenset()


@pytest.mark.parametrize(("via", "drop", "seen"), [
    ("internal", "semantic", {"semantic"}), ("mcp", "semantic", set()),
    ("internal", "media", set()), ("internal", "", set())])
def test_only_an_internal_fetch_may_drop_and_only_a_droppable_channel(via, drop, seen) -> None:
    from superlocalmemory.server.recall_core import RecallCall, _engine_call

    got: list = []
    engine = SimpleNamespace(recall=lambda *a, **k: got.append(set(ab.current())))
    _engine_call(engine, RecallCall(query="q", limit=3, session_id="", agent_id="a", fast=True,
                                    via=via, drop_channel=drop))
    assert got == [seen]


def test_a_dropped_fetch_never_shares_a_cached_answer() -> None:
    from superlocalmemory.server import recall_cache

    assert "drop_channel" not in recall_cache._NOT_IN_KEY


from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_engine_skips_the_dropped_channel_and_says_so(mode_a_engine) -> None:  # noqa: F811
    from superlocalmemory.retrieval import channel_status as chstat

    mode_a_engine.store("The invoice run happens on the last working day of the month.")
    with ab.dropping("bm25"):
        dropped = mode_a_engine.recall("invoice run")
    full = mode_a_engine.recall("invoice run")
    assert dropped.channel_status.get("bm25") == chstat.DISABLED
    assert full.channel_status.get("bm25") != chstat.DISABLED


# -- the record -----------------------------------------------------------------------------------

def _begun(tmp_path, *, arm):
    for i in range(5000):
        b = receipts.begin(slm_home=tmp_path, path="session_init")
        if b is not None and b.treatment.arm == arm:
            return b
    raise AssertionError(f"no {arm} arm in 5000 draws")


def test_a_channel_drop_is_on_record_with_nothing_withheld(store, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "on")
    ids = [_fact(store, f"memory number {i}") for i in range(4)]
    b = _begun(tmp_path, arm=r.CHANNEL_ARM)
    p = receipts.plan(slm_home=tmp_path, db_path=store.db_path, profile_id="default",
                      session_id="conv", path="session_init", query="q",
                      candidates=[r.Candidate(f, "recall", 0.5) for f in ids], begun=b)
    assert p.delivery_id == b.delivery_id and not p.withheld
    assert receipts.record(store.db_path, p, frozenset(ids))
    ((arm, channel, share, design),) = _rows(
        store.db_path, "SELECT arm, dropped_channel, memory_share, design FROM "
                       "injection_deliveries")
    assert (arm, channel, share, design) == ("channel", b.treatment.channel, 0.5, "one-out-v2")
    props = [x[0] for x in _rows(store.db_path, "SELECT propensity FROM injection_receipts")]
    assert len(props) == 4 and all(p == pytest.approx(1 - EPS * 0.5 / 4) for p in props)


def test_the_schema_refuses_a_channel_without_an_arm_or_draws(store) -> None:
    import sqlite3

    conn = sqlite3.connect(str(store.db_path))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
                     " holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                     " dropped_channel) VALUES ('d', 'p', 'session_init', 'x', 'on', 0.1, 0,"
                     " 0, 't', 'bm25')")
    conn.close()


# -- session_init: draw first, fetch with the drop, undo it if unrecorded -------------------------

def _session_init_with(monkeypatch, *, channel, stored):
    from superlocalmemory.mcp import session_start

    fetches: list = []

    async def fake_recall(*a, drop_channel=None, **k):
        fetches.append(drop_channel)
        return SimpleNamespace(results=[]), False

    begun = receipts.Begun("dlv_x", r.Treatment(0.01, 0.9, 0.5, "channel", channel, 0.5))
    rules = SimpleNamespace(get_recall_config=lambda: {}, should_recall=lambda _: True)
    monkeypatch.setattr(session_start, "_recall", fake_recall)
    monkeypatch.setattr(session_start, "_begin", lambda engine: begun)
    monkeypatch.setattr(session_start, "_start_inputs", lambda e, p: ("", "default", rules, None))
    monkeypatch.setattr(session_start, "_relevant", lambda *a: [])
    monkeypatch.setattr(session_start, "_jev_ordered", lambda e, q, rel: rel)
    monkeypatch.setattr(session_start, "_priority_facts", lambda *a: ([], [], []))
    monkeypatch.setattr(session_start, "_deliver_session",
                        lambda *a: (("ctx", []), stored if a[-1] is begun else True))
    monkeypatch.setattr(session_start, "_envelope", lambda *a, **k: {"ok": True})
    import superlocalmemory.mcp.tools_mesh as mesh
    monkeypatch.setattr(mesh, "bind_local_project", lambda p: None)
    asyncio.run(session_start.session_init(lambda: SimpleNamespace(), project_path="/p",
                                           query="q"))
    return fetches


def test_the_fetch_carries_the_drop(monkeypatch) -> None:
    assert _session_init_with(monkeypatch, channel="bm25", stored=True) == ["bm25"]


def test_a_drop_whose_receipt_was_not_stored_is_fetched_again_in_full(monkeypatch) -> None:
    assert _session_init_with(monkeypatch, channel="bm25", stored=False) == ["bm25", None]


def test_the_drop_travels_from_session_init_to_the_daemon_request(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _pool_adapter, session_start
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    seen: dict = {}

    def fake_pool_recall(query, limit=10, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(_pool_adapter, "pool_recall", fake_pool_recall)
    asyncio.run(session_start._recall("q", "", "", "default", 5, 30, drop_channel="bm25"))
    assert seen["drop_channel"] == "bm25" and seen["via"] == "internal"
    paths: list[str] = []
    monkeypatch.setattr(daemon, "daemon_request", lambda m, p, **k: paths.append(p) or {})
    DaemonPoolProxy(port=1).recall("q", via="internal", drop_channel="bm25")
    DaemonPoolProxy(port=1).recall("q")
    assert "drop_channel=bm25" in paths[0] and "drop_channel" not in paths[1]
