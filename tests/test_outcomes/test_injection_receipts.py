# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Injection receipts and randomised withholding (4.2.30 D-14, D-22, D-28; M075)."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import FlagState, write_flag
from superlocalmemory.outcomes import randomiser, receipts
from superlocalmemory.outcomes.randomiser import Candidate
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M075_injection_receipts as M075
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

KEY = b"k" * 32


@pytest.fixture
def holdout_on(monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "on")


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _fact(db, content: str) -> str:
    record = MemoryRecord(content=content, session_date="2026-11-16")
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=content)
    db.store_fact(fact)
    return fact.fact_id


def _rows(db_path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [tuple(r) for r in conn.execute(sql)]
    finally:
        conn.close()


# -- the migration ------------------------------------------------------------------------------

def test_m075_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M075.NAME] is M075
    assert M075.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M075.DOWNGRADE_FLOOR == 51
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M075.verify(conn)
    M075.apply(conn)
    M075.apply(conn)
    assert M075.verify(conn)
    conn.execute("DROP TRIGGER trg_injection_receipts_fact_erased")
    assert not M075.verify(conn)
    M075.repair(conn)
    assert M075.verify(conn)


def _schema() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M075.apply(conn)
    return conn


_RECEIPT = ("INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
            " score, eligible, exclusion, propensity, draw, included, delivered)"
            " VALUES ('d', ?, 'p', 'recall', 1, 0.5, ?, ?, ?, ?, ?, ?)")


@pytest.mark.parametrize("row", [
    ("f1", 0, "pinned", 0.9, None, 1, 1),      # ineligible but p < 1
    ("f2", 0, "pinned", 1.0, None, 0, 0),      # ineligible but withheld
    ("f3", 1, None, 0.9, 0.95, 0, 1),          # withheld yet delivered
    ("f4", 1, None, 0.9, None, 1, 1),          # eligible with no draw
    ("f5", 1, "pinned", 0.9, 0.1, 1, 1),       # eligible with an exclusion
    ("f6", 0, "pinned", 1.0, 0.3, 1, 1),       # ineligible with a draw
    ("f7", 1, None, 0.0, 0.3, 0, 0),           # zero propensity: no IPW weight
])
def test_the_schema_refuses_a_receipt_that_breaks_the_design(row) -> None:
    conn = _schema()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(_RECEIPT, row)


def test_the_schema_refuses_a_withholding_probability_above_the_cap() -> None:
    conn = _schema()
    sql = ("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
           " holdout_state, epsilon, candidates, delivered_at)"
           " VALUES (?, 'p', 'session_init', 'bernoulli-v1', 'on', ?, 0, 'now')")
    conn.execute(sql, ("ok", 0.10))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, ("too-much", 0.11))


# -- the randomiser -----------------------------------------------------------------------------

def test_the_draw_is_reproducible_keyed_and_uniform_in_range() -> None:
    assert randomiser.draw(KEY, "d", "f") == randomiser.draw(KEY, "d", "f")
    assert randomiser.draw(KEY, "d", "f") != randomiser.draw(b"x" * 32, "d", "f")
    assert randomiser.draw(KEY, "d", "f") != randomiser.draw(KEY, "d2", "f")
    assert all(0.0 <= randomiser.draw(KEY, f"d{i}", "f") < 1.0 for i in range(200))


def test_pinned_rules_and_the_core_block_are_never_withheld() -> None:
    for i in range(500):
        choices = randomiser.choose(KEY, f"d{i}", "session_init", [
            Candidate("p", "pinned"), Candidate("s", "standing"), Candidate("c", "core")],
            holdout_on=True)
        assert all(c.included and c.propensity == 1.0 and not c.eligible and c.draw is None
                   for c in choices)
    assert [c.exclusion for c in choices] == ["pinned", "standing_rule", "core_block"]


def test_an_eligible_candidate_is_included_iff_its_draw_is_below_its_logged_probability() -> None:
    for i in range(300):
        (c,) = randomiser.choose(KEY, f"d{i}", "session_context",
                                 [Candidate("f", "recent", 0.4)], holdout_on=True)
        assert c.eligible and c.exclusion is None
        assert c.propensity == pytest.approx(1 - randomiser.EPSILON)
        assert c.included is (c.draw < c.propensity)


def test_the_withheld_share_matches_epsilon_within_five_standard_errors() -> None:
    n = 20_000
    withheld = sum(
        not randomiser.choose(KEY, f"d{i}", "session_init", [Candidate("f", "recall")],
                              holdout_on=True)[0].included
        for i in range(n))
    eps = randomiser.EPSILON
    assert abs(withheld - n * eps) < 5 * math.sqrt(n * eps * (1 - eps))


def test_the_draw_ignores_the_score_so_strong_memories_are_withheld_as_often() -> None:
    for i in range(200):
        low = randomiser.choose(KEY, f"d{i}", "session_init", [Candidate("f", "recall", 0.01)],
                                holdout_on=True)[0]
        high = randomiser.choose(KEY, f"d{i}", "session_init", [Candidate("f", "recall", 0.99)],
                                 holdout_on=True)[0]
        assert low.included is high.included and low.draw == high.draw


def test_holdout_off_logs_every_candidate_with_probability_one() -> None:
    choices = randomiser.choose(KEY, "d", "session_init",
                                [Candidate("a", "recall"), Candidate("b", "session_summary")],
                                holdout_on=False, epsilon=0.0)
    assert all(c.included and c.propensity == 1.0 and c.exclusion == randomiser.HOLDOUT_OFF
               for c in choices)


def test_a_fact_listed_twice_gets_one_choice_and_its_first_role() -> None:
    choices = randomiser.choose(KEY, "d", "session_init", [
        Candidate("a", "pinned"), Candidate("a", "recall"), Candidate("", "recall"),
        Candidate("b", "recall")], holdout_on=True)
    assert [(c.fact_id, c.role, c.position) for c in choices] == [("a", "pinned", 1),
                                                                   ("b", "recall", 2)]


@pytest.mark.parametrize("kwargs", [
    {"path": "web", "candidates": [], "epsilon": 0.1},
    {"path": "session_init", "candidates": [Candidate("a", "guess")], "epsilon": 0.1},
    {"path": "session_init", "candidates": [], "epsilon": 0.11},
])
def test_unknown_paths_roles_and_an_epsilon_above_the_cap_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        randomiser.choose(KEY, "d", kwargs["path"], kwargs["candidates"], holdout_on=True,
                          epsilon=kwargs["epsilon"])


# -- planning -----------------------------------------------------------------------------------

def _plan(home: Path, candidates, **kw):
    return receipts.plan(slm_home=home, profile_id="default", session_id="s1",
                         path=kw.pop("path", "session_init"), query=kw.pop("query", "q"),
                         candidates=candidates, **kw)


def test_no_plan_when_receipts_are_off_or_the_kill_switch_is_set(tmp_path, monkeypatch) -> None:
    write_flag(tmp_path, "outcomes.receipts", FlagState.OFF)
    assert _plan(tmp_path, [Candidate("a", "recall")]) is None
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert _plan(fresh, [Candidate("a", "recall")]) is None


def test_holdout_off_plans_receipts_without_withholding(tmp_path) -> None:
    plan = _plan(tmp_path, [Candidate(f"f{i}", "recall") for i in range(50)])
    assert plan is not None and plan.holdout_state == "off" and plan.epsilon == 0.0
    assert plan.withheld == frozenset()


def test_the_query_is_kept_only_as_a_keyed_hash(tmp_path, holdout_on) -> None:
    plan = _plan(tmp_path, [], query="my secret project")
    assert plan.query_digest and "secret" not in plan.query_digest
    assert _plan(tmp_path, [], query="").query_digest is None


# -- delivering and recording -------------------------------------------------------------------

def _plan_withholding(home: Path, fact_ids: list[str], path="session_init"):
    """A plan that withholds at least one fact (draws are random per delivery)."""
    for _ in range(200):
        plan = _plan(home, [Candidate(f, "recall", 0.5) for f in fact_ids], path=path)
        if plan.withheld:
            return plan
    raise AssertionError("no delivery withheld anything in 200 tries")


def _render(items):
    return "\n".join(f"- {i} (provenance: fact_id={i})" for i in items)


def test_a_withheld_memory_is_not_delivered_and_every_candidate_has_a_receipt(
        store, tmp_path, holdout_on) -> None:
    ids = [_fact(store, f"memory number {i}") for i in range(6)]
    plan = _plan_withholding(tmp_path, ids)
    out = receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i, render=_render,
                           shown=receipts.shown_in_text)
    assert not any(f"fact_id={f}" in out for f in plan.withheld)
    rows = _rows(store.db_path, "SELECT fact_id, eligible, propensity, included, delivered"
                                " FROM injection_receipts ORDER BY position")
    assert [r[0] for r in rows] == ids
    for fact_id, eligible, propensity, included, delivered in rows:
        assert eligible == 1 and propensity == pytest.approx(0.9)
        assert included == int(fact_id not in plan.withheld) == delivered
    ((path, state, eps, n),) = _rows(store.db_path, "SELECT path, holdout_state, epsilon,"
                                                     " candidates FROM injection_deliveries")
    assert (path, state, eps, n) == ("session_init", "on", pytest.approx(0.1), 6)


def test_nothing_is_withheld_when_the_receipt_cannot_be_stored(tmp_path, holdout_on) -> None:
    bare = tmp_path / "bare.db"
    sqlite3.connect(str(bare)).close()  # no M075 tables: the write fails
    ids = [f"f{i}" for i in range(6)]
    plan = _plan_withholding(tmp_path, ids)
    out = receipts.deliver(plan, bare, ids, fact_id=lambda i: i, render=_render,
                           shown=receipts.shown_in_text)
    assert receipts.shown_in_text(out) == frozenset(ids)


def test_a_candidate_the_renderer_dropped_is_recorded_as_not_delivered(store, tmp_path) -> None:
    ids = [_fact(store, f"budgeted memory {i}") for i in range(3)]
    plan = _plan(tmp_path, [Candidate(f, "recall") for f in ids])
    receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i,
                     render=lambda items: _render(items[:1]), shown=receipts.shown_in_text)
    assert _rows(store.db_path, "SELECT included, delivered FROM injection_receipts"
                                " ORDER BY position") == [(1, 1), (1, 0), (1, 0)]


def test_erasing_a_fact_erases_its_receipts_and_keeps_the_delivery(store, tmp_path) -> None:
    ids = [_fact(store, f"erasable memory {i}") for i in range(2)]
    plan = _plan(tmp_path, [Candidate(f, "recall") for f in ids])
    receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i, render=_render,
                     shown=receipts.shown_in_text)
    store.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert _rows(store.db_path, "SELECT fact_id FROM injection_receipts") == [(ids[1],)]
    assert len(_rows(store.db_path, "SELECT delivery_id FROM injection_deliveries")) == 1


def test_a_busy_writer_skips_the_receipt_within_its_wait_budget(store, tmp_path,
                                                                holdout_on) -> None:
    import threading
    import time

    from superlocalmemory.storage.write_lock import get_write_lock

    ids = [f"f{i}" for i in range(6)]
    plan = _plan_withholding(tmp_path, ids)
    lock = get_write_lock(store.db_path)
    held, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(5)
    try:
        started = time.monotonic()
        out = receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i, render=_render,
                               shown=receipts.shown_in_text)
        assert time.monotonic() - started < receipts.WRITE_WAIT_S + 0.5
    finally:
        release.set()
        thread.join(5)
    assert receipts.shown_in_text(out) == frozenset(ids)
    assert _rows(store.db_path, "SELECT COUNT(*) FROM injection_receipts") == [(0,)]


# -- the two session-start surfaces -------------------------------------------------------------

def _result(fact_id: str, score: float = 0.8):
    fact = SimpleNamespace(fact_id=fact_id, content=f"recalled memory {fact_id}",
                           memory_id=f"m-{fact_id}", created_at="2026-11-16T00:00:00",
                           importance=0.5, access_count=0)
    return SimpleNamespace(fact=fact, score=score, tier="")


def test_session_init_withholds_recall_rows_on_record_but_never_pins(store, tmp_path,
                                                                     holdout_on) -> None:
    from superlocalmemory.mcp import session_start

    pin_id = _fact(store, "a pinned rule that is always delivered")
    pin = SimpleNamespace(fact_id=pin_id, content="a pinned rule that is always delivered",
                          importance=1.0, access_count=0, kind="rule")
    engine = SimpleNamespace(config=SimpleNamespace(base_dir=tmp_path, injection=None),
                             db=store, mode="B")
    response = SimpleNamespace(results=[], calibration_status="uncalibrated")
    relevant = [_result(_fact(store, f"recall candidate {i}")) for i in range(8)]
    for attempt in range(100):
        context, rows = session_start._deliver_session(
            engine, "", "default", f"s{attempt}", "query", response, [pin], [], relevant, 10)
        withheld = _rows(store.db_path, "SELECT fact_id FROM injection_receipts"
                                        " WHERE included = 0")
        if withheld:
            break
    else:
        raise AssertionError("no session withheld a recall row in 100 sessions")
    shown = {r["fact_id"] for r in rows}
    assert pin_id in shown and f"fact_id={pin_id}" in context
    assert not {w for (w,) in withheld} & shown
    roles = dict(_rows(store.db_path, "SELECT role, MAX(propensity) FROM injection_receipts"
                                      " GROUP BY role"))
    assert roles["pinned"] == 1.0 and roles["recall"] == pytest.approx(0.9)
    assert _rows(store.db_path, "SELECT DISTINCT path FROM injection_deliveries") == [
        ("session_init",)]


def test_session_init_without_a_real_store_delivers_everything(tmp_path, holdout_on) -> None:
    from superlocalmemory.mcp import session_start

    engine = SimpleNamespace(config=SimpleNamespace(base_dir="not-a-path", injection=None),
                             db=object(), mode="B")
    response = SimpleNamespace(results=[], calibration_status="uncalibrated")
    relevant = [_result(f"f{i}") for i in range(8)]
    _, rows = session_start._deliver_session(engine, "", "default", "s", "q", response,
                                             [], [], relevant, 10)
    assert {r["fact_id"] for r in rows} == {f"f{i}" for i in range(8)}


def test_the_session_start_hook_context_goes_through_the_receipts(store, tmp_path,
                                                                  monkeypatch) -> None:
    from superlocalmemory.cli import commands
    from superlocalmemory.core.injection import InjectableMemory, render_context

    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-session-7")
    ids = [_fact(store, f"recent important memory {i}") for i in range(3)]
    mems = [("recent", InjectableMemory(content=f"recent important memory {i}", score=0.5,
                                        fact_id=f)) for i, f in enumerate(ids)]
    config = SimpleNamespace(base_dir=tmp_path)
    out = commands._session_context_delivery(
        config, store.db_path, "default", "", mems,
        lambda kept: render_context([m for _, m in kept], mode="B", wrap=True))
    assert receipts.shown_in_text(out) == frozenset(ids)
    assert _rows(store.db_path, "SELECT path, session_id, holdout_state FROM"
                                " injection_deliveries") == [
        ("session_context", "claude-session-7", "off")]
    assert _rows(store.db_path, "SELECT role, delivered FROM injection_receipts") == [
        ("recent", 1)] * 3



def test_a_key_another_process_created_first_is_read_not_an_error(tmp_path, monkeypatch) -> None:
    from superlocalmemory.core import receipt_key

    def racing_open(path, flags, mode=0o777):
        Path(path).write_bytes(b"z" * 32)  # the other process won the race
        raise FileExistsError(path)

    monkeypatch.setattr(receipt_key.os, "open", racing_open)
    assert receipt_key.load_or_create(tmp_path) == b"z" * 32
