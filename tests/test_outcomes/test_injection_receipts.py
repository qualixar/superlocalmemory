# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Injection receipts and randomised withholding (4.2.30 D-14, D-22, D-33, D-34; M075)."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
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
EPS = randomiser.EPSILON


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
            " score, eligible, exclusion, propensity, included, delivered)"
            " VALUES ('d', ?, 'p', 'recall', 1, 0.5, ?, ?, ?, ?, ?)")


@pytest.mark.parametrize("row", [
    ("f1", 0, "pinned", 0.9, 1, 1),     # ineligible but p < 1
    ("f2", 0, "pinned", 1.0, 0, 0),     # ineligible but withheld
    ("f3", 1, None, 0.9, 0, 1),         # withheld yet delivered
    ("f4", 1, "pinned", 0.9, 1, 1),     # eligible with an exclusion
    ("f5", 1, None, 0.0, 0, 0),         # zero propensity: no IPW weight
])
def test_the_schema_refuses_a_receipt_that_breaks_the_design(row) -> None:
    conn = _schema()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(_RECEIPT, row)


_DELIVERY = ("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
             " holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
             " candidates, delivered_at) VALUES (?, 'p', 'session_init', 'one-out-v1', 'on',"
             " ?, 3, ?, ?, ?, 3, 'now')")


@pytest.mark.parametrize("row", [
    ("too-much", 0.11, 0.5, 0.5, "k"),   # above the D-28 cap
    ("one-draw", 0.10, 0.5, None, "k"),  # the two draws come together
    ("no-key", 0.10, 0.5, 0.5, None),    # a draw nobody can check
    ("bad-draw", 0.10, 1.0, 0.5, "k"),   # a draw outside [0, 1)
])
def test_the_schema_refuses_a_delivery_that_breaks_the_design(row) -> None:
    conn = _schema()
    conn.execute(_DELIVERY, ("ok", 0.10, 0.5, 0.5, "k"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(_DELIVERY, row)


# -- the randomiser (one-out-v1) ----------------------------------------------------------------

def test_the_draw_is_reproducible_keyed_and_uniform_in_range() -> None:
    assert randomiser.draw(KEY, "d", "pick") == randomiser.draw(KEY, "d", "pick")
    assert randomiser.draw(KEY, "d", "pick") != randomiser.draw(b"x" * 32, "d", "pick")
    assert randomiser.draw(KEY, "d", "pick") != randomiser.draw(KEY, "d", "withhold")
    assert all(0.0 <= randomiser.draw(KEY, f"d{i}", "pick") < 1.0 for i in range(200))


def _cands(k: int, role="recall") -> list[Candidate]:
    return [Candidate(f"f{i}", role, 0.5) for i in range(k)]


def test_pinned_rules_and_the_core_block_are_never_withheld() -> None:
    for i in range(2000):
        a = randomiser.choose(KEY, f"d{i}", "session_init", [
            Candidate("p", "pinned"), Candidate("s", "standing"), Candidate("c", "core"),
            Candidate("r", "recall")], holdout_on=True)
        assert all(c.included and c.propensity == 1.0 and not c.eligible
                   for c in a.choices[:3])
    assert [c.exclusion for c in a.choices[:3]] == ["pinned", "standing_rule", "core_block"]


#: P(memory arm | treated) per path (D-44): session_init shares with the channel arm.
SHARE = {"session_init": 0.5, "session_context": 1.0}


@pytest.mark.parametrize("path", sorted(SHARE))
def test_at_most_one_memory_leaves_a_delivery_with_its_exact_logged_probability(path) -> None:
    for i in range(3000):
        a = randomiser.choose(KEY, f"d{i}", path, _cands(7), holdout_on=True)
        withheld = [c for c in a.choices if not c.included]
        memory_arm = a.withhold_draw < EPS and a.treatment.arm == randomiser.MEMORY_ARM
        assert len(withheld) == (1 if memory_arm else 0)
        assert a.eligible_count == 7
        assert all(c.propensity == pytest.approx(1 - EPS * SHARE[path] / 7) for c in a.choices)
        if withheld:
            assert withheld[0].position - 1 == int(a.pick_draw * 7)


@pytest.mark.parametrize("path", sorted(SHARE))
def test_the_share_of_deliveries_and_of_each_candidate_matches_the_design(path) -> None:
    n, k = 20_000, 10
    lost, by_position = 0, Counter()
    for i in range(n):
        out = [c.position for c in randomiser.choose(
            KEY, f"d{i}", path, _cands(k), holdout_on=True).choices if not c.included]
        lost += bool(out)
        by_position.update(out)
    e = EPS * SHARE[path]
    assert abs(lost - n * e) < 5 * math.sqrt(n * e * (1 - e))  # D-28: <= 0.10 of deliveries
    q = e / k
    assert all(abs(by_position[p] - n * q) < 5 * math.sqrt(n * q * (1 - q))
               for p in range(1, k + 1))  # uniform: each candidate's 1 - p_i is honest


def test_the_draw_ignores_the_score_so_strong_memories_are_withheld_as_often() -> None:
    for i in range(300):
        low = randomiser.choose(KEY, f"d{i}", "session_init", [Candidate("f", "recall", 0.01)],
                                holdout_on=True)
        high = randomiser.choose(KEY, f"d{i}", "session_init", [Candidate("f", "recall", 0.99)],
                                 holdout_on=True)
        assert low.choices[0].included is high.choices[0].included


def test_holdout_off_draws_nothing_and_logs_probability_one() -> None:
    a = randomiser.choose(KEY, "d", "session_init", _cands(3), holdout_on=False, epsilon=0.0)
    assert (a.eligible_count, a.withhold_draw, a.pick_draw) == (0, None, None)
    assert all(c.included and c.propensity == 1.0 and c.exclusion == randomiser.HOLDOUT_OFF
               for c in a.choices)


def test_a_fact_listed_twice_gets_one_choice_and_its_first_role() -> None:
    a = randomiser.choose(KEY, "d", "session_init", [
        Candidate("a", "pinned"), Candidate("a", "recall"), Candidate("", "recall"),
        Candidate("b", "recall")], holdout_on=True)
    assert [(c.fact_id, c.role, c.position) for c in a.choices] == [("a", "pinned", 1),
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


# -- planning: defaults and protection from the store (D-34, W0-01) -----------------------------

def _plan(home: Path, candidates, db_path=None, **kw):
    return receipts.plan(slm_home=home, db_path=db_path, profile_id="default",
                         session_id=kw.pop("session_id", "s1"),
                         path=kw.pop("path", "session_init"), query=kw.pop("query", "q"),
                         candidates=candidates, **kw)


def test_no_plan_when_receipts_are_off_or_the_kill_switch_is_set(tmp_path, monkeypatch) -> None:
    write_flag(tmp_path, "outcomes.receipts", FlagState.OFF)
    assert _plan(tmp_path, [Candidate("a", "recall")]) is None
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert _plan(fresh, [Candidate("a", "recall")]) is None


def test_withholding_is_off_by_default_until_its_shipping_conditions_exist(
        store, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SLM_FEATURE_OUTCOMES_HOLDOUT", raising=False)  # the shipped default
    ids = [_fact(store, f"default memory {i}") for i in range(10)]
    for _ in range(50):
        plan = _plan(tmp_path, [Candidate(f, "recall") for f in ids], db_path=store.db_path)
        assert plan.holdout_state == "off" and plan.withheld == frozenset()


def test_the_query_is_kept_only_as_a_keyed_digest(tmp_path, holdout_on) -> None:
    plan = _plan(tmp_path, [], query="my secret project")
    assert plan.query_digest and "secret" not in plan.query_digest
    assert _plan(tmp_path, [], query="").query_digest is None


def _protect_in_store(db, fact_id: str, how: str) -> None:
    if how == "pinned_table":
        from superlocalmemory.storage import schema_v3411

        with db.raw_connection() as conn:  # the store this test builds has no pin table yet
            conn.executescript(schema_v3411._PINNED_FACTS_DDL)
        db.execute("INSERT INTO pinned_facts (fact_id, profile_id, pinned_at) "
                   "VALUES (?, 'default', 'now')", (fact_id,))
    elif how == "pinned_column":
        db.execute("UPDATE atomic_facts SET pinned = 1 WHERE fact_id = ?", (fact_id,))
    else:  # a standing rule the user confirmed, whatever the session config says
        db.execute("UPDATE atomic_facts SET memory_kind = 'rule', memory_kind_source = 'user' "
                   "WHERE fact_id = ?", (fact_id,))


@pytest.mark.parametrize("how,role", [("pinned_table", "pinned"), ("pinned_column", "pinned"),
                                      ("standing", "standing")])
def test_a_protected_fact_is_never_withheld_however_the_surface_labels_it(
        store, tmp_path, holdout_on, how, role) -> None:
    protected = _fact(store, "a rule the user pinned or confirmed")
    _protect_in_store(store, protected, how)
    others = [_fact(store, f"plain memory {i}") for i in range(2)]
    withheld_any = False
    for _ in range(300):  # with k = 2 eligible, a delivery loses one 10% of the time
        plan = _plan(tmp_path, [Candidate(protected, "recent")] + [
            Candidate(f, "recent") for f in others], db_path=store.db_path)
        choice = plan.choices[0]
        assert (choice.role, choice.eligible, choice.included) == (role, False, True)
        withheld_any |= bool(plan.withheld)
    assert withheld_any  # the others are still randomised


def test_an_unreadable_store_means_nothing_is_withheld(tmp_path, holdout_on) -> None:
    bare = tmp_path / "bare.db"
    sqlite3.connect(str(bare)).close()
    for _ in range(100):
        plan = _plan(tmp_path, _cands(5), db_path=bare)
        assert plan.holdout_state == "off" and plan.withheld == frozenset()


# -- delivering and recording -------------------------------------------------------------------

def _plan_withholding(home: Path, fact_ids: list[str], db_path, path="session_init", **kw):
    """A plan that withholds one fact (one delivery in ten does)."""
    for _ in range(400):
        plan = _plan(home, [Candidate(f, "recall", 0.5) for f in fact_ids], db_path=db_path,
                     path=path, **kw)
        if plan.withheld:
            return plan
    raise AssertionError("no delivery withheld anything in 400 tries")


def _render(items):
    return "\n".join(f"- {i} (provenance: fact_id={i})" for i in items)


def test_a_withheld_memory_is_not_delivered_and_every_candidate_has_a_receipt(
        store, tmp_path, holdout_on) -> None:
    ids = [_fact(store, f"memory number {i}") for i in range(6)]
    plan = _plan_withholding(tmp_path, ids, store.db_path, session_id="", client="claude-code")
    out = receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i, render=_render,
                           shown=receipts.shown_in_text)
    (gone,) = plan.withheld
    assert f"fact_id={gone}" not in out
    rows = _rows(store.db_path, "SELECT fact_id, eligible, propensity, included, delivered"
                                " FROM injection_receipts ORDER BY position")
    assert [r[0] for r in rows] == ids
    for fact_id, eligible, propensity, included, delivered in rows:
        assert eligible == 1 and propensity == pytest.approx(1 - EPS / 6)  # planned after the fetch: memory arm only
        assert included == int(fact_id != gone) == delivered
    ((row),) = _rows(store.db_path, "SELECT path, design, holdout_state, epsilon, "
                                     "eligible_count, withhold_draw < epsilon, key_id IS NOT NULL,"
                                     " session_id, client FROM injection_deliveries")
    assert row == ("session_init", "one-out-v2", "on", pytest.approx(EPS), 6, 1, 1, None,
                   "claude-code")


def test_nothing_is_withheld_when_the_receipt_cannot_be_stored(store, tmp_path,
                                                               holdout_on) -> None:
    ids = [_fact(store, f"unrecorded memory {i}") for i in range(6)]
    plan = _plan_withholding(tmp_path, ids, store.db_path)
    store.execute("DROP TABLE injection_receipts")  # the write will fail
    out = receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i, render=_render,
                           shown=receipts.shown_in_text)
    assert receipts.shown_in_text(out) == frozenset(ids)


def test_a_candidate_the_renderer_dropped_is_recorded_as_not_delivered(store, tmp_path) -> None:
    ids = [_fact(store, f"budgeted memory {i}") for i in range(3)]
    plan = _plan(tmp_path, [Candidate(f, "recall") for f in ids], db_path=store.db_path)
    receipts.deliver(plan, store.db_path, ids, fact_id=lambda i: i,
                     render=lambda items: _render(items[:1]), shown=receipts.shown_in_text)
    assert _rows(store.db_path, "SELECT included, delivered FROM injection_receipts"
                                " ORDER BY position") == [(1, 1), (1, 0), (1, 0)]


def test_erasing_a_fact_erases_its_receipts_and_keeps_the_delivery(store, tmp_path) -> None:
    ids = [_fact(store, f"erasable memory {i}") for i in range(2)]
    plan = _plan(tmp_path, [Candidate(f, "recall") for f in ids], db_path=store.db_path)
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

    ids = [_fact(store, f"busy memory {i}") for i in range(6)]
    plan = _plan_withholding(tmp_path, ids, store.db_path)
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
        assert time.monotonic() - started < receipts.WRITE_WAIT_S + 0.25
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


def _engine(store, tmp_path):
    return SimpleNamespace(config=SimpleNamespace(base_dir=tmp_path, injection=None),
                           db=store, mode="B")


def test_session_init_never_withholds_a_pin_that_arrives_through_recall(store, tmp_path,
                                                                        holdout_on) -> None:
    from superlocalmemory.mcp import session_start

    pinned = _fact(store, "pin number twenty-one, reached by recall")
    _protect_in_store(store, pinned, "pinned_table")
    relevant = [_result(pinned)] + [_result(_fact(store, f"recall candidate {i}"))
                                    for i in range(7)]
    response = SimpleNamespace(results=[], calibration_status="uncalibrated")
    withheld = set()
    for attempt in range(200):
        (context, rows), _stored = session_start._deliver_session(
            _engine(store, tmp_path), "", "default", f"s{attempt}", "query", response, [], [],
            relevant, 10, "codex-agent")
        assert pinned in {r["fact_id"] for r in rows}
        withheld |= {w for (w,) in _rows(store.db_path, "SELECT fact_id FROM "
                                                        "injection_receipts WHERE included = 0")}
    assert withheld and pinned not in withheld
    assert _rows(store.db_path, "SELECT DISTINCT client FROM injection_deliveries") == [
        ("codex-agent",)]


def test_session_init_without_a_real_store_delivers_everything(tmp_path, holdout_on) -> None:
    from superlocalmemory.mcp import session_start

    engine = SimpleNamespace(config=SimpleNamespace(base_dir="not-a-path", injection=None),
                             db=object(), mode="B")
    response = SimpleNamespace(results=[], calibration_status="uncalibrated")
    relevant = [_result(f"f{i}") for i in range(8)]
    (_, rows), _stored = session_start._deliver_session(engine, "", "default", "s", "q",
                                                        response, [], [], relevant, 10)
    assert {r["fact_id"] for r in rows} == {f"f{i}" for i in range(8)}


def _hook_context(store, tmp_path, ids):
    from superlocalmemory.cli import commands
    from superlocalmemory.core.injection import InjectableMemory, render_context

    mems = [("recent", InjectableMemory(content=f"hook memory {f}", score=0.5, fact_id=f))
            for f in ids]
    return commands._session_context_delivery(
        SimpleNamespace(base_dir=tmp_path), store.db_path, "default", "", mems,
        lambda kept: render_context([m for _, m in kept], mode="B", wrap=True))


def test_the_session_start_hook_records_its_session_and_client(store, tmp_path,
                                                               monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-session-7")
    monkeypatch.setenv("SLM_CLIENT", "claude-code")
    ids = [_fact(store, f"recent important memory {i}") for i in range(3)]
    assert receipts.shown_in_text(_hook_context(store, tmp_path, ids)) == frozenset(ids)
    assert _rows(store.db_path, "SELECT path, session_id, client, holdout_state FROM"
                                " injection_deliveries") == [
        ("session_context", "claude-session-7", "claude-code", "off")]
    assert _rows(store.db_path, "SELECT role, delivered FROM injection_receipts") == [
        ("recent", 1)] * 3


def test_the_hook_never_withholds_a_pinned_fact_among_its_recent_rows(store, tmp_path,
                                                                     holdout_on) -> None:
    pinned = _fact(store, "a pinned rule in the hook's top ten")
    _protect_in_store(store, pinned, "pinned_column")
    ids = [pinned] + [_fact(store, f"recent row {i}") for i in range(4)]
    for _ in range(200):
        assert f"fact_id={pinned}" in _hook_context(store, tmp_path, ids)
    assert _rows(store.db_path, "SELECT DISTINCT role FROM injection_receipts "
                                "WHERE fact_id = '" + pinned + "'") == [("pinned",)]
    assert _rows(store.db_path, "SELECT COUNT(*) > 0 FROM injection_receipts "
                                "WHERE included = 0") == [(1,)]


def test_a_key_another_process_created_first_is_read_not_an_error(tmp_path, monkeypatch) -> None:
    from superlocalmemory.core import receipt_key

    def racing_open(path, flags, mode=0o777):
        Path(path).write_bytes(b"z" * 32)  # the other process won the race
        raise FileExistsError(path)

    monkeypatch.setattr(receipt_key.os, "open", racing_open)
    assert receipt_key.load_or_create(tmp_path) == b"z" * 32
