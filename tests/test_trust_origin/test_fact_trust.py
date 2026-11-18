# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M079 and the class and taint of each fact: join through parents, cycles, erase, batch."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M079_fact_trust as M079
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import fact_trust as ft
from tests.test_core.test_coverage_backfill import _db

FLAG = "SLM_FEATURE_TRUST_TAINT"


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


def _store(db, text, origin=None, *, memory_id=None):
    if memory_id is None:
        record = MemoryRecord(content=text, session_date="2026-11-15")
        db.store_memory(record)
        memory_id = record.memory_id
    fact = AtomicFact(memory_id=memory_id, content=text)
    if origin is None:
        db.store_fact(fact)
    else:
        with fo.declare(origin):
            db.store_fact(fact)
    return fact


def _person():
    return fo.Origin("remember_http", "verbatim", source_type="http", role="user")


def _web(**kw):
    return fo.Origin("ingest_adapter", "verbatim", source_type="gmail", **kw)


def _query(db):
    return lambda sql, params: db.execute(sql, params)


def _rows(db):
    return {r["fact_id"]: (r["origin_class"], r["taint_class"], r["basis"])
            for r in db.execute("SELECT * FROM fact_trust")}


# -- the migration ------------------------------------------------------------------------------

def test_m079_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M079.NAME] is M079
    assert M079.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M079.DOWNGRADE_FLOOR == 51
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M079.verify(conn)
    M079.apply(conn)
    M079.apply(conn)
    assert M079.verify(conn)
    conn.execute("DROP TRIGGER trg_fact_trust_fact_erased")
    assert not M079.verify(conn)
    M079.repair(conn)
    assert M079.verify(conn)


def test_the_profile_fold_moves_it() -> None:
    from superlocalmemory.storage.profile_fold import DECISIONS, MOVE

    assert DECISIONS["fact_trust"][0] == MOVE


# -- class and taint ----------------------------------------------------------------------------

def test_a_fact_with_no_origin_row_is_unknown(db) -> None:
    fact = _store(db, "Saved before origins were kept.")
    db.execute("DELETE FROM fact_origin WHERE fact_id = ?", (fact.fact_id,))
    got = ft.compute(_query(db), fact.fact_id)
    assert got == ft.Trust("unknown", "unknown", "v2:no_origin_row")


def test_a_fact_that_does_not_exist_has_no_trust(db) -> None:
    assert ft.compute(_query(db), "nope") is None


def test_a_persons_fact_is_user_direct_with_its_own_taint(db) -> None:
    fact = _store(db, "I prefer tea.", _person())
    got = ft.compute(_query(db), fact.fact_id)
    assert (got.origin_class, got.taint_class, got.basis) == (
        "user_direct", "user_direct", "v2:remember.person")


def test_a_correction_inherits_the_taint_of_what_it_corrects(db) -> None:
    parent = _store(db, "The page says the port is 8080.", _web())
    child = _store(db, "The port is 9090.", fo.Origin(
        "correction", "verbatim", source_type="correction", parent_fact_id=parent.fact_id))
    got = ft.compute(_query(db), child.fact_id)
    assert got.origin_class == "agent_inferred" and got.taint_class == "web"


def test_taint_is_the_lowest_class_through_a_chain(db) -> None:
    a = _store(db, "Mail said X.", _web())
    b = _store(db, "A person edited X.", fo.Origin("correction", "verbatim", parent_fact_id=a.fact_id))
    c = _store(db, "A person edited X again.", fo.Origin(
        "correction", "verbatim", parent_fact_id=b.fact_id))
    assert ft.compute(_query(db), c.fact_id).taint_class == "web"


def test_a_better_parent_does_not_lift_the_taint(db) -> None:
    parent = _store(db, "I prefer tea.", _person())
    child = _store(db, "The user prefers tea, an agent says.", fo.Origin(
        "correction", "verbatim", parent_fact_id=parent.fact_id))
    assert ft.compute(_query(db), child.fact_id).taint_class == "agent_inferred"


def test_a_cycle_terminates_with_the_join_of_the_cycle(db) -> None:
    a = _store(db, "Fact A about the tea.", fo.Origin("correction", "verbatim"))
    b = _store(db, "Fact B about the tea.", _web())
    db.execute("UPDATE fact_origin SET parent_fact_id = ? WHERE fact_id = ?", (b.fact_id, a.fact_id))
    db.execute("UPDATE fact_origin SET parent_fact_id = ? WHERE fact_id = ?", (a.fact_id, b.fact_id))
    assert ft.compute(_query(db), a.fact_id).taint_class == "web"
    assert ft.compute(_query(db), b.fact_id).taint_class == "web"
    db.execute("UPDATE fact_origin SET parent_fact_id = ? WHERE fact_id = ?", (a.fact_id, a.fact_id))
    assert ft.compute(_query(db), a.fact_id).taint_class == "agent_inferred"  # a fact is its own parent


def test_a_long_chain_is_cut_at_the_depth_bound(db) -> None:
    prev = _store(db, "Chain start from mail.", _web())
    for i in range(ft.MAX_DEPTH + 3):
        prev = _store(db, f"Chain link number {i} restated.", fo.Origin(
            "correction", "verbatim", parent_fact_id=prev.fact_id))
    got = ft.compute(_query(db), prev.fact_id)
    assert got.taint_class == "agent_inferred"  # the web start is deeper than MAX_DEPTH


def test_an_erased_parent_leaves_nothing_to_inherit(db) -> None:
    parent = _store(db, "Mail said X.", _web())
    child = _store(db, "Edited X.", fo.Origin("correction", "verbatim", parent_fact_id=parent.fact_id))
    with db.raw_connection() as conn:
        conn.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (parent.fact_id,))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (parent.fact_id,))
    assert ft.compute(_query(db), child.fact_id).taint_class == "agent_inferred"


def test_a_derived_fact_is_tainted_by_the_source_facts_of_its_memory(db) -> None:
    source = _store(db, "A mail about the budget.", _web())
    cut = _store(db, "The budget.", fo.Origin("derived", "split", source_type="coverage-backfill",
                                              parent_memory_id=source.memory_id),
                 memory_id=source.memory_id)
    got = ft.compute(_query(db), cut.fact_id)
    assert got.origin_class == "system" and got.taint_class == "web"


def test_siblings_in_one_memory_do_not_taint_each_other(db) -> None:
    first = _store(db, "I prefer tea.", _person())
    sibling = _store(db, "The user prefers tea.", fo.Origin(
        "remember_http", "model_rewritten", source_type="http", role="user"),
        memory_id=first.memory_id)
    assert ft.compute(_query(db), first.fact_id).taint_class == "user_direct"
    assert ft.compute(_query(db), sibling.fact_id).taint_class == "model_generated"


def test_a_model_rewrite_of_a_persons_memory_is_model_generated(db) -> None:
    fact = _store(db, "A paraphrase.", fo.Origin("remember_http", "model_rewritten", role="user"))
    got = ft.compute(_query(db), fact.fact_id)
    assert got.origin_class == "model_generated" and "+model_rewritten" in got.basis


# -- persistence --------------------------------------------------------------------------------

def test_with_the_flag_off_a_save_computes_nothing(db) -> None:
    _store(db, "I prefer tea.", _person())
    assert _rows(db) == {}


@pytest.mark.parametrize("state", ["shadow", "on"])
def test_with_the_flag_observing_admission_records_the_class(db, monkeypatch, state) -> None:
    monkeypatch.setenv(FLAG, state)
    fact = _store(db, "I prefer tea.", _person())
    assert _rows(db) == {fact.fact_id: ("user_direct", "user_direct", "v2:remember.person")}


def test_the_kill_switch_stops_admission(db, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    monkeypatch.setenv("SLM_4230_OFF", "1")
    _store(db, "I prefer tea.", _person())
    assert _rows(db) == {}


def test_admission_never_blocks_a_save(db, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    monkeypatch.setattr(ft, "compute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    fact = _store(db, "A memory that must survive.", _person())
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?",
                      (fact.fact_id,))[0]["n"] == 1


def test_erasing_a_fact_erases_its_trust_row(db, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    fact = _store(db, "A memory to forget.", _person())
    assert fact.fact_id in _rows(db)
    with db.raw_connection() as conn:
        conn.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (fact.fact_id,))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact.fact_id,))
    assert _rows(db) == {}


def test_the_batch_fills_what_admission_did_not_and_is_resumable(db, monkeypatch) -> None:
    facts = [_store(db, f"Memory number {i} about tea.", _person()) for i in range(5)]
    assert ft.run_batch(db, limit=3) == 3
    assert len(_rows(db)) == 3
    assert ft.run_batch(db, limit=3) == 2
    assert set(_rows(db)) == {f.fact_id for f in facts}
    assert ft.run_batch(db, limit=3) == 0  # nothing left: a current row is never selected again


def test_the_batch_recomputes_rows_of_an_older_mapping_but_not_a_persons_confirmation(db) -> None:
    a = _store(db, "Mail about tea.", _web())
    b = _store(db, "Another mail about tea.", _web())
    ft.run_batch(db)
    with db.raw_connection() as conn:
        conn.execute("UPDATE fact_trust SET origin_class='user_direct', taint_class='user_direct', "
                     "basis='v0:old' WHERE fact_id = ?", (a.fact_id,))
        ft.mark_confirmed(conn, b.fact_id, by="varun", channel="dashboard")
    assert ft.run_batch(db) == 1
    rows = _rows(db)
    assert rows[a.fact_id][:2] == ("web", "web")
    assert rows[b.fact_id] == ("user_confirmed", "user_confirmed", "confirmed:dashboard:varun")


def test_the_batch_marks_facts_saved_before_origins_unknown(db) -> None:
    fact = _store(db, "Saved before origins were kept.")
    db.execute("DELETE FROM fact_origin WHERE fact_id = ?", (fact.fact_id,))
    ft.run_batch(db)
    assert _rows(db)[fact.fact_id] == ("unknown", "unknown", "v2:no_origin_row")


def test_the_batch_is_the_step_only_while_the_flag_observes(db, monkeypatch) -> None:
    _store(db, "I prefer tea.", _person())
    assert ft.maintenance_step(db) == 0 and _rows(db) == {}
    monkeypatch.setenv(FLAG, "shadow")
    assert ft.maintenance_step(db) == 1 and len(_rows(db)) == 1


def test_mark_confirmed_refuses_a_fact_that_does_not_exist(db) -> None:
    with db.raw_connection() as conn, pytest.raises(LookupError):
        ft.mark_confirmed(conn, "nope", by="varun", channel="cli")


# -- reads never write --------------------------------------------------------------------------

def test_origin_of_reads_the_row_or_computes_without_writing(db) -> None:
    fact = _store(db, "Mail about tea.", _web())
    with db.raw_connection() as conn:
        got = ft.origin_of(conn, fact.fact_id)
    assert got["class"] == "web" and got["label"] == "a web page" and got["basis"].startswith("v2:")
    assert _rows(db) == {}  # computed, not stored
    assert ft.origin_of(sqlite3.connect(":memory:"), "x") is None


def test_for_results_is_one_batched_select_plus_a_bounded_live_fill(db, monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "on")
    stored = _store(db, "I prefer tea.", _person())
    monkeypatch.delenv(FLAG)
    live = [_store(db, f"Mail number {i} about tea.", _web()) for i in range(13)]
    with db.raw_connection() as conn:
        got = ft.for_results(conn, [stored.fact_id] + [f.fact_id for f in live] + ["", "nope"])
    assert got[stored.fact_id]["class"] == "user_direct"
    assert sum(1 for f in live if f.fact_id in got) == 10  # the bound is part of the contract
    assert len(_rows(db)) == 1  # nothing written by a read


def test_a_store_before_m079_reads_as_computed(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    assert ft.for_results(conn, []) == {}
    assert ft.for_results(conn, ["a"]) == {}


def test_a_cycle_is_walked_once_not_until_the_bound(db) -> None:
    a = _store(db, "Fact A about the tea.", fo.Origin("correction", "verbatim"))
    b = _store(db, "Fact B about the tea.", fo.Origin("correction", "verbatim"))
    db.execute("UPDATE fact_origin SET parent_fact_id = ? WHERE fact_id = ?", (b.fact_id, a.fact_id))
    db.execute("UPDATE fact_origin SET parent_fact_id = ? WHERE fact_id = ?", (a.fact_id, b.fact_id))
    asked = []

    def query(sql, params):
        asked.append(params)
        return db.execute(sql, params)

    ft.compute(query, a.fact_id)
    assert len(asked) == 2  # a, then b; a is not asked about again


def test_a_confirmation_survives_a_later_computation(db) -> None:
    fact = _store(db, "Mail about tea.", _web())
    with db.raw_connection() as conn:
        ft.mark_confirmed(conn, fact.fact_id, by="varun", channel="dashboard")
    ft.store(lambda sql, params: db.execute(sql, params), fact.fact_id, "default",
             ft.compute(_query(db), fact.fact_id))
    assert _rows(db)[fact.fact_id][0] == "user_confirmed"
