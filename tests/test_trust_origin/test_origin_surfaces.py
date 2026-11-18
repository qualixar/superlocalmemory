# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Origin on recall results (`why.origin`), the memory detail, and the maintenance step."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.server import recall_core
from superlocalmemory.server.answer_receipt import why, with_why
from superlocalmemory.trust import fact_trust
from superlocalmemory.truth.fact_truth import for_fact
from tests.test_core.test_coverage_backfill import _db
from tests.test_trust_origin.test_origin_supersession import PERSON, WEB, _fact

TAINT = env_name("trust.taint")


@pytest.fixture
def store(tmp_path):
    db = _db(tmp_path / "memory.db")
    mine = _fact(db, "I prefer tea in the morning.", "2026-11-01T00:00:00Z", PERSON)
    page = _fact(db, "A newsletter says tea is bad.", "2026-11-02T00:00:00Z", WEB)
    old = _fact(db, "Saved before 4.2.30.", "2026-01-01T00:00:00Z", None)
    db.execute("DELETE FROM fact_origin WHERE fact_id = ?", (old,))
    return db, mine, page, old


def _notes(db, *fact_ids):
    engine = SimpleNamespace(_db=db, profile_id="default")
    call = SimpleNamespace(profile_id="default", caller_view="")
    return recall_core._truth_notes(engine, call, [{"fact_id": f, "content": "x"} for f in fact_ids])


def _count(db) -> int:
    return db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"]


# -- recall ------------------------------------------------------------------------------------

def test_results_carry_origin_in_words_only_while_the_flag_observes(store, monkeypatch) -> None:
    db, mine, page, old = store
    monkeypatch.setenv(TAINT, "on")
    got = {r["fact_id"]: r for r in _notes(db, mine, page, old)}
    assert got[mine]["origin"] == {"class": "user_direct", "taint": "user_direct",
                                   "label": "you said this", "taint_label": "you said this"}
    assert got[page]["origin"]["class"] == "web" and got[page]["origin"]["label"] == "a web page"
    assert got[old]["origin"]["class"] == "unknown"
    assert "basis" not in got[mine]["origin"]


def test_nothing_is_added_with_the_flag_off_or_the_kill_switch_on(store, monkeypatch) -> None:
    db, mine, page, _old = store
    assert all(set(r) == {"fact_id", "content"} for r in _notes(db, mine, page))
    monkeypatch.setenv(TAINT, "on")
    monkeypatch.setenv(KILL_ENV, "1")
    assert all(set(r) == {"fact_id", "content"} for r in _notes(db, mine, page))


def test_recall_never_writes_the_class(store, monkeypatch) -> None:
    db, mine, page, _old = store
    monkeypatch.setenv(TAINT, "shadow")
    before = _count(db)
    _notes(db, mine, page)
    assert _count(db) == before == 0


def test_a_failing_origin_read_leaves_the_results_as_they_were(store, monkeypatch) -> None:
    db, mine, _page, _old = store
    monkeypatch.setenv(TAINT, "on")
    monkeypatch.setattr(fact_trust, "for_results",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert _notes(db, mine) == [{"fact_id": mine, "content": "x"}]


def test_origin_and_the_truth_notes_are_independent(store, monkeypatch) -> None:
    db, mine, _page, _old = store
    monkeypatch.setenv(env_name("truth.slots"), "shadow")
    monkeypatch.setenv(TAINT, "off")
    assert _notes(db, mine) == [{"fact_id": mine, "content": "x"}]
    monkeypatch.setenv(TAINT, "on")
    assert "origin" in _notes(db, mine)[0]


def test_why_copies_the_origin_and_never_reads_a_database() -> None:
    item = {"fact_id": "f", "age_label": "2 days ago",
            "origin": {"class": "web", "taint": "web", "label": "a web page",
                       "taint_label": "a web page", "basis": "v1:adapter.service"}}
    block = why(item)
    assert block["origin"] == {"class": "web", "taint": "web", "label": "a web page"}
    assert block["standing"] == "current" and block["said"] == "2 days ago"
    assert "origin" not in why({"fact_id": "f"})
    assert "origin" not in why({"fact_id": "f", "origin": "web"})
    assert "origin" not in why({"fact_id": "f", "origin": {"class": ""}})
    assert with_why([item])[0]["why"]["origin"]["class"] == "web"


# -- the memory detail -------------------------------------------------------------------------

def test_the_memory_detail_says_where_it_came_from_whatever_the_flag(store) -> None:
    db, mine, page, old = store
    with db.raw_connection() as conn:
        assert for_fact(conn, "default", mine)["origin"]["label"] == "you said this"
        assert for_fact(conn, "default", page)["origin"]["class"] == "web"
        unknown = for_fact(conn, "default", old)["origin"]
        assert unknown["class"] == "unknown" and "4.2.30" in unknown["label"]
        assert "basis" not in unknown
    assert _count(db) == 0  # reading wrote nothing


# -- the maintenance step ----------------------------------------------------------------------

def test_the_scheduler_step_classifies_in_bounded_batches(store, monkeypatch, caplog) -> None:
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    db, mine, page, _old = store
    sched = SimpleNamespace(_running=True, _db=db, steps=[])
    sched._record_step = lambda name, ok, err="": sched.steps.append((name, ok))
    MaintenanceScheduler._trust_pass(sched)
    assert _count(db) == 0 and sched.steps == [("trust pass", True)]
    monkeypatch.setenv(TAINT, "shadow")
    with caplog.at_level(logging.INFO):
        MaintenanceScheduler._trust_pass(sched)
    assert _count(db) == 3
    sched._running = False
    MaintenanceScheduler._trust_pass(sched)
    assert len(sched.steps) == 2  # a stopped scheduler does nothing


def test_a_failing_step_is_recorded_not_raised(store, monkeypatch) -> None:
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    db, *_ = store
    monkeypatch.setattr(fact_trust, "maintenance_step",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    sched = SimpleNamespace(_running=True, _db=db, steps=[])
    sched._record_step = lambda name, ok, err="": sched.steps.append((name, ok, err))
    MaintenanceScheduler._trust_pass(sched)
    assert sched.steps == [("trust pass", False, "disk")]


def test_origin_of_is_what_slm_why_will_call(store) -> None:
    db, mine, *_ = store
    with db.raw_connection() as conn:
        got = fact_trust.origin_of(conn, mine)
    assert got["class"] == "user_direct" and got["basis"] == "v1:remember.person"
