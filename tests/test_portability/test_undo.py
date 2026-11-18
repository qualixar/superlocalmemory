# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9: undo restores the store exactly (row-level diff), and only this run's rows."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from superlocalmemory.core import decisions
from superlocalmemory.portability import rollback, runner, service
from superlocalmemory.portability.jobs import JOBS
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available
from tests.test_portability import fixtures as fx
from tests.test_portability.conftest import WHO, diff, snapshot
from tests.test_truth.test_slot_pass import _fact


def test_undo_of_a_saved_but_not_yet_enriched_run_is_exact(world) -> None:
    _fact(world.db, "An older memory about the weekly review.", "2026-03-01T00:00:00+00:00")
    before = snapshot(world.db)
    run = world.run("conversations.json", fx.dumps(fx.chatgpt_list()))
    assert world.count("memories") == 5 and diff(before, snapshot(world.db))
    done = world.undo(run["run_id"])
    assert done["state"] == "rolled_back"
    assert diff(before, snapshot(world.db)) == {}
    assert world.count("import_run_items") == 0


def test_undo_of_an_enriched_run_is_exact_including_everything_derived(world) -> None:
    _fact(world.db, "An older memory about the weekly review.", "2026-03-01T00:00:00+00:00")
    before = snapshot(world.db)
    run = world.run("conversations.json", fx.dumps(fx.chatgpt_list()))
    world.enrich()  # extraction, entities, scenes, lineage, trust score: the daemon's own passes
    grown = diff(before, snapshot(world.db))
    for table in ("atomic_facts", "memory_scenes", "provenance", "derivation_lineage",
                  "consolidation_log", "trust_scores", "fact_context", "bm25_tokens"):
        assert table in grown, f"{table} did not grow: the test would prove nothing"
    done = world.undo(run["run_id"])
    assert done["state"] == "rolled_back", done["note"]
    assert diff(before, snapshot(world.db)) == {}


def test_every_importer_undoes_exactly(world) -> None:
    before = snapshot(world.db)
    runs_ = [world.run("a.json", fx.dumps(fx.claude_list())),
             world.run("m.json", fx.dumps({"results": fx.MEM0_ITEMS})),
             world.run("a.af", fx.dumps(fx.letta_file())),
             world.run("AGENTS.md", fx.AGENTS_MD.encode()),
             world.run("k.md", fx.OKF_CONCEPT.encode()),
             world.run("t.txt", fx.CLAUDE_MEMORY_TXT.encode())]
    world.enrich()
    for run in reversed(runs_):
        assert world.undo(run["run_id"])["state"] == "rolled_back"
    assert diff(before, snapshot(world.db)) == {}


def test_undo_takes_only_this_run_and_nothing_that_was_there_before(world) -> None:
    keep = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    world.enrich()
    middle = snapshot(world.db)
    other = world.run("c.json", fx.dumps(fx.chatgpt_list()))
    world.enrich()
    assert world.undo(other["run_id"])["state"] == "rolled_back"
    assert diff(middle, snapshot(world.db)) == {}
    still = service.get_run(world.engine, world.profile, keep["run_id"])
    assert still["state"] == "done" and world.count("import_run_items", "run_id = ?",
                                                    (keep["run_id"],)) == 2


def test_undo_can_be_done_once_and_says_so(world) -> None:
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    world.undo(run["run_id"])
    with pytest.raises(service.ImportRefused, match="already undone"):
        world.undo(run["run_id"])
    with pytest.raises(LookupError):
        world.undo("imp_missing")


def test_undo_is_recorded_and_marks_the_import_decision_undone(world) -> None:
    from superlocalmemory.truth.invariants import open_read_only

    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    undoer = decisions.Decider("dashboard", "dashboard")
    service.start_undo(world.engine, world.runtime, profile_id=world.profile, run_id=run["run_id"],
                       decider=undoer, wait_s=60)
    conn = open_read_only(str(world.db.db_path))
    rows = {d["action"]: d for d in decisions.recent(conn, world.profile) if d["kind"] == "import"}
    conn.close()
    assert rows["run"]["undone"] and (rows["run"]["undone_by"], rows["run"]["undone_channel"]) == (
        "dashboard", "dashboard")
    assert rows["rollback"]["actor"] == "dashboard" and not rows["rollback"]["can_undo"]
    final = service.get_run(world.engine, world.profile, run["run_id"])
    assert final["state"] == "rolled_back" and final["rolled_back_by"] == "dashboard"


def test_a_memory_that_gains_a_fact_while_the_undo_runs_is_still_removed(world, monkeypatch) -> None:
    """The save path enriches in the background: a fact can land after the pass has listed one."""
    from superlocalmemory.storage.models import AtomicFact

    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    ids = runs_items(world, run["run_id"])
    real = rollback._delete_fact
    calls = {"n": 0}

    def delete(engine, runtime, profile_id, run_id, fact_id, memory_id):
        real(engine, runtime, profile_id, run_id, fact_id, memory_id)
        calls["n"] += 1
        if calls["n"] == 1:  # a late fact lands on a memory this pass has not reached yet
            other = next(m for m in ids if m != memory_id)
            world.db.store_fact(AtomicFact(memory_id=other, profile_id=profile_id,
                                           content="A late fact derived after the pass began."))

    monkeypatch.setattr(rollback, "_delete_fact", delete)
    monkeypatch.setattr(rollback.C, "ROLLBACK_SETTLE_S", 0.1)
    done = world.undo(run["run_id"])
    assert done["state"] == "rolled_back"
    assert world.count("atomic_facts", "content LIKE 'A late fact%'") == 0
    assert world.count("memories") == 0 and world.count("import_run_items") == 0


def test_an_undo_that_cannot_finish_stays_rolling_back_and_says_how_to_finish(world, monkeypatch) -> None:
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    monkeypatch.setattr(rollback, "_delete_fact", lambda *a, **k: None)  # nothing can be deleted
    monkeypatch.setattr(rollback.C, "ROLLBACK_PASSES", 1)
    done = world.undo(run["run_id"])
    assert done["state"] == "rolling_back" and "Undo it again" in done["note"]
    monkeypatch.undo()
    assert world.undo(run["run_id"])["state"] == "rolled_back"


def test_cancel_keeps_what_was_saved_and_undo_then_removes_it_exactly(world, monkeypatch) -> None:
    before = snapshot(world.db)
    gate, reached = threading.Event(), threading.Event()
    real = runner._let_others_through

    def hold(ctx, stop):
        reached.set()
        gate.wait(20)
        real(ctx, stop)

    monkeypatch.setattr(runner._let_others_through.__globals__["C"], "BATCH", 2)
    monkeypatch.setattr(runner, "_let_others_through", hold)
    run = world.run("conversations.json", fx.dumps(fx.chatgpt_list()), wait=False)
    assert reached.wait(20)
    cancelled = service.cancel(world.engine, world.profile, run["run_id"])
    gate.set()
    assert JOBS.join(run["run_id"], 30)
    final = service.get_run(world.engine, world.profile, run["run_id"])
    assert final["state"] == "cancelled" and final["saved_items"] == 2
    assert cancelled["cancel_requested"] and "kept" in final["note"]
    assert world.undo(run["run_id"])["state"] == "rolled_back"
    assert diff(before, snapshot(world.db)) == {}


def test_undoing_a_run_that_is_still_importing_stops_it_first(world, monkeypatch) -> None:
    before = snapshot(world.db)
    gate, reached = threading.Event(), threading.Event()
    real = runner._let_others_through

    def hold(ctx, stop):
        reached.set()
        gate.wait(5)
        real(ctx, stop)

    def hold_until_stopped(ctx, stop):
        reached.set()
        stop.wait(20)  # an import that only ends when it is told to stop
        real(ctx, stop)

    monkeypatch.setattr(runner.C, "BATCH", 1)
    monkeypatch.setattr(runner, "_let_others_through", hold_until_stopped)
    run = world.run("conversations.json", fx.dumps(fx.chatgpt_list()), wait=False)
    assert reached.wait(20)
    done = world.undo(run["run_id"])
    assert done["state"] == "rolled_back"
    assert done["saved_items"] == 1  # it was stopped after its first save, not left to finish
    assert diff(before, snapshot(world.db)) == {}
    gate.set()


def test_a_run_the_daemon_lost_is_closed_with_words_and_can_still_be_undone(world) -> None:
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    with world.db.raw_connection() as conn:
        conn.execute("UPDATE import_runs SET state = 'running' WHERE run_id = ?", (run["run_id"],))
    listed = service.overview(world.engine, world.profile)
    lost = next(r for r in listed["runs"] if r["run_id"] == run["run_id"])
    assert lost["state"] == "failed" and "stopped while this was running" in lost["note"]
    assert world.undo(run["run_id"])["state"] == "rolled_back"


def test_erasing_a_memory_removes_the_runs_record_of_it(world) -> None:
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    memory_id = runs_items(world, run["run_id"])[0]
    world.db.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
    assert memory_id not in runs_items(world, run["run_id"])


def runs_items(world, run_id):
    return [dict(r)["memory_id"] for r in world.db.execute(
        "SELECT memory_id FROM import_run_items WHERE run_id = ?", (run_id,))]


def test_erasing_a_profile_removes_its_runs_and_items(world) -> None:
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    other = service.start_run(world.engine, world.runtime, profile_id="work",
                              docs=world.docs("m.json", fx.dumps(fx.MEM0_ITEMS)), fmt=None,
                              label="m.json", decider=WHO)
    assert JOBS.join(other["run_id"], 60)
    kept = world.run("c.json", fx.dumps(fx.chatgpt_list()))
    assert world.count("import_runs", "profile_id = 'work'") == 1
    GDPRCompliance(world.db, engine=world.engine,
                   data_root=world.db.db_path.parent).forget_profile("work")
    assert world.count("import_runs", "profile_id = 'work'") == 0
    assert world.count("import_run_items", "profile_id = 'work'") == 0
    assert world.count("decision_log", "profile_id = 'work'") == 0
    assert world.count("import_runs", "run_id = ?", (kept["run_id"],)) == 1


def test_the_profile_fold_has_a_decision_for_both_tables() -> None:
    from superlocalmemory.storage import profile_fold

    assert profile_fold.DECISIONS["import_runs"][0] == profile_fold.MOVE
    assert profile_fold.DECISIONS["import_run_items"][0] == profile_fold.MOVE


@pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)
def test_folding_a_profile_moves_its_runs_to_the_target(world) -> None:
    from superlocalmemory.storage import profile_fold

    run = service.start_run(world.engine, world.runtime, profile_id="work",
                            docs=world.docs("m.json", fx.dumps(fx.MEM0_ITEMS)), fmt=None,
                            label="m.json", decider=WHO)
    assert JOBS.join(run["run_id"], 60)
    conn = sqlite3.connect(world.db.db_path, isolation_level=None)
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("BEGIN IMMEDIATE")
        profile_fold.fold_profile(conn, "work", "default")
        conn.execute("DELETE FROM profiles WHERE profile_id = 'work'")
        conn.execute("COMMIT")
    finally:
        conn.close()
    moved = service.get_run(world.engine, "default", run["run_id"])
    assert moved["state"] == "done"
    assert world.undo(run["run_id"])["state"] == "rolled_back"
    assert world.count("memories", "profile_id = 'default'") == 0


def test_a_fact_that_cannot_be_deleted_keeps_its_copies_and_the_undo_says_so(world, monkeypatch) -> None:
    """A fact a reviewed correction protects stays whole: nothing derived from it is half removed."""
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    world.enrich()
    protected = dict(world.db.execute("SELECT fact_id FROM atomic_facts ORDER BY rowid LIMIT 1")[0])["fact_id"]
    tokens = world.count("bm25_tokens", "fact_id = ?", (protected,))
    assert tokens > 0
    real = world.runtime.delete_fact

    def refuse(profile_id, fact_id, **kwargs):
        if fact_id == protected:
            raise CanonicalMutationConflict("a reviewed correction protects this memory")
        return real(profile_id, fact_id, **kwargs)

    monkeypatch.setattr(world.runtime, "delete_fact", refuse)
    monkeypatch.setattr(rollback.C, "ROLLBACK_PASSES", 1)
    done = world.undo(run["run_id"])
    assert done["state"] == "rolling_back" and "Undo it again" in done["note"]
    assert world.count("atomic_facts", "fact_id = ?", (protected,)) == 1
    assert world.count("bm25_tokens", "fact_id = ?", (protected,)) == tokens
    assert world.count("memories") == 1 and world.count("atomic_facts") == 1  # the other one is gone
    monkeypatch.undo()
    assert world.undo(run["run_id"])["state"] == "rolled_back"
    assert world.count("memories") == 0
