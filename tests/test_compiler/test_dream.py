# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The dream cycle (WS-6): lint, propose, verify, queue; the W4 rollback acceptance scenario."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.compiler import dream, proposals
from superlocalmemory.compiler.proposals import Basis
from tests.test_compiler.conftest import PROFILE, raw, table_rows
from tests.test_compiler.helpers import add_fact, bind_slot

PERSON = Basis.person("alice", "cli")
#: Append-only records of the decision itself: they say what happened and are not rolled back.
AUDIT = {"proposal_sets", "proposals", "proposal_facts", "correction_cases", "correction_events",
         "correction_case_until_source", "artifact_versions", "artifact_version_facts",
         "write_commits", "projection_outbox", "fact_search_changes", "judge_receipts",
         "slm_schema_version", "migration_log", "decision_log"}
LATER = datetime.now(UTC) + timedelta(hours=7)


@pytest.fixture
def busy(store):
    """Three topic facts, an exact duplicate pair, a slot with a head and a dispute."""
    db, ids = store
    first = add_fact(db, "The nightly backup runs at two in the morning.", "2026-02-01T00:00:00+00:00",
                     entity=None)
    again = add_fact(db, "the nightly backup runs at two in the morning", "2026-02-09T00:00:00+00:00",
                     entity=None)
    head = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    bind_slot(db, head, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    return db, ids, (first, again)


def _state(db) -> dict:
    names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                      "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' "
                                      "AND name NOT LIKE '%_vec%' ORDER BY name")]
    return {n: table_rows(db, n) for n in names if n not in AUDIT}


def test_normal_text_folds_case_space_and_end_punctuation() -> None:
    assert dream.normal_text("  The  Cat sat.  ") == dream.normal_text("the cat sat")
    assert dream.normal_text("a b!") != dream.normal_text("a c!")


def test_lint_counts_duplicates_stale_sources_and_disputes(busy) -> None:
    db, ids, (first, again) = busy
    db.execute("INSERT INTO memory_leases (fact_id, profile_id, source_kind, locator, fingerprint, "
               "state, leased_at, checked_at) VALUES (?, ?, 'file', '/x/a.md', 'f', 'missing', "
               "'2026-01-01', '2026-01-02')", (ids[0], PROFILE))
    conn = raw(db)
    try:
        assert dream.duplicate_groups(conn, PROFILE) == [(first, [again])]
        got = dream.lint(conn, PROFILE)
    finally:
        conn.close()
    assert got == {"rule": "exact-text-v1", "duplicate_groups": 1, "stale_sources": 1,
                   "open_disputes": 0}


def test_short_text_and_pinned_facts_are_never_duplicates(store) -> None:
    db, _ids = store
    add_fact(db, "ok", entity=None)
    add_fact(db, "ok", entity=None)
    a = add_fact(db, "A pinned reminder that must stay.", entity=None)
    b = add_fact(db, "A pinned reminder that must stay.", entity=None)
    db.execute("UPDATE atomic_facts SET pinned = 1 WHERE fact_id = ?", (b,))
    conn = raw(db)
    try:
        assert dream.duplicate_groups(conn, PROFILE) == []
    finally:
        conn.close()
    assert a


def test_a_cycle_queues_one_set_and_changes_no_memory(busy) -> None:
    db, _ids, (first, again) = busy
    before = _state(db)
    got = dream.run_profile(db, PROFILE)
    assert got["set_id"] and got["merges"] == 1 and got["pages"] >= 2
    assert _state(db) == before  # only the proposal tables moved
    shown = proposals.get_set(raw(db), PROFILE, got["set_id"])
    assert shown["state"] == "proposed" and shown["origin"] == "dream"
    assert shown["title"] == f"Dream cycle: {got['pages']} page(s), 1 duplicate group(s)"
    kinds = [p["kind"] for p in shown["proposals"]]
    assert kinds.count("merge") == 1 and kinds.count("create_page") == got["pages"]


def test_dream_rollback_scenario_leaves_every_table_byte_identical(busy) -> None:
    """W4 exit gate: run the cycle, apply its proposal set, roll it back."""
    db, _ids, (first, again) = busy
    before = _state(db)
    got = dream.run_profile(db, PROFILE)
    conn = raw(db)
    try:
        applied = proposals.apply_set(conn, profile_id=PROFILE, set_id=got["set_id"], basis=PERSON)
        assert applied["state"] == "applied"
        during = _state(db)
        assert during != before
        assert table_rows(db, "compiled_pages") and during["compiled_pages"]
        retired = db.execute("SELECT system_expired_at FROM fact_temporal_validity WHERE fact_id = ?",
                             (again,))
        assert retired[0][0] is not None  # the duplicate really left
        done = proposals.rollback_set(conn, profile_id=PROFILE, set_id=got["set_id"], basis=PERSON)
        assert done["state"] == "rolled_back"
    finally:
        conn.close()
    assert _state(db) == before


def test_a_waiting_set_blocks_a_second_one_and_a_recent_set_too(busy) -> None:
    db, _ids, _dups = busy
    first = dream.run_profile(db, PROFILE)
    assert dream.run_profile(db, PROFILE, now=LATER)["skipped"] == "a dream set is still waiting for review"
    proposals.reject_set(raw(db), profile_id=PROFILE, set_id=first["set_id"], actor="alice",
                         channel="cli")
    assert dream.run_profile(db, PROFILE)["skipped"] == "the last dream set is recent"


def test_a_rejected_proposal_is_not_proposed_again_unchanged(busy) -> None:
    db, _ids, _dups = busy
    first = dream.run_profile(db, PROFILE)
    proposals.reject_set(raw(db), profile_id=PROFILE, set_id=first["set_id"], actor="alice",
                         channel="cli")
    assert dream.run_profile(db, PROFILE, now=LATER)["skipped"] == "nothing to propose"


def test_an_applied_page_is_not_proposed_again_until_it_changes(store) -> None:
    db, _ids = store
    first = dream.run_profile(db, PROFILE)
    proposals.apply_set(raw(db), profile_id=PROFILE, set_id=first["set_id"], basis=PERSON)
    again = dream.run_profile(db, PROFILE, now=LATER)
    assert again["skipped"] == "nothing to propose"
    add_fact(db, "Another SuperLocalMemory fact.", "2026-09-01T00:00:00+00:00")
    third = dream.run_profile(db, PROFILE, now=LATER)
    assert third["pages"] == 1 and third["set_id"]


def test_every_queued_sentence_passed_the_span_check(busy) -> None:
    db, _ids, _dups = busy
    got = dream.run_profile(db, PROFILE)
    conn = raw(db)
    try:
        from superlocalmemory.compiler import page_templates as tpl

        shown = proposals.get_set(conn, PROFILE, got["set_id"])
        rows = conn.execute("SELECT payload_json FROM proposals WHERE kind = 'create_page' AND set_id = ?",
                            (got["set_id"],)).fetchall()
        assert rows
        for (payload,) in rows:
            for s in json.loads(payload)["page"]["sentences"]:
                assert tpl.verify(conn, PROFILE, s)[0]
        assert shown["state"] == "proposed"
    finally:
        conn.close()


def test_the_cycle_has_nothing_to_do_on_an_empty_store(tmp_path) -> None:
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    assert dream.run_profile(db, PROFILE)["skipped"] == "nothing to propose"


def test_the_scheduler_step_runs_only_while_the_flag_is_on(busy, monkeypatch, tmp_path) -> None:
    db, _ids, _dups = busy
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SLM_FEATURE_COMPILER_DREAM", raising=False)
    assert dream.maintenance_step(db, (PROFILE,)) == []
    assert table_rows(db, "proposal_sets") == []
    monkeypatch.setenv("SLM_FEATURE_COMPILER_DREAM", "on")
    got = dream.maintenance_step(db, (PROFILE,))
    assert len(got) == 1 and got[0]["set_id"]
    monkeypatch.setenv("SLM_4230_OFF", "1")  # the kill switch stops it
    assert dream.maintenance_step(db, (PROFILE,)) == []


def test_the_scheduler_runs_the_dream_pass_last_and_records_failures(busy, monkeypatch, tmp_path) -> None:
    import inspect
    from types import SimpleNamespace

    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    source = inspect.getsource(MaintenanceScheduler)
    assert source.index("self._cue_pass()") < source.index("self._dream_pass()")
    db, _ids, _dups = busy
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    sched = SimpleNamespace(_running=True, _db=db, steps=[], _profile_ids=lambda: (PROFILE,))
    sched._record_step = lambda name, ok, err="": sched.steps.append((name, ok, err))
    monkeypatch.delenv("SLM_FEATURE_COMPILER_DREAM", raising=False)
    MaintenanceScheduler._dream_pass(sched)
    assert sched.steps == [("dream cycle", True, "")] and table_rows(db, "proposal_sets") == []
    monkeypatch.setenv("SLM_FEATURE_COMPILER_DREAM", "on")
    MaintenanceScheduler._dream_pass(sched)
    assert len(table_rows(db, "proposal_sets")) == 1
    monkeypatch.setattr(dream, "maintenance_step",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    MaintenanceScheduler._dream_pass(sched)
    assert sched.steps[-1] == ("dream cycle", False, "disk")
    sched._running = False
    MaintenanceScheduler._dream_pass(sched)
    assert len(sched.steps) == 3  # a stopped scheduler does nothing


def test_a_draft_sentence_that_fails_the_check_is_left_out_and_counted(busy, monkeypatch) -> None:
    from superlocalmemory.compiler import page_sources

    db, _ids, _dups = busy
    conn = raw(db)
    try:
        real = page_sources.build_drafts(conn, PROFILE, ("topic",))[0]
        bad = {**real, "sentences": [*real["sentences"],
                                     {**real["sentences"][0], "text": "Not in the memory at all."}]}
        monkeypatch.setattr(page_sources, "build_drafts", lambda *a, **k: [bad])
        got = dream.plan(conn, PROFILE, merges=False)
    finally:
        conn.close()
    (item,) = got["items"]
    page = item["payload"]["page"]
    assert len(page["sentences"]) == len(real["sentences"]) and page["withheld"]["unverifiable"] == 1
    assert got["report"]["withheld_sentences"] >= 1 and got["report"]["merges"] == 0
