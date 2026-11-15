# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Memories saved before 4.2.5 get a fact for every sentence, and nothing else.

The backfill only reaches memories whose whole-memory fact is live: that fact
already makes every sentence retrievable, so the pass never exposes text that
recall could not already find. Anything forgotten, corrected, archived or
tombstoned stays as it is.
"""

from __future__ import annotations

import json

from superlocalmemory.core.coverage_backfill import CoverageBackfill, pending
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from tests.test_core.test_source_fidelity_guard import _db as _base_db


def _db(path):
    """A store migrated like a live one (tombstones, valid_until, merged_into)."""
    from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

    db = _base_db(path)
    learning = path.parent / f"{path.stem}-learning.db"
    apply_all(learning, path)
    apply_deferred(learning, path)
    return db

NOTE = (
    "Launch plan for Wednesday.\n"
    "- X Article goes out at 9:30 AM.\n"
    "- Product Hunt page fix needed: license MIT to AGPL-3.0."
)
LICENSE = "Product Hunt page fix needed: license MIT to AGPL-3.0."


def _memory(db, content=NOTE, *, whole=True, **whole_cols):
    record = MemoryRecord(content=content, session_date="2026-10-13")
    db.store_memory(record)
    if whole:
        fact = AtomicFact(memory_id=record.memory_id, content=content)
        db.store_fact(fact)
        if whole_cols:
            sets = ", ".join(f"{k}=?" for k in whole_cols)
            db.execute(f"UPDATE atomic_facts SET {sets} WHERE fact_id=?",
                       (*whole_cols.values(), fact.fact_id))
    db.store_fact(AtomicFact(memory_id=record.memory_id, content="Launch plan for Wednesday."))
    return record


def _run(db, tmp_path, **kw):
    written: list[AtomicFact] = []
    job = CoverageBackfill(db, written.append, state_dir=tmp_path, **kw)
    job.run_until_done()
    return job, written


def test_an_old_memory_gets_its_missing_sentences(tmp_path):
    db = _db(tmp_path / "memory.db")
    record = _memory(db)
    job, written = _run(db, tmp_path)
    assert [f.content for f in written] == ["X Article goes out at 9:30 AM.", LICENSE]
    assert all(f.memory_id == record.memory_id and f.profile_id == "default" for f in written)
    assert job.finished


def test_every_added_fact_is_in_the_ledger(tmp_path):
    db = _db(tmp_path / "memory.db")
    _memory(db)
    _, written = _run(db, tmp_path)
    lines = (tmp_path / "coverage-backfill.jsonl").read_text().splitlines()
    assert [json.loads(line)["fact_id"] for line in lines] == [f.fact_id for f in written]


def test_a_finished_pass_adds_nothing_again(tmp_path):
    db = _db(tmp_path / "memory.db")
    _memory(db)
    _, first = _run(db, tmp_path)
    for fact in first:  # what the daemon's writer would have stored
        db.store_fact(fact)
    _, second = _run(db, tmp_path)
    assert second == []
    state = json.loads((tmp_path / "coverage-backfill.json").read_text())
    assert state["finished"] is True and state["before_rowid"] >= 1


def test_untouched_when_the_whole_memory_is_not_live(tmp_path):
    for cols in ({"lifecycle": "archived"}, {"quarantined": 1}, {"non_answer": 1},
                 {"merged_into": "other"}, {"valid_until": "2026-10-01"}):
        db = _db(tmp_path / f"{next(iter(cols))}.db")
        _memory(db, **cols)
        _, written = _run(db, tmp_path / next(iter(cols)))
        assert written == [], cols


def test_untouched_without_a_whole_memory_fact(tmp_path):
    db = _db(tmp_path / "memory.db")
    _memory(db, whole=False)
    assert _run(db, tmp_path)[1] == []


def test_untouched_after_an_erasure(tmp_path):
    db = _db(tmp_path / "memory.db")
    record = _memory(db)
    db.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
               "created_at) VALUES ('default', 'gone', 'e1', ?, 1.0)",
               (record.memory_id,))
    assert _run(db, tmp_path)[1] == []


def test_a_corrected_sentence_is_not_brought_back(tmp_path):
    db = _db(tmp_path / "memory.db")
    record = _memory(db)
    # The extractor's fact for the license line, since superseded by a correction.
    db.store_fact(AtomicFact(memory_id=record.memory_id, content=LICENSE))
    db.execute("UPDATE atomic_facts SET lifecycle='archived' WHERE content=?", (LICENSE,))
    assert LICENSE not in [f.content for f in _run(db, tmp_path)[1]]


def test_pending_counts_without_writing(tmp_path):
    db = _db(tmp_path / "memory.db")
    _memory(db)
    _memory(db, "A one sentence memory that stays as it is today.")
    assert pending(db.execute) == (1, 2)
    assert not (tmp_path / "coverage-backfill.jsonl").exists()


def test_a_stop_between_batches_resumes(tmp_path):
    db = _db(tmp_path / "memory.db")
    for _ in range(3):
        _memory(db)
    job, written = _run(db, tmp_path, batch=1, max_batches=1)
    assert len(written) == 2 and not job.finished
    _, rest = _run(db, tmp_path, batch=1)
    assert len(rest) == 4


def test_the_daemon_scheduler_runs_it_with_its_writer(tmp_path, monkeypatch):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    db = _db(tmp_path / "memory.db")
    _memory(db)
    written: list[AtomicFact] = []
    config = SLMConfig()
    config.encoding.coverage_backfill_per_memory = 10  # turned on, as config.json does
    scheduler = MaintenanceScheduler(db, config, fact_writer=written.append)
    scheduler._running = True
    scheduler._coverage_pass()
    assert LICENSE in [f.content for f in written]
    assert scheduler.failing_steps() == {}
    ledger = tmp_path / "data" / "backfill" / "coverage-backfill.jsonl"
    assert len(ledger.read_text().splitlines()) == len(written)


def test_a_second_process_waits_its_turn(tmp_path):
    import os

    from superlocalmemory.core.coverage_backfill import _try_lock, _unlock

    db = _db(tmp_path / "memory.db")
    _memory(db)
    fd = os.open(tmp_path / "coverage-backfill.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert _try_lock(fd)
        assert _run(db, tmp_path)[1] == []
        _unlock(fd)
    finally:
        os.close(fd)
    assert len(_run(db, tmp_path)[1]) == 2


def test_newest_memories_are_covered_first(tmp_path):
    db = _db(tmp_path / "memory.db")
    _memory(db, NOTE.replace("Wednesday", "Monday"))
    newest = _memory(db)
    _, written = _run(db, tmp_path, batch=1, max_batches=1)
    assert {f.memory_id for f in written} == {newest.memory_id}


def test_a_cap_of_zero_turns_the_pass_off(tmp_path, monkeypatch):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    db = _db(tmp_path / "memory.db")
    _memory(db)
    config = SLMConfig()
    config.encoding.coverage_backfill_per_memory = 0
    written: list[AtomicFact] = []
    scheduler = MaintenanceScheduler(db, config, fact_writer=written.append)
    scheduler._running = True
    scheduler._coverage_pass()
    assert written == []
    assert not (tmp_path / "data" / "backfill").exists()  # not even a scan


def test_the_backfill_is_off_by_default_and_config_json_turns_it_on(tmp_path):
    from superlocalmemory.core.config import SLMConfig

    assert SLMConfig().encoding.coverage_backfill_per_memory == 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "a", "encoding": {
        "coverage_backfill_per_memory": 10, "coverage_facts_per_memory": 30}}), encoding="utf-8")
    loaded = SLMConfig.load(path)
    assert loaded.encoding.coverage_backfill_per_memory == 10
    assert loaded.encoding.coverage_facts_per_memory == 30


def test_a_bad_coverage_setting_keeps_the_default(tmp_path):
    from superlocalmemory.core.config import SLMConfig

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "a", "encoding": {
        "coverage_backfill_per_memory": -3, "coverage_facts_per_memory": "lots"}}), encoding="utf-8")
    loaded = SLMConfig.load(path)
    assert loaded.encoding.coverage_backfill_per_memory == 0
    assert loaded.encoding.coverage_facts_per_memory == 40


def test_a_memory_whose_whole_fact_was_replaced_is_left_alone(tmp_path):
    """4.2.30: retirement lives in fact_temporal_validity, not the atomic_facts mirror."""
    db = _db(tmp_path / "memory.db")
    record = _memory(db)
    whole = db.execute("SELECT fact_id FROM atomic_facts WHERE memory_id=? AND content=?",
                       (record.memory_id, NOTE))[0]["fact_id"]
    db.execute("INSERT OR REPLACE INTO fact_temporal_validity (fact_id, profile_id, "
               "system_created_at, system_expired_at) VALUES (?, 'default', "
               "'2026-10-13T00:00:00+00:00', '2026-10-14T00:00:00+00:00')", (whole,))
    assert _run(db, tmp_path)[1] == []
    assert pending(db.execute)[0] == 0
