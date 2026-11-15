# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-21: every fact's origin is recorded once, when it is admitted (M074)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M074_fact_origin as M074
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


def _origins(db) -> list[tuple]:
    return [tuple(r) for r in db.execute(
        "SELECT channel, transform, source_type, actor_id FROM fact_origin ORDER BY recorded_at")]


@pytest.fixture
def db(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _store(db, content, *, memory_id=None):
    record = MemoryRecord(content=content, session_date="2026-11-15")
    if memory_id is None:
        db.store_memory(record)
        memory_id = record.memory_id
    fact = AtomicFact(memory_id=memory_id, content=content)
    db.store_fact(fact)
    return fact


# -- the migration ------------------------------------------------------------------------------

def test_m074_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M074.NAME] is M074
    assert M074.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M074.DOWNGRADE_FLOOR == 51
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M074.verify(conn)
    M074.apply(conn)
    M074.apply(conn)
    assert M074.verify(conn)
    conn.execute("DROP TRIGGER trg_fact_origin_fact_erased")
    assert not M074.verify(conn)
    M074.repair(conn)
    assert M074.verify(conn)


# -- the vocabulary -----------------------------------------------------------------------------

@pytest.mark.parametrize("source_type,metadata,channel", [
    ("http", {}, "remember_http"),
    ("http", {"tags": "hook-file-change"}, "hook_file_change"),
    ("http", {"tags": ["session-end", "x"]}, "hook_session_summary"),
    ("http-observe", {}, "observe"), ("cli-observe", {}, "cli_observe"),
    ("gmail", {}, "ingest_adapter"), ("folder", {}, "folder"), ("media", {}, "media"),
    ("restore-reimport", {}, "restore"), ("something-new", {}, "unknown"),
])
def test_the_channel_is_named_only_where_it_is_certain(source_type, metadata, channel) -> None:
    assert fo.channel_for(source_type, metadata) == channel


def test_the_vocabulary_is_closed() -> None:
    with pytest.raises(ValueError):
        fo.Origin("email-ish", "verbatim")
    with pytest.raises(ValueError):
        fo.Origin("folder", "paraphrased")


# -- recording ----------------------------------------------------------------------------------

def test_a_fact_stored_without_a_declaration_is_recorded_as_unknown(db) -> None:
    _store(db, "The office plant is a fern.")
    assert _origins(db) == [("unknown", "unknown", None, None)]


def test_a_declared_origin_is_recorded_and_the_first_one_stands(db) -> None:
    with fo.declare(fo.Origin("folder", "split", source_type="folder", actor_id="a1")):
        fact = _store(db, "The office plant is a fern.")
    with fo.declare(fo.Origin("import", "copied")):
        db.store_fact(AtomicFact(fact_id=fact.fact_id, memory_id=fact.memory_id,
                                 content="The office plant is a fern, again."))
    assert _origins(db)[0] == ("folder", "split", "folder", "a1")
    assert len(_origins(db)) == 1


def test_the_declaration_ends_with_its_block(db) -> None:
    with fo.declare(fo.Origin("folder", "split")):
        pass
    _store(db, "Another memory about the fern.")
    assert _origins(db) == [("unknown", "unknown", None, None)]


def test_recording_never_blocks_a_save(db, monkeypatch) -> None:
    def broken(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(fo, "current", broken)
    fact = _store(db, "A memory that must survive.")
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?",
                      (fact.fact_id,))[0]["n"] == 1


def test_erasing_a_fact_erases_its_origin(db) -> None:
    fact = _store(db, "A memory to forget.")
    with db.raw_connection() as conn:
        conn.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (fact.fact_id,))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact.fact_id,))
    assert _origins(db) == []


def test_derived_facts_carry_their_memorys_channel_and_their_own_transform(db) -> None:
    content = "Release on Friday. Alex reviews the notes first."
    origin = fo.derived_from(content, source_type="http", metadata={"agent_id": "claude"},
                             actor_id="daemon-capability:x", model_mode="b")
    record = MemoryRecord(content=content, session_date="2026-11-15")
    db.store_memory(record)
    with fo.declare(origin):
        for text in (content, "Alex reviews the notes first.", "Alex will review the notes."):
            db.store_fact(AtomicFact(memory_id=record.memory_id, content=text))
    rows = [tuple(r) for r in db.execute(
        "SELECT channel, transform, agent_id, parent_memory_id FROM fact_origin")]
    assert sorted(r[1] for r in rows) == ["model_rewritten", "split", "verbatim"]
    assert {(r[0], r[2], r[3]) for r in rows} == {("remember_http", "claude", record.memory_id)}


# -- end to end: a real save through the canonical runtime ---------------------------------------

def test_a_save_through_the_canonical_writer_records_its_origin(tmp_path: Path,
                                                               monkeypatch) -> None:
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    data_dir = tmp_path / "slm"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    store = DatabaseManager(data_dir / "memory.db")
    store.initialize(schema)
    with store.raw_connection() as conn:
        for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                          M033_projection_transactions, M034_obligation_integrity, M074):
            migration.apply(conn)
    runtime = CanonicalRememberRuntime(
        db=store, profile_id="default",
        writer=build_immediate_admission_handler(store, profile_id="default"),
        journal_path=data_dir / "admission_journal.db", owner_id="origin-test")
    runtime.start()
    try:
        receipt = runtime.remember(
            RememberRequest(content="The standup moved to 10:30 on Tuesdays.",
                            profile_id="default", source_type="http",
                            idempotency_key="origin:1", trusted_actor_id="daemon-capability:t",
                            metadata={"agent_id": "codex", "agent_id_source": "connection"}),
            Actor("daemon-capability:t", frozenset({"default"}), frozenset({"personal"})),
            deadline_ms=2_000, accept_after_ms=1_200)
        assert receipt.payload["status"] in {"queryable", "accepted"}
        assert runtime.wait_for_deferred(timeout=30.0)
    finally:
        runtime.stop()
    rows = store.execute("SELECT channel, transform, source_type, actor_id, agent_id, "
                         "agent_id_source FROM fact_origin")
    assert [tuple(r) for r in rows] == [
        ("remember_http", "verbatim", "http", "daemon-capability:t", "codex", "connection")]
