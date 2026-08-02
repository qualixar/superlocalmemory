# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import pytest


def _store(engine, text: str):
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    return canonical_store(
        engine,
        text,
        source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True,
        return_receipt=True,
    )


@pytest.fixture
def stored_fact(engine_with_mock_deps):
    engine = engine_with_mock_deps
    operation = _store(
        engine, "Priya joined the Berlin platform team on 2025-01-10 as staff engineer"
    )
    fact_ids = list(operation.final_fact_ids)
    assert fact_ids
    return engine, fact_ids[0]


def test_authorized_delete_produces_verified_erasure(stored_fact):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.transactions import (
        fetch_receipt,
        is_tombstoned,
        verify_receipt,
    )

    engine, fid = stored_fact
    before = engine._db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", (fid,)
    )
    assert dict(before[0])["c"] >= 1

    result = delete_fact_authorized(
        engine, fid,
        trusted_actor_id=local_trusted_actor_id("python-api"),
        source_agent_id="python-api",
    )

    assert result["ok"] is True
    assert result["erasure_verified"] is True
    assert result["erasure_id"]

    after = engine._db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", (fid,)
    )
    assert dict(after[0])["c"] == 0

    with engine._db.raw_connection() as conn:
        receipt = fetch_receipt(conn, result["erasure_id"])
        assert receipt is not None
        assert receipt.state == "COMPLETE"
        assert receipt.subject_type == "fact"
        assert verify_receipt(conn, result["erasure_id"]) is True
        assert is_tombstoned(conn, engine._profile_id, fid) is True


def test_canonical_runtime_path_purges_before_canonical_delete(stored_fact):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.transactions import verify_receipt

    engine, fid = stored_fact
    observed: dict = {}

    class _FakeWorker:
        def __init__(self, engine):
            self._engine = engine

        def delete_fact(self, profile_id, fact_id, *, idempotency_key=None):
            rows = self._engine._db.execute(
                "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", (fact_id,)
            )
            observed["bm25_at_canonical_delete"] = dict(rows[0])["c"]
            tomb = self._engine._db.execute(
                "SELECT COUNT(*) AS c FROM projection_tombstones "
                "WHERE profile_id = ? AND fact_id = ?",
                (profile_id, fact_id),
            )
            observed["tombstone_at_canonical_delete"] = dict(tomb[0])["c"] == 1
            self._engine._db.delete_fact(fact_id, profile_id=profile_id)
            return {"ok": True, "deleted": fact_id}

    before = engine._db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", (fid,)
    )
    assert dict(before[0])["c"] >= 1

    result = delete_fact_authorized(
        engine, fid,
        trusted_actor_id=local_trusted_actor_id("dashboard"),
        source_agent_id="dashboard",
        canonical_runtime=_FakeWorker(engine),
    )

    assert observed["bm25_at_canonical_delete"] == 0
    assert observed["tombstone_at_canonical_delete"] is True
    assert result["erasure_verified"] is True
    assert result["erasure_state"] == "COMPLETE"
    after = engine._db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", (fid,)
    )
    assert dict(after[0])["c"] == 0
    with engine._db.raw_connection() as conn:
        assert verify_receipt(conn, result["erasure_id"]) is True


def test_multi_fact_memory_delete_preserves_siblings(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    pid = engine.profile_id
    db = engine._db
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, ?)",
        ("m1", pid, "raw shared memory"),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES (?, ?, ?, ?)",
        ("fa", "m1", pid, "fact a"),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES (?, ?, ?, ?)",
        ("fb", "m1", pid, "fact b"),
    )
    db.store_bm25_tokens("fa", pid, ["alpha"])
    db.store_bm25_tokens("fb", pid, ["beta"])

    result = delete_fact_authorized(
        engine, "fa",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        source_agent_id="python-api",
    )
    assert result["ok"] is True

    assert db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = 'fa'") == []
    assert db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = 'fb'") != []
    assert db.execute("SELECT 1 FROM memories WHERE memory_id = 'm1'") != []
    fa_bm25 = db.execute("SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = 'fa'")
    fb_bm25 = db.execute("SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = 'fb'")
    assert dict(fa_bm25[0])["c"] == 0
    assert dict(fb_bm25[0])["c"] == 1


def test_idempotent_retry_reverifies_absence(stored_fact):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.transactions import fetch_receipt

    engine, fid = stored_fact

    class _IdempotentWorker:
        def __init__(self, engine):
            self._engine = engine

        def delete_fact(self, profile_id, fact_id, *, idempotency_key=None):
            present = self._engine._db.execute(
                "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if present:
                self._engine._db.delete_fact(fact_id, profile_id=profile_id)
            return {"ok": True, "deleted": fact_id}

    worker = _IdempotentWorker(engine)
    actor = local_trusted_actor_id("dashboard")

    first = delete_fact_authorized(
        engine, fid, trusted_actor_id=actor, source_agent_id="dashboard",
        canonical_runtime=worker,
    )
    assert first["erasure_verified"] is True

    second = delete_fact_authorized(
        engine, fid, trusted_actor_id=actor, source_agent_id="dashboard",
        canonical_runtime=worker,
    )
    assert second["ok"] is True
    assert second["erasure_verified"] is True
    assert second["erasure_state"] == "COMPLETE"
    with engine._db.raw_connection() as conn:
        assert fetch_receipt(conn, second["erasure_id"]) is not None


def test_worker_ok_without_delete_is_not_verified(stored_fact):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine, fid = stored_fact

    class _LyingWorker:
        def delete_fact(self, profile_id, fact_id, *, idempotency_key=None):
            return {"ok": True, "deleted": fact_id}

    result = delete_fact_authorized(
        engine, fid,
        trusted_actor_id=local_trusted_actor_id("dashboard"),
        source_agent_id="dashboard",
        canonical_runtime=_LyingWorker(),
    )

    # MED-5: lying worker with ok=True but fact not deleted → result ok=False + retryable
    assert result["ok"] is False
    assert result.get("retryable") is True
    assert result["erasure_verified"] is False
    assert result["erasure_state"] == "FAILED"
    still_there = engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fid,)
    )
    assert still_there != []


def test_materializer_skips_tombstoned_fact(engine_with_mock_deps):
    from superlocalmemory.core.store_pipeline import _fact_is_tombstoned

    engine = engine_with_mock_deps
    pid = engine.profile_id
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO projection_tombstones "
            "(profile_id, fact_id, erasure_id, memory_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (pid, "erased-fact", "e1", "m1", 0),
        )
        conn.commit()

    assert _fact_is_tombstoned(engine._db, pid, "erased-fact") is True
    assert _fact_is_tombstoned(engine._db, pid, "live-fact") is False


def test_failclosed_blocks_canonical_when_spine_residue(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    pid = engine.profile_id
    db = engine._db
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, ?)",
        ("mF", pid, "raw"),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES (?, ?, ?, ?)",
        ("fF", "mF", pid, "fact"),
    )
    db.store_bm25_tokens("fF", pid, ["alpha"])
    orig = db.delete_bm25_tokens_for_fact
    db.delete_bm25_tokens_for_fact = lambda *a, **k: None
    try:
        result = delete_fact_authorized(
            engine, "fF",
            trusted_actor_id=local_trusted_actor_id("python-api"),
            source_agent_id="python-api",
        )
    finally:
        db.delete_bm25_tokens_for_fact = orig

    assert result["ok"] is False
    assert result.get("retryable") is True
    assert result["erasure_state"] == "FAILED"
    assert db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = 'fF'") != []


def test_delete_missing_fact_reports_not_found(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    result = delete_fact_authorized(
        engine, "no-such-fact",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        source_agent_id="python-api",
    )
    assert result["ok"] is False
    assert "erasure_id" not in result


def test_tombstone_provenance_conflict_retains_canonical(stored_fact):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.transactions.erasure import write_tombstones

    engine, fid = stored_fact
    pid = engine._profile_id

    # Pre-seed a tombstone whose memory_id disagrees with the fact's real one.
    assert write_tombstones(engine._db, pid, (fid,), "e-prior", 0.0, "mBOGUS") is True

    result = delete_fact_authorized(
        engine, fid,
        trusted_actor_id=local_trusted_actor_id("python-api"),
        source_agent_id="python-api",
    )

    # No valid anti-resurrection tombstone => fail closed, canonical retained.
    assert result["ok"] is False
    assert result.get("retryable") is True
    assert engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fid,)
    ) != []


def test_retry_resumes_partial_erasure_and_reclaims_memory(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.transactions.erasure import write_tombstones

    engine = engine_with_mock_deps
    pid = engine.profile_id
    db = engine._db

    # A prior erasure got partway: the fact's canonical row is already gone and
    # a tombstone (with memory_id) was written, but the source memory survived a
    # transient delete failure. A retry must resume, not report "not found".
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, ?)",
        ("mR", pid, "orphaned raw memory"),
    )
    assert write_tombstones(db, pid, ("fR",), "e-prior", 0.0, "mR") is True

    result = delete_fact_authorized(
        engine, "fR",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        source_agent_id="python-api",
    )

    assert result["ok"] is True
    assert result["erasure_verified"] is True
    assert db.execute("SELECT 1 FROM memories WHERE memory_id = 'mR'") == []
