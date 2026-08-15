"""Persistence gates for the v4.0.2 Agent Experience learning plane."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import superlocalmemory.storage.migration_runner as migration_runner
from superlocalmemory.storage import schema
from superlocalmemory.storage.agent_experience import (
    AgentExperienceConflictError,
    AgentExperienceStore,
    CognitiveTurnTransitionError,
    LearningWriteBusyError,
    ProfileAdmissionError,
    get_profile_receipt_summary,
    purge_profile_receipts,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _experience(profile_id: str = "alpha") -> dict:
    return {
        "experience_id": "experience-1",
        "profile_id": profile_id,
        "occurred_at": "2026-08-15T00:00:00+00:00",
        "task_class": "code",
        "project_scope": "project-digest",
        "route": {
            "harness": "codex",
            "provider": "openai",
            "model": "gpt-5.6",
            "effort": "high",
            "machine": "machine-digest",
        },
        "verification": {"authority": "deterministic_gate", "evidence_digest": "a" * 64},
        "producer_claim": "success",
        "terminal_status": "succeeded",
    }


def _turn(profile_id: str = "alpha") -> dict:
    return {
        "receipt_id": "turn-1",
        "task_id": "task-1",
        "profile_id": profile_id,
        "project_scope": "project-digest",
        "query_digest": "b" * 64,
        "fact_decisions": {"fact-1": "used"},
        "state": "open",
    }


@pytest.fixture
def store(tmp_path: Path) -> AgentExperienceStore:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
    return AgentExperienceStore(
        path, is_profile_active=lambda profile_id: profile_id in {"alpha", "beta"}
    )


def test_m040_is_eager_learning_only_and_never_creates_receipts_in_memory(tmp_path: Path) -> None:
    learning_db, memory_db = tmp_path / "learning.db", tmp_path / "memory.db"
    result = migration_runner.apply_all(learning_db, memory_db)
    assert result["failed"] == []
    assert "M040_agent_experience_receipts" in result["applied"]
    with sqlite3.connect(learning_db) as conn:
        assert conn.execute("SELECT 1 FROM agent_experiences").fetchone() is None
    with sqlite3.connect(memory_db) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_experiences'"
            ).fetchone()
            is None
        )


def test_absent_receipt_store_is_honestly_unavailable(tmp_path: Path) -> None:
    summary = get_profile_receipt_summary(tmp_path / "missing-learning.db", "alpha")
    assert summary["is_real"] is False
    assert summary["availability"] == "unavailable"
    assert summary["experiences_total"] == 0


def test_experience_is_profile_scoped_idempotent_and_reconstructs_contract(
    store: AgentExperienceStore,
) -> None:
    payload = _experience()
    assert store.record_experience(payload) is True
    assert store.record_experience(payload) is False
    assert store.get_experience("alpha", "experience-1") == payload
    assert store.get_experience("beta", "experience-1") is None
    assert store.record_experience(_experience("beta")) is True
    assert store.get_experience("beta", "experience-1") == _experience("beta")


def test_experience_rejects_unknown_profile_and_changed_idempotency_key(
    store: AgentExperienceStore,
) -> None:
    with pytest.raises(ProfileAdmissionError):
        store.record_experience(_experience("gone"))
    assert store.record_experience(_experience()) is True
    altered = _experience()
    altered["task_class"] = "security"
    with pytest.raises(AgentExperienceConflictError, match="different evidence"):
        store.record_experience(altered)


def test_cognitive_turn_finalization_is_atomic_and_idempotent(store: AgentExperienceStore) -> None:
    turn = _turn()
    outcome = {"authority": "deterministic_gate", "receipt_digest": "c" * 64, "reference": "gate-1"}
    assert store.create_cognitive_turn(turn) is True
    assert store.create_cognitive_turn(turn) is False
    assert store.finalize_cognitive_turn("alpha", "turn-1", outcome) is True
    assert store.finalize_cognitive_turn("alpha", "turn-1", outcome) is False
    assert store.get_cognitive_turn("alpha", "turn-1") == {
        **turn,
        "state": "finalized",
        "outcome": outcome,
    }


def test_cognitive_turn_rejects_cross_profile_and_changed_finalization(
    store: AgentExperienceStore,
) -> None:
    assert store.create_cognitive_turn(_turn()) is True
    outcome = {"authority": "deterministic_gate", "receipt_digest": "c" * 64, "reference": "gate-1"}
    with pytest.raises(CognitiveTurnTransitionError, match="not found"):
        store.finalize_cognitive_turn("beta", "turn-1", outcome)
    assert store.finalize_cognitive_turn("alpha", "turn-1", outcome) is True
    with pytest.raises(AgentExperienceConflictError, match="finalized differently"):
        store.finalize_cognitive_turn("alpha", "turn-1", {**outcome, "reference": "gate-2"})


def test_profile_erasure_removes_all_learning_receipts_and_checks_residue(
    store: AgentExperienceStore,
) -> None:
    assert store.record_experience(_experience())
    assert store.create_cognitive_turn(_turn())
    assert store.record_experience(_experience("beta"))
    assert store.erase_profile("alpha") == 2
    assert store.get_experience("alpha", "experience-1") is None
    assert store.get_cognitive_turn("alpha", "turn-1") is None
    assert store.get_experience("beta", "experience-1") == _experience("beta")


def test_profile_receipt_purger_is_legacy_safe_and_stops_half_schema(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    assert purge_profile_receipts(path, "alpha") == 0
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE agent_experiences (profile_id TEXT)")
    with pytest.raises(sqlite3.OperationalError, match="incomplete"):
        purge_profile_receipts(path, "alpha")


def test_closed_profile_rejects_new_receipts_after_purge(store: AgentExperienceStore) -> None:
    assert store.record_experience(_experience())
    assert store.erase_profile("alpha") == 1
    with pytest.raises(ProfileAdmissionError, match="closing"):
        store.record_experience(_experience())


def test_durable_closure_blocks_a_fresh_store_after_profile_erasure(tmp_path: Path) -> None:
    """The durable sidecar closes the race a process-local gate cannot see."""
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
    first = AgentExperienceStore(path, is_profile_active=lambda _: True)
    second = AgentExperienceStore(path, is_profile_active=lambda _: True)
    assert first.record_experience(_experience())
    assert first.erase_profile("alpha") == 1
    with pytest.raises(ProfileAdmissionError, match="inactive or closing"):
        second.record_experience(_experience())


def test_learning_reset_purges_receipts_without_closing_active_profile(
    store: AgentExperienceStore,
) -> None:
    from superlocalmemory.learning.database import LearningDatabase

    assert store.record_experience(_experience())
    LearningDatabase(store._path).reset("alpha")
    assert store.record_experience(_experience())


def test_learning_reset_purges_agent_receipts_when_m040_exists(store: AgentExperienceStore) -> None:
    from superlocalmemory.learning.database import LearningDatabase

    assert store.record_experience(_experience())
    assert store.create_cognitive_turn(_turn())
    LearningDatabase(store._path).reset("alpha")
    assert store.get_experience("alpha", "experience-1") is None
    assert store.get_cognitive_turn("alpha", "turn-1") is None


def test_normal_profile_deletion_purges_learning_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from superlocalmemory.server.routes.helpers import delete_profile_from_db
    from superlocalmemory.storage import schema

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    memory_db, learning_db = tmp_path / "memory.db", tmp_path / "learning.db"
    with sqlite3.connect(memory_db) as conn:
        schema.create_all_tables(conn)
        conn.execute("INSERT INTO profiles (profile_id, name) VALUES ('alpha', 'Alpha')")
    with sqlite3.connect(learning_db) as conn:
        m040.apply(conn)
    receipt_store = AgentExperienceStore(learning_db, is_profile_active=lambda _: True)
    assert receipt_store.record_experience(_experience())

    delete_profile_from_db("alpha")

    assert receipt_store.get_experience("alpha", "experience-1") is None
    with sqlite3.connect(memory_db) as conn:
        assert conn.execute("SELECT 1 FROM profiles WHERE profile_id='alpha'").fetchone() is None


def test_write_busy_deadline_is_bounded_and_visible(
    store: AgentExperienceStore, tmp_path: Path
) -> None:
    locked = sqlite3.connect(tmp_path / "learning.db", isolation_level=None)
    try:
        locked.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(LearningWriteBusyError):
            store.record_experience(_experience())
        assert time.monotonic() - started < 1.1
    finally:
        locked.rollback()
        locked.close()


def test_concurrent_receipt_writes_complete_without_deadlock(store: AgentExperienceStore) -> None:
    def write(number: int) -> bool:
        payload = _experience()
        payload["experience_id"] = f"experience-{number}"
        return store.record_experience(payload)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(write, range(32)))
    assert outcomes == [True] * 32
    assert time.monotonic() - started < 2.0


def test_receipt_load_does_not_block_memory_remember_or_recall(tmp_path: Path) -> None:
    """Separate DB domains keep foreground memory operations inside 2 seconds.

    This is intentionally a mixed workload, not a microbenchmark of one
    SQLite statement: concurrent receipt writes hit ``learning.db`` while
    foreground remembers and recalls use ``memory.db``.  A deadlock or an
    accidental cross-database write would surface as a timeout or outlier.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    with sqlite3.connect(learning_db) as conn:
        m040.apply(conn)
    receipt_store = AgentExperienceStore(
        learning_db, is_profile_active=lambda profile_id: profile_id == "default"
    )
    memory = DatabaseManager(memory_db)
    memory.initialize(schema)
    durations: list[float] = []

    def receipt(number: int) -> bool:
        payload = _experience("default")
        payload["experience_id"] = f"load-{number}"
        return receipt_store.record_experience(payload)

    def foreground(number: int) -> int:
        started = time.monotonic()
        record = MemoryRecord(profile_id="default", content=f"foreground {number}")
        memory.store_memory(record)
        memory.store_fact(
            AtomicFact(
                profile_id="default", memory_id=record.memory_id,
                content=f"foreground fact {number}", fact_type=FactType.SEMANTIC,
            )
        )
        count = len(memory.get_all_facts("default"))
        durations.append(time.monotonic() - started)
        return count

    with ThreadPoolExecutor(max_workers=12) as executor:
        receipt_futures = [executor.submit(receipt, number) for number in range(48)]
        foreground_futures = [executor.submit(foreground, number) for number in range(20)]
        assert [future.result(timeout=5) for future in receipt_futures] == [True] * 48
        counts = [future.result(timeout=5) for future in foreground_futures]

    assert all(count >= 1 for count in counts)
    # Nearest-rank p95: users care about the slow tail, not just average speed.
    p95 = sorted(durations)[int(len(durations) * 0.95) - 1]
    assert p95 < 2.0, f"foreground remember+recall p95 {p95:.3f}s exceeded 2s"


def test_m040_refuses_malformed_populated_tables_but_repairs_an_index_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        conn.execute("DROP INDEX idx_agent_experiences_profile_occurred")
        conn.execute(
            "CREATE INDEX idx_agent_experiences_profile_occurred "
            "ON agent_experiences (experience_id)"
        )
        assert m040.verify(conn) is False
        m040.repair(conn)
        assert m040.verify(conn) is True
        conn.execute(
            "INSERT INTO agent_experiences (profile_id, experience_id, occurred_at, task_class, "
            "project_scope, route_json, verification_authority, verification_digest, "
            "producer_claim, "
            "terminal_status, artifact_digests_json, payload_sha256, created_at) "
            "VALUES ('alpha', 'e', 'x', 'x', 'x', '{}', 'deterministic_gate', 'x', "
            "'success', 'succeeded', '[]', 'x', 'x')"
        )
        conn.execute("DROP INDEX idx_agent_experiences_profile_occurred")
        conn.execute("ALTER TABLE agent_experiences ADD COLUMN unsafe TEXT")
        with pytest.raises(sqlite3.OperationalError, match="malformed"):
            m040.repair(conn)


def test_m040_rebuilds_same_named_index_from_the_wrong_table(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE stale (profile_id TEXT, occurred_at TEXT)")
        conn.execute(
            "CREATE INDEX idx_agent_experiences_profile_occurred "
            "ON stale (profile_id, occurred_at)"
        )
        m040.apply(conn)
        assert m040.verify(conn) is True
        owner = conn.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?",
            ("idx_agent_experiences_profile_occurred",),
        ).fetchone()
        assert owner == ("agent_experiences",)
