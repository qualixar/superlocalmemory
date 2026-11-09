"""Synthetic, redacted an earlier stage quality contracts for SLM 4.0.5.

These tests deliberately use generated identifiers and fixture text only.  They
exercise the existing temporal and receipt boundaries without reading the live
SLM data directory.  Where the review-gated correction ledger is not connected
to retrieval yet, the test proves the safe pre-integration behaviour instead
of pretending that the future seam already exists.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.integrations.bounded_loops_mcp import bridge_payload
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.agent_experience import (
    AgentExperienceStore,
    get_profile_receipt_summary,
)
from superlocalmemory.storage.correction_cases import (
    CorrectionActor,
    CorrectionCaseStore,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.external_evidence import (
    ExternalEvidenceStore,
    get_profile_external_evidence_summary,
)
from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040
from superlocalmemory.storage.migrations import M041_external_evidence_receipts as m041
from superlocalmemory.storage.migrations import M042_correction_case_ledger as m042
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def memory_db(tmp_path: Path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    return db


def _seed_fact(
    db: DatabaseManager,
    *,
    fact_id: str,
    content: str,
    profile_id: str = "default",
    scope: str = "personal",
) -> None:
    memory_id = f"memory-{fact_id}"
    db.store_memory(MemoryRecord(memory_id=memory_id, profile_id=profile_id, content="fixture"))
    db.store_fact(
        AtomicFact(
            fact_id=fact_id,
            memory_id=memory_id,
            profile_id=profile_id,
            scope=scope,
            content=content,
            fact_type=FactType.SEMANTIC,
        )
    )


def _reviewer() -> CorrectionActor:
    return CorrectionActor(
        actor_id="synthetic-reviewer",
        actor_kind="human",
        trust_tier="operator_verified",
    )


def _correction_store(db: DatabaseManager) -> CorrectionCaseStore:
    with sqlite3.connect(db.db_path) as conn:
        m042.apply(conn)
    return CorrectionCaseStore(
        db.db_path,
        is_profile_active=lambda profile_id: profile_id == "default",
        is_actor_trusted=lambda actor: actor == _reviewer(),
    )


def _propose(store: CorrectionCaseStore, *, case_id: str, key: str) -> None:
    store.propose(
        case_id=case_id,
        profile_id="default",
        scope="project",
        predecessor_fact_id=f"old-{case_id}",
        successor_fact_id=f"new-{case_id}",
        reason_code="synthetic_release_state_replaced",
        actor=_reviewer(),
        idempotency_key=key,
    )


def _experience() -> dict[str, object]:
    return {
        "experience_id": "synthetic-experience",
        "profile_id": "default",
        "occurred_at": "2026-08-16T00:00:00+00:00",
        "task_class": "quality_harness",
        "project_scope": "synthetic",
        "route": {
            "harness": "synthetic",
            "provider": "fixture",
            "model": "fixture",
            "effort": "low",
            "machine": "fixture",
        },
        "verification": {
            "authority": "deterministic_gate",
            "evidence_digest": "a" * 64,
        },
        "producer_claim": "success",
        "terminal_status": "succeeded",
    }


def _external_evidence() -> dict[str, object]:
    return {
        "contract": "bounded-loops.dev/slm-bridge/v1",
        "profile_id": "default",
        "workspace_id": "sha256:" + "a" * 64,
        "run_ref": "synthetic-run",
        "run_id": "synthetic-run-id",
        "outcome": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "demonstration": False,
        "eligible_for_learning": False,
        "terminal_at": "2026-08-16T00:00:00Z",
        "graph_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "policy_digest": "sha256:" + "d" * 64,
        "receipt": {
            "sequence": 1,
            "head_digest": "sha256:" + "e" * 64,
            "trust": "local_hash_chain_only",
        },
        "nodes": [],
    }


def test_q01_current_correction_excludes_predecessor_and_keeps_successor(
    memory_db: DatabaseManager,
) -> None:
    _seed_fact(memory_db, fact_id="old-release", content="synthetic old state")
    _seed_fact(memory_db, fact_id="new-release", content="synthetic current state")
    memory_db.invalidate_fact_temporal(
        "old-release", invalidated_by="new-release", invalidation_reason="reviewed_fixture"
    )

    assert memory_db.get_invalidated_fact_ids(["old-release", "new-release"], "default") == {
        "old-release"
    }
    assert {"new-release"} <= set(memory_db.get_valid_facts("default"))


def test_q02_historical_correction_recovers_predecessor_before_system_expiry(
    memory_db: DatabaseManager,
) -> None:
    _seed_fact(memory_db, fact_id="old-history", content="synthetic prior state")
    memory_db.execute(
        "UPDATE fact_temporal_validity SET system_created_at=?, system_expired_at=? "
        "WHERE fact_id=?",
        ("2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "old-history"),
    )

    assert (
        memory_db.get_strict_temporal_inadmissible_fact_ids(
            ["old-history"], "default", known_as_of="2026-01-15T00:00:00+00:00"
        )
        == set()
    )
    assert memory_db.get_strict_temporal_inadmissible_fact_ids(
        ["old-history"], "default", known_as_of="2026-02-01T00:00:00+00:00"
    ) == {"old-history"}


def test_q03_two_clocks_are_independent(memory_db: DatabaseManager) -> None:
    _seed_fact(memory_db, fact_id="future-event", content="synthetic scheduled event")
    memory_db.execute(
        "UPDATE fact_temporal_validity SET system_created_at=?, valid_from=? WHERE fact_id=?",
        ("2026-01-01T00:00:00+00:00", "2030-01-01T00:00:00+00:00", "future-event"),
    )

    assert (
        memory_db.get_strict_temporal_inadmissible_fact_ids(
            ["future-event"], "default", known_as_of="2026-02-01T00:00:00+00:00"
        )
        == set()
    )
    assert memory_db.get_strict_temporal_inadmissible_fact_ids(
        ["future-event"], "default", valid_at="2026-02-01T00:00:00+00:00"
    ) == {"future-event"}


def test_q04_reviewed_case_changes_current_truth_only_after_apply(
    memory_db: DatabaseManager,
) -> None:
    """A pending successor is excluded; applying flips both identities atomically."""
    _seed_fact(
        memory_db, fact_id="old-case", content="synthetic old release", scope="project"
    )
    _seed_fact(
        memory_db, fact_id="new-case", content="synthetic new release", scope="project"
    )
    store = _correction_store(memory_db)
    _propose(store, case_id="case", key="release-case-propose")
    assert memory_db.get_nonapplied_correction_successor_ids(["new-case"], "default") == {
        "new-case"
    }
    applied = store.apply(
        "case", expected_version=0, actor=_reviewer(), operation_id="release-case-apply"
    )

    assert applied.status == "applied"
    assert memory_db.get_invalidated_fact_ids(["old-case"], "default") == {"old-case"}
    assert memory_db.get_nonapplied_correction_successor_ids(["new-case"], "default") == set()
    with sqlite3.connect(memory_db.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM correction_events WHERE event_type='applied'"
        ).fetchone() == (1,)


def test_q05_latest_project_decision_wins_without_destroying_historical_plan(
    memory_db: DatabaseManager,
) -> None:
    _seed_fact(memory_db, fact_id="plan-v1", content="synthetic project decision v1")
    _seed_fact(memory_db, fact_id="plan-v2", content="synthetic project decision v2")
    memory_db.execute(
        "UPDATE fact_temporal_validity SET system_created_at=?, system_expired_at=? "
        "WHERE fact_id=?",
        ("2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00", "plan-v1"),
    )

    assert (
        memory_db.get_strict_temporal_inadmissible_fact_ids(
            ["plan-v1"], "default", known_as_of="2026-02-01T00:00:00+00:00"
        )
        == set()
    )
    assert memory_db.get_strict_temporal_inadmissible_fact_ids(
        ["plan-v1"], "default", known_as_of="2026-03-01T00:00:00+00:00"
    ) == {"plan-v1"}
    assert memory_db.get_fact("plan-v1") is not None
    assert memory_db.get_fact("plan-v2") is not None


def test_q06_duplicate_proposals_are_idempotent_and_do_not_fabricate_correction(
    memory_db: DatabaseManager,
) -> None:
    _seed_fact(memory_db, fact_id="duplicate-a", content="synthetic duplicate wording")
    _seed_fact(memory_db, fact_id="duplicate-b", content="synthetic duplicate wording")
    store = _correction_store(memory_db)
    _propose(store, case_id="duplicate-case", key="duplicate-proposal")
    _propose(store, case_id="duplicate-case", key="duplicate-proposal")

    assert memory_db.get_invalidated_fact_ids(["duplicate-a", "duplicate-b"], "default") == set()
    with sqlite3.connect(memory_db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM correction_events").fetchone() == (1,)


def test_q07_profile_scoped_correction_never_leaks_to_another_profile(
    memory_db: DatabaseManager,
) -> None:
    memory_db.execute("INSERT INTO profiles (profile_id, name) VALUES (?, ?)", ("alpha", "Alpha"))
    memory_db.execute("INSERT INTO profiles (profile_id, name) VALUES (?, ?)", ("beta", "Beta"))
    _seed_fact(memory_db, fact_id="alpha-old", content="synthetic alpha state", profile_id="alpha")
    memory_db.invalidate_fact_temporal(
        "alpha-old", invalidated_by="alpha-new", invalidation_reason="reviewed_fixture"
    )

    assert memory_db.get_invalidated_fact_ids(["alpha-old"], "alpha") == {"alpha-old"}
    assert memory_db.get_invalidated_fact_ids(["alpha-old"], "beta") == set()


def test_q08_native_host_assets_declare_distinct_identities_and_one_code_profile() -> None:
    """Codex and Claude use the same portable MCP contract with honest identity."""
    root = Path(__file__).resolve().parents[2]
    codex = (root / "codex-plugin" / ".codex" / "config.toml").read_text(encoding="utf-8")
    codex_template = (root / "ide" / "configs" / "codex-mcp.toml").read_text(encoding="utf-8")
    claude = (root / "plugin" / ".mcp.json").read_text(encoding="utf-8")

    assert 'SLM_AGENT_ID = "codex"' in codex
    # Checked on the env ASSIGNMENT, not the whole file: both configs now carry
    # a comment explaining why the profile is absent, and a bare "not in" over
    # the text matches that comment and fails on correct config.
    import re as _re

    def _env_line(text: str) -> str:
        m = _re.search(r"^env = \{.*\}\s*$", text, _re.MULTILINE)
        return m.group(0) if m else ""

    assert "SLM_MCP_PROFILE" not in _env_line(codex), (
        "4.1.3: the host chooses its own tool set; the plugin must not pin one"
    )
    assert 'SLM_AGENT_ID = "codex"' in codex_template
    assert "SLM_MCP_PROFILE" not in _env_line(codex_template)
    assert '"SLM_AGENT_ID": "claude_code"' in claude
    assert '"SLM_MCP_PROFILE"' not in claude


def test_q09_receipts_and_memory_activity_remain_separate(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.db"
    memory_db = DatabaseManager(memory_path)
    memory_db.initialize(real_schema)
    _seed_fact(memory_db, fact_id="ordinary-memory", content="synthetic remembered text")
    learning_path = tmp_path / "learning.db"
    with sqlite3.connect(learning_path) as conn:
        m040.apply(conn)
        m041.apply(conn)
    experiences = AgentExperienceStore(learning_path, is_profile_active=lambda _: True)
    evidence = ExternalEvidenceStore(learning_path, is_profile_active=lambda _: True)

    assert experiences.record_experience(_experience()) is True
    assert evidence.record(_external_evidence()) is True
    receipt_summary = get_profile_receipt_summary(learning_path, "default")
    evidence_summary = get_profile_external_evidence_summary(learning_path, "default")

    assert memory_db.get_fact("ordinary-memory") is not None
    assert receipt_summary["experiences_total"] == 1
    assert evidence_summary["total"] == 1
    assert evidence_summary["control_plane"] == "observation_only"


def test_q10_optional_bridge_receipt_stays_outside_memory_recall_plane(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.db"
    memory_db = DatabaseManager(memory_path)
    memory_db.initialize(real_schema)
    _seed_fact(memory_db, fact_id="ordinary-memory", content="synthetic ordinary fact")
    learning_path = tmp_path / "learning.db"
    with sqlite3.connect(learning_path) as conn:
        m041.apply(conn)
    evidence = ExternalEvidenceStore(learning_path, is_profile_active=lambda _: True)

    bridge = bridge_payload(_external_evidence(), profile_id="default")
    assert evidence.record(bridge) is True
    assert memory_db.get_valid_facts("default") == ["ordinary-memory"]
    assert get_profile_external_evidence_summary(learning_path, "default")["total"] == 1


def test_q11_small_concurrent_write_harness_completes_without_deadlock(
    memory_db: DatabaseManager,
) -> None:
    """A bounded smoke test records liveness metrics; release SLOs use a larger runner."""
    store = _correction_store(memory_db)
    started = time.monotonic()

    def write(index: int) -> str:
        _propose(store, case_id=f"concurrent-{index}", key=f"concurrent-key-{index}")
        return f"concurrent-{index}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        completed = list(executor.map(write, range(6), timeout=10))
    elapsed = time.monotonic() - started

    assert completed == [f"concurrent-{index}" for index in range(6)]
    assert elapsed >= 0
    with sqlite3.connect(memory_db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (6,)


def test_q12_approved_correction_fails_closed_when_admission_state_is_unavailable() -> None:
    """A lifecycle read fault yields explicit empty candidate paths, never stale truth."""
    from superlocalmemory.retrieval.temporal_validity_filter import admit_correction_candidates

    db = MagicMock()
    db.get_invalidated_fact_ids.side_effect = sqlite3.OperationalError("injected fault")
    admitted = admit_correction_candidates(
        {"semantic": [("approved-stale", 0.99)]}, "default", db
    )

    assert admitted == {"semantic": []}
