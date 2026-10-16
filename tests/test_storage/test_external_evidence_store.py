"""Persistence contract for external MCP evidence in SLM's learning plane."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from superlocalmemory.storage.agent_experience import AgentExperienceStore, ProfileAdmissionError
from superlocalmemory.storage.external_evidence import (
    ExternalEvidenceConflictError,
    ExternalEvidenceStore,
    ExternalEvidenceValidationError,
)
from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040
from superlocalmemory.storage.migrations import M041_external_evidence_receipts as m041

# Pure liveness/deadlock guard for the concurrency test below, not a
# performance target: 32 writes serialised behind one in-process lock take
# well under a second normally.  This only exists so a genuine deadlock in
# the store's process lock fails the test instead of hanging the suite.
_DEADLOCK_GUARD_S = 15.0


def _evidence(profile_id: str = "alpha") -> dict:
    return {
        "contract": "bounded-loops.dev/slm-bridge/v1",
        "profile_id": profile_id,
        "workspace_id": "sha256:" + "a" * 64,
        "run_ref": "nightly-1",
        "run_id": "sandbox-demo-run",
        "outcome": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "demonstration": True,
        "eligible_for_learning": False,
        "terminal_at": "2026-08-15T15:53:42Z",
        "graph_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "policy_digest": "sha256:" + "d" * 64,
        "receipt": {
            "sequence": 9,
            "head_digest": "sha256:" + "e" * 64,
            "trust": "local_hash_chain_only",
        },
        "nodes": [
            {
                "node_id": "sandbox_probe",
                "state": "SUCCEEDED",
                "gate_passed": True,
                "attempts": 1,
                "artifact_digests": ["sha256:" + "f" * 64],
            }
        ],
    }


@pytest.fixture
def store(tmp_path: Path) -> ExternalEvidenceStore:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
    return ExternalEvidenceStore(path, is_profile_active=lambda profile_id: profile_id == "alpha")


def test_external_evidence_is_profile_scoped_and_idempotent(store: ExternalEvidenceStore) -> None:
    payload = _evidence()

    assert store.record(payload) is True
    assert store.record(payload) is False
    assert store.get("alpha", payload["workspace_id"], payload["run_ref"]) == payload
    assert store.get("beta", payload["workspace_id"], payload["run_ref"]) is None


def test_changed_terminal_head_is_quarantined_not_rewritten(store: ExternalEvidenceStore) -> None:
    payload = _evidence()
    assert store.record(payload) is True
    changed = _evidence()
    changed["receipt"] = {**changed["receipt"], "head_digest": "sha256:" + "1" * 64}

    with pytest.raises(ExternalEvidenceConflictError, match="different receipt head"):
        store.record(changed)
    assert store.get("alpha", payload["workspace_id"], payload["run_ref"]) == payload


def test_demo_and_learning_refusal_are_persisted_as_typed_values(
    store: ExternalEvidenceStore,
) -> None:
    payload = _evidence()
    assert store.record(payload) is True

    stored = store.get("alpha", payload["workspace_id"], payload["run_ref"])
    assert stored is not None
    assert stored["demonstration"] is True
    assert stored["eligible_for_learning"] is False
    assert stored["nodes"][0]["gate_passed"] is True
    assert stored["nodes"][0]["attempts"] == 1


def test_profile_erasure_purges_external_evidence_and_durably_closes_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
    external = ExternalEvidenceStore(path, is_profile_active=lambda _: True)
    receipts = AgentExperienceStore(path, is_profile_active=lambda _: True)

    assert external.record(_evidence()) is True
    assert receipts.erase_profile("alpha") == 1
    assert external.get("alpha", _evidence()["workspace_id"], "nightly-1") is None
    with pytest.raises(ProfileAdmissionError, match="inactive or closing"):
        external.record(_evidence())


def test_invalid_timestamp_is_refused_before_sqlite_write(store: ExternalEvidenceStore) -> None:
    payload = _evidence()
    payload["terminal_at"] = "yesterday"
    with pytest.raises(ValueError, match="RFC3339"):
        store.record(payload)


def test_m041_refuses_a_preexisting_malformed_table(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE external_evidence_receipts (profile_id TEXT)")
        with pytest.raises(sqlite3.OperationalError, match="malformed"):
            m041.apply(conn)


def test_m041_repairs_missing_derived_index_without_rebuilding_evidence(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m041.apply(conn)
        conn.execute("DROP INDEX idx_external_evidence_profile_workspace")
        assert not m041.verify(conn)
        m041.repair(conn)
        assert m041.verify(conn)


def test_erasure_does_not_depend_on_m041_performance_indexes(tmp_path: Path) -> None:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
        conn.execute("DROP INDEX idx_external_evidence_profile_workspace")
    external = ExternalEvidenceStore(path, is_profile_active=lambda _: True)
    assert external.record(_evidence())
    assert AgentExperienceStore(path, is_profile_active=lambda _: True).erase_profile("alpha") == 1


def test_evidence_limits_protect_the_learning_writer(store: ExternalEvidenceStore) -> None:
    too_many_nodes = _evidence()
    too_many_nodes["nodes"] *= 257
    with pytest.raises(ExternalEvidenceValidationError, match="node count"):
        store.record(too_many_nodes)

    too_many_artifacts = _evidence()
    too_many_artifacts["nodes"][0]["artifact_digests"] *= 65
    with pytest.raises(ExternalEvidenceValidationError, match="artifact count"):
        store.record(too_many_artifacts)


def test_read_only_uri_handles_reserved_path_characters(tmp_path: Path) -> None:
    path = tmp_path / "learning#brain.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
    evidence = ExternalEvidenceStore(path, is_profile_active=lambda _: True)
    payload = _evidence()
    assert evidence.record(payload)
    assert evidence.get("alpha", payload["workspace_id"], payload["run_ref"]) == payload


def test_concurrent_external_observations_complete_without_deadlock(
    store: ExternalEvidenceStore,
) -> None:
    def record(number: int) -> bool:
        payload = _evidence()
        payload["run_ref"] = f"terminal-{number}"
        payload["run_id"] = f"run-{number}"
        return store.record(payload)

    # `ThreadPoolExecutor.map` blocks the calling thread until every submission
    # finishes, with no timeout of its own — a real deadlock in the store's
    # process lock would hang this test (and the whole suite) forever instead
    # of failing it.  Running the pool from a background thread lets us
    # `join()` with a bound and fail cleanly if it is ever exceeded, rather
    # than asserting an exact wall-clock duration for 32 in-process-serialised
    # writes, which is noisy on a machine shared with other test runs.
    outcomes: list[bool] = []

    def run_pool() -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes.extend(executor.map(record, range(32)))

    pool_thread = threading.Thread(target=run_pool, daemon=True)
    pool_thread.start()
    pool_thread.join(timeout=_DEADLOCK_GUARD_S)

    assert not pool_thread.is_alive(), (
        f"32 concurrent record() calls did not finish within "
        f"{_DEADLOCK_GUARD_S:.0f}s — the store's process lock likely deadlocked"
    )
    assert outcomes == [True] * 32


# -- Bounded Loops nodes that never ran (slm-bridge/v1, bounded-loops 0.7.6) ---------------
# gate_passed is tri-state in the contract: None means "no gate ran". A node that never ran
# (an approval node, a join, a node that failed before its gate) honestly reports
# attempts 0 with gate_passed None. Refusing it dropped the whole run.


def _node(node_id: str, attempts, gate_passed, state: str = "SUCCEEDED") -> dict:
    return {
        "node_id": node_id,
        "state": state,
        "gate_passed": gate_passed,
        "attempts": attempts,
        "artifact_digests": [],
    }


def test_never_attempted_node_without_gate_verdict_is_accepted(
    store: ExternalEvidenceStore,
) -> None:
    payload = _evidence()
    payload["nodes"] = [_node("approval", 0, None, state="PENDING")]
    assert store.record(payload) is True


@pytest.mark.parametrize("gate_passed", [True, False])
def test_zero_attempts_with_a_gate_verdict_is_refused(
    store: ExternalEvidenceStore, gate_passed: bool,
) -> None:
    payload = _evidence()
    payload["nodes"] = [_node("probe", 0, gate_passed)]
    with pytest.raises(ExternalEvidenceValidationError, match="node gate metadata"):
        store.record(payload)


@pytest.mark.parametrize("attempts", [-1, True, False, "1", 1.0, None])
def test_invalid_attempts_values_are_refused(store: ExternalEvidenceStore, attempts) -> None:
    payload = _evidence()
    payload["nodes"] = [_node("probe", attempts, None)]
    with pytest.raises(ExternalEvidenceValidationError, match="node gate metadata"):
        store.record(payload)


def test_bounded_loops_076_document_with_approval_node_is_accepted(
    store: ExternalEvidenceStore,
) -> None:
    payload = _evidence()
    payload.update({
        "run_ref": "release-gate-1",
        "run_id": "graph-run-076",
        "outcome": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "demonstration": False,
        "eligible_for_learning": False,
        "receipt": {
            "sequence": 14,
            "head_digest": "sha256:" + "9" * 64,
            "trust": "local_hash_chain_only",
        },
        "nodes": [
            _node("human_approval", 0, None, state="SKIPPED"),
            {
                "node_id": "build_and_test",
                "state": "SUCCEEDED",
                "gate_passed": True,
                "attempts": 1,
                "artifact_digests": ["sha256:" + "8" * 64],
            },
        ],
    })
    assert store.record(payload) is True
    assert store.get("alpha", payload["workspace_id"], "release-gate-1") == payload


@pytest.mark.parametrize("gate_passed", [1, 0, 1.0, "true"])
def test_a_gate_verdict_must_be_a_real_true_or_false(
    store: ExternalEvidenceStore, gate_passed,
) -> None:
    payload = _evidence()
    payload["nodes"] = [_node("probe", 1, gate_passed)]
    with pytest.raises(ExternalEvidenceValidationError, match="node gate metadata"):
        store.record(payload)


def _hold_learning_db(path: Path, seconds: float):
    """Another process's writer: holds learning.db for ``seconds``."""
    import threading

    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    done = threading.Event()

    def release() -> None:
        time.sleep(seconds)
        holder.execute("COMMIT")
        holder.close()
        done.set()

    threading.Thread(target=release, daemon=True).start()
    return done


def test_evidence_waits_out_another_writer_like_every_other_writer(
    store: ExternalEvidenceStore, tmp_path: Path,
) -> None:
    """4.2.8: the busy deadline was 0.9 s, so a 1.5 s write by another process
    failed the observation with LearningWriteBusyError. It is now the shared
    busy timeout (SLM_DB_BUSY_TIMEOUT_MS, default 10 s)."""
    released = _hold_learning_db(tmp_path / "learning.db", 1.5)
    try:
        assert store.record(_evidence()) is True
    finally:
        released.wait(5)


def test_evidence_still_says_busy_after_the_shared_deadline(
    store: ExternalEvidenceStore, tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.storage.agent_experience import LearningWriteBusyError

    monkeypatch.setenv("SLM_DB_BUSY_TIMEOUT_MS", "200")
    released = _hold_learning_db(tmp_path / "learning.db", 1.0)
    try:
        with pytest.raises(LearningWriteBusyError, match="deadline exceeded"):
            store.record(_evidence())
    finally:
        released.wait(5)
