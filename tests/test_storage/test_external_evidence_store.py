"""Persistence contract for external MCP evidence in SLM's learning plane."""

from __future__ import annotations

import sqlite3
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

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(record, range(32)))
    assert outcomes == [True] * 32
    assert time.monotonic() - started < 2.0
