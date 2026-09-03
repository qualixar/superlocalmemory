"""SLM bridge-v2 execution-learning storage contract."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.execution_learning import (
    ExecutionLearningStore,
    ExecutionLearningValidationError,
)
from superlocalmemory.storage.agent_experience import ProfileAdmissionError
from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040
from superlocalmemory.storage.migrations import M041_external_evidence_receipts as m041
from superlocalmemory.storage.migrations import M050_execution_learning_v2 as m050


def _receipt(*, run_id: str = "run-1", demonstration: bool = False,
             run_state: str = "SUCCEEDED", eligible: bool = True) -> dict:
    return {
        "contract": "bounded-loops.dev/slm-bridge/v2",
        "profile_id": "alpha",
        "workspace_id": "sha256:" + "a" * 64,
        "run_ref": f"ref-{run_id}",
        "run_id": run_id,
        "outcome": run_state,
        "run_state": run_state,
        "demonstration": demonstration,
        "eligible_for_learning": eligible,
        "terminal_at": "2026-09-03T00:00:00Z",
        "graph_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "policy_digest": "sha256:" + "d" * 64,
        "receipt": {"sequence": 1, "head_digest": "sha256:" + "e" * 64,
                    "trust": "local_hash_chain_only"},
        "nodes": [{"node_id": "gate", "state": "SUCCEEDED", "gate_passed": True,
                   "attempts": 1, "artifact_digests": []}],
        "learning_authority": {
            "scope": "execution_reliability_only",
            "reason_code": "verified_terminal_receipt",
            "verification_state": "reconciled",
            "gate_authority": "deterministic_gate",
            "trust_class": "local_hash_chain_only",
        },
        "route": {"runner": "hermes", "provider": "local", "model": "model-a",
                  "effort": "medium"},
        "usage": {"attempts": 1},
    }


def _verified_receipt(**kwargs: object):
    """Fixture-only evidence after the bridge boundary has verified it."""
    from superlocalmemory.storage.execution_learning import _seal_verified_execution_evidence

    return _seal_verified_execution_evidence(
        _receipt(**kwargs),
        producer_identity="test:bounded-loops-mcp",
        capability_digest="sha256:" + "1" * 64,
        terminal_listing_digest="sha256:" + "2" * 64,
    )


@pytest.fixture
def store(tmp_path: Path) -> ExecutionLearningStore:
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
        m050.apply(conn)
    return ExecutionLearningStore(path, is_profile_active=lambda profile_id: profile_id == "alpha")


def test_eligible_receipt_is_immutable_and_replay_is_idempotent(store: ExecutionLearningStore) -> None:
    payload = _receipt()
    # A schema-shaped document is not provenance.  The store must accept only
    # evidence sealed by the negotiated bridge boundary.
    with pytest.raises(ExecutionLearningValidationError, match="provenance"):
        store.ingest(payload)
    sealed = _verified_receipt()
    assert store.ingest(sealed) is True
    assert store.ingest(sealed) is False
    assert store.status("alpha")["receipts_total"] == 1


def test_immutable_receipt_persists_verified_bridge_provenance(
    store: ExecutionLearningStore,
) -> None:
    """An evidence receipt must preserve the local proof that admitted it."""
    assert store.ingest(_verified_receipt()) is True
    with sqlite3.connect(store._path) as conn:  # noqa: SLF001 - storage contract assertion
        row = conn.execute(
            "SELECT producer_identity, capability_digest, terminal_listing_digest "
            "FROM execution_learning_receipts"
        ).fetchone()
    assert row == (
        "test:bounded-loops-mcp",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
    )


def test_sealed_v2_evidence_is_rejected_if_modified_after_boundary_verification(
    store: ExecutionLearningStore,
) -> None:
    sealed = _verified_receipt()
    sealed.payload["run_ref"] = "another-run"

    with pytest.raises(ExecutionLearningValidationError, match="modified after verification"):
        store.ingest(sealed)
    assert store.status("alpha")["learning_events_total"] == 0


@pytest.mark.parametrize("kwargs", [
    {"demonstration": True},
    {"run_state": "CANCELLED"},
    {"eligible": False},
])
def test_forbidden_receipts_create_no_learning_rows(
    store: ExecutionLearningStore, kwargs: dict,
) -> None:
    with pytest.raises(ExecutionLearningValidationError):
        store.ingest(_verified_receipt(run_id=f"blocked-{len(kwargs)}", **kwargs))
    assert store.status("alpha")["learning_events_total"] == 0


@pytest.mark.parametrize("mutate", [
    lambda receipt: receipt["receipt"].update({"head_digest": "sha256:not-a-digest"}),
    lambda receipt: receipt["nodes"][0].update({"unbounded": "x" * 100_000}),
    lambda receipt: receipt.update({"route": {"runner": "x" * 129}}),
    lambda receipt: receipt.update({"usage": {"attempts": True}}),
])
def test_malformed_or_overlong_nested_values_are_never_persisted(
    store: ExecutionLearningStore, mutate,
) -> None:
    receipt = _receipt(run_id="nested-negative")
    mutate(receipt)
    from superlocalmemory.storage.execution_learning import _seal_verified_execution_evidence
    with pytest.raises(ExecutionLearningValidationError):
        store.ingest(_seal_verified_execution_evidence(
            receipt,
            producer_identity="test:bounded-loops-mcp",
            capability_digest="sha256:" + "1" * 64,
            terminal_listing_digest="sha256:" + "2" * 64,
        ))
    assert store.status("alpha") == {
        "receipts_total": 0, "learning_events_total": 0,
        "positive_events": 0, "negative_events": 0,
    }


def test_profile_erasure_removes_v2_receipts_and_derived_events(tmp_path: Path) -> None:
    from superlocalmemory.storage.agent_experience import AgentExperienceStore

    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
        m050.apply(conn)
    execution = ExecutionLearningStore(path, is_profile_active=lambda profile_id: profile_id == "alpha")
    assert execution.ingest(_verified_receipt()) is True

    erased = AgentExperienceStore(path, is_profile_active=lambda _: True).erase_profile(
        "alpha", close_profile=False
    )

    assert erased == 2
    assert execution.status("alpha") == {
        "receipts_total": 0, "learning_events_total": 0,
        "positive_events": 0, "negative_events": 0,
    }


def test_closed_profile_cannot_be_resurrected_by_a_delayed_v2_receipt(tmp_path: Path) -> None:
    """M050 rechecks the durable closure inside its write transaction."""
    from superlocalmemory.storage.agent_experience import AgentExperienceStore

    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        m040.apply(conn)
        m041.apply(conn)
        m050.apply(conn)
    execution = ExecutionLearningStore(path, is_profile_active=lambda profile_id: profile_id == "alpha")
    AgentExperienceStore(path, is_profile_active=lambda _: True).erase_profile(
        "alpha", close_profile=True
    )

    with pytest.raises(ProfileAdmissionError):
        execution.ingest(_verified_receipt())
    assert execution.status("alpha")["learning_events_total"] == 0


def test_negotiated_producer_v2_tool_payload_reaches_consumer_once(store: ExecutionLearningStore) -> None:
    """Exercise the public producer capability/tool names, not a private adapter."""
    from superlocalmemory.integrations.bounded_loops_mcp import observe_terminal_runs_v2

    evidence = _receipt()
    evidence.pop("profile_id")
    calls: list[str] = []

    async def call_tool(name: str, arguments: dict) -> dict:
        calls.append(name)
        if name == "bl_capabilities":
            return {"status": "ok", "capabilities": {"evidence_contracts": [{
                "id": "bounded-loops.dev/slm-bridge/v2",
                "tool": "bl_graph_execution_evidence",
                "operation": "observe_verified_terminal_run",
            }]}}
        if name == "bl_graph_terminal_runs":
            assert arguments == {"limit": 100}
            return {"status": "ok", "contract": "bounded-loops.dev/slm-bridge/v1", "runs": [{
                "run_ref": evidence["run_ref"],
                "run_id": evidence["run_id"],
                "run_state": evidence["run_state"],
                "terminal_at": evidence["terminal_at"],
            }]}
        if name == "bl_graph_execution_evidence":
            assert arguments == {"run_ref": evidence["run_ref"]}
            return {"status": "ok", "evidence": evidence}
        raise AssertionError(name)

    payloads = asyncio.run(observe_terminal_runs_v2(
        call_tool, profile_id="alpha", producer_identity="test:bounded-loops-mcp"
    ))

    assert calls == ["bl_capabilities", "bl_graph_terminal_runs", "bl_graph_execution_evidence"]
    assert len(payloads) == 1
    assert store.ingest(payloads[0]) is True
    assert store.ingest(payloads[0]) is False


def test_v2_evidence_with_listing_mismatch_is_never_sealed_or_persisted(
    store: ExecutionLearningStore,
) -> None:
    from superlocalmemory.integrations.bounded_loops_mcp import (
        BridgeUnavailable,
        observe_terminal_runs_v2,
    )

    evidence = _receipt()
    evidence.pop("profile_id")

    async def call_tool(name: str, arguments: dict) -> dict:
        if name == "bl_capabilities":
            return {"status": "ok", "capabilities": {"evidence_contracts": [{
                "id": "bounded-loops.dev/slm-bridge/v2",
                "tool": "bl_graph_execution_evidence",
                "operation": "observe_verified_terminal_run",
            }]}}
        if name == "bl_graph_terminal_runs":
            return {"status": "ok", "contract": "bounded-loops.dev/slm-bridge/v1", "runs": [{
                "run_ref": evidence["run_ref"], "run_id": "different-run",
                "run_state": evidence["run_state"], "terminal_at": evidence["terminal_at"],
            }]}
        if name == "bl_graph_execution_evidence":
            return {"status": "ok", "evidence": evidence}
        raise AssertionError(name)

    with pytest.raises(BridgeUnavailable, match="does not match terminal listing"):
        asyncio.run(observe_terminal_runs_v2(
            call_tool, profile_id="alpha", producer_identity="test:bounded-loops-mcp"
        ))
    assert store.status("alpha")["learning_events_total"] == 0
