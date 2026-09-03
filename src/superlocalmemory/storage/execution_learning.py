"""Bridge-v2 immutable receipt storage and strictly bounded execution learning.

This module deliberately has no dependency on recall, semantic facts, or user
preferences.  Bounded Loops receipts can only update the rebuildable execution
event projection after the consumer validates every eligibility condition.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.storage.agent_experience import (
    _ProfileAdmissionGate,
    _PROCESS_LOCKS_GUARD,
    _PROFILE_GATES,
    ProfileAdmissionError,
)

_CONTRACT = "bounded-loops.dev/slm-bridge/v2"
_SHA256_PREFIX = "sha256:"
_ALLOWED_STATES = frozenset({"SUCCEEDED", "FAILED"})
_MAX_PAYLOAD_BYTES = 32 * 1024
_MAX_NODES = 256
_MAX_ARTIFACTS_PER_NODE = 64
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExecutionLearningValidationError(ValueError):
    """A v2 receipt is not eligible for execution-learning ingestion."""


# This is deliberately an in-process capability, not a serializable field in
# the producer payload.  An MCP producer can send JSON; it cannot mint this
# boundary witness.  It prevents an accidental future caller from turning a
# schema-valid JSON document directly into learned execution behaviour.
_BRIDGE_SEAL = object()


@dataclass(frozen=True)
class VerifiedExecutionEvidence:
    """A v2 payload bound to one negotiated, locally observed producer session.

    ``payload`` intentionally remains available for the immutable receipt
    record.  The private seal is checked by :class:`ExecutionLearningStore`
    and never crosses MCP or disk boundaries.
    """

    payload: dict[str, Any]
    producer_identity: str
    capability_digest: str
    terminal_listing_digest: str
    payload_sha256: str
    _seal: object


def _seal_verified_execution_evidence(
    payload: dict[str, Any],
    *,
    producer_identity: str,
    capability_digest: str,
    terminal_listing_digest: str,
) -> VerifiedExecutionEvidence:
    """Create the sole accepted execution-learning input at the MCP boundary."""
    _validate(payload)
    for label, value in {
        "producer identity": producer_identity,
        "capability digest": capability_digest,
        "terminal listing digest": terminal_listing_digest,
    }.items():
        if not isinstance(value, str) or not value:
            raise ExecutionLearningValidationError(f"v2 receipt has no verified {label}")
    sealed_payload = deepcopy(payload)
    return VerifiedExecutionEvidence(
        payload=sealed_payload,
        producer_identity=producer_identity,
        capability_digest=capability_digest,
        terminal_listing_digest=terminal_listing_digest,
        payload_sha256=_digest(sealed_payload),
        _seal=_BRIDGE_SEAL,
    )


class ExecutionLearningStore:
    """Atomically persist v2 receipts before their derived event projection."""

    def __init__(self, path: str | Path, *, is_profile_active: Callable[[str], bool]) -> None:
        self._path = Path(path).resolve()
        self._is_profile_active = is_profile_active
        with _PROCESS_LOCKS_GUARD:
            self._gate = _PROFILE_GATES.setdefault(str(self._path), _ProfileAdmissionGate())

    def ingest(self, evidence: VerifiedExecutionEvidence) -> bool:
        if (
            not isinstance(evidence, VerifiedExecutionEvidence)
            or evidence._seal is not _BRIDGE_SEAL
        ):
            raise ExecutionLearningValidationError(
                "v2 execution learning requires verified bridge provenance"
            )
        payload = evidence.payload
        if evidence.payload_sha256 != _digest(payload):
            raise ExecutionLearningValidationError(
                "v2 execution-learning provenance was modified after verification"
            )
        _validate(payload)
        profile_id = payload["profile_id"]
        self._gate.admit(profile_id, self._is_profile_active)
        digest = _digest(payload)
        receipt = payload["receipt"]
        route_key = _route_key(payload["route"])
        signal = 1 if payload["run_state"] == "SUCCEEDED" else -1
        now = datetime.now(timezone.utc).isoformat()
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self._path), timeout=1.0, isolation_level=None)
            conn.execute("BEGIN IMMEDIATE")
            # The same process-wide admission gate drains in-flight writes
            # before erasure. This durable closure check also prevents a late
            # transaction from resurrecting a profile after a completed close.
            closed = conn.execute(
                "SELECT 1 FROM agent_receipt_profile_closures WHERE profile_id=?", (profile_id,)
            ).fetchone()
            if closed is not None:
                raise ProfileAdmissionError("profile is inactive or closing for erasure")
            existing = conn.execute(
                "SELECT payload_sha256 FROM execution_learning_receipts "
                "WHERE profile_id=? AND workspace_id=? AND run_ref=?",
                (profile_id, payload["workspace_id"], payload["run_ref"]),
            ).fetchone()
            if existing is not None:
                if existing[0] != digest:
                    raise ExecutionLearningValidationError(
                        "external run address has a different receipt head"
                    )
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                "INSERT INTO execution_learning_receipts "
                "(profile_id, workspace_id, run_ref, run_id, receipt_head_digest, "
                "payload_json, payload_sha256, producer_identity, capability_digest, "
                "terminal_listing_digest, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile_id, payload["workspace_id"], payload["run_ref"], payload["run_id"],
                 receipt["head_digest"], _json(payload), digest, evidence.producer_identity,
                 evidence.capability_digest, evidence.terminal_listing_digest, now),
            )
            conn.execute(
                "INSERT INTO execution_learning_events "
                "(profile_id, workspace_id, run_ref, receipt_head_digest, route_key, signal, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (profile_id, payload["workspace_id"], payload["run_ref"],
                 receipt["head_digest"], route_key, signal, now),
            )
            conn.execute("COMMIT")
            return True
        except Exception:
            if conn is not None and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            if conn is not None:
                conn.close()
            self._gate.release(profile_id)

    def status(self, profile_id: str) -> dict[str, int]:
        conn = sqlite3.connect(str(self._path), timeout=0.5)
        try:
            receipts = conn.execute(
                "SELECT COUNT(*) FROM execution_learning_receipts WHERE profile_id=?", (profile_id,)
            ).fetchone()[0]
            totals = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(signal = 1), 0), COALESCE(SUM(signal = -1), 0) "
                "FROM execution_learning_events WHERE profile_id=?", (profile_id,),
            ).fetchone()
            return {"receipts_total": int(receipts), "learning_events_total": int(totals[0]),
                    "positive_events": int(totals[1]), "negative_events": int(totals[2])}
        finally:
            conn.close()


def _validate(payload: dict[str, Any]) -> None:
    required = {"contract", "profile_id", "workspace_id", "run_ref", "run_id", "outcome",
                "run_state", "demonstration", "eligible_for_learning", "terminal_at",
                "graph_digest", "plan_digest", "policy_digest", "receipt", "nodes",
                "learning_authority", "route", "usage"}
    if set(payload) != required or payload.get("contract") != _CONTRACT:
        raise ExecutionLearningValidationError("v2 receipt does not match the bounded contract")
    if len(_json(payload).encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ExecutionLearningValidationError("v2 receipt exceeds the bounded payload size")
    for field in ("profile_id", "workspace_id", "run_ref", "run_id"):
        if not isinstance(payload[field], str) or not _SAFE_ID.fullmatch(payload[field]):
            raise ExecutionLearningValidationError(f"v2 receipt has an invalid {field}")
    for field in ("workspace_id", "graph_digest", "plan_digest", "policy_digest"):
        if not isinstance(payload[field], str) or not _DIGEST.fullmatch(payload[field]):
            raise ExecutionLearningValidationError(f"v2 receipt has an invalid {field}")
    if not isinstance(payload["terminal_at"], str) or len(payload["terminal_at"]) > 64:
        raise ExecutionLearningValidationError("v2 receipt has an invalid terminal timestamp")
    if payload["demonstration"] is not False or payload["eligible_for_learning"] is not True:
        raise ExecutionLearningValidationError("receipt is not eligible for execution learning")
    if payload["run_state"] not in _ALLOWED_STATES or payload["outcome"] != payload["run_state"]:
        raise ExecutionLearningValidationError("only terminal succeeded or executed-gate failure is learnable")
    receipt, authority = payload["receipt"], payload["learning_authority"]
    if not isinstance(receipt, dict) or not isinstance(authority, dict):
        raise ExecutionLearningValidationError("receipt authority is malformed")
    if set(receipt) != {"sequence", "head_digest", "trust"}:
        raise ExecutionLearningValidationError("receipt chain metadata is malformed")
    if (
        isinstance(receipt["sequence"], bool)
        or not isinstance(receipt["sequence"], int)
        or receipt["sequence"] < 1
        or receipt["trust"] != "local_hash_chain_only"
        or not isinstance(receipt.get("head_digest"), str)
        or not _DIGEST.fullmatch(receipt["head_digest"])
    ):
        raise ExecutionLearningValidationError("receipt chain head is invalid")
    if authority != {"scope": "execution_reliability_only", "reason_code": "verified_terminal_receipt",
                     "verification_state": "reconciled", "gate_authority": "deterministic_gate",
                     "trust_class": "local_hash_chain_only"}:
        raise ExecutionLearningValidationError("receipt lacks bounded deterministic learning authority")
    _validate_nodes(payload["nodes"])
    _validate_route(payload["route"])
    _validate_usage(payload["usage"])


def _validate_nodes(nodes: Any) -> None:
    if not isinstance(nodes, list) or not nodes or len(nodes) > _MAX_NODES:
        raise ExecutionLearningValidationError("receipt nodes are malformed")
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {
            "node_id", "state", "gate_passed", "attempts", "artifact_digests"
        }:
            raise ExecutionLearningValidationError("receipt node schema is malformed")
        if (
            not isinstance(node["node_id"], str)
            or not _SAFE_ID.fullmatch(node["node_id"])
            or not isinstance(node["state"], str)
            or not _SAFE_ID.fullmatch(node["state"])
            or node["gate_passed"] not in (True, False, None)
            or isinstance(node["attempts"], bool)
            or not isinstance(node["attempts"], int)
            or not 0 <= node["attempts"] <= 10_000
            or not isinstance(node["artifact_digests"], list)
            or len(node["artifact_digests"]) > _MAX_ARTIFACTS_PER_NODE
            or not all(isinstance(item, str) and _DIGEST.fullmatch(item)
                       for item in node["artifact_digests"])
        ):
            raise ExecutionLearningValidationError("receipt node values are malformed")


def _validate_route(route: Any) -> None:
    if not isinstance(route, dict) or set(route) - {"runner", "provider", "model", "effort"}:
        raise ExecutionLearningValidationError("route or usage is malformed")
    if not all(isinstance(value, str) and _SAFE_ID.fullmatch(value) for value in route.values()):
        raise ExecutionLearningValidationError("route or usage is malformed")


def _validate_usage(usage: Any) -> None:
    if (
        not isinstance(usage, dict)
        or set(usage) != {"attempts"}
        or isinstance(usage["attempts"], bool)
        or not isinstance(usage["attempts"], int)
        or not 0 <= usage["attempts"] <= 1_000_000
    ):
        raise ExecutionLearningValidationError("route or usage is malformed")


def _route_key(route: dict[str, Any]) -> str:
    return "|".join(str(route.get(key, "")) for key in ("runner", "provider", "model", "effort"))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
