"""Typed storage for versioned, observation-only MCP evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.storage.agent_experience import (
    LearningWriteBusyError,
    ProfileAdmissionError,
)

_CONTRACT = "bounded-loops.dev/slm-bridge/v1"
_SHA256 = re.compile(r"\Asha256:[a-f0-9]{64}\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_RUN_STATES = frozenset({"SUCCEEDED", "FAILED", "HALTED", "CANCELLED", "EXPIRED"})
_OUTCOMES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
_INSERT = (
    "INSERT INTO external_evidence_receipts (profile_id, contract_id, workspace_id, "
    "run_ref, run_id, outcome, run_state, demonstration, "
    "eligible_for_learning, terminal_at, graph_digest, plan_digest, "
    "policy_digest, receipt_sequence, receipt_head_digest, receipt_trust, "
    "nodes_json, artifact_digests_json, payload_sha256, observed_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(profile_id, contract_id, workspace_id, run_ref) DO NOTHING"
)


class ExternalEvidenceConflictError(ValueError):
    """A stable external run address produced a different terminal receipt head."""


class ExternalEvidenceValidationError(ValueError):
    """An external evidence document does not satisfy the public v1 contract."""


class ExternalEvidenceStore:
    """Persist evidence without entering SLM's memory/recall lock domain."""

    def __init__(self, path: str | Path, *, is_profile_active: Callable[[str], bool]) -> None:
        self._path = Path(path)
        self._is_profile_active = is_profile_active

    def record(self, payload: dict[str, Any]) -> bool:
        _validate(payload)
        profile_id = payload["profile_id"]
        if not self._is_profile_active(profile_id):
            raise ExternalEvidenceValidationError("profile is inactive or closing for erasure")
        digest = _payload_digest(payload)
        deadline = time.monotonic() + 0.90
        while True:
            conn: sqlite3.Connection | None = None
            try:
                conn = sqlite3.connect(str(self._path), timeout=0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=0")
                conn.execute("BEGIN IMMEDIATE")
                _assert_profile_open(conn, profile_id)
                row = _row(payload, digest)
                cursor = conn.execute(_INSERT, row)
                if cursor.rowcount:
                    conn.execute("COMMIT")
                    return True
                existing = _get_conn(
                    conn,
                    profile_id,
                    payload["contract"],
                    payload["workspace_id"],
                    payload["run_ref"],
                )
                conn.execute("ROLLBACK")
                if existing == payload:
                    return False
                raise ExternalEvidenceConflictError(
                    "external run address has a different receipt head"
                )
            except sqlite3.OperationalError as exc:
                if conn is not None and conn.in_transaction:
                    conn.execute("ROLLBACK")
                busy = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                if not busy or time.monotonic() >= deadline:
                    if busy:
                        raise LearningWriteBusyError(
                            "external evidence write deadline exceeded"
                        ) from exc
                    raise
                time.sleep(0.02)
            finally:
                if conn is not None:
                    conn.close()

    def get(self, profile_id: str, workspace_id: str, run_ref: str) -> dict[str, Any] | None:
        conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=0.5)
        conn.row_factory = sqlite3.Row
        try:
            return _get_conn(conn, profile_id, _CONTRACT, workspace_id, run_ref)
        finally:
            conn.close()


def get_profile_external_evidence_summary(path: str | Path, profile_id: str) -> dict[str, Any]:
    """Return indexed Living Brain totals without opening SLM's memory database."""
    empty = {
        "is_real": False,
        "availability": "unavailable",
        "total": 0,
        "by_run_state": {},
        "demonstrations": 0,
    }
    target = Path(path)
    if not target.exists():
        return empty
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=0.5)
        total = conn.execute(
            "SELECT COUNT(*) FROM external_evidence_receipts WHERE profile_id=?", (profile_id,)
        ).fetchone()[0]
        demo = conn.execute(
            "SELECT COUNT(*) FROM external_evidence_receipts "
            "WHERE profile_id=? AND demonstration=1",
            (profile_id,),
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT run_state, COUNT(*) FROM external_evidence_receipts "
            "WHERE profile_id=? GROUP BY run_state",
            (profile_id,),
        ).fetchall()
    except sqlite3.Error:
        return empty
    finally:
        if conn is not None:
            conn.close()
    return {
        "is_real": True,
        "availability": "available",
        "total": int(total),
        "by_run_state": {str(k): int(v) for k, v in rows},
        "demonstrations": int(demo),
        "control_plane": "observation_only",
    }


def _validate(payload: dict[str, Any]) -> None:
    required = {
        "contract",
        "profile_id",
        "workspace_id",
        "run_ref",
        "run_id",
        "outcome",
        "run_state",
        "demonstration",
        "eligible_for_learning",
        "terminal_at",
        "graph_digest",
        "plan_digest",
        "policy_digest",
        "receipt",
        "nodes",
    }
    if set(payload) != required:
        raise ExternalEvidenceValidationError("external evidence fields do not match v1")
    if payload["contract"] != _CONTRACT:
        raise ExternalEvidenceValidationError("unsupported external evidence contract")
    for name in ("profile_id", "run_ref", "run_id"):
        if not isinstance(payload[name], str) or not _IDENTIFIER.match(payload[name]):
            raise ExternalEvidenceValidationError(f"{name} must be a safe identifier")
    for name in ("workspace_id", "graph_digest", "plan_digest", "policy_digest"):
        if not isinstance(payload[name], str) or not _SHA256.match(payload[name]):
            raise ExternalEvidenceValidationError(f"{name} must be a sha256 digest")
    if payload["outcome"] not in _OUTCOMES or payload["run_state"] not in _RUN_STATES:
        raise ExternalEvidenceValidationError("outcome or run_state is unsupported")
    if payload["run_state"] == "SUCCEEDED" and payload["outcome"] != "SUCCEEDED":
        raise ExternalEvidenceValidationError("SUCCEEDED run_state must keep its outcome")
    if not isinstance(payload["terminal_at"], str):
        raise ExternalEvidenceValidationError("terminal_at must be an RFC3339 timestamp")
    try:
        datetime.fromisoformat(payload["terminal_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExternalEvidenceValidationError("terminal_at must be an RFC3339 timestamp") from exc
    if (
        not isinstance(payload["demonstration"], bool)
        or payload["eligible_for_learning"] is not False
    ):
        raise ExternalEvidenceValidationError("v1 evidence is observation-only")
    receipt = payload["receipt"]
    if not isinstance(receipt, dict) or set(receipt) != {
        "sequence",
        "head_digest",
        "trust",
    }:
        raise ExternalEvidenceValidationError("receipt shape is invalid")
    if (
        not isinstance(receipt["sequence"], int)
        or receipt["sequence"] < 1
        or receipt["trust"] != "local_hash_chain_only"
    ):
        raise ExternalEvidenceValidationError("receipt metadata is invalid")
    if not isinstance(receipt["head_digest"], str) or not _SHA256.match(receipt["head_digest"]):
        raise ExternalEvidenceValidationError("receipt head digest is invalid")
    if not isinstance(payload["nodes"], list):
        raise ExternalEvidenceValidationError("nodes must be a list")
    for node in payload["nodes"]:
        if not isinstance(node, dict) or set(node) != {
            "node_id",
            "state",
            "gate_passed",
            "attempts",
            "artifact_digests",
        }:
            raise ExternalEvidenceValidationError("node shape is invalid")
        valid_node = _IDENTIFIER.match(str(node["node_id"])) and _IDENTIFIER.match(
            str(node["state"])
        )
        if not valid_node:
            raise ExternalEvidenceValidationError("node identifiers are invalid")
        if (
            node["gate_passed"] not in (True, False, None)
            or not isinstance(node["attempts"], int)
            or node["attempts"] < 1
        ):
            raise ExternalEvidenceValidationError("node gate metadata is invalid")
        if not isinstance(node["artifact_digests"], list) or any(
            not isinstance(item, str) or not _SHA256.match(item)
            for item in node["artifact_digests"]
        ):
            raise ExternalEvidenceValidationError("node artifact digests are invalid")


def _row(payload: dict[str, Any], digest: str) -> tuple[Any, ...]:
    artifacts = sorted({item for node in payload["nodes"] for item in node["artifact_digests"]})
    receipt = payload["receipt"]
    return (
        payload["profile_id"],
        payload["contract"],
        payload["workspace_id"],
        payload["run_ref"],
        payload["run_id"],
        payload["outcome"],
        payload["run_state"],
        int(payload["demonstration"]),
        0,
        payload["terminal_at"],
        payload["graph_digest"],
        payload["plan_digest"],
        payload["policy_digest"],
        receipt["sequence"],
        receipt["head_digest"],
        receipt["trust"],
        _json(payload["nodes"]),
        _json(artifacts),
        digest,
        datetime.now(timezone.utc).isoformat(),
    )


def _assert_profile_open(conn: sqlite3.Connection, profile_id: str) -> None:
    """Use M040's durable tombstone inside this writer transaction."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='agent_receipt_profile_closures'"
    ).fetchone()
    if (
        table is not None
        and conn.execute(
            "SELECT 1 FROM agent_receipt_profile_closures WHERE profile_id=?", (profile_id,)
        ).fetchone()
        is not None
    ):
        raise ProfileAdmissionError("profile is inactive or closing for erasure")


def _get_conn(
    conn: sqlite3.Connection,
    profile_id: str,
    contract_id: str,
    workspace_id: str,
    run_ref: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM external_evidence_receipts WHERE profile_id=? AND contract_id=? "
        "AND workspace_id=? AND run_ref=?",
        (profile_id, contract_id, workspace_id, run_ref),
    ).fetchone()
    if row is None:
        return None
    return {
        "contract": row["contract_id"],
        "profile_id": row["profile_id"],
        "workspace_id": row["workspace_id"],
        "run_ref": row["run_ref"],
        "run_id": row["run_id"],
        "outcome": row["outcome"],
        "run_state": row["run_state"],
        "demonstration": bool(row["demonstration"]),
        "eligible_for_learning": bool(row["eligible_for_learning"]),
        "terminal_at": row["terminal_at"],
        "graph_digest": row["graph_digest"],
        "plan_digest": row["plan_digest"],
        "policy_digest": row["policy_digest"],
        "receipt": {
            "sequence": row["receipt_sequence"],
            "head_digest": row["receipt_head_digest"],
            "trust": row["receipt_trust"],
        },
        "nodes": json.loads(row["nodes_json"]),
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
