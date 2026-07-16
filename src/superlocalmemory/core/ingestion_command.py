# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Canonical durable-ingestion command and operation state machine.

The command deliberately depends on injected queryable/materialization
functions.  This keeps the durable contract testable while legacy write paths
are migrated through an expand-migrate-contract rollout.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from superlocalmemory.storage.database import DatabaseManager

_MATERIALIZATION_LOCKS = tuple(threading.RLock() for _ in range(64))


def _materialization_lock(operation_id: str) -> threading.RLock:
    bucket = int(hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:8], 16)
    return _MATERIALIZATION_LOCKS[bucket % len(_MATERIALIZATION_LOCKS)]


class IngestionState(str, Enum):
    RAW = "raw"
    QUERYABLE = "queryable"
    ENRICHING = "enriching"
    COMPLETE = "complete"
    FAILED = "failed"


class IdempotencyConflict(ValueError):
    """The same idempotency key was reused for different immutable evidence."""


class InvalidStateTransition(RuntimeError):
    """An ingestion operation attempted an illegal or stale transition."""


class OperationInProgress(RuntimeError):
    """Another live lease owner is materializing this operation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    content: str
    profile_id: str
    source_type: str
    idempotency_key: str
    metadata: dict[str, Any] = field(default_factory=dict)
    scope: str = "personal"
    shared_with: tuple[str, ...] = ()
    trusted_actor_id: str = ""
    session_id: str = ""
    session_date: str = ""
    speaker: str = ""
    role: str = "user"

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("content is required")
        for name in ("profile_id", "source_type", "idempotency_key"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.scope not in {"personal", "project", "shared", "global"}:
            raise ValueError(f"unsupported scope: {self.scope}")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "shared_with", tuple(self.shared_with))

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IngestionOperation:
    operation_id: str
    profile_id: str
    source_type: str
    idempotency_key: str
    source_hash: str
    raw_content: str
    metadata: dict[str, Any]
    scope: str
    shared_with: tuple[str, ...]
    trusted_actor_id: str
    session_id: str
    session_date: str
    speaker: str
    role: str
    state: IngestionState
    queryable_fact_ids: tuple[str, ...]
    final_fact_ids: tuple[str, ...]
    derivation_version: str
    derivation_state: dict[str, bool]
    lease_owner: str
    lease_expires_at: float
    next_retry_at: float
    attempt_count: int
    last_error: str
    created_at: str
    updated_at: str

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return self.final_fact_ids or self.queryable_fact_ids


class IngestionOperationRepository:
    """Persistence and compare-and-swap transitions for M018 operations."""

    _ALLOWED: dict[IngestionState, frozenset[IngestionState]] = {
        IngestionState.RAW: frozenset(
            {IngestionState.QUERYABLE, IngestionState.FAILED}
        ),
        IngestionState.QUERYABLE: frozenset(
            {IngestionState.ENRICHING, IngestionState.FAILED}
        ),
        IngestionState.ENRICHING: frozenset(
            {IngestionState.COMPLETE, IngestionState.FAILED}
        ),
        IngestionState.FAILED: frozenset({IngestionState.ENRICHING}),
        IngestionState.COMPLETE: frozenset(),
    }

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def _from_row(row: Any) -> IngestionOperation:
        data = dict(row)
        return IngestionOperation(
            operation_id=data["operation_id"],
            profile_id=data["profile_id"],
            source_type=data["source_type"],
            idempotency_key=data["idempotency_key"],
            source_hash=data["source_hash"],
            raw_content=data["raw_content"],
            metadata=json.loads(data["raw_metadata_json"] or "{}"),
            scope=data["scope"],
            shared_with=tuple(json.loads(data["shared_with_json"] or "[]")),
            trusted_actor_id=data["trusted_actor_id"],
            session_id=data["session_id"],
            session_date=data["session_date"],
            speaker=data["speaker"],
            role=data["role"],
            state=IngestionState(data["state"]),
            queryable_fact_ids=tuple(
                json.loads(data["queryable_fact_ids_json"] or "[]")
            ),
            final_fact_ids=tuple(json.loads(data["final_fact_ids_json"] or "[]")),
            derivation_version=data["derivation_version"],
            derivation_state=json.loads(data.get("derivation_state_json") or "{}"),
            lease_owner=data.get("lease_owner") or "",
            lease_expires_at=float(data.get("lease_expires_at") or 0),
            next_retry_at=float(data.get("next_retry_at") or 0),
            attempt_count=int(data["attempt_count"]),
            last_error=data["last_error"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def _find_request(self, request: IngestionRequest) -> IngestionOperation | None:
        rows = self.db.execute(
            "SELECT * FROM ingestion_operations "
            "WHERE profile_id=? AND source_type=? AND idempotency_key=?",
            (request.profile_id, request.source_type, request.idempotency_key),
        )
        return self._from_row(rows[0]) if rows else None

    @staticmethod
    def _assert_same_request(
        existing: IngestionOperation, request: IngestionRequest
    ) -> None:
        comparable = (
            existing.source_hash == request.source_hash,
            existing.raw_content == request.content,
            existing.metadata == request.metadata,
            existing.scope == request.scope,
            existing.shared_with == request.shared_with,
            existing.trusted_actor_id == request.trusted_actor_id,
            existing.session_id == request.session_id,
            existing.session_date == request.session_date,
            existing.speaker == request.speaker,
            existing.role == request.role,
        )
        if not all(comparable):
            raise IdempotencyConflict(
                "idempotency key already belongs to different immutable evidence"
            )

    def create(self, request: IngestionRequest) -> IngestionOperation:
        operation, _created = self.create_with_status(request)
        return operation

    def create_with_status(
        self, request: IngestionRequest,
    ) -> tuple[IngestionOperation, bool]:
        existing = self._find_request(request)
        if existing is not None:
            self._assert_same_request(existing, request)
            return existing, False

        operation_id = uuid.uuid4().hex
        try:
            self.db.execute(
                "INSERT INTO ingestion_operations "
                "(operation_id, profile_id, source_type, idempotency_key, "
                "source_hash, raw_content, raw_metadata_json, scope, "
                "shared_with_json, trusted_actor_id, session_id, session_date, "
                "speaker, role) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    operation_id,
                    request.profile_id,
                    request.source_type,
                    request.idempotency_key,
                    request.source_hash,
                    request.content,
                    _canonical_json(request.metadata),
                    request.scope,
                    _canonical_json(list(request.shared_with)),
                    request.trusted_actor_id,
                    request.session_id,
                    request.session_date,
                    request.speaker,
                    request.role,
                ),
            )
        except sqlite3.IntegrityError:
            concurrent = self._find_request(request)
            if concurrent is None:
                raise
            self._assert_same_request(concurrent, request)
            return concurrent, False
        return self.get(operation_id), True

    def get(self, operation_id: str) -> IngestionOperation:
        rows = self.db.execute(
            "SELECT * FROM ingestion_operations WHERE operation_id=?",
            (operation_id,),
        )
        if not rows:
            raise KeyError(operation_id)
        return self._from_row(rows[0])

    def list_operations(self) -> list[IngestionOperation]:
        return [
            self._from_row(row)
            for row in self.db.execute(
                "SELECT * FROM ingestion_operations ORDER BY created_at, operation_id"
            )
        ]

    def list_materializable(
        self,
        *,
        limit: int = 50,
        min_queryable_age_seconds: float = 0.0,
    ) -> list[IngestionOperation]:
        """Return durable work in FIFO order for the background materializer.

        A short age gate can protect a freshly admitted receipt from racing
        the user's immediate recall on single-queue local model runtimes such
        as Ollama.  Failed retries and expired leases remain immediately due.
        """
        now = time.time()
        grace_modifier = f"-{max(0.0, float(min_queryable_age_seconds))} seconds"
        return [
            self._from_row(row)
            for row in self.db.execute(
                "SELECT * FROM ingestion_operations "
                "WHERE (state='queryable' AND created_at <= "
                "strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)) "
                "OR (state='failed' AND next_retry_at <= ?) "
                "OR (state='enriching' AND lease_expires_at <= ?) "
                "ORDER BY created_at, rowid LIMIT ?",
                (grace_modifier, now, now, max(1, int(limit))),
            )
        ]

    def claim_enriching(
        self,
        operation_id: str,
        *,
        owner: str,
        lease_seconds: float,
    ) -> IngestionOperation:
        """Claim queryable/failed work or reclaim an expired enrichment."""
        now = time.time()
        rows = self.db.execute(
            "UPDATE ingestion_operations SET state='enriching', "
            "lease_owner=?, lease_expires_at=?, "
            "next_retry_at=0, attempt_count=attempt_count + "
            "CASE WHEN state='enriching' AND lease_owner=? THEN 0 ELSE 1 END, "
            "last_error='', "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE operation_id=? AND ("
            "state IN ('queryable', 'failed') OR "
            "(state='enriching' AND (lease_owner=? OR lease_expires_at <= ?))"
            ") RETURNING *",
            (owner, now + lease_seconds, owner, operation_id, owner, now),
        )
        if rows:
            return self._from_row(rows[0])
        current = self.get(operation_id)
        if current.state is IngestionState.COMPLETE:
            return current
        raise OperationInProgress(
            f"operation {operation_id} is leased by another worker"
        )

    def transition(
        self,
        operation_id: str,
        *,
        expected: IngestionState,
        target: IngestionState,
        queryable_fact_ids: tuple[str, ...] | None = None,
        final_fact_ids: tuple[str, ...] | None = None,
        derivation_version: str | None = None,
        derivation_state: dict[str, bool] | None = None,
        last_error: str | None = None,
    ) -> IngestionOperation:
        if target not in self._ALLOWED[expected]:
            raise InvalidStateTransition(f"{expected.value} -> {target.value}")
        rows = self.db.execute(
            "UPDATE ingestion_operations SET state=?, "
            "queryable_fact_ids_json=COALESCE(?, queryable_fact_ids_json), "
            "final_fact_ids_json=COALESCE(?, final_fact_ids_json), "
            "derivation_version=COALESCE(?, derivation_version), "
            "derivation_state_json=COALESCE(?, derivation_state_json), "
            "attempt_count=attempt_count + ?, "
            "last_error=?, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE operation_id=? AND state=? RETURNING *",
            (
                target.value,
                _canonical_json(list(queryable_fact_ids))
                if queryable_fact_ids is not None
                else None,
                _canonical_json(list(final_fact_ids))
                if final_fact_ids is not None
                else None,
                derivation_version,
                _canonical_json(derivation_state)
                if derivation_state is not None
                else None,
                1 if target is IngestionState.ENRICHING else 0,
                last_error or "",
                operation_id,
                expected.value,
            ),
        )
        if not rows:
            actual = self.get(operation_id).state
            raise InvalidStateTransition(
                f"expected {expected.value}, found {actual.value}"
            )
        return self._from_row(rows[0])

    def checkpoint_enriching(
        self,
        operation_id: str,
        *,
        final_fact_ids: tuple[str, ...],
        derivation_version: str,
        derivation_state: dict[str, bool],
        lease_owner: str,
        lease_seconds: float,
    ) -> IngestionOperation:
        """Durably checkpoint relational derivation before external indexes."""
        rows = self.db.execute(
            "UPDATE ingestion_operations SET "
            "final_fact_ids_json=?, derivation_version=?, "
            "derivation_state_json=?, lease_expires_at=?, last_error='', "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE operation_id=? AND state='enriching' AND lease_owner=? "
            "RETURNING *",
            (
                _canonical_json(list(final_fact_ids)),
                derivation_version,
                _canonical_json(derivation_state),
                time.time() + max(1.0, float(lease_seconds)),
                operation_id,
                lease_owner,
            ),
        )
        if not rows:
            raise InvalidStateTransition("enriching checkpoint lost ownership")
        operation = self._from_row(rows[0])
        from superlocalmemory.core.derivation_lineage import capture_operation_lineage

        capture_operation_lineage(
            self.db,
            operation_id=operation.operation_id,
            profile_id=operation.profile_id,
            raw_content=operation.raw_content,
            fact_ids=operation.final_fact_ids,
            derivation_version=operation.derivation_version,
        )
        return operation

    def finish_enriching(
        self,
        operation_id: str,
        *,
        owner: str,
        target: IngestionState,
        final_fact_ids: tuple[str, ...] | None = None,
        derivation_version: str | None = None,
        derivation_state: dict[str, bool] | None = None,
        last_error: str = "",
    ) -> IngestionOperation:
        """Finish only work owned by the caller's durable lease."""
        if target not in {IngestionState.COMPLETE, IngestionState.FAILED}:
            raise InvalidStateTransition(f"enriching -> {target.value}")
        current = self.get(operation_id)
        retry_at = 0.0
        if target is IngestionState.FAILED:
            delay = min(2 ** min(max(current.attempt_count, 1), 10), 300)
            retry_at = time.time() + delay
        rows = self.db.execute(
            "UPDATE ingestion_operations SET state=?, "
            "final_fact_ids_json=COALESCE(?, final_fact_ids_json), "
            "derivation_version=COALESCE(?, derivation_version), "
            "derivation_state_json=COALESCE(?, derivation_state_json), "
            "lease_owner='', lease_expires_at=0, next_retry_at=?, last_error=?, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE operation_id=? AND state='enriching' AND lease_owner=? "
            "RETURNING *",
            (
                target.value,
                _canonical_json(list(final_fact_ids))
                if final_fact_ids is not None
                else None,
                derivation_version,
                _canonical_json(derivation_state)
                if derivation_state is not None
                else None,
                retry_at,
                last_error,
                operation_id,
                owner,
            ),
        )
        if not rows:
            raise InvalidStateTransition("enriching lease ownership was lost")
        return self._from_row(rows[0])


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Final fact IDs plus the declared derivation stages actually completed."""

    fact_ids: tuple[str, ...]
    derivation_state: dict[str, bool]


QueryableWriter = Callable[[IngestionRequest, str], list[str]]
Materializer = Callable[
    [IngestionOperation],
    list[str] | tuple[str, ...] | MaterializationResult,
]
Projector = Callable[[IngestionOperation], dict[str, bool]]


class IngestionCommand:
    """Coordinates durable submission and idempotent enrichment."""

    def __init__(
        self,
        repository: IngestionOperationRepository,
        *,
        write_queryable: QueryableWriter,
        materialize: Materializer,
        project: Projector | None = None,
        derivation_version: str = "v3.7-ingestion-1",
        lease_seconds: float = 900.0,
    ) -> None:
        self.repository = repository
        self._write_queryable = write_queryable
        self._materializer = materialize
        self._projector = project
        self._derivation_version = derivation_version
        self._lease_seconds = max(1.0, float(lease_seconds))
        self._owner = f"ingestion-worker:{uuid.uuid4().hex}"

    def submit(self, request: IngestionRequest) -> IngestionOperation:
        operation, _created = self.submit_with_status(request)
        return operation

    def submit_with_status(
        self, request: IngestionRequest,
    ) -> tuple[IngestionOperation, bool]:
        """Submit once and report whether this call created the operation."""
        with self.repository.db.transaction():
            operation, created = self.repository.create_with_status(request)
            if operation.state is not IngestionState.RAW:
                return operation, created
            fact_ids = tuple(self._write_queryable(request, operation.operation_id))
            if not fact_ids:
                raise RuntimeError("ingestion produced no queryable facts")
            receipt = self.repository.transition(
                operation.operation_id,
                expected=IngestionState.RAW,
                target=IngestionState.QUERYABLE,
                queryable_fact_ids=fact_ids,
            )
            return receipt, created

    def materialize(self, operation_id: str) -> IngestionOperation:
        # Coalesce in-process HTTP/background/worker races for the same
        # operation. Database compare-and-swap remains the cross-process gate.
        with _materialization_lock(operation_id):
            return self._materialize_locked(operation_id)

    def _materialize_locked(self, operation_id: str) -> IngestionOperation:
        operation = self.repository.get(operation_id)
        if operation.state is IngestionState.COMPLETE:
            return operation
        if operation.state not in {
            IngestionState.QUERYABLE,
            IngestionState.ENRICHING,
            IngestionState.FAILED,
        }:
            raise InvalidStateTransition(
                f"cannot materialize operation in {operation.state.value}"
            )
        enriching = self.repository.claim_enriching(
            operation_id,
            owner=self._owner,
            lease_seconds=self._lease_seconds,
        )
        if enriching.state is IngestionState.COMPLETE:
            return enriching
        if enriching.final_fact_ids:
            return self._project_and_complete(enriching)
        try:
            with self.repository.db.transaction():
                materialized = self._materializer(enriching)
                if isinstance(materialized, MaterializationResult):
                    fact_ids = tuple(materialized.fact_ids)
                    derivation_state = dict(materialized.derivation_state)
                else:
                    fact_ids = tuple(materialized)
                    derivation_state = {"materializer": True}
                if not fact_ids:
                    raise RuntimeError("materialization produced no final facts")
                incomplete = sorted(
                    name for name, complete in derivation_state.items()
                    if not complete
                )
                if incomplete:
                    raise RuntimeError(
                        "incomplete derivation stages: " + ", ".join(incomplete)
                    )
                checkpointed = self.repository.checkpoint_enriching(
                    operation_id,
                    final_fact_ids=fact_ids,
                    derivation_version=self._derivation_version,
                    derivation_state=derivation_state,
                    lease_owner=self._owner,
                    lease_seconds=self._lease_seconds,
                )
        except Exception as exc:
            return self.repository.finish_enriching(
                operation_id,
                owner=self._owner,
                target=IngestionState.FAILED,
                last_error=str(exc),
            )
        return self._project_and_complete(checkpointed)

    def _project_and_complete(
        self, operation: IngestionOperation,
    ) -> IngestionOperation:
        try:
            # The relational checkpoint extends ownership before optional ANN
            # and vector projectors, whose cold-start latency can be material.
            # Re-claiming with the same owner renews the lease atomically.
            operation = self.repository.claim_enriching(
                operation.operation_id,
                owner=self._owner,
                lease_seconds=self._lease_seconds,
            )
            projection_state = (
                dict(self._projector(operation))
                if self._projector is not None
                else {}
            )
            combined = {**operation.derivation_state, **projection_state}
            incomplete = sorted(
                name for name, complete in combined.items() if not complete
            )
            if incomplete:
                raise RuntimeError(
                    "incomplete derivation stages: " + ", ".join(incomplete)
                )
            return self.repository.finish_enriching(
                operation.operation_id,
                owner=self._owner,
                target=IngestionState.COMPLETE,
                final_fact_ids=operation.final_fact_ids,
                derivation_version=self._derivation_version,
                derivation_state=combined,
            )
        except Exception as exc:
            return self.repository.finish_enriching(
                operation.operation_id,
                owner=self._owner,
                target=IngestionState.FAILED,
                last_error=str(exc),
            )

    def retry(self, operation_id: str) -> IngestionOperation:
        operation = self.repository.get(operation_id)
        if operation.state is not IngestionState.FAILED:
            raise InvalidStateTransition(
                f"cannot retry operation in {operation.state.value}"
            )
        return self.materialize(operation_id)
