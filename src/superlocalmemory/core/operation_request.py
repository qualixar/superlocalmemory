# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Admission envelope (OperationRequest) and status types for V4.

OperationRequest is the typed, immutable boundary between every caller and the
policy/ingestion layer. The registry uses only the fields it can evaluate
purely in memory (kind, actor, scope, payload_hash) — no I/O.

Part of SuperLocalMemory V4 | Phase 4: Admission/Policy Layer
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport

if TYPE_CHECKING:
    pass  # reserved for future typing imports


class OperationKind(str, Enum):
    """Canonical names for every mutation kind the system accepts.

    The registry stores one OperationPolicy per OperationKind. An operation
    whose kind is absent from the registry is treated as "unknown" and
    evaluated by the mode-specific fallback rule (fail-open local /
    fail-closed company).
    """

    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
    ARCHIVE = "archive"
    RESTORE = "restore"
    CORRECT = "correct"
    ERASE = "erase"
    CONSOLIDATE = "consolidate"
    BACKUP = "backup"
    RESTORE_BACKUP = "restore_backup"
    MESH_SEND = "mesh_send"
    MESH_LOCK = "mesh_lock"
    PROVIDER_TEST = "provider_test"
    MODE_CHANGE = "mode_change"
    PROFILE_SWITCH = "profile_switch"
    SCHEMA_MIGRATE = "schema_migrate"
    VECTOR_MIGRATE = "vector_migrate"
    EVOLVE_SKILL = "evolve_skill"


class OperationStatus(str, Enum):
    """Lifecycle states for a durable operation."""

    ACCEPTED = "accepted"       # Journaled, not yet committed
    COMMITTED = "committed"     # Written to canonical store
    PROJECTING = "projecting"   # Running external projections
    COMPLETE = "complete"       # All obligations met
    DEGRADED = "degraded"       # Some projections failed, retryable
    FAILED = "failed"           # Non-retryable failure
    ROLLED_BACK = "rolled_back" # Compensation complete


def _default_actor() -> ActorContext:
    """Zero-value ActorContext for the internal Python-API path.

    Used as the default in OperationRequest when a caller does not supply an
    explicit actor. The principal_id "local-operator" signals an in-process
    call but is NOT authenticated (no session token). Callers that need a
    fully-authenticated context must supply one explicitly.
    """
    return ActorContext(
        principal_id="local-operator",
        roles=frozenset({ActorRole.OWNER}),
        transport=Transport.INTERNAL,
        client_host="",
    )


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """Immutable admission envelope for every mutation entering the system.

    Construction rules (NON-NEGOTIABLE)
    ------------------------------------
    1. ``actor`` is ALWAYS server-derived — session store, RBAC, daemon
       descriptor. NEVER populated from the HTTP/MCP request body.
    2. ``operation_id`` is auto-generated when omitted (UUID hex).
    3. ``idempotency_key`` falls back to ``operation_id`` in __post_init__.
    4. ``payload_hash`` is the SHA-256 of the raw content before any scrubbing;
       callers may leave it empty for lightweight envelopes.
    """

    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = ""
    kind: OperationKind = OperationKind.REMEMBER
    actor: ActorContext = field(default_factory=_default_actor)
    profile_id: str = ""
    profile_generation: int = 0
    resource_ids: tuple[str, ...] = ()
    scope: str = "personal"
    deadline_ms: int = 30_000
    schema_capability: str = ""
    payload_hash: str = ""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        # Guarantee: idempotency_key is never empty after construction.
        if not self.idempotency_key:
            object.__setattr__(self, "idempotency_key", self.operation_id)


__all__ = [
    "OperationKind",
    "OperationRequest",
    "OperationStatus",
]
