# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Declarative per-operation policy record for the V4 admission layer.

OperationPolicy is a plain frozen dataclass — a pure data declaration with no
behaviour. The OperationPolicyRegistry owns all evaluation logic. A policy
declares WHAT is required; the registry decides WHAT to DO when requirements
are not met.

Part of SuperLocalMemory V4 | Phase 4: Admission/Policy Layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

from superlocalmemory.core.actor_context import ActorRole, Transport
from superlocalmemory.core.operation_request import OperationKind

# Pre-compute the frozenset of all transports once; used as the default for
# policies that allow every transport (most operations).
_ALL_TRANSPORTS: frozenset[Transport] = frozenset(Transport)

# Sensitive/administrative operations that are restricted to fewer transports
# by default. The registry may override these at construction time.
_ADMIN_TRANSPORTS: frozenset[Transport] = frozenset({
    Transport.CLI,
    Transport.INTERNAL,
    Transport.DASHBOARD,
})
_MESH_TRANSPORTS: frozenset[Transport] = frozenset({
    Transport.MESH,
    Transport.MCP,
    Transport.HTTP,
    Transport.INTERNAL,
})


@dataclass(frozen=True, slots=True)
class OperationPolicy:
    """Declarative policy for one operation kind.

    The registry stores one OperationPolicy per OperationKind (and
    optionally per transport when transport-specific overrides are needed,
    though the V4 default table uses one per kind).

    Field semantics
    ---------------
    required_roles          Actor must hold at least ONE of these roles.
                            Empty frozenset → deny all (explicit deny policy).
    required_authentication Actor.is_authenticated must be True.
    allowed_transports      Actor.transport must be in this set.
    requires_schema_capability  Non-empty → DB must expose this capability.
    max_payload_bytes       Advisory only — payload > this is ANNOTATED, not
                            rejected (double-rejection is avoided; the ingest
                            gate already enforces the 1 MiB hard cap).
    deadline_budget_ms      Advisory request deadline.
    audit_level             "none" | "standard" | "full" — governs log detail.
    redaction_policy        "none" | "scrub" | "full" — governs PII scrub.
    resource_ownership_check Whether the coordinator must verify ownership.
    allow_cross_profile     Whether the actor may target a non-active profile.
    fail_closed_in_company_mode Informational — the registry uses the ``mode``
                            argument to evaluate(), not this field, for
                            unknown-kind fallback behaviour.
    """

    kind: OperationKind
    required_roles: FrozenSet[ActorRole] = field(
        default_factory=lambda: frozenset({ActorRole.OWNER})
    )
    required_authentication: bool = True
    allowed_transports: FrozenSet[Transport] = field(
        default_factory=lambda: _ALL_TRANSPORTS
    )
    requires_schema_capability: str = ""
    max_payload_bytes: int = 1_048_576          # 1 MiB — advisory only
    deadline_budget_ms: int = 30_000
    audit_level: str = "standard"              # "none" | "standard" | "full"
    redaction_policy: str = "scrub"            # "none" | "scrub" | "full"
    resource_ownership_check: bool = True
    allow_cross_profile: bool = False
    fail_closed_in_company_mode: bool = True    # informational; see docstring


__all__ = [
    "OperationPolicy",
    "_ALL_TRANSPORTS",
    "_ADMIN_TRANSPORTS",
    "_MESH_TRANSPORTS",
]
