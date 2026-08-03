# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""OperationPolicyRegistry — declarative admission / policy layer (V4 Phase 4).

Design contract (NON-NEGOTIABLE):
1. evaluate() is PURE and CPU-ONLY. No file, network, or DB access. Microseconds.
2. Default policy is ADDITIVE: every operation kind that succeeds today continues
   to succeed after this module is imported. The registry adds checks on top of
   the existing trust hook + RBAC + ingest gate — it does NOT replace them.
3. Unknown kind: fail-OPEN in local/single-user mode (annotate with audit=True).
              fail-CLOSED in company/remote mode (deny with reason string).
4. Payload > max_payload_bytes: ANNOTATED only — never rejected here. The ingest
   gate already enforces the 1 MiB hard cap; double-rejection is forbidden.
5. The module-level _DEFAULT_REGISTRY singleton is the safe default for all
   call sites. It never rejects any REMEMBER that the existing stack accepts.

Part of SuperLocalMemory V4 | Phase 4: Admission/Policy Layer
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
from superlocalmemory.core.operation_policy import (
    _ADMIN_TRANSPORTS,
    _ALL_TRANSPORTS,
    _MESH_TRANSPORTS,
    OperationPolicy,
)
from superlocalmemory.core.operation_request import OperationKind

# ---------------------------------------------------------------------------
# Mode sentinel sets — used by evaluate() to decide unknown-kind behaviour.
# ---------------------------------------------------------------------------

_LOCAL_MODES: frozenset[str] = frozenset({
    "local", "single-user", "single_user", "personal",
})
_COMPANY_MODES: frozenset[str] = frozenset({
    "company", "remote", "multi-user", "multi_user", "enterprise",
})

# ---------------------------------------------------------------------------
# PolicyDecision — the return value of evaluate()
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result of one OperationPolicyRegistry.evaluate() call.

    ``allowed``     Whether the operation is admitted by policy.
    ``reason``      Machine-readable reason code (never a user-visible message).
    ``annotations`` Supplementary key/value pairs for audit, telemetry, and
                    downstream enrichment. The dict is NOT a copy — callers
                    must not mutate it.
    """

    allowed: bool
    reason: str
    annotations: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OperationPolicyRegistry
# ---------------------------------------------------------------------------


class OperationPolicyRegistry:
    """Declarative, immutable registry mapping OperationKind → OperationPolicy.

    Construction
    ------------
    Use ``OperationPolicyRegistry.default()`` to obtain the safe production
    default. All known OperationKind values are pre-populated with policies
    that preserve the current single-user allow behaviour.

    Immutability
    ------------
    The internal policy table is wrapped in ``types.MappingProxyType`` to
    prevent external mutation. ``register_deny`` and ``with_policy`` return
    NEW registry instances (immutable-update pattern).

    evaluate() contract
    -------------------
    - Pure: no I/O, no global mutation, no side effects.
    - Fast: dict lookup + frozenset.isdisjoint() + field comparisons.
    - Thread-safe: all state is read-only after construction.
    """

    def __init__(
        self,
        policies: dict[OperationKind, OperationPolicy],
        explicit_denies: frozenset[OperationKind] = frozenset(),
    ) -> None:
        # MappingProxyType enforces read-only access after construction.
        self._policies: types.MappingProxyType[
            OperationKind, OperationPolicy
        ] = types.MappingProxyType(dict(policies))
        self._explicit_denies: frozenset[OperationKind] = frozenset(explicit_denies)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> OperationPolicyRegistry:
        """Build the production-default registry.

        Every known OperationKind is assigned a policy whose required_roles
        includes ActorRole.OWNER and allowed_transports includes
        Transport.INTERNAL — so the existing in-process Python API path and the
        single-user HTTP path continue to be admitted without any new rejection.

        Policy table rationale
        ----------------------
        REMEMBER / RECALL      Core R/W — open to OWNER, ADMIN, MEMBER.
        FORGET / CORRECT       Reversible mutations — OWNER and ADMIN only.
        ERASE                  Irreversible — OWNER only.
        CONSOLIDATE / ARCHIVE  Maintenance — OWNER and ADMIN.
        BACKUP / RESTORE_BACKUP Administrative — OWNER only, any transport.
        MESH_*                 Collaboration — OWNER, ADMIN, MEMBER; mesh transports.
        PROVIDER_TEST          Internal probing — OWNER and ADMIN; CLI/INTERNAL only.
        MODE_CHANGE            System config — OWNER only; admin transports only.
        PROFILE_SWITCH         Profile ops — OWNER and ADMIN; all transports.
        SCHEMA_MIGRATE         Dangerous DDL — OWNER only; CLI/INTERNAL only.
        VECTOR_MIGRATE         Index DDL — OWNER and ADMIN; CLI/INTERNAL only.
        """
        _owner_only = frozenset({ActorRole.OWNER})
        _owner_admin = frozenset({ActorRole.OWNER, ActorRole.ADMIN})
        _owner_admin_member = frozenset({
            ActorRole.OWNER, ActorRole.ADMIN, ActorRole.MEMBER,
        })
        _all_reads = frozenset({
            ActorRole.OWNER, ActorRole.ADMIN, ActorRole.MEMBER, ActorRole.VIEWER,
        })

        policies: dict[OperationKind, OperationPolicy] = {
            OperationKind.REMEMBER: OperationPolicy(
                kind=OperationKind.REMEMBER,
                required_roles=_owner_admin_member,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.RECALL: OperationPolicy(
                kind=OperationKind.RECALL,
                required_roles=_all_reads,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="none",
            ),
            OperationKind.FORGET: OperationPolicy(
                kind=OperationKind.FORGET,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="full",
            ),
            OperationKind.ARCHIVE: OperationPolicy(
                kind=OperationKind.ARCHIVE,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.RESTORE: OperationPolicy(
                kind=OperationKind.RESTORE,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.CORRECT: OperationPolicy(
                kind=OperationKind.CORRECT,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="full",
            ),
            OperationKind.ERASE: OperationPolicy(
                kind=OperationKind.ERASE,
                required_roles=_owner_only,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="full",
                redaction_policy="full",
                resource_ownership_check=True,
            ),
            OperationKind.CONSOLIDATE: OperationPolicy(
                kind=OperationKind.CONSOLIDATE,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.BACKUP: OperationPolicy(
                kind=OperationKind.BACKUP,
                required_roles=_owner_only,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.RESTORE_BACKUP: OperationPolicy(
                kind=OperationKind.RESTORE_BACKUP,
                required_roles=_owner_only,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="full",
            ),
            OperationKind.MESH_SEND: OperationPolicy(
                kind=OperationKind.MESH_SEND,
                required_roles=_owner_admin_member,
                required_authentication=True,
                allowed_transports=_MESH_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.MESH_LOCK: OperationPolicy(
                kind=OperationKind.MESH_LOCK,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=frozenset({
                    Transport.MESH, Transport.INTERNAL, Transport.MCP,
                }),
                audit_level="full",
            ),
            OperationKind.PROVIDER_TEST: OperationPolicy(
                kind=OperationKind.PROVIDER_TEST,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=frozenset({
                    Transport.CLI, Transport.INTERNAL, Transport.DASHBOARD,
                }),
                audit_level="standard",
            ),
            OperationKind.MODE_CHANGE: OperationPolicy(
                kind=OperationKind.MODE_CHANGE,
                required_roles=_owner_only,
                required_authentication=True,
                allowed_transports=_ADMIN_TRANSPORTS | frozenset({Transport.MCP}),
                audit_level="full",
            ),
            OperationKind.PROFILE_SWITCH: OperationPolicy(
                kind=OperationKind.PROFILE_SWITCH,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="standard",
            ),
            OperationKind.SCHEMA_MIGRATE: OperationPolicy(
                kind=OperationKind.SCHEMA_MIGRATE,
                required_roles=_owner_only,
                required_authentication=True,
                allowed_transports=frozenset({Transport.CLI, Transport.INTERNAL}),
                audit_level="full",
            ),
            OperationKind.VECTOR_MIGRATE: OperationPolicy(
                kind=OperationKind.VECTOR_MIGRATE,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=frozenset({Transport.CLI, Transport.INTERNAL}),
                audit_level="full",
            ),
            OperationKind.EVOLVE_SKILL: OperationPolicy(
                kind=OperationKind.EVOLVE_SKILL,
                required_roles=_owner_admin,
                required_authentication=True,
                allowed_transports=_ALL_TRANSPORTS,
                audit_level="full",
            ),
        }
        return cls(policies)

    # ------------------------------------------------------------------
    # Immutable-update helpers
    # ------------------------------------------------------------------

    def with_policy(self, policy: OperationPolicy) -> OperationPolicyRegistry:
        """Return a new registry with one policy replaced or added."""
        updated = dict(self._policies)
        updated[policy.kind] = policy
        return OperationPolicyRegistry(updated, self._explicit_denies)

    def register_deny(self, kind: OperationKind) -> OperationPolicyRegistry:
        """Return a new registry that explicitly denies ``kind`` regardless of actor."""
        return OperationPolicyRegistry(
            dict(self._policies),
            self._explicit_denies | {kind},
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def coverage(self) -> dict[str, dict]:
        """Return a coverage summary for every known OperationKind.

        Returns a mapping of {kind_value: {"has_policy": bool}} for every
        value in OperationKind. Used by the startup self-check and unit tests.

        Example::

            cov = _DEFAULT_REGISTRY.coverage()
            assert cov["remember"]["has_policy"] is True
        """
        return {
            kind.value: {"has_policy": kind in self._policies}
            for kind in OperationKind
        }

    def get_policy(
        self,
        kind: OperationKind,
        transport: Transport | None = None,  # reserved for future per-transport overrides
    ) -> OperationPolicy | None:
        """Return the policy for ``kind``, or None if no policy is registered.

        The ``transport`` argument is reserved for future per-transport policy
        overrides. In V4 the table is keyed only by kind; the argument is
        accepted but not used for lookup.
        """
        return self._policies.get(kind)

    # ------------------------------------------------------------------
    # Core: pure, CPU-only evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        kind: OperationKind | str,
        actor: ActorContext,
        mode: str = "local",
        *,
        payload_bytes: int = 0,
    ) -> PolicyDecision:
        """Evaluate whether ``actor`` may perform ``kind`` under ``mode``.

        This method is PURE. No I/O of any kind. Microseconds.

        Parameters
        ----------
        kind            The operation being requested.  May be an OperationKind
                        enum value or a raw string (unknown strings are treated
                        as unknown kinds).
        actor           Server-derived actor identity. NEVER from request body.
        mode            Deployment mode: "local" / "single-user" → fail-open for
                        unknown kinds. "company" / "remote" → fail-closed.
                        Defaults to "local" (safe for internal path).
        payload_bytes   Optional advisory payload size. If > policy.max_payload_bytes,
                        the decision is annotated but NOT denied (double-rejection
                        is forbidden — the ingest gate already enforces the cap).

        Returns
        -------
        PolicyDecision  ``allowed=True`` → proceed; ``allowed=False`` → deny.
                        Never raises; all errors produce ``allowed=False``.
        """
        annotations: dict = {}

        # ----------------------------------------------------------------
        # Step 1: Resolve kind string → OperationKind enum.
        # ----------------------------------------------------------------
        if not isinstance(kind, OperationKind):
            try:
                kind = OperationKind(str(kind))
            except ValueError:
                return self._unknown_kind_decision(str(kind), mode)

        # ----------------------------------------------------------------
        # Step 2: Check explicit deny list (highest priority).
        # ----------------------------------------------------------------
        if kind in self._explicit_denies:
            return PolicyDecision(
                allowed=False,
                reason="explicit_deny",
                annotations={"kind": kind.value},
            )

        # ----------------------------------------------------------------
        # Step 3: Look up policy for this kind.
        # ----------------------------------------------------------------
        policy = self._policies.get(kind)
        if policy is None:
            # Known enum value but no policy registered — treat as unknown.
            return self._unknown_kind_decision(kind.value, mode)

        # ----------------------------------------------------------------
        # Step 4: Role check — actor must hold at least one required role.
        # An empty required_roles set means "deny all" (explicit deny policy).
        # ----------------------------------------------------------------
        if not policy.required_roles:
            return PolicyDecision(
                allowed=False,
                reason="policy_denies_all_roles",
                annotations={"kind": kind.value},
            )

        # ----------------------------------------------------------------
        # Step 5: Authentication before authorization — an unauthenticated
        # actor is rejected for authentication, not role, so the reason is
        # actionable.
        # ----------------------------------------------------------------
        if policy.required_authentication and not actor.is_authenticated:
            return PolicyDecision(
                allowed=False,
                reason="authentication_required",
                annotations={"kind": kind.value, "principal_id": actor.principal_id},
            )

        # ----------------------------------------------------------------
        # Step 6: Role check — actor must hold at least one required role.
        # ----------------------------------------------------------------
        if actor.roles.isdisjoint(policy.required_roles):
            return PolicyDecision(
                allowed=False,
                reason="insufficient_roles",
                annotations={
                    "kind": kind.value,
                    "actor_roles": sorted(r.value for r in actor.roles),
                    "required_roles": sorted(r.value for r in policy.required_roles),
                },
            )

        # ----------------------------------------------------------------
        # Step 7: Transport check.
        # ----------------------------------------------------------------
        if actor.transport not in policy.allowed_transports:
            return PolicyDecision(
                allowed=False,
                reason="transport_not_allowed",
                annotations={
                    "kind": kind.value,
                    "actor_transport": actor.transport.value,
                    "allowed_transports": sorted(
                        t.value for t in policy.allowed_transports
                    ),
                },
            )

        # ----------------------------------------------------------------
        # Step 7: Payload size — annotate only, never reject.
        # The ingest gate already enforces the 1 MiB hard cap. Annotating
        # here gives the audit log a signal without double-rejecting.
        # ----------------------------------------------------------------
        if payload_bytes > 0 and payload_bytes > policy.max_payload_bytes:
            annotations["payload_oversized"] = True
            annotations["payload_bytes"] = payload_bytes
            annotations["max_payload_bytes"] = policy.max_payload_bytes

        # ----------------------------------------------------------------
        # Step 8: Admit.
        # ----------------------------------------------------------------
        return PolicyDecision(
            allowed=True,
            reason="allow",
            annotations=annotations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unknown_kind_decision(self, kind_raw: str, mode: str) -> PolicyDecision:
        """Fail-open in local mode, fail-closed in company/remote mode."""
        if mode in _LOCAL_MODES:
            return PolicyDecision(
                allowed=True,
                reason="unknown_kind_allow_local",
                annotations={"audit": True, "kind_raw": kind_raw},
            )
        # Company / remote mode: fail closed. An unknown kind may be a
        # future privileged operation or a typo from an untrusted caller.
        return PolicyDecision(
            allowed=False,
            reason="unknown_kind_deny_company",
            annotations={"kind_raw": kind_raw},
        )

    def __repr__(self) -> str:
        return (
            f"OperationPolicyRegistry("
            f"kinds={sorted(k.value for k in self._policies)}, "
            f"explicit_denies={sorted(k.value for k in self._explicit_denies)})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — safe default for all wire-up call sites.
#
# CRIT: This singleton is constructed ONCE at import time. It is read-only
# (MappingProxyType internal storage). Thread-safe by construction.
# The default evaluate() for REMEMBER + OWNER + local ALWAYS returns
# allowed=True — zero new rejections on the existing happy path.
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: OperationPolicyRegistry = OperationPolicyRegistry.default()


__all__ = [
    "OperationPolicyRegistry",
    "PolicyDecision",
    "_DEFAULT_REGISTRY",
]
