# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Server-derived actor identity for the V4 admission / policy layer.

INVARIANT: ActorContext is ALWAYS constructed from server-authenticated state —
session tokens, daemon descriptors, profile runtime, RBAC result — NEVER from
the request body. This constraint is enforced by construction convention and
audited in OperationPolicyRegistry.evaluate().

Part of SuperLocalMemory V4 | Phase 4: Admission/Policy Layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet

# Hosts that indicate a local-machine origin (loopback or in-process).
_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", ""})


class ActorRole(str, Enum):
    """Coarse role for an authenticated principal.

    Roles form a permission lattice: OWNER ≥ ADMIN ≥ MEMBER ≥ VIEWER.
    SYSTEM is for internal daemon operations; ANONYMOUS marks unauthenticated.
    """

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"
    SYSTEM = "system"
    ANONYMOUS = "anonymous"


class Transport(str, Enum):
    """Physical transport through which the operation arrived.

    Used by OperationPolicy.allowed_transports to restrict sensitive operations
    to known-safe channels (e.g. SCHEMA_MIGRATE only via CLI or INTERNAL).
    """

    HTTP = "http"
    MCP = "mcp"
    CLI = "cli"
    MESH = "mesh"
    HOOK = "hook"
    ADAPTER = "adapter"
    DASHBOARD = "dashboard"
    INTERNAL = "internal"  # In-process Python API path


@dataclass(frozen=True, slots=True)
class ActorContext:
    """Immutable, server-derived actor identity for a single operation.

    All fields come from server-side resolution (session store, RBAC, daemon
    descriptor). None may originate from the request body. Callers that build
    an ActorContext from untrusted input violate this contract — the registry
    evaluator cannot detect this, so the constraint must be audited at the
    construction site.

    Field semantics
    ---------------
    principal_id        Stable user/operator identifier (never a raw token).
    roles               Coarse permission set; OWNER means unrestricted writes.
    allowed_profiles    Empty frozenset ≡ unrestricted (all profiles allowed).
                        Non-empty ≡ explicit allowlist; any other profile is
                        denied at the coordinator level, not here.
    active_profile_id   Profile this operation targets.
    scopes              Scope labels this actor may write to.
    delegations         Signed capability tokens (future use).
    transport           Channel through which the call arrived.
    client_host         Resolved remote address; empty string ≡ in-process.
    session_token_hash  First 16 hex chars of SHA-256 of session token.
                        The raw token is NEVER stored here.
    """

    principal_id: str = ""
    roles: FrozenSet[ActorRole] = field(
        default_factory=lambda: frozenset({ActorRole.OWNER})
    )
    # empty = all profiles allowed; non-empty = explicit allowlist
    allowed_profiles: FrozenSet[str] = field(default_factory=frozenset)
    active_profile_id: str = ""
    active_profile_generation: int = 0
    scopes: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"personal", "project", "shared", "global"})
    )
    delegations: tuple[str, ...] = ()
    transport: Transport = Transport.HTTP
    client_host: str = ""
    session_token_hash: str = ""  # SHA-256[:16] of session token — NEVER raw

    # ------------------------------------------------------------------
    # Derived predicates (pure, no I/O)
    # ------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when the request originates from the local machine (loopback)."""
        return self.client_host in _LOCAL_HOSTS

    @property
    def is_authenticated(self) -> bool:
        """True when principal_id is non-empty and not an anonymous role."""
        return self.principal_id != "" and ActorRole.ANONYMOUS not in self.roles


# ---------------------------------------------------------------------------
# Convenience constructors (server-side only — never call from request parsers)
# ---------------------------------------------------------------------------

def make_internal_owner_context(
    *,
    principal_id: str,
    profile_id: str = "",
) -> ActorContext:
    """Build the standard ActorContext for the in-process Python API path.

    Use this when the caller is the daemon itself or an authenticated local
    operator using the Python SDK directly. The transport is INTERNAL and the
    client_host is empty (treated as local).
    """
    return ActorContext(
        principal_id=principal_id,
        roles=frozenset({ActorRole.OWNER}),
        active_profile_id=profile_id,
        transport=Transport.INTERNAL,
        client_host="",
    )


def make_http_actor_context(
    *,
    principal_id: str,
    profile_id: str = "",
    client_host: str = "",
    session_token_hash: str = "",
    roles: FrozenSet[ActorRole] | None = None,
) -> ActorContext:
    """Build the standard ActorContext for the HTTP /remember endpoint.

    ``principal_id`` comes from ``_require_write_actor`` (server-side); it is
    never the raw session token and never from the request body.
    """
    return ActorContext(
        principal_id=principal_id,
        roles=roles if roles is not None else frozenset({ActorRole.OWNER}),
        active_profile_id=profile_id,
        transport=Transport.HTTP,
        client_host=client_host,
        session_token_hash=session_token_hash,
    )


__all__ = [
    "ActorContext",
    "ActorRole",
    "Transport",
    "make_internal_owner_context",
    "make_http_actor_context",
]
