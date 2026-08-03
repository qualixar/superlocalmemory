# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 admission gateway — resolve_actor + admit + @admits decorator.

This module is the single shared admission API for MCP, CLI, and HTTP transports.
HTTP already uses the registry directly (unified_daemon.py:3740); this module
wires the same evaluation to MCP tools and CLI commands.

INVARIANT: resolve_actor() derives ActorContext from server-side facts only.
           Never call it with data from the MCP/CLI request body.

Key design choices
------------------
- Personal/single-user mode → OWNER (frictionless). Zero new friction.
- Enterprise mode + no principal → ANONYMOUS → denied for mutations.
- admit() raises AdmissionDenied on deny so the caller can return a clean error.
- @admits(kind) is a thin async decorator for MCP tools.
- coverage_self_check() is called at daemon startup.
- Fail-closed: config.toml present but unreadable → enterprise (not personal).

Part of SuperLocalMemory V4 | Phase 1: Admission Gateway
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, FrozenSet

from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
from superlocalmemory.core.operation_policy_registry import (
    PolicyDecision,
    _DEFAULT_REGISTRY,
)
from superlocalmemory.core.operation_request import OperationKind

if TYPE_CHECKING:
    from superlocalmemory.core.config import DeploymentConfig
    from superlocalmemory.core.operation_policy_registry import OperationPolicyRegistry

logger = logging.getLogger(__name__)

_COMPANY_MODES: frozenset[str] = frozenset({
    "company", "remote", "enterprise", "multi-user", "multi_user",
})

# ---------------------------------------------------------------------------
# Tool inventory tracking (populated at decoration time by @admits)
# ---------------------------------------------------------------------------

# Auto-populated as each @admits decorator is applied. Checked at startup.
_GATED_MCP_TOOLS: set[str] = set()

# Minimal set of MCP mutating tools that MUST be decorated with @admits.
# Extend this set when new mutating tools are added (Tranche B extends it).
# coverage_self_check verifies _REQUIRED_MCP_GATES ⊆ _GATED_MCP_TOOLS.
_REQUIRED_MCP_GATES: frozenset[str] = frozenset({
    # Core mutations (tools_core.py)
    "remember",
    "delete_memory",
    "update_memory",
    "correct_pattern",
    "switch_profile",
    "build_graph",
    # Forgetting / consolidation (tools_v33.py)
    "forget",
    "consolidate_cognitive",
    "quantize",
    # Mode mutations (tools_v3.py)
    "set_mode",
    # Mesh mutations (tools_mesh.py)
    "mesh_send",
    "mesh_lock",
    "mesh_state",
    "mesh_summary",
    # Evolution (tools_evolution.py)
    "evolve_skill",
    # Learning / feedback (tools_learning.py, tools_v28.py, tools_active.py)
    "reinforce_assertion",
    "contradict_assertion",
    "report_outcome",
    "report_feedback",
    "observe",
    "close_session",
    # Optimize / cache (tools_optimize.py)
    "slm_cache_set",
    "slm_compress",
    # Code graph (tools_code_graph.py)
    "update_code_graph",
})


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class AdmissionDenied(Exception):
    """Raised by admit() when the policy evaluation returns allowed=False.

    Callers map this to the appropriate transport error:
      - MCP → {"success": False, "error": "not_authorized", "reason": ...}
      - CLI → sys.exit(1) + message
      - HTTP → already handled via PermissionError→403 in unified_daemon.py
    """

    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__(decision.reason)
        self.decision: PolicyDecision = decision


# ---------------------------------------------------------------------------
# Deployment config resolution (fail-closed on present-but-unreadable)
# ---------------------------------------------------------------------------

def _resolve_deployment() -> "DeploymentConfig":
    """Load DeploymentConfig. Fail-closed when config.toml is present but unreadable.

    Distinction:
      - config.toml absent   → legitimate fresh personal install → PERSONAL (frictionless)
      - config.toml present + unreadable/corrupt → unknown enterprise state → ENTERPRISE
      - config.toml present + readable → use canonical loader result
    """
    from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE, DEPLOYMENT_PERSONAL

    # Step 1: resolve the expected config path (same root as load_deployment_config).
    try:
        from superlocalmemory.infra.data_root import state_path
        config_path = state_path("config.toml")
        config_exists = config_path.exists()
    except Exception as exc:
        logger.debug("admission: cannot resolve config path, personal default: %s", exc)
        return DEPLOYMENT_PERSONAL

    # Step 2: absent → personal (fresh install, no config ever written).
    if not config_exists:
        return DEPLOYMENT_PERSONAL

    # Step 3: present → probe raw to detect corrupt/unreadable before delegating.
    try:
        import tomllib
        raw = config_path.read_text(encoding="utf-8")
        tomllib.loads(raw)  # raises on corrupt TOML
    except Exception as exc:
        # File exists but cannot be parsed → fail-closed (treat as enterprise).
        logger.warning(
            "admission: config.toml present but unreadable — fail-closed "
            "(treating as enterprise). Cause: %s", exc,
        )
        return DEPLOYMENT_ENTERPRISE

    # Step 4: readable → delegate to canonical loader (handles mode/fields).
    try:
        from superlocalmemory.core.config import load_deployment_config
        return load_deployment_config(config_toml_path=config_path)
    except Exception as exc:
        logger.debug("admission: load_deployment_config raised (unexpected): %s", exc)
        return DEPLOYMENT_PERSONAL


# ---------------------------------------------------------------------------
# resolve_actor
# ---------------------------------------------------------------------------

def resolve_actor(
    transport: Transport,
    *,
    profile: str = "",
    principal: str = "",
    session: str = "",
    tier: str = "personal",
    mode: str = "personal",
    client_host: str = "",
    roles: FrozenSet[ActorRole] | None = None,
) -> ActorContext:
    """Build a server-derived ActorContext for MCP or CLI transport.

    Rules
    -----
    - Personal tier (default) → OWNER, no authentication required.
      This preserves the existing single-user UX with zero new friction.
    - Enterprise tier + no principal → ANONYMOUS.
      Mutations will be denied (authentication_required) by admit().
    - Enterprise tier + principal → use supplied roles (default: MEMBER).

    Parameters
    ----------
    transport    : MCP, CLI, INTERNAL, etc.
    profile      : Active profile id (metadata only).
    principal    : Authenticated principal id (from session store, never from
                   request body). Empty string → anonymous.
    session      : Session token (only the first 16 hex chars are stored).
    tier         : Deployment tier: "personal" or "enterprise".
    mode         : Deployment mode string; "company"/"remote"/"enterprise" are
                   treated as enterprise. Checked in addition to ``tier``.
    client_host  : Resolved remote address (for is_local check).
    roles        : Explicit role set for authenticated enterprise actor.
                   When None, defaults to {ActorRole.MEMBER}.
    """
    is_enterprise = tier == "enterprise" or mode in _COMPANY_MODES

    if not is_enterprise:
        return ActorContext(
            principal_id="local-operator",
            roles=frozenset({ActorRole.OWNER}),
            active_profile_id=profile,
            transport=transport,
            client_host=client_host,
        )

    if not principal:
        return ActorContext(
            principal_id="",
            roles=frozenset({ActorRole.ANONYMOUS}),
            active_profile_id=profile,
            transport=transport,
            client_host=client_host,
        )

    effective_roles: FrozenSet[ActorRole] = (
        roles if roles is not None else frozenset({ActorRole.MEMBER})
    )
    return ActorContext(
        principal_id=principal,
        roles=effective_roles,
        active_profile_id=profile,
        transport=transport,
        client_host=client_host,
        session_token_hash=session[:16] if session else "",
    )


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------

def admit(
    kind: OperationKind,
    actor: ActorContext,
    *,
    resource_ids: tuple[str, ...] = (),
    scope: str | None = None,
    mode: str = "local",
    registry: "OperationPolicyRegistry | None" = None,
) -> PolicyDecision:
    """Evaluate kind + actor against the policy registry.

    Raises AdmissionDenied on deny. Returns PolicyDecision on allow.

    Parameters
    ----------
    kind         : Operation being requested.
    actor        : Server-derived ActorContext (never from request body).
    resource_ids : Resource identifiers for future ownership checks.
    scope        : Scope label for future scoped-read checks.
    mode         : Deployment mode string forwarded to registry.evaluate().
                   "local"/"personal" → fail-open for unknown kinds.
                   "company"/"remote"/"enterprise" → fail-closed.
    registry     : Override the default registry (for testing).
    """
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    decision = reg.evaluate(kind, actor, mode)
    if not decision.allowed:
        raise AdmissionDenied(decision)
    return decision


# ---------------------------------------------------------------------------
# @admits decorator for async MCP tools
# ---------------------------------------------------------------------------

def admits(kind: OperationKind):
    """Decorator that gates an async MCP tool function via the policy registry.

    Usage (inside register_*_tools):

        @server.tool()
        @admits(OperationKind.REMEMBER)
        async def remember(content: str, ...) -> dict:
            ...

    Also registers the tool name in _GATED_MCP_TOOLS at decoration time,
    enabling coverage_self_check() to verify tool inventory at startup.

    On AdmissionDenied the decorator returns the error dict directly without
    calling the wrapped function.
    """
    def decorator(fn):
        _GATED_MCP_TOOLS.add(fn.__name__)  # register at decoration time

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            deployment = _resolve_deployment()
            tier = "enterprise" if deployment.is_enterprise else "personal"
            mode = "company" if deployment.is_enterprise else "local"
            actor = resolve_actor(Transport.MCP, tier=tier, mode=mode)
            try:
                admit(kind, actor, mode=mode)
            except AdmissionDenied as exc:
                return {
                    "success": False,
                    "error": "not_authorized",
                    "reason": exc.decision.reason,
                }
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# CLI gate helper
# ---------------------------------------------------------------------------

def gate_cli_mutation(
    kind: OperationKind,
    *,
    principal: str = "",
    roles: FrozenSet[ActorRole] | None = None,
) -> None:
    """Gate a CLI mutation command. Exits with code 1 if denied.

    Call this at the top of any CLI mutation handler that bypasses the daemon.
    In personal mode this is a no-op (OWNER always admitted). In enterprise
    mode without a principal it exits with a clear message.

    Parameters
    ----------
    kind      : Operation being performed.
    principal : Authenticated CLI principal (from session store / login token).
    roles     : Explicit roles for an authenticated enterprise user.
    """
    import sys
    deployment = _resolve_deployment()
    tier = "enterprise" if deployment.is_enterprise else "personal"
    mode = "company" if deployment.is_enterprise else "local"
    actor = resolve_actor(
        Transport.CLI,
        tier=tier,
        mode=mode,
        principal=principal,
        roles=roles,
    )
    try:
        admit(kind, actor, mode=mode)
    except AdmissionDenied as exc:
        print(
            f"[slm] Operation denied ({exc.decision.reason}). "
            "This workspace requires authentication. "
            "Log in with 'slm login' or contact your workspace administrator.",
            flush=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Startup coverage self-check (non-vacuous)
# ---------------------------------------------------------------------------

def coverage_self_check(
    deployment: "DeploymentConfig",
    registry: "OperationPolicyRegistry | None" = None,
) -> None:
    """Assert comprehensive policy coverage at daemon startup.

    Checks:
      1. Every OperationKind has a registered policy (existing check).
      2. No policy has an empty allowed_transports set (unreachable = bug).
      3. Every tool in _REQUIRED_MCP_GATES appears in _GATED_MCP_TOOLS.

    In personal/local mode: logs warnings for any gap (non-fatal).
    In enterprise mode: raises RuntimeError on the first gap (fatal startup).

    Parameters
    ----------
    deployment : Loaded DeploymentConfig (from unified_daemon startup).
    registry   : Override the default registry (for testing).
    """
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    is_enterprise = deployment.is_enterprise
    messages: list[str] = []

    # Check 1: every OperationKind has a policy entry.
    cov = reg.coverage()
    missing_kinds = [
        kind.value
        for kind in OperationKind
        if not cov.get(kind.value, {}).get("has_policy", False)
    ]
    if missing_kinds:
        messages.append(f"policy coverage gap — no policy for: {missing_kinds}")

    # Check 2: no policy has empty allowed_transports (unreachable).
    empty_transport_kinds = [
        info["kind"]
        for info in cov.values()
        if info.get("has_policy") and not info.get("has_transports", True)
    ]
    if empty_transport_kinds:
        messages.append(
            f"empty_transports in policies for: {empty_transport_kinds} "
            "(no reachable transport — these operations can never be invoked)"
        )

    # Check 3: tool inventory — every required MCP gate is wired.
    ungated = sorted(_REQUIRED_MCP_GATES - _GATED_MCP_TOOLS)
    if ungated:
        messages.append(f"ungated MCP tools (missing @admits): {ungated}")

    if not messages:
        logger.debug("admission: coverage self-check passed (%d kinds)", len(list(OperationKind)))
        return

    for msg in messages:
        full = f"admission: {msg}"
        if is_enterprise:
            raise RuntimeError(full)
        logger.warning(full)


__all__ = [
    "AdmissionDenied",
    "admit",
    "admits",
    "coverage_self_check",
    "gate_cli_mutation",
    "resolve_actor",
    "_resolve_deployment",
    "_GATED_MCP_TOOLS",
    "_REQUIRED_MCP_GATES",
]
