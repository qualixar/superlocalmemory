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
import os
import sqlite3
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


def _tool_read_only_hint(tool: object) -> object:
    """Return the tool's read-only annotation under mcp 1.x or 2.x naming.

    MCP wire protocol uses camelCase ``readOnlyHint``. mcp==2.0.0's
    ``ToolAnnotations`` pydantic model stores the field as snake_case
    ``read_only_hint`` (alias ``readOnlyHint``) — attribute access by alias
    is not available, so a camelCase-only getattr always returns None and
    every annotated read tool is misclassified as a mutator.
    """
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return None
    val = getattr(ann, "readOnlyHint", None)
    if val is not None:
        return val
    return getattr(ann, "read_only_hint", None)

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
    # Scoped reads (Tranche C — tools_core.py)
    "recall",
    "search",
    # Tranche E — remaining MCP mutators
    "slm_loop_run",
    "set_retention_policy",
    "compact_memories",
    "log_tool_event",
    "core_memory",
    "run_maintenance",
    "reap_processes",
    "build_code_graph",
    "apply_refactor",
    "link_memory_to_code",
    # Tranche G — mesh_inbox marks messages as read (POST to mesh broker)
    "mesh_inbox",
    "mesh_wait",
    "fetch",
    "list_recent",
    "session_init",
    # Memory kinds (tools_kinds.py) — writes a kind onto an existing memory
    "set_memory_kind",
    "confirm_memory_kinds",
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
        parsed_toml = tomllib.loads(raw)  # raises on corrupt TOML
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
        result = load_deployment_config(config_toml_path=config_path)
    except Exception as exc:  # noqa: BLE001
        # config.toml exists and parsed as TOML; only its interpretation failed.
        # Returning PERSONAL here would hand owner access to a store that may
        # well be enterprise. Step 2 already returned PERSONAL for the fresh
        # install with no config at all, which is the case that needs to stay
        # frictionless.
        logger.warning(
            "admission: config.toml is present but could not be interpreted "
            "(%s) -- fail-closed (treating as enterprise).", exc,
        )
        return DEPLOYMENT_ENTERPRISE

    # D1 fail-closed: [deployment] section present but mode is unrecognized or
    # absent means someone tried to configure enterprise and a typo/omission
    # should not silently grant personal (OWNER) access on an enterprise box.
    # Explicit mode="personal" is a deliberate choice — honour it.
    # Explicit mode="enterprise" → canonical loader already returned ENTERPRISE.
    _KNOWN_MODES = ("personal", "enterprise")
    dep_section = parsed_toml.get("deployment", {})
    declared_mode = str(dep_section.get("mode", "")).strip().lower()
    if "deployment" in parsed_toml and declared_mode not in _KNOWN_MODES:
        logger.warning(
            "admission: config.toml has [deployment] section but mode %r is "
            "unrecognized/absent — fail-closed → ENTERPRISE.", declared_mode or "<missing>",
        )
        return DEPLOYMENT_ENTERPRISE
    return result


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
    # Store the SHA-256 prefix of the session token (matching the canonical HTTP
    # actor), never the raw token material, for audit-log attribution only.
    import hashlib as _hashlib

    _session_hash = (
        _hashlib.sha256(session.encode("utf-8")).hexdigest()[:16] if session else ""
    )
    return ActorContext(
        principal_id=principal,
        roles=effective_roles,
        active_profile_id=profile,
        transport=transport,
        client_host=client_host,
        session_token_hash=_session_hash,
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


# ---------------------------------------------------------------------------
# Company mode has two switches, and this is where they become one
# ---------------------------------------------------------------------------

#: Environment variable carrying the caller's user session over a transport that
#: has no request to put a header on. The dashboard issues the same token, so
#: this is not a second credential system -- it is the only channel the MCP
#: surface offers for presenting the one that already exists.
_SESSION_ENV = "SLM_USER_SESSION"

_RBAC_ROLE_TO_ACTOR = {
    "admin": ActorRole.ADMIN,
    "member": ActorRole.MEMBER,
    "viewer": ActorRole.VIEWER,
}


def _rbac_engine():
    """The workspace's role store, or None when there is not one.

    Built from the data root rather than from an HTTP app state, because the
    callers here have no request. Failures return None, which then reads as
    "personal mode" -- safe, because a workspace with no role store has no roles
    to enforce.
    """
    try:
        from superlocalmemory.access.rbac import RbacEngine
        from superlocalmemory.infra.data_root import canonical_data_root

        path = canonical_data_root() / "memory.db"
        if not path.exists():
            return None
        return RbacEngine(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission: no role store available: %s", exc)
        return None


def _company_mode_active(deployment) -> bool:
    """Whether a login is required, by EITHER of the two switches.

    THE DEFECT THIS CLOSES

    "Company mode" was two independent settings that nobody had joined up:
    ``deployment`` in config.toml, which this module read, and ``require_login``
    in the workspace's own settings, which the dashboard toggle writes and which
    the HTTP routes read. Turning company mode on from the dashboard therefore
    changed what HTTP would allow and changed nothing here.

    Measured on a real store: with ``require_login`` on, two users configured,
    and the viewer's role denying WRITE, an MCP write resolved to
    ``local-operator`` with role ``owner`` and succeeded -- while the same write
    over HTTP returned 401. The role check was not bypassed by a missing call;
    it was bypassed because this path was still being told the workspace was
    personal.

    Either switch now means the same thing on every transport.
    """
    if getattr(deployment, "is_enterprise", False):
        return True
    rbac = _rbac_engine()
    if rbac is None:
        return False
    try:
        return bool(rbac.require_login())
    except sqlite3.OperationalError as exc:
        # "No such table" means the role tables were never created, which means
        # roles were never set up, which is a personal install. Failing closed on
        # THIS is not caution -- it is refusing every write on every store that
        # has never used company mode, which is nearly all of them. It was caught
        # by an existing test whose second MCP write started failing once a store
        # file appeared in the data root.
        if "no such table" in str(exc).lower():
            logger.debug("admission: no role tables; personal workspace")
            return False
        logger.warning(
            "admission: the login policy is unreadable (%s); treating the "
            "workspace as requiring one", exc,
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- an unreadable policy is not a licence
        logger.warning(
            "admission: cannot read the login policy (%s); treating the "
            "workspace as requiring one", exc,
        )
        return True


def _target_profile(explicit: str = "") -> str:
    """The workspace a call will actually touch.

    An explicit argument wins when the caller supplies one. Otherwise this reads
    the same ``profiles.json`` the engine and the HTTP layer read, because that
    is where the write is going to land.

    THE DEFECT THIS CLOSES

    The role check used to key off ``kwargs.get("profile_id")``. ``remember``
    has no such parameter, so every role lookup resolved against ``default``
    while the write went to whichever workspace was active. A user who is an
    admin on ``default`` and a viewer on ``team`` passed the check on
    ``default`` and wrote to ``team`` -- which HTTP would have refused.
    """
    name = (explicit or "").strip()
    if name:
        return name
    try:
        from superlocalmemory.server.profile_runtime import current_request_profile

        runtime = current_request_profile()
        if runtime:
            return str(runtime)
    except Exception as exc:  # noqa: BLE001 -- no request context is normal off HTTP
        logger.debug("admission: no request profile in scope: %s", exc)
    try:
        import json as _json

        from superlocalmemory.infra.data_root import canonical_data_root

        config_file = canonical_data_root() / "profiles.json"
        if config_file.exists():
            data = _json.loads(config_file.read_text(encoding="utf-8"))
            active = str(data.get("active_profile", "") or "").strip()
            if active:
                return active
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission: cannot read the active workspace: %s", exc)
    return "default"


def _session_principal(profile: str) -> tuple[str, str, FrozenSet[ActorRole] | None]:
    """Resolve the caller from a session token in the environment.

    Returns ``(principal_id, raw_token, roles)``. An empty principal means the
    caller could not be identified, which ``resolve_actor`` turns into ANONYMOUS
    and ``admit`` then denies -- so an unset or expired token fails closed.
    """
    token = os.environ.get(_SESSION_ENV, "").strip()
    if not token:
        return "", "", None
    rbac = _rbac_engine()
    if rbac is None:
        return "", "", None
    try:
        user = rbac.resolve_session(token)
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission: session lookup failed: %s", exc)
        return "", "", None
    if not user:
        return "", "", None
    role = None
    try:
        role = rbac.get_role(user["user_id"], profile or "default")
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission: role lookup failed: %s", exc)
    # No membership on this workspace is not the same as MEMBER. Falling back to
    # a write-capable default here would hand every authenticated user write
    # access to every workspace on the machine.
    actor_role = _RBAC_ROLE_TO_ACTOR.get(
        getattr(role, "value", role) if role is not None else "", None,
    )
    if actor_role is None:
        return user["user_id"], token, frozenset({ActorRole.ANONYMOUS})
    return user["user_id"], token, frozenset({actor_role})


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
            # Either switch means company mode. Reading only the config file is
            # what let a dashboard toggle change HTTP and leave this transport
            # writing as the machine owner.
            company = _company_mode_active(deployment)
            tier = "enterprise" if company else "personal"
            mode = "company" if company else "local"
            principal, token, roles = ("", "", None)
            if company:
                principal, token, roles = _session_principal(
                    _target_profile(kwargs.get("profile_id", "") or ""),
                )
            actor = resolve_actor(
                Transport.MCP, tier=tier, mode=mode,
                principal=principal, session=token, roles=roles,
            )
            try:
                admit(kind, actor, mode=mode)
            except AdmissionDenied as exc:
                return {
                    "success": False,
                    "error": "not_authorized",
                    "code": "NOT_AUTHORIZED",
                    "retryable": False,
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
    profile: str = "",
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
    profile   : Workspace the command will touch. Empty means the active one.
    """
    import sys
    deployment = _resolve_deployment()
    # Either switch means company mode -- the same rule the MCP gate uses. This
    # gate used to read config.toml alone, so turning per-user access on from
    # the dashboard changed HTTP and MCP and left every CLI write running as the
    # machine owner.
    company = _company_mode_active(deployment)
    tier = "enterprise" if company else "personal"
    mode = "company" if company else "local"
    session = ""
    if company and not principal:
        # A CLI caller identifies itself the same way an MCP caller does. With
        # no token the principal stays empty, resolve_actor returns ANONYMOUS,
        # and admit denies -- which is the intended answer for an unauthenticated
        # write on a workspace that requires a login.
        principal, session, resolved_roles = _session_principal(
            _target_profile(profile),
        )
        if roles is None:
            roles = resolved_roles
    actor = resolve_actor(
        Transport.CLI,
        tier=tier,
        mode=mode,
        principal=principal,
        session=session,
        roles=roles,
    )
    try:
        admit(kind, actor, mode=mode)
    except AdmissionDenied as exc:
        # Name something the reader can actually do. This used to say
        # "log in with 'slm login'", and there is no such command -- so the one
        # instruction the message gave was a dead end.
        print(
            f"[slm] Operation denied ({exc.decision.reason}). "
            "This workspace requires a signed-in user. Sign in on the dashboard "
            "(slm dashboard), copy your session token, and put it in the "
            "SLM_USER_SESSION environment variable -- or ask whoever "
            "administers this workspace for access.",
            flush=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Startup coverage self-check (non-vacuous)
# ---------------------------------------------------------------------------

def coverage_self_check(
    deployment: "DeploymentConfig",
    registry: "OperationPolicyRegistry | None" = None,
    server: "object | None" = None,
) -> None:
    """Assert comprehensive policy coverage at daemon startup.

    Checks:
      1. Every OperationKind has a registered policy (existing check).
      2. No policy has an empty allowed_transports set (unreachable = bug).
      3. Every tool in _REQUIRED_MCP_GATES appears in _GATED_MCP_TOOLS.
      4. (F1) Dynamic: every mutating tool in server._tool_manager._tools
         that is NOT flagged readOnlyHint=True must be in _GATED_MCP_TOOLS.

    In personal/local mode: logs warnings for any gap (non-fatal).
    In enterprise mode: raises RuntimeError on the first gap (fatal startup).

    Parameters
    ----------
    deployment : Loaded DeploymentConfig (from unified_daemon startup).
    registry   : Override the default registry (for testing).
    server     : Optional FastMCP server; when supplied, its tool registry
                 is enumerated for check 4 (dynamic mutator coverage).
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

    # Check 4 (F1): dynamic discovery — enumerate server tool registry and flag
    # any mutating tool (readOnlyHint / read_only_hint != True) not in
    # _GATED_MCP_TOOLS. See _tool_read_only_hint for mcp 2.0 naming.
    if server is not None:
        try:
            tool_dict = server._tool_manager._tools  # type: ignore[attr-defined]
            dynamic_ungated = sorted(
                name
                for name, tool in tool_dict.items()
                if name not in _GATED_MCP_TOOLS
                and _tool_read_only_hint(tool) is not True
            )
            if dynamic_ungated:
                messages.append(
                    f"dynamic ungated MCP mutators (not in _GATED_MCP_TOOLS): {dynamic_ungated}"
                )
        except AttributeError:
            logger.debug(
                "admission: server does not expose _tool_manager._tools — skipping dynamic check"
            )

    if not messages:
        logger.debug("admission: coverage self-check passed (%d kinds)", len(list(OperationKind)))
        return

    for msg in messages:
        full = f"admission: {msg}"
        if is_enterprise:
            raise RuntimeError(full)
        logger.warning(full)


def enforce_read_scope(
    include_global: "bool | None",
    include_shared: "bool | None",
    *,
    registry: "OperationPolicyRegistry | None" = None,
) -> "tuple[bool | None, bool | None]":
    """Clamp cross-profile read flags to the RECALL policy's ``allow_cross_profile``.

    In personal mode the OWNER is unrestricted — flags pass through unchanged.
    In enterprise mode with ``allow_cross_profile=False`` (the default), any
    explicit ``True`` is silently clamped to ``False`` to prevent client
    escalation beyond the configured scope authority.  ``None`` (not specified)
    is left alone so the server default applies.
    """
    deployment = _resolve_deployment()
    # Both switches, for the same reason the write gates read both: a dashboard
    # toggle used to leave this path unclamped, so a viewer could ask for
    # include_global=True over MCP and pull another workspace's facts into the
    # candidate set while HTTP refused the same request.
    if not _company_mode_active(deployment):
        return include_global, include_shared

    reg = registry if registry is not None else _DEFAULT_REGISTRY
    from superlocalmemory.core.operation_request import OperationKind as _OK
    policy = reg._policies.get(_OK.RECALL)
    if policy is None or policy.allow_cross_profile:
        return include_global, include_shared

    clamped_global = False if include_global is True else include_global
    clamped_shared = False if include_shared is True else include_shared
    if clamped_global is not include_global or clamped_shared is not include_shared:
        logger.debug(
            "admission: enforce_read_scope clamped cross-profile flags "
            "(include_global=%s→%s, include_shared=%s→%s)",
            include_global, clamped_global, include_shared, clamped_shared,
        )
    return clamped_global, clamped_shared


__all__ = [
    "AdmissionDenied",
    "admit",
    "admits",
    "coverage_self_check",
    "enforce_read_scope",
    "gate_cli_mutation",
    "resolve_actor",
    "_resolve_deployment",
    "_GATED_MCP_TOOLS",
    "_REQUIRED_MCP_GATES",
]
