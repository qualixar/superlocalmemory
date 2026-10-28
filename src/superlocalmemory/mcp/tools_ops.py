# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Operational Recovery & Admin Remediation MCP tools (2 tools).

list_failed_operations — Surface dead-letter, degraded, and exhausted ops.
resolve_operation      — Admin retry/force-reconcile/cancel for stuck ops.

RBAC: both tools require OWNER or ADMIN role (OPS_INSPECT / OPS_RESOLVE policy).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import Callable


from superlocalmemory.mcp.tool_hints import DESTROYS, READ_ONLY
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"retry", "force_reconcile", "cancel"})
_NOT_FOUND = "not_found — operation not in a workspace you manage"


def _caller_scope(kind: OperationKind, profiles: set[str]) -> set[str] | None:
    """``None`` for the installation owner, else the subset the caller may act on.

    Workspace administration is scoped (4.1.25 audit OPS-22): the decorator
    checked the caller's role on one workspace, and the tool then listed or
    changed operations in every workspace. Each workspace is checked here with
    the same rule the decorator uses.
    """
    from superlocalmemory.core.actor_context import ActorRole, Transport
    from superlocalmemory.core.admission import AdmissionDenied, admit, caller_actor

    actor, mode = caller_actor(Transport.MCP)
    if mode == "local" or ActorRole.OWNER in actor.roles:
        return None
    allowed: set[str] = set()
    for profile in profiles:
        candidate, candidate_mode = caller_actor(Transport.MCP, profile)
        try:
            admit(kind, candidate, mode=candidate_mode)
        except AdmissionDenied:
            continue
        allowed.add(profile)
    return allowed


def _db_path(get_engine: Callable):
    engine = get_engine()
    db_path = getattr(getattr(engine, "_config", None), "db_path", None)
    if db_path is None:
        from superlocalmemory.infra.data_root import state_path
        db_path = state_path("memory.db")
    return engine, db_path


def _all_profiles(db_path) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        return {str(r[0]) for r in conn.execute("SELECT profile_id FROM profiles")}
    finally:
        conn.close()


def register_ops_tools(server, get_engine: Callable) -> None:
    """Register operational-recovery MCP tools on *server*."""

    # ------------------------------------------------------------------
    # 1. list_failed_operations — surface all stuck/failed/degraded ops
    # ------------------------------------------------------------------
    @server.tool(title="List Failed Operations", annotations=READ_ONLY)
    @admits(OperationKind.OPS_INSPECT)
    async def list_failed_operations(
        profile_id: str = "",
    ) -> dict:
        """List all failed, stuck, or degraded memory operations.

        Returns three categories of troubled operations:
        - dead_letter: ingestion ops that exhausted all automatic retries.
        - degraded_manifests: completion records where some projections failed.
        - exhausted_obligations: graph/projection tasks that could not apply.

        An admin can then call resolve_operation to retry, force-reconcile, or
        cancel each entry.  Non-technical users see these in the dashboard
        Operations & Health panel.

        Args:
            profile_id: Filter to a specific profile (empty = all profiles).
        """
        try:
            _engine, db_path = _db_path(get_engine)
            from superlocalmemory.core.ops_remediation import (
                list_failed_operations as _list,
            )
            named = {profile_id} if profile_id else _all_profiles(db_path)
            scope = _caller_scope(OperationKind.OPS_INSPECT, named)
            # success is False, with the categories named, when the store
            # could not be read: unknown is never reported as "no failures".
            return _list(db_path, profile_id=profile_id or None, allowed_profiles=scope)
        except Exception as exc:
            logger.exception("list_failed_operations tool failed")
            return {"success": False, "error": type(exc).__name__,
                    "message": "The failed-operations list could not be read."}

    # ------------------------------------------------------------------
    # 2. resolve_operation — admin remediation: retry / force / cancel
    # ------------------------------------------------------------------
    @server.tool(title="Resolve Failed Operation", annotations=DESTROYS)
    @admits(OperationKind.OPS_RESOLVE)
    async def resolve_operation(
        operation_id: str,
        action: str,
    ) -> dict:
        """Resolve a stuck, failed, or degraded operation.

        Performs one of three remediation actions:
        - retry:            Re-queue the dead-letter entry for ingestion.
        - force_reconcile:  Force projection obligations to re-apply now.
        - cancel:           Remove the entry from failure surfaces permanently.

        After a cancel, the entry no longer appears in list_failed_operations.
        After retry/force_reconcile, results depend on whether the underlying
        issue (e.g. missing embedding model) is now resolved.

        Args:
            operation_id: The operation_id string shown in list_failed_operations.
            action:       One of: retry, force_reconcile, cancel.
        """
        if action not in _VALID_ACTIONS:
            return {
                "success": False,
                "error": f"invalid action '{action}': must be one of {sorted(_VALID_ACTIONS)}",
            }
        try:
            engine, db_path = _db_path(get_engine)
            from superlocalmemory.core.ops_remediation import (
                operation_profiles,
            )
            from superlocalmemory.core.ops_remediation import (
                resolve_operation as _resolve,
            )
            owners = operation_profiles(db_path, operation_id)
            scope = _caller_scope(OperationKind.OPS_RESOLVE, owners)
            if scope is not None and (not owners or scope != owners):
                return {"success": False, "action": action,
                        "operation_id": operation_id, "reason": _NOT_FOUND}
            return _resolve(db_path, engine, operation_id, action, profiles=scope)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("resolve_operation tool failed op=%s action=%s", operation_id, action)
            return {"success": False, "error": str(exc)}
