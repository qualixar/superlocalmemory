# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Wave-3 Operational Recovery & Admin Remediation MCP tools (2 tools).

list_failed_operations — Surface dead-letter, degraded, and exhausted ops.
resolve_operation      — Admin retry/force-reconcile/cancel for stuck ops.

RBAC: both tools require OWNER or ADMIN role (OPS_INSPECT / OPS_RESOLVE policy).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"retry", "force_reconcile", "cancel"})


def register_ops_tools(server, get_engine: Callable) -> None:
    """Register Wave-3 operational-recovery MCP tools on *server*."""

    # ------------------------------------------------------------------
    # 1. list_failed_operations — surface all stuck/failed/degraded ops
    # ------------------------------------------------------------------
    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
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
            engine = get_engine()
            db_path = getattr(getattr(engine, "_config", None), "db_path", None)
            if db_path is None:
                from superlocalmemory.infra.data_root import state_path
                db_path = state_path("memory.db")

            from superlocalmemory.core.ops_remediation import (
                list_failed_operations as _list,
            )
            result = _list(db_path, profile_id=profile_id or None)
            return {"success": True, **result}
        except Exception as exc:
            logger.exception("list_failed_operations tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. resolve_operation — admin remediation: retry / force / cancel
    # ------------------------------------------------------------------
    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
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
            engine = get_engine()
            db_path = getattr(getattr(engine, "_config", None), "db_path", None)
            if db_path is None:
                from superlocalmemory.infra.data_root import state_path
                db_path = state_path("memory.db")

            from superlocalmemory.core.ops_remediation import (
                resolve_operation as _resolve,
            )
            return _resolve(db_path, engine, operation_id, action)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("resolve_operation tool failed op=%s action=%s", operation_id, action)
            return {"success": False, "error": str(exc)}
