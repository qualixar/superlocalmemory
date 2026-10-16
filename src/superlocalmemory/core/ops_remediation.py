# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Operational Recovery & Admin Remediation helpers (resilience slice).

Provides two primary functions used by HTTP endpoints, MCP tools, and CLI:

    list_failed_operations(db_path, profile_id=None) -> dict
        - dead_letter: ingestion ops that exhausted automatic retries (M031)
        - degraded_manifests: completion_manifests in DEGRADED state
        - exhausted_obligations: projection_obligations FAILED with attempts >= MAX

    resolve_operation(db_path, engine, operation_id, action) -> dict
        action ∈ {"retry", "force_reconcile", "cancel"}
        - retry: re-enqueue a dead-lettered ingestion op via IngestionCommand.retry()
        - force_reconcile: immediate obligation redrive (bypasses 30s throttle)
        - cancel: mark op terminally cancelled and remove from failure surfaces

The listing and counts live in ``core/ops_inspection`` (4.1.25: an unreadable
store is reported, never "zero failures"; listings and remediation are scoped
to the workspaces a caller manages) and are re-exported here.

Design constraints (NON-NEGOTIABLE):
    - No schema migration — surface via existing tables only
    - Pure reads in list_failed_operations (no mutation)
    - Additive & backward-compatible — healthy-path unaffected
    - Immutable return dicts; explicit error handling; no silent swallowing

Part of SuperLocalMemory V4 | Operational Recovery
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from superlocalmemory.core.ops_inspection import (  # re-exported for callers
    MAX_ATTEMPTS as _MAX_ATTEMPTS,
)
from superlocalmemory.core.ops_inspection import (
    get_failure_counts,
    list_failed_operations,
    operation_profiles,
)
from superlocalmemory.storage.memory_write import memory_write

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"retry", "force_reconcile", "cancel"})


# ---------------------------------------------------------------------------
# Public API — resolve_operation
# ---------------------------------------------------------------------------


def resolve_operation(
    db_path: str | Path,
    engine: Any,
    operation_id: str,
    action: str,
    *,
    profiles: "set[str] | None" = None,
) -> dict[str, Any]:
    """Resolve a failed/stuck operation.

    Parameters
    ----------
    db_path
        Path to memory.db (used for DLQ + obligation queries).
    engine
        The running engine instance (may be None for cancel-only operations).
    operation_id
        The operation to act upon.
    action
        One of: "retry", "force_reconcile", "cancel".
    profiles
        The workspaces the caller was authorized to manage for this call
        (``None`` = the installation owner, no limit). Rows of the operation
        in any other workspace are never touched (OPS-22).

    Returns a dict: {"success": bool, "action": str, "operation_id": str, ...}
    Raises ValueError for invalid action.
    """
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"invalid action {action!r}; must be one of {sorted(_VALID_ACTIONS)}"
        )

    db_path = Path(db_path)

    if action == "cancel":
        return _action_cancel(db_path, operation_id, profiles)
    if action == "retry":
        return _action_retry(db_path, engine, operation_id, profiles)
    if action == "force_reconcile":
        return _action_force_reconcile(db_path, engine, operation_id)

    # Unreachable — guarded by _VALID_ACTIONS check above
    raise ValueError(f"unhandled action {action!r}")  # pragma: no cover


def _scoped(where: str, params: tuple, profiles: "set[str] | None") -> tuple[str, tuple]:
    """``where`` narrowed to the caller's workspaces (``None`` = no limit)."""
    if profiles is None:
        return where, params
    names = tuple(sorted(profiles))
    if not names:
        return where + " AND 0", params
    return where + f" AND profile_id IN ({','.join('?' * len(names))})", params + names


def _action_cancel(db_path: Path, operation_id: str,
                   profiles: "set[str] | None" = None) -> dict:
    """Cancel an operation: remove from DLQ, mark obligations as admin-cancelled."""
    import json

    cancel_detail = json.dumps({"admin_cancel": True, "cancelled_at": time.time()})
    statements = (
        ("DELETE FROM dead_letter_operations WHERE original_op_id = ?", (operation_id,)),
        # Exhausted obligations are marked cancelled (so they don't resurface).
        ("UPDATE projection_obligations SET state = 'failed', detail = ?, "
         "attempts = MAX(attempts, ?) WHERE operation_id = ? AND state = 'failed'",
         (cancel_detail, _MAX_ATTEMPTS, operation_id)),
        ("UPDATE completion_manifests SET state = 'FAILED' "
         "WHERE operation_id = ? AND state = 'DEGRADED'", (operation_id,)),
    )
    found = False
    try:
        # memory_write: the process write lock and the shared busy timeout, so
        # the daemon's own memory writer is waited for, not raced (4.2.8).
        with memory_write(db_path) as conn:
            for sql, params in statements:
                scoped_sql, scoped_params = _scoped(sql, params, profiles)
                if conn.execute(scoped_sql, scoped_params).rowcount > 0:
                    found = True
    except Exception as exc:  # noqa: BLE001
        logger.error("resolve_operation cancel %s: %s", operation_id, exc)
        return {
            "success": False,
            "action": "cancel",
            "operation_id": operation_id,
            "reason": str(exc),
        }

    if not found:
        return {
            "success": False,
            "action": "cancel",
            "operation_id": operation_id,
            "reason": "not_found — operation not in any failure surface",
        }

    return {
        "success": True,
        "action": "cancel",
        "operation_id": operation_id,
        "message": "Operation cancelled. It will no longer appear in the attention list.",
    }


def _dead_letter_present(db_path: Path, operation_id: str,
                         profiles: "set[str] | None") -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        sql, params = _scoped(
            "SELECT id FROM dead_letter_operations WHERE original_op_id = ?",
            (operation_id,), profiles,
        )
        return conn.execute(sql + " LIMIT 1", params).fetchone() is not None
    finally:
        conn.close()


def _retry_result(operation_id: str, success: bool, **fields) -> dict:
    return {"success": success, "action": "retry", "operation_id": operation_id, **fields}


def _action_retry(db_path: Path, engine: Any, operation_id: str,
                  profiles: "set[str] | None" = None) -> dict:
    """Retry a dead-lettered ingestion operation via IngestionCommand.retry().

    The dead-letter row is removed only after the canonical operation has
    verifiably left FAILED. With no running engine nothing can retry it, so the
    answer is "not retried" and the row stays: before 4.1.25 (audit OPS-23)
    the row was deleted and the call reported success, so a failure vanished
    from every list while nothing had been retried.
    """
    try:
        present = _dead_letter_present(db_path, operation_id, profiles)
    except Exception as exc:  # noqa: BLE001
        logger.error("resolve_operation retry DLQ check %s: %s", operation_id, exc)
        return _retry_result(operation_id, False, reason=type(exc).__name__)
    if not present:
        return _retry_result(operation_id, False,
                             reason="not_found — operation not in dead-letter queue")
    if engine is None:
        return _retry_result(
            operation_id, False, code="engine_unavailable", retryable=True,
            reason=("engine_unavailable — the SLM service is not running, so "
                    "nothing can retry this save. It stays in the list. Start "
                    "the service ('slm restart') and retry again."),
        )
    try:
        # A fully-wired ingestion command from the live engine (db + queryable
        # writer + materializer); IngestionCommand(engine) would lack them.
        from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
        from superlocalmemory.core.ingestion_command import IngestionState

        operation = build_engine_ingestion_command(engine).retry(operation_id)
        # G-10: only remove from DLQ if the retry actually moved the
        # operation out of FAILED state.
        if operation.state is IngestionState.FAILED:
            return _retry_result(operation_id, False, reason=(
                "retry_still_failed — operation remains in FAILED state;"
                " DLQ row preserved for further investigation"))
        with memory_write(db_path) as conn:
            sql, params = _scoped(
                "DELETE FROM dead_letter_operations WHERE original_op_id = ?",
                (operation_id,), profiles)
            conn.execute(sql, params)
        return _retry_result(operation_id, True, message=(
            f"Operation re-queued for retry (state: {operation.state.value})."))
    except Exception as exc:  # noqa: BLE001
        logger.error("resolve_operation retry %s: %s", operation_id, exc)
        return _retry_result(operation_id, False, reason=str(exc))


def _action_force_reconcile(db_path: Path, engine: Any, operation_id: str) -> dict:
    """Force an immediate projection reconcile for this operation_id."""
    if engine is None:
        return {
            "success": False,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "reason": "engine not available — daemon must be running for force_reconcile",
        }

    try:
        from superlocalmemory.core.transactions.concrete_owners import (
            build_transaction_service,
        )
        from superlocalmemory.server.unified_daemon import _context_for_operation

        db_obj = getattr(engine, "_db", None)
        if db_obj is None:
            return {
                "success": False,
                "action": "force_reconcile",
                "operation_id": operation_id,
                "reason": "engine._db not available",
            }

        # An erasure id has no ingestion row by design: re-prove it instead of
        # reporting "no canonical record". Erase work goes first so a manifest
        # written by the apply reconcile below already reflects it.
        erasure = (
            _force_reconcile_erasure(engine, operation_id)
            if _has_erase_obligations(db_obj, operation_id) else None
        )

        context = _context_for_operation(engine, operation_id)
        if context is None:
            return erasure or {
                "success": False,
                "action": "force_reconcile",
                "operation_id": operation_id,
                "reason": "not_found — no canonical record for this operation",
            }

        service = build_transaction_service(engine)
        service.reconcile_operation(db_obj, context)
        if erasure is not None and not erasure["success"]:
            return erasure
        return {
            "success": True,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "message": "Reconcile triggered immediately (30s throttle bypassed).",
        }
    except Exception as exc:
        logger.error("resolve_operation force_reconcile %s: %s", operation_id, exc)
        return {
            "success": False,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "reason": str(exc),
        }


def _has_erase_obligations(db_obj: Any, operation_id: str) -> bool:
    return bool(db_obj.execute(
        "SELECT 1 FROM projection_obligations "
        "WHERE operation_id = ? AND kind = 'erase' LIMIT 1",
        (operation_id,),
    ))


def _erasure_profiles(db_obj: Any, operation_id: str) -> list[str]:
    """The profiles an erasure's obligations belong to (normally exactly one)."""
    rows = db_obj.execute(
        "SELECT DISTINCT profile_id FROM projection_obligations "
        "WHERE operation_id = ? AND kind = 'erase' ORDER BY profile_id",
        (operation_id,),
    )
    return [dict(row)["profile_id"] for row in rows]


def _force_reconcile_erasure(engine: Any, operation_id: str) -> dict:
    """Re-prove an erasure now. It closes only if nothing of the memory remains.

    Proven in the erasure's own profile, which need not be the active one:
    ``slm ops`` lists every profile's unfinished erasures, and the background
    redrive re-proves them all (core/transactions/erase_redrive).
    """
    from superlocalmemory.core.transactions.erase_redrive import (
        reconcile_erase_operation,
    )

    outcomes = [
        reconcile_erase_operation(engine, operation_id, profile_id=profile_id)
        for profile_id in _erasure_profiles(engine._db, operation_id)
    ] or [reconcile_erase_operation(engine, operation_id)]
    outcome = next((o for o in outcomes if not o.closed), outcomes[0])
    if outcome.closed:
        return {
            "success": True,
            "action": "force_reconcile",
            "operation_id": operation_id,
            "message": "Deletion confirmed: nothing of the memory remains.",
        }
    return {
        "success": False,
        "action": "force_reconcile",
        "operation_id": operation_id,
        "reason": f"deletion not confirmed: {outcome.reason}",
    }


__all__ = [
    "get_failure_counts",
    "list_failed_operations",
    "operation_profiles",
    "resolve_operation",
]
