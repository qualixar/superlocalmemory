# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping

from superlocalmemory.core.transactions.manifest import CompletionManifest
from superlocalmemory.core.transactions.obligations import ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
    ProjectionOwner,
)
from superlocalmemory.core.transactions.reconciler import Reconciler

logger = logging.getLogger("superlocalmemory.core.transactions.service")

MAX_APPLY_ATTEMPTS = 10


class MemoryTransactionService:
    def __init__(
        self,
        owners: Mapping[str, ProjectionOwner] | None = None,
        *,
        ledger: ObligationLedger | None = None,
        reconciler: Reconciler | None = None,
    ) -> None:
        self._owners: dict[str, ProjectionOwner] = dict(owners or {})
        self._ledger = ledger or ObligationLedger()
        self._reconciler = reconciler or Reconciler(self._ledger)

    @property
    def owners(self) -> Mapping[str, ProjectionOwner]:
        return dict(self._owners)

    def register(self, owner: ProjectionOwner) -> None:
        self._owners[owner.name] = owner

    def record(
        self,
        conn: sqlite3.Connection,
        context: OperationContext,
        *,
        owners: Iterable[str] | None = None,
        kind: ObligationKind = ObligationKind.APPLY,
    ) -> None:
        names = list(owners) if owners is not None else list(self._owners)
        self._ledger.record_many(conn, context, names, kind)

    def apply(
        self, conn: sqlite3.Connection, context: OperationContext,
    ) -> None:
        for obligation in self._ledger.fetch(conn, context.operation_id):
            if obligation.kind is not ObligationKind.APPLY:
                continue
            if (
                obligation.state in (ObligationState.FAILED, ObligationState.APPLIED)
                and obligation.attempts >= MAX_APPLY_ATTEMPTS
            ):
                continue
            self._reconcile_owner(conn, context, obligation)

    def erase(
        self, conn: sqlite3.Connection, context: OperationContext,
    ) -> None:
        for obligation in self._ledger.fetch(conn, context.operation_id):
            if obligation.kind is not ObligationKind.ERASE:
                continue
            if obligation.state is ObligationState.ERASED:
                continue
            self._erase_one(conn, context, obligation.owner)

    def compensate(
        self, conn: sqlite3.Connection, context: OperationContext, owner_name: str,
    ) -> None:
        owner = self._owners.get(owner_name)
        if owner is None:
            self._ledger.mark(
                conn, context.operation_id, owner_name, ObligationKind.APPLY,
                ObligationState.FAILED,
                detail={"phase": "compensate", "error": "owner not registered"},
                bump_attempts=True,
            )
            return
        try:
            result = owner.compensate(context)
        except Exception as exc:  # noqa: BLE001
            self._ledger.mark(
                conn, context.operation_id, owner_name, ObligationKind.APPLY,
                ObligationState.FAILED,
                detail={"phase": "compensate", "error": _err(exc)},
                bump_attempts=True,
            )
            return
        state = ObligationState.COMPENSATED if result.ok else ObligationState.FAILED
        self._ledger.mark(
            conn, context.operation_id, owner_name, ObligationKind.APPLY, state,
            checksum=result.checksum,
            detail={"phase": "compensate", **dict(result.detail)},
            bump_attempts=True,
        )

    def reconcile(
        self,
        conn: sqlite3.Connection,
        operation_id: str,
        profile_id: str,
        *,
        canonical_committed: bool | None = None,
    ) -> CompletionManifest:
        return self._reconciler.reconcile(
            conn, operation_id, profile_id,
            canonical_committed=canonical_committed,
        )

    def run(
        self,
        conn: sqlite3.Connection,
        context: OperationContext,
        *,
        canonical_committed: bool | None = None,
    ) -> CompletionManifest:
        self.apply(conn, context)
        return self.reconcile(
            conn, context.operation_id, context.profile_id,
            canonical_committed=canonical_committed,
        )

    def reconcile_operation(
        self,
        db: object,
        context: OperationContext,
        *,
        canonical_committed: bool | None = None,
    ) -> CompletionManifest:
        op = context.operation_id
        with db.raw_connection() as conn:
            obligations = self._ledger.fetch(conn, op)
        for obligation in obligations:
            if obligation.kind is not ObligationKind.APPLY:
                continue
            if (
                obligation.state in (ObligationState.FAILED, ObligationState.APPLIED)
                and obligation.attempts >= MAX_APPLY_ATTEMPTS
            ):
                continue
            self._reconcile_owner_unlocked(db, context, obligation)
        with db.raw_connection() as conn:
            return self._reconciler.reconcile(
                conn, op, context.profile_id,
                canonical_committed=canonical_committed,
            )

    def _reconcile_owner_unlocked(
        self, db: object, context: OperationContext, obligation: object,
    ) -> None:
        op = context.operation_id
        owner_name = obligation.owner
        owner = self._owners.get(owner_name)
        if owner is None:
            with db.raw_connection() as conn:
                self._ledger.mark(
                    conn, op, owner_name, ObligationKind.APPLY,
                    ObligationState.FAILED,
                    detail={"phase": "apply", "error": "owner not registered"},
                    bump_attempts=True,
                )
            return
        ok, checksum, detail = self._safe_verify(owner, context)
        if (
            ok
            and obligation.state is ObligationState.VERIFIED
            and obligation.checksum is not None
            and checksum != obligation.checksum
        ):
            ok = False
            detail = {"error": "projection drift: content changed since verification"}
        if not ok:
            applied_ok, a_checksum, a_detail = self._safe_apply(owner, context)
            with db.raw_connection() as conn:
                if applied_ok:
                    self._ledger.mark(
                        conn, op, owner_name, ObligationKind.APPLY,
                        ObligationState.APPLIED, checksum=a_checksum,
                        bump_attempts=True,
                    )
                else:
                    self._ledger.mark(
                        conn, op, owner_name, ObligationKind.APPLY,
                        ObligationState.FAILED, checksum=a_checksum,
                        detail={"phase": "apply", **a_detail}, bump_attempts=True,
                    )
            if applied_ok:
                ok, checksum, detail = self._safe_verify(owner, context)
        with db.raw_connection() as conn:
            if ok:
                updated = self._ledger.mark(
                    conn, op, owner_name, ObligationKind.APPLY,
                    ObligationState.VERIFIED, checksum=checksum,
                    bump_verify_attempts=True, set_verified_at=True,
                )
            else:
                updated = self._ledger.mark(
                    conn, op, owner_name, ObligationKind.APPLY,
                    ObligationState.FAILED, checksum=checksum,
                    detail={"phase": "verify", **detail},
                    bump_verify_attempts=True,
                )
        if updated == 0:
            logger.warning(
                "obligation %s/%s missing during reconcile mark", op, owner_name,
            )

    @staticmethod
    def _safe_verify(
        owner: ProjectionOwner, context: OperationContext,
    ) -> tuple[bool, str | None, dict]:
        try:
            result = owner.verify(context)
        except Exception as exc:  # noqa: BLE001
            return False, None, {"error": _err(exc)}
        return result.ok, result.checksum, dict(result.detail)

    @staticmethod
    def _safe_apply(
        owner: ProjectionOwner, context: OperationContext,
    ) -> tuple[bool, str | None, dict]:
        try:
            result = owner.apply(context)
        except Exception as exc:  # noqa: BLE001
            return False, None, {"error": _err(exc)}
        return result.ok, result.checksum, dict(result.detail)

    def verify_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> bool:
        return self._reconciler.verify_manifest(conn, operation_id)

    def fetch_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> CompletionManifest | None:
        return self._reconciler.fetch_manifest(conn, operation_id)

    def _reconcile_owner(
        self, conn: sqlite3.Connection, context: OperationContext, obligation: object,
    ) -> None:
        op = context.operation_id
        owner_name = obligation.owner
        owner = self._owners.get(owner_name)
        if owner is None:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                detail={"phase": "apply", "error": "owner not registered"},
                bump_attempts=True,
            )
            return
        ok, checksum, detail = self._safe_verify(owner, context)
        if (
            ok
            and obligation.state is ObligationState.VERIFIED
            and obligation.checksum is not None
            and checksum != obligation.checksum
        ):
            ok = False
            detail = {"error": "projection drift: content changed since verification"}
        if not ok:
            applied_ok, a_checksum, a_detail = self._safe_apply(owner, context)
            if applied_ok:
                self._ledger.mark(
                    conn, op, owner_name, ObligationKind.APPLY,
                    ObligationState.APPLIED, checksum=a_checksum, bump_attempts=True,
                )
                ok, checksum, detail = self._safe_verify(owner, context)
            else:
                self._ledger.mark(
                    conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                    checksum=a_checksum, detail={"phase": "apply", **a_detail},
                    bump_attempts=True,
                )
        if ok:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.VERIFIED,
                checksum=checksum, bump_verify_attempts=True, set_verified_at=True,
            )
        else:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                checksum=checksum, detail={"phase": "verify", **detail},
                bump_verify_attempts=True,
            )

    def _erase_one(
        self, conn: sqlite3.Connection, context: OperationContext, owner_name: str,
    ) -> None:
        owner = self._owners.get(owner_name)
        op = context.operation_id
        if owner is None:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.ERASE, ObligationState.FAILED,
                detail={"phase": "erase", "error": "owner not registered"},
                bump_attempts=True,
            )
            return
        try:
            proof = owner.erase(context)
        except Exception as exc:  # noqa: BLE001
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.ERASE, ObligationState.FAILED,
                detail={"phase": "erase", "error": _err(exc)}, bump_attempts=True,
            )
            return
        state = ObligationState.ERASED if proof.erased else ObligationState.FAILED
        self._ledger.mark(
            conn, op, owner_name, ObligationKind.ERASE, state,
            checksum=proof.checksum,
            detail={"phase": "erase", **dict(proof.detail)},
            bump_attempts=True,
        )


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


__all__ = ["MemoryTransactionService"]
