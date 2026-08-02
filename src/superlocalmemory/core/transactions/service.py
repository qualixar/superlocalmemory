# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping

from superlocalmemory.core.transactions.manifest import CompletionManifest
from superlocalmemory.core.transactions.obligations import ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
    ProjectionOwner,
    is_terminal_success,
)
from superlocalmemory.core.transactions.reconciler import Reconciler


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
            if is_terminal_success(obligation.state):
                continue
            self._apply_one(conn, context, obligation.owner)

    def erase(
        self, conn: sqlite3.Connection, context: OperationContext,
    ) -> None:
        for obligation in self._ledger.fetch(conn, context.operation_id):
            if obligation.kind is not ObligationKind.ERASE:
                continue
            if is_terminal_success(obligation.state):
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
        canonical_committed: bool = True,
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
        canonical_committed: bool = True,
    ) -> CompletionManifest:
        self.apply(conn, context)
        return self.reconcile(
            conn, context.operation_id, context.profile_id,
            canonical_committed=canonical_committed,
        )

    def verify_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> bool:
        return self._reconciler.verify_manifest(conn, operation_id)

    def fetch_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> CompletionManifest | None:
        return self._reconciler.fetch_manifest(conn, operation_id)

    def _apply_one(
        self, conn: sqlite3.Connection, context: OperationContext, owner_name: str,
    ) -> None:
        owner = self._owners.get(owner_name)
        op = context.operation_id
        if owner is None:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                detail={"phase": "apply", "error": "owner not registered"},
                bump_attempts=True,
            )
            return
        try:
            applied = owner.apply(context)
        except Exception as exc:  # noqa: BLE001
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                detail={"phase": "apply", "error": _err(exc)}, bump_attempts=True,
            )
            return
        if not applied.ok:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                checksum=applied.checksum,
                detail={"phase": "apply", **dict(applied.detail)},
                bump_attempts=True,
            )
            return
        self._ledger.mark(
            conn, op, owner_name, ObligationKind.APPLY, ObligationState.APPLIED,
            checksum=applied.checksum, bump_attempts=True,
        )
        try:
            verified = owner.verify(context)
        except Exception as exc:  # noqa: BLE001
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                detail={"phase": "verify", "error": _err(exc)},
            )
            return
        if verified.ok:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.VERIFIED,
                checksum=verified.checksum,
            )
        else:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.APPLY, ObligationState.FAILED,
                checksum=verified.checksum,
                detail={"phase": "verify", **dict(verified.detail)},
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
        if proof.erased:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.ERASE, ObligationState.ERASED,
                checksum=proof.checksum,
                detail={"phase": "erase", **dict(proof.detail)},
                bump_attempts=True,
            )
        else:
            self._ledger.mark(
                conn, op, owner_name, ObligationKind.ERASE, ObligationState.FAILED,
                checksum=proof.checksum,
                detail={"phase": "erase", **dict(proof.detail)},
                bump_attempts=True,
            )


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


__all__ = ["MemoryTransactionService"]
