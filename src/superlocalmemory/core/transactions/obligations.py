# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
)


class ObligationConflictError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Obligation:
    operation_id: str
    profile_id: str
    owner: str
    kind: ObligationKind
    subject_id: str
    revision: int
    state: ObligationState
    checksum: str | None
    detail: Mapping[str, Any] | None
    attempts: int
    context_digest: str | None
    verify_attempts: int
    last_verified_at: float | None
    created_at: float
    updated_at: float


_SELECT_COLUMNS = (
    "operation_id, profile_id, owner, kind, subject_id, revision, state, "
    "checksum, detail, attempts, context_digest, verify_attempts, "
    "last_verified_at, created_at, updated_at"
)


def context_digest(
    context: OperationContext, owner: str, kind: ObligationKind,
) -> str:
    payload = "\0".join([
        context.operation_id,
        context.profile_id,
        owner,
        str(kind),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ObligationLedger:
    def record(
        self,
        conn: sqlite3.Connection,
        context: OperationContext,
        owner: str,
        kind: ObligationKind,
    ) -> None:
        digest = context_digest(context, owner, kind)
        existing = conn.execute(
            "SELECT context_digest, profile_id FROM projection_obligations "
            "WHERE operation_id = ? AND owner = ? AND kind = ?",
            (context.operation_id, owner, str(kind)),
        ).fetchone()
        if existing is not None:
            stored, stored_profile = existing[0], existing[1]
            if stored_profile != context.profile_id:
                raise ObligationConflictError(
                    f"obligation {context.operation_id}/{owner}/{kind} "
                    "already recorded for a different profile"
                )
            if stored is None:
                conn.execute(
                    "UPDATE projection_obligations SET context_digest = ? "
                    "WHERE operation_id = ? AND owner = ? AND kind = ?",
                    (digest, context.operation_id, owner, str(kind)),
                )
                return
            if stored != digest:
                raise ObligationConflictError(
                    f"obligation {context.operation_id}/{owner}/{kind} "
                    "already recorded with a different context"
                )
            return
        now = time.time()
        conn.execute(
            "INSERT INTO projection_obligations "
            "(operation_id, profile_id, owner, kind, subject_id, revision, "
            "state, checksum, detail, attempts, context_digest, "
            "verify_attempts, last_verified_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, 0, NULL, ?, ?)",
            (
                context.operation_id,
                context.profile_id,
                owner,
                str(kind),
                context.subject_id,
                context.revision,
                str(ObligationState.PENDING),
                digest,
                now,
                now,
            ),
        )

    def record_many(
        self,
        conn: sqlite3.Connection,
        context: OperationContext,
        owners: Iterable[str],
        kind: ObligationKind,
    ) -> None:
        for owner in owners:
            self.record(conn, context, owner, kind)

    def mark(
        self,
        conn: sqlite3.Connection,
        operation_id: str,
        owner: str,
        kind: ObligationKind,
        state: ObligationState,
        *,
        checksum: str | None = None,
        detail: Mapping[str, Any] | None = None,
        bump_attempts: bool = False,
        bump_verify_attempts: bool = False,
        set_verified_at: bool = False,
    ) -> int:
        detail_json = None if detail is None else json.dumps(
            dict(detail), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        now = time.time()
        cursor = conn.execute(
            "UPDATE projection_obligations SET "
            "state = ?, "
            "checksum = COALESCE(?, checksum), "
            "detail = ?, "
            "attempts = attempts + ?, "
            "verify_attempts = verify_attempts + ?, "
            "last_verified_at = CASE WHEN ? THEN ? ELSE last_verified_at END, "
            "updated_at = ? "
            "WHERE operation_id = ? AND owner = ? AND kind = ?",
            (
                str(state),
                checksum,
                detail_json,
                1 if bump_attempts else 0,
                1 if bump_verify_attempts else 0,
                1 if set_verified_at else 0,
                now,
                now,
                operation_id,
                owner,
                str(kind),
            ),
        )
        return cursor.rowcount

    def fetch(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> tuple[Obligation, ...]:
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM projection_obligations "
            "WHERE operation_id = ? ORDER BY owner, kind",
            (operation_id,),
        ).fetchall()
        return tuple(_row_to_obligation(row) for row in rows)

    def fetch_by_state(
        self,
        conn: sqlite3.Connection,
        state: ObligationState,
        *,
        limit: int = 100,
    ) -> tuple[Obligation, ...]:
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM projection_obligations "
            "WHERE state = ? ORDER BY updated_at LIMIT ?",
            (str(state), limit),
        ).fetchall()
        return tuple(_row_to_obligation(row) for row in rows)

    def pending_operation_ids(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str | None = None,
        limit: int = 100,
        max_attempts: int = 10,
    ) -> tuple[str, ...]:
        where = "state NOT IN (?, ?) AND attempts < ?"
        params: list[Any] = [
            str(ObligationState.VERIFIED), str(ObligationState.ERASED), max_attempts,
        ]
        if profile_id is not None:
            where += " AND profile_id = ?"
            params.append(profile_id)
        params.append(limit)
        rows = conn.execute(
            f"SELECT DISTINCT operation_id FROM projection_obligations "
            f"WHERE {where} ORDER BY updated_at LIMIT ?",
            tuple(params),
        ).fetchall()
        return tuple(row[0] for row in rows)

    def operations_missing_manifest(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str | None = None,
        limit: int = 100,
        max_attempts: int = 10,
    ) -> tuple[str, ...]:
        where = "m.operation_id IS NULL AND o.attempts < ?"
        params: list[Any] = [max_attempts]
        if profile_id is not None:
            where += " AND o.profile_id = ?"
            params.append(profile_id)
        params.append(limit)
        rows = conn.execute(
            f"SELECT DISTINCT o.operation_id FROM projection_obligations o "
            f"LEFT JOIN completion_manifests m ON m.operation_id = o.operation_id "
            f"WHERE {where} ORDER BY o.operation_id LIMIT ?",
            tuple(params),
        ).fetchall()
        return tuple(row[0] for row in rows)


def _row_to_obligation(row: Any) -> Obligation:
    detail_raw = row[8]
    detail: Mapping[str, Any] | None
    if detail_raw is None:
        detail = None
    else:
        try:
            parsed = json.loads(detail_raw)
            detail = parsed if isinstance(parsed, dict) else {"value": parsed}
        except (TypeError, json.JSONDecodeError):
            detail = None
    return Obligation(
        operation_id=row[0],
        profile_id=row[1],
        owner=row[2],
        kind=ObligationKind(row[3]),
        subject_id=row[4],
        revision=int(row[5]),
        state=ObligationState(row[6]),
        checksum=row[7],
        detail=detail,
        attempts=int(row[9]),
        context_digest=row[10],
        verify_attempts=int(row[11]) if row[11] is not None else 0,
        last_verified_at=float(row[12]) if row[12] is not None else None,
        created_at=float(row[13]),
        updated_at=float(row[14]),
    )


__all__ = ["Obligation", "ObligationConflictError", "ObligationLedger", "context_digest"]
