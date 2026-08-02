# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

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
    created_at: float
    updated_at: float


class ObligationLedger:
    def record(
        self,
        conn: sqlite3.Connection,
        context: OperationContext,
        owner: str,
        kind: ObligationKind,
    ) -> None:
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO projection_obligations "
            "(operation_id, profile_id, owner, kind, subject_id, revision, "
            "state, checksum, detail, attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?)",
            (
                context.operation_id,
                context.profile_id,
                owner,
                str(kind),
                context.subject_id,
                context.revision,
                str(ObligationState.PENDING),
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
    ) -> None:
        detail_json = None if detail is None else json.dumps(
            dict(detail), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        conn.execute(
            "UPDATE projection_obligations SET "
            "state = ?, "
            "checksum = COALESCE(?, checksum), "
            "detail = COALESCE(?, detail), "
            "attempts = attempts + ?, "
            "updated_at = ? "
            "WHERE operation_id = ? AND owner = ? AND kind = ?",
            (
                str(state),
                checksum,
                detail_json,
                1 if bump_attempts else 0,
                time.time(),
                operation_id,
                owner,
                str(kind),
            ),
        )

    def fetch(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> tuple[Obligation, ...]:
        rows = conn.execute(
            "SELECT operation_id, profile_id, owner, kind, subject_id, revision, "
            "state, checksum, detail, attempts, created_at, updated_at "
            "FROM projection_obligations WHERE operation_id = ? "
            "ORDER BY owner, kind",
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
            "SELECT operation_id, profile_id, owner, kind, subject_id, revision, "
            "state, checksum, detail, attempts, created_at, updated_at "
            "FROM projection_obligations WHERE state = ? "
            "ORDER BY updated_at LIMIT ?",
            (str(state), limit),
        ).fetchall()
        return tuple(_row_to_obligation(row) for row in rows)


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
        created_at=float(row[10]),
        updated_at=float(row[11]),
    )


__all__ = ["Obligation", "ObligationLedger"]
