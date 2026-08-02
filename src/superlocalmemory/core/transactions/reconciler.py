# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3
import time

from superlocalmemory.core.transactions.manifest import (
    CompletionManifest,
    ManifestState,
    OwnerEvidence,
    build_evidence,
    compute_manifest_hash,
    derive_state,
    evidence_json,
)
from superlocalmemory.core.transactions.obligations import ObligationLedger


class Reconciler:
    def __init__(self, ledger: ObligationLedger | None = None) -> None:
        self._ledger = ledger or ObligationLedger()

    def reconcile(
        self,
        conn: sqlite3.Connection,
        operation_id: str,
        profile_id: str,
        *,
        canonical_committed: bool = True,
    ) -> CompletionManifest:
        obligations = self._ledger.fetch(conn, operation_id)
        evidence = build_evidence(obligations)
        state, all_met = derive_state(
            obligations, canonical_committed=canonical_committed
        )
        manifest_hash = compute_manifest_hash(evidence)
        payload = evidence_json(evidence)
        now = time.time()
        existing = conn.execute(
            "SELECT created_at FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        created_at = float(existing[0]) if existing is not None else now
        conn.execute(
            "INSERT INTO completion_manifests "
            "(operation_id, profile_id, state, all_met, obligation_count, "
            "owner_evidence_json, manifest_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(operation_id) DO UPDATE SET "
            "profile_id = excluded.profile_id, "
            "state = excluded.state, "
            "all_met = excluded.all_met, "
            "obligation_count = excluded.obligation_count, "
            "owner_evidence_json = excluded.owner_evidence_json, "
            "manifest_hash = excluded.manifest_hash, "
            "updated_at = excluded.updated_at",
            (
                operation_id,
                profile_id,
                str(state),
                1 if all_met else 0,
                len(obligations),
                payload,
                manifest_hash,
                created_at,
                now,
            ),
        )
        return CompletionManifest(
            operation_id=operation_id,
            profile_id=profile_id,
            state=state,
            all_met=all_met,
            obligation_count=len(obligations),
            owner_evidence=evidence,
            manifest_hash=manifest_hash,
            created_at=created_at,
            updated_at=now,
        )

    def fetch_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> CompletionManifest | None:
        row = conn.execute(
            "SELECT operation_id, profile_id, state, all_met, obligation_count, "
            "owner_evidence_json, manifest_hash, created_at, updated_at "
            "FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        return CompletionManifest(
            operation_id=row[0],
            profile_id=row[1],
            state=ManifestState(row[2]),
            all_met=bool(row[3]),
            obligation_count=int(row[4]),
            owner_evidence=_evidence_from_stored(conn, operation_id),
            manifest_hash=row[6],
            created_at=float(row[7]),
            updated_at=float(row[8]),
        )

    def verify_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT manifest_hash FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return False
        obligations = self._ledger.fetch(conn, operation_id)
        recomputed = compute_manifest_hash(build_evidence(obligations))
        return recomputed == row[0]


def _evidence_from_stored(
    conn: sqlite3.Connection, operation_id: str,
) -> tuple[OwnerEvidence, ...]:
    ledger = ObligationLedger()
    return build_evidence(ledger.fetch(conn, operation_id))


__all__ = ["Reconciler"]
