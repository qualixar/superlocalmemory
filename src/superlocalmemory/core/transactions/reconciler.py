# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib as _hashlib
import hmac as _hmac
import json
import sqlite3
import time

from superlocalmemory.core.transactions.manifest import (
    MANIFEST_V1,
    MANIFEST_V2,
    CompletionManifest,
    ManifestState,
    OwnerEvidence,
    build_evidence,
    compute_envelope_hash,
    compute_envelope_hmac,
    derive_state,
    evidence_json,
    hash_envelope_fields,
    verify_envelope_hmac,
)
from superlocalmemory.core.transactions.obligations import ObligationLedger
from superlocalmemory.core.transactions.owners import ObligationKind, ObligationState


class Reconciler:
    def __init__(self, ledger: ObligationLedger | None = None) -> None:
        self._ledger = ledger or ObligationLedger()

    def reconcile(
        self,
        conn: sqlite3.Connection,
        operation_id: str,
        profile_id: str,
        *,
        canonical_committed: bool | None = None,
    ) -> CompletionManifest:
        committed = self._resolve_canonical(conn, operation_id, canonical_committed)
        obligations = self._ledger.fetch(conn, operation_id)
        evidence = build_evidence(obligations)
        state, all_met = derive_state(obligations, canonical_committed=committed)

        version = _manifest_version_supported(conn)
        if version >= MANIFEST_V2:
            manifest_hash = compute_envelope_hmac(
                operation_id=operation_id,
                profile_id=profile_id,
                state=state,
                all_met=all_met,
                obligation_count=len(obligations),
                evidence=evidence,
            )
        else:
            manifest_hash = compute_envelope_hash(
                operation_id=operation_id,
                profile_id=profile_id,
                state=state,
                all_met=all_met,
                obligation_count=len(obligations),
                evidence=evidence,
            )

        payload = evidence_json(evidence)
        now = time.time()
        existing = conn.execute(
            "SELECT created_at FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        created_at = float(existing[0]) if existing is not None else now

        if version >= MANIFEST_V2:
            conn.execute(
                "INSERT INTO completion_manifests "
                "(operation_id, profile_id, state, all_met, obligation_count, "
                "owner_evidence_json, manifest_hash, manifest_version, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(operation_id) DO UPDATE SET "
                "profile_id = excluded.profile_id, "
                "state = excluded.state, "
                "all_met = excluded.all_met, "
                "obligation_count = excluded.obligation_count, "
                "owner_evidence_json = excluded.owner_evidence_json, "
                "manifest_hash = excluded.manifest_hash, "
                "manifest_version = excluded.manifest_version, "
                "updated_at = excluded.updated_at",
                (
                    operation_id, profile_id, str(state), 1 if all_met else 0,
                    len(obligations), payload, manifest_hash, MANIFEST_V2,
                    created_at, now,
                ),
            )
        else:
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
                    operation_id, profile_id, str(state), 1 if all_met else 0,
                    len(obligations), payload, manifest_hash, created_at, now,
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
            owner_evidence=_evidence_from_json(row[5]),
            manifest_hash=row[6],
            created_at=float(row[7]),
            updated_at=float(row[8]),
        )

    def verify_manifest(
        self, conn: sqlite3.Connection, operation_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT operation_id, profile_id, state, all_met, obligation_count, "
            "owner_evidence_json, manifest_hash "
            "FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            evidence_dicts = json.loads(row[5])
            if not isinstance(evidence_dicts, list):
                return False
        except (TypeError, json.JSONDecodeError):
            return False

        db_version = _manifest_version_supported(conn)
        row_version = _manifest_row_version(conn, operation_id)

        if db_version >= MANIFEST_V2:
            # On M037-capable DBs NEVER accept the unkeyed v1 SHA path —
            # an attacker can write manifest_version=1 with a valid SHA of
            # mutated content.  Any v1 row here is either a downgrade attack
            # or a legacy row that hasn't been re-sealed; both are rejected.
            if row_version < MANIFEST_V2:
                return False
            return verify_envelope_hmac(
                operation_id=row[0],
                profile_id=row[1],
                state=row[2],
                all_met=bool(row[3]),
                obligation_count=int(row[4]),
                evidence_dicts=evidence_dicts,
                stored_mac=row[6],
            )

        # v1 path (M037 absent) — use constant-time comparison to prevent
        # timing oracles on old SHA256 hex digests.
        recomputed = hash_envelope_fields(
            operation_id=row[0],
            profile_id=row[1],
            state=row[2],
            all_met=bool(row[3]),
            obligation_count=int(row[4]),
            evidence_dicts=evidence_dicts,
        )
        stored = row[6] or ""
        return _hmac.compare_digest(
            recomputed.encode("utf-8"), stored.encode("utf-8")
        )

    def _resolve_canonical(
        self,
        conn: sqlite3.Connection,
        operation_id: str,
        canonical_committed: bool | None,
    ) -> bool:
        if canonical_committed is not None:
            return canonical_committed
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='ingestion_operations'"
        ).fetchone()
        if table is None:
            return True
        row = conn.execute(
            "SELECT state FROM ingestion_operations WHERE operation_id = ? LIMIT 1",
            (operation_id,),
        ).fetchone()
        if row is None:
            return False
        return row[0] in ("queryable", "enriching", "complete")


def _manifest_version_supported(conn: sqlite3.Connection) -> int:
    """Return the highest manifest version this DB supports.

    Returns MANIFEST_V2 if the ``manifest_version`` column exists (M037 applied),
    otherwise MANIFEST_V1.
    """
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(completion_manifests)").fetchall()}
        return MANIFEST_V2 if "manifest_version" in cols else MANIFEST_V1
    except Exception:  # noqa: BLE001
        # Fail-closed: PRAGMA failure must not silently route verification through
        # the unkeyed-SHA v1 path — that allows a downgrade-forgery attack where
        # an attacker resets manifest_version=1 and recomputes a valid v1 SHA.
        return MANIFEST_V2


def _manifest_row_version(conn: sqlite3.Connection, operation_id: str) -> int:
    """Read the manifest_version column for an existing row.

    Falls back to MANIFEST_V1 when the column does not exist.
    """
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(completion_manifests)").fetchall()}
        if "manifest_version" not in cols:
            return MANIFEST_V1
        row = conn.execute(
            "SELECT manifest_version FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return int(row[0]) if row else MANIFEST_V1
    except Exception:  # noqa: BLE001
        return MANIFEST_V1


def _evidence_from_json(payload: str) -> tuple[OwnerEvidence, ...]:
    try:
        rows = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return ()
    if not isinstance(rows, list):
        return ()
    evidence: list[OwnerEvidence] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        try:
            evidence.append(OwnerEvidence(
                owner=str(entry["owner"]),
                kind=ObligationKind(entry["kind"]),
                state=ObligationState(entry["state"]),
                revision=int(entry["revision"]),
                checksum=str(entry["checksum"]),
            ))
        except (KeyError, ValueError):
            continue
    return tuple(evidence)


__all__ = ["Reconciler"]
