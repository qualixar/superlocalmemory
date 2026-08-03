# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M037 — additive columns for HMAC manifest/receipt versioning.

Adds ``manifest_version`` to ``completion_manifests`` and
``receipt_version`` to ``erasure_receipts``.  Default value 1 means the
existing rows were written with the unkeyed SHA-256 scheme (v1).  New rows
are written with version 2 (HMAC-SHA256).  The verify path reads the
version column and selects the appropriate verification algorithm.

Dependency: M033 (completion_manifests), M035 (erasure_receipts).
"""

from __future__ import annotations

import json
import sqlite3

NAME = "M037_manifest_hmac_version"
DB_TARGET = "memory"

_ADDED_COLUMNS: dict[str, tuple[str, str]] = {
    "manifest_version_in_completion_manifests": (
        "completion_manifests",
        "manifest_version INTEGER NOT NULL DEFAULT 1",
    ),
    "receipt_version_in_erasure_receipts": (
        "erasure_receipts",
        "receipt_version INTEGER NOT NULL DEFAULT 1",
    ),
}

DDL = "\n".join(
    f"ALTER TABLE {tbl} ADD COLUMN {col_def};"
    for _, (tbl, col_def) in _ADDED_COLUMNS.items()
)


def apply(conn: sqlite3.Connection) -> None:
    for _key, (table, col_def) in _ADDED_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        col_name = col_def.split()[0]
        if col_name not in _columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

    # Re-seal pre-existing v1 rows to v2 HMAC so verify_manifest / verify_receipt
    # continue to PASS post-migration.  Without re-seal, the verify path rejects
    # v1 rows on v2 DBs as potential downgrade attacks, silently breaking every
    # evidence chain written before M037 was applied.
    _reseal_manifests(conn)
    _reseal_receipts(conn)


def _reseal_manifests(conn: sqlite3.Connection) -> None:
    """Re-seal completion_manifests rows from v1 (unkeyed SHA) to v2 (HMAC)."""
    if not _table_exists(conn, "completion_manifests"):
        return
    if "manifest_version" not in _columns(conn, "completion_manifests"):
        return  # column not yet present — nothing to re-seal

    from superlocalmemory.core.transactions.manifest import (
        MANIFEST_V2,
        _canonical_envelope,
    )
    from superlocalmemory.core.transactions.manifest_key import (
        compute_hmac,
        derive_manifest_hmac_key,
    )

    key = derive_manifest_hmac_key()
    rows = conn.execute(
        "SELECT operation_id, profile_id, state, all_met, obligation_count, "
        "owner_evidence_json FROM completion_manifests WHERE manifest_version = 1"
    ).fetchall()
    for row in rows:
        evidence_dicts = json.loads(row[5] or "[]")
        canonical = _canonical_envelope(
            operation_id=row[0],
            profile_id=row[1],
            state=row[2],
            all_met=bool(row[3]),
            obligation_count=int(row[4]),
            evidence_dicts=evidence_dicts,
            manifest_version=MANIFEST_V2,
        )
        new_hash = compute_hmac(key, canonical)
        conn.execute(
            "UPDATE completion_manifests SET manifest_hash = ?, manifest_version = ? "
            "WHERE operation_id = ?",
            (new_hash, MANIFEST_V2, row[0]),
        )


def _reseal_receipts(conn: sqlite3.Connection) -> None:
    """Re-seal erasure_receipts rows from v1 (unkeyed SHA) to v2 (HMAC)."""
    if not _table_exists(conn, "erasure_receipts"):
        return
    if "receipt_version" not in _columns(conn, "erasure_receipts"):
        return  # column not yet present — nothing to re-seal

    from superlocalmemory.core.transactions.erasure import (
        _RECEIPT_V2,
        _erasure_canonical,
    )
    from superlocalmemory.core.transactions.manifest_key import (
        compute_hmac,
        derive_receipt_hmac_key,
    )

    key = derive_receipt_hmac_key()
    rows = conn.execute(
        "SELECT erasure_id, profile_id, subject_type, subject_id, requested_by, "
        "fact_count, state, all_erased, owner_evidence_json, requested_at, completed_at "
        "FROM erasure_receipts WHERE receipt_version = 1"
    ).fetchall()
    for row in rows:
        canonical = _erasure_canonical(
            erasure_id=row[0],
            profile_id=row[1],
            subject_type=row[2],
            subject_id=row[3],
            requested_by=row[4],
            fact_count=int(row[5]),
            state=row[6],
            all_erased=bool(row[7]),
            evidence_json=row[8],
            requested_at=float(row[9]),
            completed_at=float(row[10]),
            receipt_version=_RECEIPT_V2,
        )
        new_hash = compute_hmac(key, canonical)
        conn.execute(
            "UPDATE erasure_receipts SET audit_hash = ?, receipt_version = ? "
            "WHERE erasure_id = ?",
            (new_hash, _RECEIPT_V2, row[0]),
        )


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)


def verify(conn: sqlite3.Connection) -> bool:
    for _key, (table, col_def) in _ADDED_COLUMNS.items():
        if not _table_exists(conn, table):
            return False
        col_name = col_def.split()[0]
        if col_name not in _columns(conn, table):
            return False
    return True


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
