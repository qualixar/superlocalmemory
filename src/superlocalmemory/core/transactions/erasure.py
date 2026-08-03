# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import hmac as _hmac_mod
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

_RECEIPT_V1: int = 1
_RECEIPT_V2: int = 2

from superlocalmemory.core.transactions.obligations import ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
    ProjectionOwner,
)

logger = logging.getLogger("superlocalmemory.core.transactions.erasure")

MAX_ERASE_ATTEMPTS = 10

VALID_SUBJECT_TYPES = frozenset({"fact", "entity", "profile"})

AuditLogger = Callable[[Mapping[str, object]], None]


class ErasureState:
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ErasureProofRecord:
    owner: str
    erased: bool
    checksum: str
    residue: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "erased": bool(self.erased),
            "checksum": self.checksum,
            "residue": list(self.residue),
        }


@dataclass(frozen=True, slots=True)
class RemoveResult:
    proofs: tuple["ErasureProofRecord", ...]
    spine_ok: bool
    tombstoned: bool


@dataclass(frozen=True, slots=True)
class ErasureReceipt:
    erasure_id: str
    profile_id: str
    subject_type: str
    subject_id: str
    requested_by: str
    fact_count: int
    state: str
    all_erased: bool
    proofs: tuple[ErasureProofRecord, ...]
    audit_hash: str
    requested_at: float
    completed_at: float
    persisted: bool


def _proof_dicts(proofs: Iterable[ErasureProofRecord]) -> list[dict[str, object]]:
    return sorted((p.as_dict() for p in proofs), key=lambda d: d["owner"])


def compute_erasure_hash(
    *,
    erasure_id: str,
    profile_id: str,
    subject_type: str,
    subject_id: str,
    requested_by: str,
    fact_count: int,
    state: str,
    all_erased: bool,
    evidence_json: str,
    requested_at: float,
    completed_at: float,
) -> str:
    """Unkeyed SHA-256 hash for erasure receipts (v1, backward-compat)."""
    canonical = _erasure_canonical(
        erasure_id=erasure_id,
        profile_id=profile_id,
        subject_type=subject_type,
        subject_id=subject_id,
        requested_by=requested_by,
        fact_count=fact_count,
        state=state,
        all_erased=all_erased,
        evidence_json=evidence_json,
        requested_at=requested_at,
        completed_at=completed_at,
    )
    return hashlib.sha256(canonical).hexdigest()


def _erasure_canonical(
    *,
    erasure_id: str,
    profile_id: str,
    subject_type: str,
    subject_id: str,
    requested_by: str,
    fact_count: int,
    state: str,
    all_erased: bool,
    evidence_json: str,
    requested_at: float,
    completed_at: float,
    receipt_version: int | None = None,
) -> bytes:
    """Produce the deterministic byte representation of the erasure receipt envelope.

    When ``receipt_version`` is provided (v2+), it is bound into the canonical
    bytes so a downgrade from v2 → v1 is detected by the MAC mismatch.
    """
    envelope: dict[str, object] = {
        "erasure_id": erasure_id,
        "profile_id": profile_id,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "requested_by": requested_by,
        "fact_count": fact_count,
        "state": state,
        "all_erased": bool(all_erased),
        "evidence": evidence_json,
        "requested_at": repr(float(requested_at)),
        "completed_at": repr(float(completed_at)),
    }
    if receipt_version is not None:
        envelope["receipt_version"] = receipt_version
    return json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def compute_erasure_hmac(
    *,
    erasure_id: str,
    profile_id: str,
    subject_type: str,
    subject_id: str,
    requested_by: str,
    fact_count: int,
    state: str,
    all_erased: bool,
    evidence_json: str,
    requested_at: float,
    completed_at: float,
    key: bytes | None = None,
) -> str:
    """HMAC-SHA256 keyed hash for erasure receipts (v2).

    ``key`` is injected in tests; production code passes ``key=None`` to
    auto-derive from the installation key.
    """
    from superlocalmemory.core.transactions.manifest_key import (
        compute_hmac,
        derive_receipt_hmac_key,
    )

    actual_key = key if key is not None else derive_receipt_hmac_key()
    canonical = _erasure_canonical(
        erasure_id=erasure_id,
        profile_id=profile_id,
        subject_type=subject_type,
        subject_id=subject_id,
        requested_by=requested_by,
        fact_count=fact_count,
        state=state,
        all_erased=all_erased,
        evidence_json=evidence_json,
        requested_at=requested_at,
        completed_at=completed_at,
        receipt_version=_RECEIPT_V2,
    )
    return compute_hmac(actual_key, canonical)


class ErasureService:
    def __init__(
        self,
        owners: Mapping[str, ProjectionOwner],
        *,
        ledger: ObligationLedger | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._owners: dict[str, ProjectionOwner] = dict(owners)
        self._ledger = ledger or ObligationLedger()
        self._audit_logger = audit_logger

    def erase(
        self,
        db: object,
        context: OperationContext,
        *,
        subject_type: str,
        subject_id: str,
        requested_by: str = "",
    ) -> ErasureReceipt:
        requested_at = time.time()
        self.remove(db, context, memory_id=None)
        return self.finalize(
            db, context,
            subject_type=subject_type, subject_id=subject_id,
            requested_by=requested_by, requested_at=requested_at,
        )

    def remove(
        self, db: object, context: OperationContext, *, memory_id: str | None = None,
    ) -> RemoveResult:
        self._record_obligations(db, context)
        tombstoned = write_tombstones(
            db, context.profile_id, tuple(sorted(set(context.fact_ids))),
            context.operation_id, time.time(), memory_id,
        )
        proofs = [self._erase_owner(db, context, name) for name in sorted(self._owners)]
        spine_ok = bool(proofs) and all(p.erased for p in proofs)
        return RemoveResult(
            proofs=tuple(proofs), spine_ok=spine_ok, tombstoned=tombstoned,
        )

    def finalize(
        self,
        db: object,
        context: OperationContext,
        *,
        subject_type: str,
        subject_id: str,
        requested_by: str = "",
        requested_at: float | None = None,
        extra_proofs: Iterable[ErasureProofRecord] = (),
    ) -> ErasureReceipt:
        if subject_type not in VALID_SUBJECT_TYPES:
            raise ValueError(f"invalid subject_type: {subject_type!r}")
        requested_at = time.time() if requested_at is None else requested_at
        proofs: list[ErasureProofRecord] = [
            self._prove_owner(context, name) for name in sorted(self._owners)
        ]
        proofs.extend(extra_proofs)
        all_erased = bool(proofs) and all(p.erased for p in proofs)
        state = ErasureState.COMPLETE if all_erased else ErasureState.FAILED
        evidence_json = json.dumps(
            {"fact_ids": sorted(set(context.fact_ids)), "proofs": _proof_dicts(proofs)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        fact_count = len(set(context.fact_ids))
        completed_at = time.time()
        receipt_version = _receipt_version_from_db(db)
        if receipt_version >= _RECEIPT_V2:
            audit_hash = compute_erasure_hmac(
                erasure_id=context.operation_id,
                profile_id=context.profile_id,
                subject_type=subject_type,
                subject_id=subject_id,
                requested_by=requested_by,
                fact_count=fact_count,
                state=state,
                all_erased=all_erased,
                evidence_json=evidence_json,
                requested_at=requested_at,
                completed_at=completed_at,
            )
        else:
            audit_hash = compute_erasure_hash(
                erasure_id=context.operation_id,
                profile_id=context.profile_id,
                subject_type=subject_type,
                subject_id=subject_id,
                requested_by=requested_by,
                fact_count=fact_count,
                state=state,
                all_erased=all_erased,
                evidence_json=evidence_json,
                requested_at=requested_at,
                completed_at=completed_at,
            )
        persisted = self._persist(
            db,
            erasure_id=context.operation_id,
            profile_id=context.profile_id,
            subject_type=subject_type,
            subject_id=subject_id,
            requested_by=requested_by,
            fact_count=fact_count,
            state=state,
            all_erased=all_erased,
            evidence_json=evidence_json,
            audit_hash=audit_hash,
            requested_at=requested_at,
            completed_at=completed_at,
            fact_ids=tuple(sorted(set(context.fact_ids))),
        )
        self._emit_audit(
            erasure_id=context.operation_id,
            profile_id=context.profile_id,
            subject_type=subject_type,
            subject_id=subject_id,
            requested_by=requested_by,
            state=state,
            audit_hash=audit_hash,
        )
        return ErasureReceipt(
            erasure_id=context.operation_id,
            profile_id=context.profile_id,
            subject_type=subject_type,
            subject_id=subject_id,
            requested_by=requested_by,
            fact_count=fact_count,
            state=state,
            all_erased=all_erased,
            proofs=tuple(proofs),
            audit_hash=audit_hash,
            requested_at=requested_at,
            completed_at=completed_at,
            persisted=persisted,
        )

    def _prove_owner(
        self, context: OperationContext, name: str,
    ) -> ErasureProofRecord:
        owner = self._owners[name]
        try:
            proof = owner.prove_erased(context)
            erased = bool(proof.erased)
            checksum = proof.checksum or ""
            raw_residue = dict(proof.detail).get("residue")
            residue = (
                tuple(str(x) for x in raw_residue)
                if isinstance(raw_residue, list) else ()
            )
        except Exception as exc:  # noqa: BLE001
            erased = False
            checksum = ""
            residue = ()
            logger.warning("erasure prove failed for %s: %s", name, _err(exc))
        return ErasureProofRecord(
            owner=name, erased=erased, checksum=checksum, residue=residue,
        )

    def _record_obligations(self, db: object, context: OperationContext) -> None:
        try:
            with db.raw_connection() as conn:
                for name in sorted(self._owners):
                    self._ledger.record(conn, context, name, ObligationKind.ERASE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("erasure obligation record skipped: %s", _err(exc))

    def _write_tombstones(
        self,
        db: object,
        profile_id: str,
        fact_ids: tuple[str, ...],
        erasure_id: str,
        created_at: float,
        memory_id: str | None = None,
    ) -> None:
        write_tombstones(
            db, profile_id, fact_ids, erasure_id, created_at, memory_id,
        )

    def _erase_owner(
        self, db: object, context: OperationContext, name: str,
    ) -> ErasureProofRecord:
        owner = self._owners[name]
        residue: tuple[str, ...] = ()
        try:
            proof = owner.erase(context)
            erased = bool(proof.erased)
            checksum = proof.checksum or ""
            detail = {"phase": "erase", **dict(proof.detail)}
            raw_residue = detail.get("residue")
            if isinstance(raw_residue, list):
                residue = tuple(str(x) for x in raw_residue)
        except Exception as exc:  # noqa: BLE001
            erased = False
            checksum = ""
            detail = {"phase": "erase", "error": _err(exc)}
        try:
            with db.raw_connection() as conn:
                self._ledger.mark(
                    conn,
                    context.operation_id,
                    name,
                    ObligationKind.ERASE,
                    ObligationState.ERASED if erased else ObligationState.FAILED,
                    checksum=checksum or None,
                    detail=detail,
                    bump_attempts=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("erasure obligation mark skipped for %s: %s", name, _err(exc))
        return ErasureProofRecord(
            owner=name, erased=erased, checksum=checksum, residue=residue,
        )

    def _persist(
        self,
        db: object,
        *,
        erasure_id: str,
        profile_id: str,
        subject_type: str,
        subject_id: str,
        requested_by: str,
        fact_count: int,
        state: str,
        all_erased: bool,
        evidence_json: str,
        audit_hash: str,
        requested_at: float,
        completed_at: float,
        fact_ids: tuple[str, ...],
    ) -> bool:
        try:
            with db.raw_connection() as conn:
                if not _table_exists(conn, "erasure_receipts"):
                    return False
                version = _receipt_version_supported(conn)
                if version >= _RECEIPT_V2:
                    conn.execute(
                        "INSERT INTO erasure_receipts "
                        "(erasure_id, profile_id, subject_type, subject_id, "
                        "requested_by, fact_count, state, all_erased, "
                        "owner_evidence_json, audit_hash, receipt_version, "
                        "requested_at, completed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(erasure_id) DO UPDATE SET "
                        "state = excluded.state, "
                        "all_erased = excluded.all_erased, "
                        "owner_evidence_json = excluded.owner_evidence_json, "
                        "audit_hash = excluded.audit_hash, "
                        "receipt_version = excluded.receipt_version, "
                        "requested_at = excluded.requested_at, "
                        "completed_at = excluded.completed_at",
                        (
                            erasure_id, profile_id, subject_type, subject_id,
                            requested_by, fact_count, state,
                            1 if all_erased else 0, evidence_json, audit_hash,
                            _RECEIPT_V2, requested_at, completed_at,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO erasure_receipts "
                        "(erasure_id, profile_id, subject_type, subject_id, "
                        "requested_by, fact_count, state, all_erased, "
                        "owner_evidence_json, audit_hash, requested_at, completed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(erasure_id) DO UPDATE SET "
                        "state = excluded.state, "
                        "all_erased = excluded.all_erased, "
                        "owner_evidence_json = excluded.owner_evidence_json, "
                        "audit_hash = excluded.audit_hash, "
                        "requested_at = excluded.requested_at, "
                        "completed_at = excluded.completed_at",
                        (
                            erasure_id, profile_id, subject_type, subject_id,
                            requested_by, fact_count, state,
                            1 if all_erased else 0, evidence_json, audit_hash,
                            requested_at, completed_at,
                        ),
                    )
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("erasure receipt persist skipped: %s", _err(exc))
            return False

    def _emit_audit(
        self,
        *,
        erasure_id: str,
        profile_id: str,
        subject_type: str,
        subject_id: str,
        requested_by: str,
        state: str,
        audit_hash: str,
    ) -> None:
        if self._audit_logger is None:
            return
        try:
            self._audit_logger({
                "erasure_id": erasure_id,
                "profile_id": profile_id,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "requested_by": requested_by,
                "state": state,
                "audit_hash": audit_hash,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("erasure audit-chain emit skipped: %s", _err(exc))


def fetch_receipt(
    conn: object, erasure_id: str, *, profile_id: str | None = None,
) -> ErasureReceipt | None:
    predicate = "erasure_id = ?"
    params: list[object] = [erasure_id]
    if profile_id is not None:
        predicate += " AND profile_id = ?"
        params.append(profile_id)
    row = conn.execute(
        "SELECT erasure_id, profile_id, subject_type, subject_id, requested_by, "
        "fact_count, state, all_erased, owner_evidence_json, audit_hash, "
        f"requested_at, completed_at FROM erasure_receipts WHERE {predicate}",
        tuple(params),
    ).fetchone()
    if row is None:
        return None
    return ErasureReceipt(
        erasure_id=row[0],
        profile_id=row[1],
        subject_type=row[2],
        subject_id=row[3],
        requested_by=row[4],
        fact_count=int(row[5]),
        state=row[6],
        all_erased=bool(row[7]),
        proofs=_proofs_from_json(row[8]),
        audit_hash=row[9],
        requested_at=float(row[10]),
        completed_at=float(row[11]),
        persisted=True,
    )


def verify_receipt(
    conn: object, erasure_id: str, *, profile_id: str | None = None,
) -> bool:
    """Verify the audit_hash of an erasure receipt.

    Uses HMAC (v2) when the ``receipt_version`` column is present and
    the row has version >= 2, otherwise falls back to unkeyed SHA-256 (v1).
    """
    predicate = "erasure_id = ?"
    params: list[object] = [erasure_id]
    if profile_id is not None:
        predicate += " AND profile_id = ?"
        params.append(profile_id)

    db_version = _receipt_version_supported(conn)
    row_version = _receipt_row_version(conn, erasure_id)

    row = conn.execute(
        "SELECT erasure_id, profile_id, subject_type, subject_id, requested_by, "
        "fact_count, state, all_erased, owner_evidence_json, audit_hash, "
        f"requested_at, completed_at FROM erasure_receipts WHERE {predicate}",
        tuple(params),
    ).fetchone()
    if row is None:
        return False

    kwargs = dict(
        erasure_id=row[0],
        profile_id=row[1],
        subject_type=row[2],
        subject_id=row[3],
        requested_by=row[4],
        fact_count=int(row[5]),
        state=row[6],
        all_erased=bool(row[7]),
        evidence_json=row[8],
        requested_at=float(row[10]),
        completed_at=float(row[11]),
    )

    if db_version >= _RECEIPT_V2:
        # On M037-capable DBs NEVER accept the unkeyed v1 SHA path — a
        # version-downgrade forgery rewrites receipt_version=1 with a valid
        # SHA of mutated content.  Any v1 row here is rejected; legitimate
        # old rows should be re-sealed before use.
        if row_version < _RECEIPT_V2:
            return False
        from superlocalmemory.core.transactions.manifest_key import (
            derive_receipt_hmac_key,
            verify_hmac,
        )
        canonical = _erasure_canonical(**kwargs, receipt_version=_RECEIPT_V2)
        return verify_hmac(row[9], derive_receipt_hmac_key(), canonical)

    # v1 path (M037 absent) — use constant-time comparison to prevent
    # timing oracles on old SHA256 hex digests.
    recomputed = compute_erasure_hash(**kwargs)
    stored = row[9] or ""
    return _hmac_mod.compare_digest(
        recomputed.encode("utf-8"), stored.encode("utf-8")
    )


def write_tombstones(
    db: object,
    profile_id: str,
    fact_ids: tuple[str, ...],
    erasure_id: str,
    created_at: float,
    memory_id: str | None = None,
) -> bool:
    if not fact_ids:
        return False
    try:
        with db.raw_connection() as conn:
            if not _table_exists(conn, "projection_tombstones"):
                return False
            for fact_id in fact_ids:
                conn.execute(
                    "INSERT INTO projection_tombstones "
                    "(profile_id, fact_id, erasure_id, memory_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_id, fact_id) DO UPDATE SET "
                    "memory_id = COALESCE(projection_tombstones.memory_id, excluded.memory_id)",
                    (profile_id, fact_id, erasure_id, memory_id, created_at),
                )
                stored = conn.execute(
                    "SELECT memory_id FROM projection_tombstones "
                    "WHERE profile_id = ? AND fact_id = ?",
                    (profile_id, fact_id),
                ).fetchone()
                if (
                    stored is not None
                    and memory_id is not None
                    and stored[0] is not None
                    and stored[0] != memory_id
                ):
                    logger.error(
                        "tombstone provenance conflict for %s: stored=%r != passed=%r; "
                        "failing closed",
                        fact_id[:16], stored[0], memory_id,
                    )
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                    return False
            conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("erasure tombstone write skipped: %s", _err(exc))
        return False


def is_tombstoned(conn: object, profile_id: str, fact_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM projection_tombstones WHERE profile_id = ? AND fact_id = ?",
        (profile_id, fact_id),
    ).fetchone()
    return row is not None


def tombstone_memory_id(db: object, profile_id: str, fact_id: str) -> str | None:
    try:
        with db.raw_connection() as conn:
            if not _table_exists(conn, "projection_tombstones"):
                return None
            row = conn.execute(
                "SELECT memory_id FROM projection_tombstones "
                "WHERE profile_id = ? AND fact_id = ?",
                (profile_id, fact_id),
            ).fetchone()
            return row[0] if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None


def _proofs_from_json(payload: str) -> tuple[ErasureProofRecord, ...]:
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return ()
    entries = parsed.get("proofs") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return ()
    records: list[ErasureProofRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_residue = entry.get("residue")
        residue = (
            tuple(str(x) for x in raw_residue)
            if isinstance(raw_residue, list) else ()
        )
        try:
            records.append(ErasureProofRecord(
                owner=str(entry["owner"]),
                erased=bool(entry["erased"]),
                checksum=str(entry["checksum"]),
                residue=residue,
            ))
        except KeyError:
            continue
    return tuple(records)


def _table_exists(conn: object, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def _receipt_version_supported(conn: object) -> int:
    """Return the highest receipt version this DB supports."""
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(erasure_receipts)").fetchall()}
        return _RECEIPT_V2 if "receipt_version" in cols else _RECEIPT_V1
    except Exception:  # noqa: BLE001
        return _RECEIPT_V1


def _receipt_version_from_db(db: object) -> int:
    """Return the highest receipt version supported by the DB wrapper."""
    try:
        with db.raw_connection() as conn:
            return _receipt_version_supported(conn)
    except Exception:  # noqa: BLE001
        return _RECEIPT_V1


def _receipt_row_version(conn: object, erasure_id: str) -> int:
    """Read the receipt_version for an existing row; fall back to V1."""
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(erasure_receipts)").fetchall()}
        if "receipt_version" not in cols:
            return _RECEIPT_V1
        row = conn.execute(
            "SELECT receipt_version FROM erasure_receipts WHERE erasure_id = ?",
            (erasure_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else _RECEIPT_V1
    except Exception:  # noqa: BLE001
        return _RECEIPT_V1


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


__all__ = [
    "ErasureProofRecord",
    "ErasureReceipt",
    "ErasureService",
    "ErasureState",
    "RemoveResult",
    "MAX_ERASE_ATTEMPTS",
    "VALID_SUBJECT_TYPES",
    "compute_erasure_hash",
    "compute_erasure_hmac",
    "fetch_receipt",
    "is_tombstoned",
    "tombstone_memory_id",
    "verify_receipt",
    "write_tombstones",
]
