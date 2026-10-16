# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Background redrive for erase obligations.

The 30 s projection redrive used to treat every pending operation as an
ingestion. Erasure ids have no ingestion row by design, so their obligations
were failed as orphans and, after ten passes, reported as exhausted. This
module owns the erase side of that redrive.

An erase obligation is closed by proof: it becomes ERASED only when the memory
is gone from the canonical store, its tombstone exists, and the owning
projection holds nothing for it. Proof is read-only and never restores data.
Since 4.2.8, when the memory is gone and tombstoned but an index still holds
part of it (the delete's removal step failed or was interrupted), that owner's
removal is run again, up to MAX_ERASE_ATTEMPTS times, and then proven. When
proof still fails the obligation is marked
FAILED with the reason, and once that has happened ``MAX_REPROOFS`` times the
erasure is reported by ``slm ops status`` instead of sitting silently. From
then on it is re-checked with a capped exponential back-off rather than every
pass, so a deletion that stays unconfirmed is not rewritten every 30 s forever.

Which memories an erasure covered comes from its erasure receipt: a fact
erasure names its one fact, an entity or profile erasure lists them in the
receipt's evidence. A receipt is trusted only after its seal verifies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.transactions.erasure import MAX_ERASE_ATTEMPTS
from superlocalmemory.core.transactions.obligations import Obligation, ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    OperationContext,
    is_terminal_success,
)

logger = logging.getLogger(__name__)

# Failed re-proofs after which an unfinished erasure counts as exhausted.
MAX_REPROOFS = MAX_ERASE_ATTEMPTS

# Back-off after MAX_REPROOFS: 30 s, 60 s, 120 s, ... never more than an hour.
RECHECK_BASE_S = 30.0
RECHECK_CAP_S = 3600.0

REASON_STILL_STORED = "the memory is still stored"
REASON_ONE_STILL_STORED = "a memory it covered is still stored"
REASON_NO_TOMBSTONE = "no deletion record exists for it"
REASON_RECEIPT_NO_LIST = "its erasure receipt does not list which memories it covered"
REASON_RECEIPT_UNSEALED = "its erasure receipt failed its integrity check"


def recheck_delay_s(reproofs: int) -> float:
    """Seconds to wait after the last re-check before the next one.

    Deterministic and capped: no wait for the first ``MAX_REPROOFS`` (so an
    unconfirmed deletion is reported within minutes), then doubling from
    ``RECHECK_BASE_S`` up to ``RECHECK_CAP_S``. ``unfinished_erase_operation_ids``
    applies the same rule in SQL.
    """
    if reproofs < MAX_REPROOFS:
        return 0.0
    return min(RECHECK_CAP_S, RECHECK_BASE_S * 2 ** min(reproofs - MAX_REPROOFS, 30))


@dataclass(frozen=True, slots=True)
class EraseOutcome:
    """Result of re-proving one erasure: closed, or the first reason it is not."""

    operation_id: str
    closed: bool
    reason: str = ""


def reconcile_pending_erasures(engine: Any, *, limit: int = 20) -> int:
    """Re-prove every profile's unfinished erasures. Returns how many closed.

    Not only the active profile's: an erasure made in another profile (a
    profile used earlier, or a remote key's own) must not wait until this
    computer switches back to it. Each is proven inside its own profile.
    """
    db = getattr(engine, "_db", None)
    if db is None:
        return 0
    ledger = ObligationLedger()
    with db.raw_connection() as conn:
        due = ledger.due_erasures_in_every_profile(
            conn, limit=limit,
            backoff=(MAX_REPROOFS, RECHECK_BASE_S, RECHECK_CAP_S),
        )
    closed = 0
    for operation_id, profile_id in due:
        try:
            closed += int(reconcile_erase_operation(
                engine, operation_id, profile_id=profile_id).closed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("erase redrive failed for %s: %s", operation_id, exc)
    return closed


def reconcile_erase_operation(engine: Any, operation_id: str, *,
                              profile_id: str | None = None) -> EraseOutcome:
    """Re-prove every unfinished erase obligation of one operation, now.

    Ignores the back-off: this is also what an explicit Reconcile runs.
    ``profile_id`` is the profile the erasure belongs to (default: the active
    one); only that profile's obligations and data are read or marked.
    """
    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service,
    )

    db = engine._db
    profile_id = profile_id or engine._profile_id
    ledger = ObligationLedger()
    with db.raw_connection() as conn:
        obligations = ledger.fetch(conn, operation_id)
    unfinished = [o for o in obligations if _needs_proof(o, profile_id)]
    if not unfinished:
        mine = [
            o for o in obligations
            if o.kind is ObligationKind.ERASE and o.profile_id == profile_id
        ]
        if mine and all(is_terminal_success(o.state) for o in mine):
            return EraseOutcome(operation_id, closed=True)
        return EraseOutcome(
            operation_id, closed=False,
            reason="it is not an open deletion in this profile",
        )
    has_apply = any(o.kind is ObligationKind.APPLY for o in obligations)
    service = build_erasure_service(engine)
    # One erase request normally names one fact; group by subject so a group
    # that cannot be proven never blocks one that can.
    groups: dict[str, list[Obligation]] = {}
    for ob in unfinished:
        groups.setdefault(ob.subject_id, []).append(ob)
    reasons = [
        _reconcile_group(db, ledger, service, operation_id, profile_id, obs, has_apply)
        for obs in groups.values()
    ]
    first_reason = next((r for r in reasons if r), "")
    return EraseOutcome(operation_id, closed=not first_reason, reason=first_reason)


def _needs_proof(ob: Obligation, profile_id: str) -> bool:
    return (
        ob.kind is ObligationKind.ERASE
        and ob.profile_id == profile_id
        and not is_terminal_success(ob.state)
        and not _admin_cancelled(ob)
    )


def _admin_cancelled(ob: Obligation) -> bool:
    detail = ob.detail or {}
    return bool(detail.get("admin_cancel") or detail.get("admin_cancelled"))


def _reconcile_group(
    db: Any, ledger: ObligationLedger, service: Any, operation_id: str,
    profile_id: str, obs: list[Obligation], has_apply: bool,
) -> str:
    """Close one subject's obligations, or record why not. Returns the reason."""
    subject_id = obs[0].subject_id
    fact_ids, reason = _covered_fact_ids(db, operation_id, profile_id, subject_id)
    proven: list[tuple[str, str | None]] = []
    if not reason:
        context = OperationContext(
            operation_id=operation_id,
            profile_id=profile_id,
            subject_id=subject_id,
            fact_ids=fact_ids,
        )
        reason, proven = _unproven_reason(db, service, context, obs)
    if reason:
        _record_unproven(db, ledger, obs, reason)
        return reason
    _close(db, ledger, operation_id, profile_id, proven, has_apply)
    return ""


def _covered_fact_ids(
    db: Any, operation_id: str, profile_id: str, subject_id: str,
) -> tuple[tuple[str, ...], str]:
    """The memories an erasure covered, or ``((), reason)`` if unknowable.

    No receipt, or a fact receipt: the subject is the fact (a receipt is
    written after the purge, so an erasure still in flight has none yet).
    Entity and profile erasures: the sealed receipt's ``fact_ids``.
    """
    from superlocalmemory.core.transactions.erasure import verify_receipt

    from superlocalmemory.core.transactions.tombstones import _table_exists

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT subject_type, owner_evidence_json FROM erasure_receipts "
            "WHERE erasure_id = ? AND profile_id = ?",
            (operation_id, profile_id),
        ).fetchone() if _table_exists(conn, "erasure_receipts") else None
        if row is None or row[0] == "fact":
            return (subject_id,), ""
        if not verify_receipt(conn, operation_id, profile_id=profile_id):
            return (), REASON_RECEIPT_UNSEALED
    try:
        listed = json.loads(row[1]).get("fact_ids")
    except (TypeError, ValueError, AttributeError):
        listed = None
    if not isinstance(listed, list) or not all(isinstance(f, str) for f in listed):
        return (), REASON_RECEIPT_NO_LIST
    return tuple(sorted(set(listed))), ""


def _unproven_reason(
    db: Any, service: Any, context: OperationContext, obs: list[Obligation],
) -> tuple[str, list[tuple[str, str | None]]]:
    """Proof, after finishing any removal left undone. ``("", proofs)`` or ``(reason, [])``."""
    from superlocalmemory.core.transactions.erasure import is_tombstoned

    for fact_id in context.fact_ids:
        if db.execute(
            "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ? LIMIT 1",
            (fact_id, context.profile_id),
        ):
            single = len(context.fact_ids) == 1
            return REASON_STILL_STORED if single else REASON_ONE_STILL_STORED, []
    with db.raw_connection() as conn:
        if not all(
            is_tombstoned(conn, context.profile_id, fact_id)
            for fact_id in context.fact_ids
        ):
            return REASON_NO_TOMBSTONE, []
    proven: list[tuple[str, str | None]] = []
    for ob in obs:
        proof = service.prove_erased(context, ob.owner)
        if not proof.erased and ob.attempts < MAX_ERASE_ATTEMPTS:
            # The memory is gone and tombstoned but this index still holds part of
            # it: the delete failed or was interrupted at this step. Finish it
            # (bounded by MAX_ERASE_ATTEMPTS), then prove it again.
            service.complete_owner(db, context, ob.owner)
            proof = service.prove_erased(context, ob.owner)
        if not proof.erased:
            return f"the {ob.owner} index still holds it", []
        proven.append((ob.owner, proof.checksum or None))
    return "", proven


def _record_unproven(
    db: Any, ledger: ObligationLedger, obs: list[Obligation], reason: str,
) -> None:
    """Mark each obligation FAILED with the reason, unless it changed meanwhile.

    Bumps ``verify_attempts`` (a re-proof), never ``attempts`` (an erase): the
    redrive did not try to erase anything. The compare-and-set matters while a
    delete is still in flight: if the erasure service closed an owner after
    the snapshot was read, its ERASED stands.
    """
    detail = {"phase": "redrive", "error": f"deletion not confirmed: {reason}"}
    with db.raw_connection() as conn:
        for ob in obs:
            ledger.mark_if_unchanged(
                conn, ob, ObligationState.FAILED,
                detail=detail, bump_verify_attempts=True,
            )


def _close(
    db: Any, ledger: ObligationLedger, operation_id: str, profile_id: str,
    proven: list[tuple[str, str | None]], has_apply: bool,
) -> None:
    """Mark the proven owners ERASED and re-seal an existing manifest, atomically.

    No manifest is created: manifests describe ingestions, and an erasure's
    durable record is its receipt. One already present is re-sealed in the
    same transaction so it cannot disagree with the ledger -- for an
    operation that also carries apply work, and for the FAILED manifest the
    4.1.17-4.1.20 orphan path wrote for erasure ids.
    """
    from superlocalmemory.core.transactions.reconciler import Reconciler

    with db.raw_connection() as conn:
        for owner, checksum in proven:
            ledger.mark(
                conn, operation_id, owner, ObligationKind.ERASE,
                ObligationState.ERASED,
                checksum=checksum,
                detail={"phase": "redrive-proven"},
            )
        has_manifest = conn.execute(
            "SELECT 1 FROM completion_manifests WHERE operation_id = ?",
            (operation_id,),
        ).fetchone() is not None
        if has_manifest:
            # Erase-only: the canonical side of an erasure is the delete, and
            # it was just proven. With apply work, ingestion state decides.
            Reconciler(ledger).reconcile(
                conn, operation_id, profile_id,
                canonical_committed=None if has_apply else True,
            )


__all__ = [
    "MAX_REPROOFS",
    "RECHECK_BASE_S",
    "RECHECK_CAP_S",
    "recheck_delay_s",
    "EraseOutcome",
    "reconcile_erase_operation",
    "reconcile_pending_erasures",
]
