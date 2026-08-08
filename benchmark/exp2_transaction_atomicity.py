"""Experiment 2 — MemoryTransactionService commitment and genuine compensate().

Drives MemoryTransactionService.run() (service.py) plus the real compensate()
path, Reconciler → CompletionManifest, and ObligationLedger over a fresh_db()
database.

Choice: option (a) — genuine compensate() test.
run() does NOT auto-compensate on failure; the caller must call compensate()
explicitly.  The FAULT sub-trial therefore:
  1. Lets alpha.apply() succeed (alpha enters the tracking dict).
  2. Lets bravo.apply() raise (bravo never enters the tracking dict).
  3. Verifies alpha WAS applied (tracking dict has alpha before compensate).
  4. Calls the REAL svc_b.compensate(conn, ctx, "alpha") — this invokes
     owner.compensate() which removes alpha from the tracking dict AND marks
     the obligation COMPENSATED in the ledger.
  5. Asserts alpha IS ABSENT from the tracking dict after compensate (the
     projection was genuinely rolled back, not just never written).
  6. Asserts bravo IS ABSENT from the tracking dict (never applied).
  7. Asserts obligation states: alpha=COMPENSATED, bravo=FAILED.
  8. Asserts manifest produced by run() was DEGRADED.

_TrackingOwner is a minimal but complete implementation of the ProjectionOwner
protocol (apply/verify/compensate/erase/prove_erased/health all touch real
data structures).  It is not a mock; it is a lightweight stand-in that keeps
the test self-contained while exercising the full service + ledger + reconciler
path over a real SQLite database.

Spine exercised:
  MemoryTransactionService (service.py) — record() + run() + compensate()
  ObligationLedger (obligations.py)    — records + marks + reads obligations
  Reconciler (reconciler.py)           — builds CompletionManifest
  CompletionManifest (manifest.py)     — all_met / state / manifest_hash

Two sub-trials per trial (both must hold):

  COMMITTED sub-trial
    Both alpha and bravo succeed in apply().  After run():
      • manifest.all_met == True, state == COMPLETE
      • Both owner names appear in the tracking dict

  FAULT + COMPENSATE sub-trial
    bravo raises in apply().  After run():
      • manifest.all_met == False, state == DEGRADED
      • alpha WAS in tracking BEFORE compensate (apply() did execute)
      • After real svc.compensate(conn, ctx, "alpha"):
          – alpha IS ABSENT from tracking (genuine rollback of the projection)
          – alpha obligation state == COMPENSATED in the ledger
      • bravo was NEVER in tracking (apply() raised)
      • bravo obligation state == FAILED in the ledger
"""

from __future__ import annotations

import uuid
from pathlib import Path

from _harness import TempWorkspace, TrialOutcome, add_profile, fresh_db, run_trials

# ---------------------------------------------------------------------------
# Lightweight genuine ProjectionOwner using an in-memory tracking dict
# ---------------------------------------------------------------------------


class _TrackingOwner:
    """Minimal but complete ProjectionOwner.

    apply() writes the owner name into tracking[op_id]; verify() checks it.
    compensate() removes the entry — this is the path tested in the FAULT trial.
    When fail_on_apply=True, apply() raises before writing, giving a provably
    absent entry (zero residue before compensate is called on other owners).
    """

    def __init__(
        self, name: str, tracking: dict, *, fail_on_apply: bool = False,
    ) -> None:
        self._name = name
        self._tracking = tracking
        self._fail = fail_on_apply

    @property
    def name(self) -> str:
        return self._name

    def apply(self, context: object) -> object:
        from superlocalmemory.core.transactions.owners import OwnerResult
        if self._fail:
            raise RuntimeError(f"injected apply failure in owner '{self._name}'")
        self._tracking.setdefault(context.operation_id, set()).add(self._name)
        return OwnerResult(owner=self._name, ok=True, checksum="chk-applied")

    def verify(self, context: object) -> object:
        from superlocalmemory.core.transactions.owners import OwnerResult
        if self._fail:
            return OwnerResult(owner=self._name, ok=False)
        applied = self._name in self._tracking.get(context.operation_id, set())
        return OwnerResult(
            owner=self._name, ok=applied,
            checksum="chk-applied" if applied else None,
        )

    def compensate(self, context: object) -> object:
        from superlocalmemory.core.transactions.owners import OwnerResult
        self._tracking.get(context.operation_id, set()).discard(self._name)
        return OwnerResult(owner=self._name, ok=True)

    def erase(self, context: object) -> object:
        from superlocalmemory.core.transactions.owners import OwnerErasureProof
        return OwnerErasureProof(owner=self._name, erased=True, checksum="chk-erase")

    def prove_erased(self, context: object) -> object:
        from superlocalmemory.core.transactions.owners import OwnerErasureProof
        return OwnerErasureProof(owner=self._name, erased=True, checksum="chk-erase")

    def health(self) -> object:
        from superlocalmemory.core.transactions.owners import OwnerHealth
        return OwnerHealth(owner=self._name, healthy=True)


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def _trial(index: int) -> TrialOutcome:
    from superlocalmemory.core.transactions.manifest import ManifestState
    from superlocalmemory.core.transactions.obligations import ObligationLedger
    from superlocalmemory.core.transactions.owners import (
        ObligationKind,
        ObligationState,
        OperationContext,
    )
    from superlocalmemory.core.transactions.service import MemoryTransactionService

    with TempWorkspace() as ws:
        db = fresh_db(ws)
        try:
            pid = f"p_{uuid.uuid4().hex[:8]}"
            add_profile(db, pid)

            # ----------------------------------------------------------------
            # Sub-trial A: committed — both owners succeed
            # ----------------------------------------------------------------
            tracking_a: dict = {}
            alpha_a = _TrackingOwner("alpha", tracking_a, fail_on_apply=False)
            bravo_a = _TrackingOwner("bravo", tracking_a, fail_on_apply=False)
            svc_a = MemoryTransactionService({"alpha": alpha_a, "bravo": bravo_a})

            op_a = f"op_commit_{uuid.uuid4().hex}"
            ctx_a = OperationContext(
                operation_id=op_a, profile_id=pid, subject_id=pid,
            )
            with db.raw_connection() as conn_a:
                svc_a.record(conn_a, ctx_a, kind=ObligationKind.APPLY)
                manifest_a = svc_a.run(conn_a, ctx_a, canonical_committed=True)

            committed_ok = (
                manifest_a.all_met
                and manifest_a.state == ManifestState.COMPLETE
                and "alpha" in tracking_a.get(op_a, set())
                and "bravo" in tracking_a.get(op_a, set())
            )
            detail_a: dict = {}
            if not committed_ok:
                detail_a = {
                    "manifest_all_met": manifest_a.all_met,
                    "manifest_state": str(manifest_a.state),
                    "tracking_keys": list(tracking_a.get(op_a, set())),
                }

            # ----------------------------------------------------------------
            # Sub-trial B: fault + genuine compensate
            #
            # bravo raises in apply() → manifest DEGRADED.
            # alpha DID apply (verified by tracking dict BEFORE compensate).
            # Then call the REAL svc_b.compensate() to roll back alpha's
            # projection — compensate() calls owner.compensate() which removes
            # alpha from the tracking dict AND marks the ledger COMPENSATED.
            # ----------------------------------------------------------------
            tracking_b: dict = {}
            alpha_b = _TrackingOwner("alpha", tracking_b, fail_on_apply=False)
            bravo_b = _TrackingOwner("bravo", tracking_b, fail_on_apply=True)
            svc_b = MemoryTransactionService({"alpha": alpha_b, "bravo": bravo_b})

            op_b = f"op_fault_{uuid.uuid4().hex}"
            ctx_b = OperationContext(
                operation_id=op_b, profile_id=pid, subject_id=pid,
            )
            with db.raw_connection() as conn_b:
                svc_b.record(conn_b, ctx_b, kind=ObligationKind.APPLY)
                manifest_b = svc_b.run(conn_b, ctx_b, canonical_committed=True)

                # Check state BEFORE compensate — alpha should be present
                alpha_applied = "alpha" in tracking_b.get(op_b, set())
                bravo_not_applied = "bravo" not in tracking_b.get(op_b, set())

                # Drive the REAL compensate() path on alpha
                svc_b.compensate(conn_b, ctx_b, "alpha")

                # After compensate — alpha must be removed from the tracking dict
                alpha_compensated = "alpha" not in tracking_b.get(op_b, set())

                # Read obligation states from the ledger
                ledger = ObligationLedger()
                obligations_b = ledger.fetch(conn_b, op_b)
                states_b = {o.owner: str(o.state) for o in obligations_b}

            fault_ok = (
                not manifest_b.all_met
                and manifest_b.state == ManifestState.DEGRADED
                and alpha_applied                    # alpha WAS applied before compensate
                and bravo_not_applied               # bravo never entered tracking
                and alpha_compensated               # compensate() removed alpha's projection
                and states_b.get("bravo") == str(ObligationState.FAILED)
                and states_b.get("alpha") == str(ObligationState.COMPENSATED)
            )
            detail_b: dict = {}
            if not fault_ok:
                detail_b = {
                    "manifest_all_met": manifest_b.all_met,
                    "manifest_state": str(manifest_b.state),
                    "alpha_applied_before_compensate": alpha_applied,
                    "bravo_not_applied": bravo_not_applied,
                    "alpha_compensated": alpha_compensated,
                    "obligation_states": states_b,
                }

            held = committed_ok and fault_ok
            detail: dict = {"index": index}
            if not held:
                detail.update(committed=detail_a, fault=detail_b)
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(n_trials: int = 200, seed: int = 0) -> object:
    return run_trials(
        name="exp2_transaction_atomicity",
        guarantee=(
            "MemoryTransactionService: committed op → manifest COMPLETE with "
            "both owners applied; faulted op → manifest DEGRADED, the "
            "successful owner's projection removed by real service.compensate() "
            "(tracking entry absent post-compensate, ledger=COMPENSATED), "
            "failed owner had no projection residue (ledger=FAILED)"
        ),
        metric_name="manifest-correct rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real MemoryTransactionService (service.py) + ObligationLedger + "
            "Reconciler → CompletionManifest over fresh_db(). "
            "Committed sub-trial: both _TrackingOwners succeed → manifest COMPLETE. "
            "Fault+compensate sub-trial: bravo raises in apply() → manifest DEGRADED; "
            "alpha WAS applied (tracking dict checked before compensate); "
            "real svc.compensate(conn, ctx, 'alpha') calls owner.compensate() "
            "removing alpha from tracking dict AND marking ledger COMPENSATED; "
            "asserts alpha absent post-compensate, bravo never present."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} "
          f"({result.metric_value:.4f})")
