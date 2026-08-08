"""Experiment 2b — Manifest correctness + compensate with the REAL projection owners.

Reviewers flagged that exp2's manifest/compensate claim uses synthetic _TrackingOwner
stand-ins.  This experiment replaces them with the real production owners from
concrete_owners.py:

  • Bm25Owner(db)                           — bm25_tokens table
  • TemporalOwner(db)                        — fact_temporal_validity table
  • VectorOwner(db, vector_store=None, ann_index=None) — embedding_metadata table
                                               (no sqlite-vec; SQL-only path)

sqlite-vec / ANN index limitation:
  VectorOwner._required() returns set() when vector_store is None (no embedded
  facts exist with a live store).  Because _required() is empty, apply() is never
  called and verify() returns ok=True with not_applicable=True.  embedding_metadata
  rows are seeded in trial type A to verify physical presence; VectorOwner.compensate
  removes them via the SQL DELETE branch in _remove().  ANN index coverage is
  explicitly OUT OF SCOPE for this harness; the paper scope is stated accordingly.

DB setup:
  Same fresh_db() as exp1 — full production schema including projection_obligations
  (M033 required by obligation ledger).

TWO TRIAL TYPES per trial (200 total: 100 type A + 100 type B):

  Trial type A — HAPPY PATH (all three real owners succeed):
    1. Seed memories, atomic_facts, bm25_tokens (real tokenize() output),
       fact_temporal_validity, embedding_metadata.
    2. record() APPLY obligations for bm25, temporal, vector.
    3. run() → verify each owner → all VERIFIED → manifest.state == COMPLETE.
    4. Assert physical presence in all three projection tables.

  Trial type B — FAULT + COMPENSATE (one owner naturally fails, two survive):
    Fault mechanism: Bm25Owner.apply() cannot insert BM25 tokens without a
    retrieval engine (retrieval=None → _heal() returns False without writing
    any new rows).  By NOT seeding bm25_tokens, Bm25Owner.verify() fails, then
    apply() fails (no rows written), → obligation state FAILED.
    TemporalOwner and VectorOwner succeed normally.

    1. Seed memories, atomic_facts, fact_temporal_validity ONLY (no bm25_tokens).
    2. record() APPLY obligations for bm25, temporal, vector.
    3. run() → Bm25 FAILED (no rows, no retrieval to heal), Temporal VERIFIED,
       Vector N/A (ok=True) → manifest.state == DEGRADED.
    4. Assert Bm25 left NO partial projection residue (bm25_tokens empty).
    5. Call svc.compensate(conn, ctx, "temporal") → removes fact_temporal_validity.
    6. Call svc.compensate(conn, ctx, "vector") → removes embedding_metadata
       (no rows to remove, but compensate must not raise).
    7. Assert fact_temporal_validity = 0 rows after compensate.
    8. Assert obligation states: temporal=COMPENSATED, bm25=FAILED, vector=VERIFIED/N/A.

CRIT (3 potential measurement biases):
  1. FAULT MECHANISM IS STRUCTURAL (NOT INJECTED): Bm25 fails because retrieval=None
     in the harness — the same condition that prevents heal(). This faithfully
     represents the isolated harness context but NOT production (where a real
     retrieval engine is wired). The paper must state: "fault scenario uses isolated
     Bm25Owner without retrieval engine; production Bm25 heals via live BM25 index."
  2. VECTOR OWNER SCOPE: VectorOwner operates in not_applicable mode (no store).
     The compensate call on vector removes nothing but does not raise. ANN index
     compensation is unverified. Paper scope: "embedding_metadata SQL table only."
  3. VACUOUS VECTOR SUCCESS: VectorOwner's ok=True in both A and B is structural
     (not_applicable), not a projection correctness signal. Physical presence check
     in trial A verifies embedding_metadata was seeded; trial B does not seed
     embedding_metadata so there is nothing to compensate — this is disclosed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SOURCE_ROOT = os.environ.get("SLM_SOURCE_ROOT")
if _SOURCE_ROOT:
    _source_path = Path(_SOURCE_ROOT).expanduser().resolve()
    _src_path = _source_path if _source_path.name == "src" else _source_path / "src"
    if not (_src_path / "superlocalmemory").is_dir():
        raise RuntimeError(
            "SLM_SOURCE_ROOT must point to a repository root or its src directory: "
            f"{_source_path}"
        )
    sys.path.insert(0, str(_src_path))

_EXP_DIR = str(Path(__file__).resolve().parent)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------

import superlocalmemory  # noqa: E402

_SLM_FILE = superlocalmemory.__file__
if _SOURCE_ROOT and not Path(_SLM_FILE).resolve().is_relative_to(_src_path):
    raise RuntimeError(f"superlocalmemory imported from unexpected location: {_SLM_FILE}")

from _harness import TempWorkspace, TrialOutcome, add_profile, fresh_db, run_trials  # noqa: E402

# ---------------------------------------------------------------------------
# DB seed helpers
# ---------------------------------------------------------------------------


def _seed_full(db, pid: str) -> tuple[str, str]:
    """Seed memories, atomic_facts, bm25_tokens, fact_temporal_validity,
    embedding_metadata.  Returns (memory_id, fact_id).

    All three projection tables are populated so the happy-path trial can
    verify physical presence before and after run().
    """
    from superlocalmemory.retrieval.bm25_channel import tokenize

    mid = f"m_{uuid.uuid4().hex[:12]}"
    fid = f"f_{uuid.uuid4().hex[:12]}"
    content = f"Real owner manifest witness {uuid.uuid4().hex[:8]} alpha bravo charlie"

    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        "speaker, role, created_at, metadata_json, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, pid, content, "s1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "lifecycle, created_at, scope) VALUES (?,?,?,?,?,?,?)",
        (fid, mid, pid, content, "active", "2026-01-01T00:00:00Z", "personal"),
    )
    tokens = sorted(tokenize(content))
    db.execute(
        "INSERT INTO bm25_tokens (fact_id, profile_id, tokens) VALUES (?,?,?)",
        (fid, pid, json.dumps(tokens)),
    )
    db.execute(
        "INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
        "VALUES (?,?,?)",
        (fid, pid, "2026-01-01T00:00:00Z"),
    )
    db.execute(
        "INSERT INTO embedding_metadata "
        "(fact_id, profile_id, model_name, dimension, created_at) "
        "VALUES (?,?,?,?,?)",
        (fid, pid, "stub", 0, "2026-01-01T00:00:00Z"),
    )
    return mid, fid


def _seed_no_bm25(db, pid: str) -> tuple[str, str]:
    """Seed memories, atomic_facts, fact_temporal_validity ONLY.
    NO bm25_tokens → Bm25Owner.verify() fails → fault scenario.
    NO embedding_metadata (vector is N/A anyway).
    Returns (memory_id, fact_id).
    """
    mid = f"m_{uuid.uuid4().hex[:12]}"
    fid = f"f_{uuid.uuid4().hex[:12]}"
    content = f"Fault injection witness {uuid.uuid4().hex[:8]} delta echo foxtrot"

    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        "speaker, role, created_at, metadata_json, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, pid, content, "s1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "lifecycle, created_at, scope) VALUES (?,?,?,?,?,?,?)",
        (fid, mid, pid, content, "active", "2026-01-01T00:00:00Z", "personal"),
    )
    db.execute(
        "INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
        "VALUES (?,?,?)",
        (fid, pid, "2026-01-01T00:00:00Z"),
    )
    return mid, fid


# ---------------------------------------------------------------------------
# Row count helpers (direct DB queries)
# ---------------------------------------------------------------------------


def _count_rows(db, table: str, profile_id: str, fact_id: str) -> int:
    rows = db.execute(
        f"SELECT COUNT(*) AS c FROM {table} "
        "WHERE profile_id = ? AND fact_id = ?",
        (profile_id, fact_id),
    )
    return int(dict(rows[0])["c"]) if rows else 0


# ---------------------------------------------------------------------------
# Trial A — happy path (all three real owners succeed)
# ---------------------------------------------------------------------------


def _trial_a(index: int) -> TrialOutcome:
    from superlocalmemory.core.transactions.concrete_owners import (
        Bm25Owner,
        TemporalOwner,
        VectorOwner,
    )
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
            pid = f"pa_{uuid.uuid4().hex[:8]}"
            add_profile(db, pid)
            _mid, fid = _seed_full(db, pid)

            # Pre-seeding check: all three tables must have rows before run()
            pre_bm25 = _count_rows(db, "bm25_tokens", pid, fid)
            pre_temporal = _count_rows(db, "fact_temporal_validity", pid, fid)
            pre_embed = _count_rows(db, "embedding_metadata", pid, fid)
            pre_ok = (pre_bm25 > 0 and pre_temporal > 0 and pre_embed > 0)

            bm25 = Bm25Owner(db)
            temporal = TemporalOwner(db)
            vector = VectorOwner(db, vector_store=None, ann_index=None)
            svc = MemoryTransactionService(
                {"bm25": bm25, "temporal": temporal, "vector": vector}
            )

            op_id = f"op_a_{uuid.uuid4().hex}"
            ctx = OperationContext(
                operation_id=op_id,
                profile_id=pid,
                subject_id=pid,
                fact_ids=(fid,),
            )

            with db.raw_connection() as conn:
                svc.record(conn, ctx, kind=ObligationKind.APPLY)
                manifest = svc.run(conn, ctx, canonical_committed=True)

            # Assertions
            manifest_complete = (
                manifest.all_met and manifest.state == ManifestState.COMPLETE
            )

            # Physical presence in all three projection tables
            post_bm25 = _count_rows(db, "bm25_tokens", pid, fid)
            post_temporal = _count_rows(db, "fact_temporal_validity", pid, fid)
            post_embed = _count_rows(db, "embedding_metadata", pid, fid)

            # bm25 and temporal must still be present (no erasure triggered)
            # embedding_metadata seeded but VectorOwner is N/A → still present
            physical_ok = (post_bm25 > 0 and post_temporal > 0 and post_embed > 0)

            # Obligation states
            with db.raw_connection() as conn_r:
                ledger = ObligationLedger()
                obligations = ledger.fetch(conn_r, op_id)
            states = {o.owner: o.state for o in obligations}

            # bm25 must be VERIFIED or APPLIED (seeded → verify succeeds directly)
            # temporal must be VERIFIED
            # vector must be VERIFIED (not_applicable)
            bm25_ok = states.get("bm25") in (
                ObligationState.VERIFIED, ObligationState.APPLIED
            )
            temporal_ok = states.get("temporal") in (
                ObligationState.VERIFIED, ObligationState.APPLIED
            )
            vector_ok = states.get("vector") in (
                ObligationState.VERIFIED, ObligationState.APPLIED
            )

            held = (
                pre_ok
                and manifest_complete
                and physical_ok
                and bm25_ok
                and temporal_ok
                and vector_ok
            )
            detail: dict = {"index": index, "trial_type": "A_happy_path"}
            if not held:
                detail.update(
                    pre_bm25=pre_bm25,
                    pre_temporal=pre_temporal,
                    pre_embed=pre_embed,
                    manifest_state=str(manifest.state),
                    manifest_all_met=manifest.all_met,
                    post_bm25=post_bm25,
                    post_temporal=post_temporal,
                    post_embed=post_embed,
                    states={k: str(v) for k, v in states.items()},
                )
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Trial B — fault + compensate (Bm25 fails, temporal compensated)
# ---------------------------------------------------------------------------


def _trial_b(index: int) -> TrialOutcome:
    from superlocalmemory.core.transactions.concrete_owners import (
        Bm25Owner,
        TemporalOwner,
        VectorOwner,
    )
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
            pid = f"pb_{uuid.uuid4().hex[:8]}"
            add_profile(db, pid)
            # No bm25_tokens seeded → Bm25Owner verify fails, apply cannot heal
            # (retrieval=None → _heal() returns False without writing any rows)
            _mid, fid = _seed_no_bm25(db, pid)

            # Pre-fault check: bm25_tokens must be absent, temporal must be present
            pre_bm25 = _count_rows(db, "bm25_tokens", pid, fid)
            pre_temporal = _count_rows(db, "fact_temporal_validity", pid, fid)
            pre_ok = (pre_bm25 == 0 and pre_temporal > 0)

            bm25 = Bm25Owner(db)            # retrieval=None → cannot heal
            temporal = TemporalOwner(db)
            vector = VectorOwner(db, vector_store=None, ann_index=None)
            svc = MemoryTransactionService(
                {"bm25": bm25, "temporal": temporal, "vector": vector}
            )

            op_id = f"op_b_{uuid.uuid4().hex}"
            ctx = OperationContext(
                operation_id=op_id,
                profile_id=pid,
                subject_id=pid,
                fact_ids=(fid,),
            )

            with db.raw_connection() as conn:
                svc.record(conn, ctx, kind=ObligationKind.APPLY)
                manifest = svc.run(conn, ctx, canonical_committed=True)

            # Assert DEGRADED (Bm25 FAILED → not all_met)
            manifest_degraded = (
                not manifest.all_met and manifest.state == ManifestState.DEGRADED
            )

            # Bm25 left NO partial residue (no rows inserted without retrieval)
            bm25_residue_before_compensate = _count_rows(db, "bm25_tokens", pid, fid)

            # Compensate the SURVIVING owners (temporal and vector)
            with db.raw_connection() as conn_c:
                svc.compensate(conn_c, ctx, "temporal")
                svc.compensate(conn_c, ctx, "vector")  # no-op (N/A), must not raise

            # Verify fact_temporal_validity has zero rows after compensate
            temporal_residue_after = _count_rows(db, "fact_temporal_validity", pid, fid)

            # Obligation states
            with db.raw_connection() as conn_r:
                ledger = ObligationLedger()
                obligations = ledger.fetch(conn_r, op_id)
            states = {o.owner: o.state for o in obligations}

            temporal_compensated = (
                states.get("temporal") == ObligationState.COMPENSATED
            )
            bm25_failed = states.get("bm25") == ObligationState.FAILED
            # vector: VERIFIED (not_applicable) or COMPENSATED (compensate called)
            vector_state_ok = states.get("vector") in (
                ObligationState.VERIFIED,
                ObligationState.APPLIED,
                ObligationState.COMPENSATED,
            )

            held = (
                pre_ok
                and manifest_degraded
                and bm25_residue_before_compensate == 0
                and temporal_residue_after == 0
                and temporal_compensated
                and bm25_failed
                and vector_state_ok
            )
            detail: dict = {"index": index, "trial_type": "B_fault_compensate"}
            if not held:
                detail.update(
                    pre_bm25=pre_bm25,
                    pre_temporal=pre_temporal,
                    manifest_state=str(manifest.state),
                    manifest_all_met=manifest.all_met,
                    bm25_residue_before_compensate=bm25_residue_before_compensate,
                    temporal_residue_after_compensate=temporal_residue_after,
                    states={k: str(v) for k, v in states.items()},
                )
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Combined trial dispatcher
# ---------------------------------------------------------------------------


def _trial(index: int) -> TrialOutcome:
    """Even-indexed trials run type A (happy path); odd run type B (fault)."""
    if index % 2 == 0:
        return _trial_a(index)
    return _trial_b(index)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _trial_distribution_str(n_trials: int) -> str:
    n_a = (n_trials + 1) // 2  # even indices: 0,2,4,...
    n_b = n_trials // 2  # odd indices: 1,3,5,...
    return f"{n_a} type-A (even index) + {n_b} type-B (odd index)"


def run(n_trials: int = 200, seed: int = 0) -> object:
    return run_trials(
        name="exp2b_real_owner_manifest",
        guarantee=(
            "Real Bm25Owner + TemporalOwner + VectorOwner (vector_store=None, "
            "ann_index=None) via MemoryTransactionService: "
            "(A) happy-path run() with all projections seeded → manifest COMPLETE, "
            "all three tables physically present; "
            "(B) Bm25 failure (no retrieval, no bm25_tokens) → manifest DEGRADED, "
            "Bm25 has zero partial residue, real svc.compensate('temporal') removes "
            "fact_temporal_validity rows to zero, obligation state COMPENSATED."
        ),
        metric_name="manifest-correct rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real production Bm25Owner(db), TemporalOwner(db), and "
            "VectorOwner(db, vector_store=None, ann_index=None) from concrete_owners.py. "
            "DB initialized via fresh_db() — full production schema (M033 included). "
            "Trial type A (happy path, n=100): memories + atomic_facts + bm25_tokens "
            "(real tokenize() output) + fact_temporal_validity + embedding_metadata "
            "seeded; record() then run() → all three owners VERIFIED → COMPLETE; "
            "physical row counts in all three projection tables verified. "
            "Trial type B (fault+compensate, n=100): memories + atomic_facts + "
            "fact_temporal_validity seeded; NO bm25_tokens — Bm25Owner.verify() fails, "
            "apply() calls _heal() which returns False (retrieval=None) without writing "
            "any rows → Bm25 FAILED; Temporal seeded → VERIFIED; "
            "Vector not_applicable (no store) → VERIFIED; manifest DEGRADED; "
            "bm25_tokens residue = 0 (apply never inserted); "
            "svc.compensate('temporal') → _FactScopedOwner._delete_all() calls "
            "db.delete_temporal_validity(fact_id) → fact_temporal_validity = 0; "
            "obligation states: temporal=COMPENSATED, bm25=FAILED, vector=VERIFIED. "
            "sqlite-vec ANN index: NOT EXERCISED (vector_store=None). "
            "Scope: embedding_metadata SQL table only for VectorOwner; "
            "ANN index compensation is explicitly out of scope for this harness."
        ),
        extra={
            "slm_module_file": _SLM_FILE,
            "trial_distribution": _trial_distribution_str(n_trials),
            "ann_scope": (
                "VectorOwner operates with vector_store=None. sqlite-vec ANN index "
                "is NOT loadable in this harness. VectorOwner coverage is limited to "
                "embedding_metadata SQL table (SQL DELETE branch in _remove()). "
                "ANN index compensation is out of scope; paper must state this."
            ),
            "bm25_fault_mechanism": (
                "Bm25Owner fails structurally (retrieval=None → _heal() returns "
                "False without writing rows). This faithfully represents the "
                "isolated harness but NOT production (where a live BM25 retrieval "
                "engine heals missing tokens). Paper must state: 'fault scenario "
                "uses isolated Bm25Owner without retrieval engine.'"
            ),
        },
    )


if __name__ == "__main__":
    import argparse

    from _harness import environment as _harness_env, write_result as _write_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200, help="number of trials")
    # Unified contract: --output-dir (DIR) is the canonical flag used by run_all.py.
    # --output (FILE) is retained for backward compatibility.
    parser.add_argument("--output", type=Path, default=None,
                        help="output FILE path (legacy, deprecated: use --output-dir)")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None,
                        help="output DIRECTORY (unified contract)")
    args = parser.parse_args()

    output_dir = args.output_dir

    result = run(n_trials=args.trials)

    if args.output is not None and output_dir is None:
        # Legacy FILE mode: write single payload file
        json_path = args.output.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "slm_module_file": _SLM_FILE,
            "result": result.to_dict(),
            "platform": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n=== EXP 2b REAL-OWNER MANIFEST + COMPENSATE ===")
        print(f"  Module: {_SLM_FILE}")
        print(f"  Trials: {result.trials}  Held: {result.held}  "
              f"Pass rate: {result.metric_value:.4f}")
        if result.failures:
            print(f"  FAILURES ({len(result.failures)}):")
            for f in result.failures[:5]:
                print(f"    {f}")
        else:
            print("  All trials PASSED")
        print(f"\n  Results written to {json_path}")
    elif output_dir is not None:
        # Unified DIRECTORY mode (also handles run_all.py invocation)
        out_dir = Path(output_dir).expanduser().resolve()
        json_path = _write_result(result, out_dir)
        print(f"\n=== EXP 2b REAL-OWNER MANIFEST + COMPENSATE ===")
        print(f"  Module: {_SLM_FILE}")
        print(f"  Trials: {result.trials}  Held: {result.held}  "
              f"Pass rate: {result.metric_value:.4f}")
        if result.failures:
            print(f"  FAILURES ({len(result.failures)}):")
            for f in result.failures[:5]:
                print(f"    {f}")
        else:
            print("  All trials PASSED")
        print(f"\n  Results written to {json_path}")
    else:
        # No output flag: default to harness write to ./results like other exps
        # (also supports bare `python exp2b_real_owner_manifest.py --trials N`)
        out_dir = Path(__file__).parent / "results"
        json_path = _write_result(result, out_dir)
        print(f"\n=== EXP 2b REAL-OWNER MANIFEST + COMPENSATE ===")
        print(f"  Module: {_SLM_FILE}")
        print(f"  Trials: {result.trials}  Held: {result.held}  "
              f"Pass rate: {result.metric_value:.4f}")
        if result.failures:
            print(f"  FAILURES ({len(result.failures)}):")
            for f in result.failures[:5]:
                print(f"    {f}")
        else:
            print("  All trials PASSED")
        print(f"\n  Results written to {json_path} (default output-dir)")
    # Legacy --output without --output-dir must remain valid; if neither flag
    # is given we still write to ./results for unified runner discovery.
