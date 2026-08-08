"""Experiment 1 — Verified erasure via the real V4 production projection owners.

Drives the REAL production Bm25Owner, TemporalOwner, and VectorOwner from
concrete_owners.py (NOT a synthetic _FactsOwner) through ErasureService.
Each owner is constructed with only a DatabaseManager instance — no live
retrieval engine or vector_store because sqlite-vec is not loadable in this
environment.

Stores seeded and scanned per owner:
  • bm25_tokens (Bm25Owner) — seeded with real tokenize() output;
    erased via delete_bm25_tokens_for_fact(); residue-scanned by
    _physical_present() which queries bm25_tokens by profile_id + fact_id.
  • fact_temporal_validity (TemporalOwner) — seeded with valid_from;
    erased via delete_temporal_validity(); residue-scanned by
    _physical_present() which queries fact_temporal_validity by
    profile_id + fact_id.
  • embedding_metadata (VectorOwner, no sqlite-vec) — seeded directly via
    SQL INSERT; erased via DELETE FROM embedding_metadata WHERE fact_id = ?
    (the _remove() branch when vector_store is None); residue-scanned by a
    direct query of embedding_metadata.

sqlite-vec / ANN index status: NOT LOADABLE in this environment.
  VectorOwner is included with vector_store=None, ann_index=None.
  _store_available() returns False, so the healing/fingerprint path skips ANN
  lookups.  _remove() executes the SQL DELETE from embedding_metadata, which is
  the table this experiment seeds and residue-scans. The guarantee scope is
  therefore "embedding_metadata (SQL table)"; the in-process ANN index and the
  vector_row_map mapping are not exercised without sqlite-vec.

Six independent evidence layers (all must hold):
  1. receipt.all_erased == True and state == ErasureState.COMPLETE
  2. Every ErasureProofRecord.erased == True and .residue == ()
  3. Direct scan: zero rows in bm25_tokens, fact_temporal_validity, and
     embedding_metadata for wipe-tenant fact_ids after erasure
  4. projection_tombstones row written for every erased fact_id
  5. erasure_receipts row persisted; audit_hash re-verified via verify_receipt()
  6. Keep-tenant content-hash unchanged before and after wipe (full-row hash
     across atomic_facts, bm25_tokens, fact_temporal_validity, and
     embedding_metadata — detects row tamper, not merely row presence)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from _harness import TempWorkspace, TrialOutcome, add_profile, fresh_db, run_trials

# ---------------------------------------------------------------------------
# Seed helper — seeds one fact across all three projection stores
# ---------------------------------------------------------------------------


def _seed_projection_facts(db: object, pid: str) -> list[str]:
    """Seed one fact with real projection rows in bm25_tokens, fact_temporal_validity,
    and embedding_metadata.  Returns the list of seeded fact_ids.

    bm25_tokens is seeded with the output of the real tokenize() function so that
    Bm25Owner._fingerprints() would recognise the row.  Erasure does not require
    a valid fingerprint (it iterates context.fact_ids unconditionally via
    _delete_all()), but seeding with real tokens avoids a trivial-erasure trap
    where the owner 'erases' rows it would never have seen on verify.
    """
    from superlocalmemory.retrieval.bm25_channel import tokenize

    mid = f"m_{pid}_{uuid.uuid4().hex[:10]}"
    fid = f"f_{pid}_{uuid.uuid4().hex[:10]}"
    content = f"erasure projection witness {uuid.uuid4().hex[:8]}"

    # memories (needed by atomic_facts FK and by Bm25Owner._fact_content)
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        "speaker, role, created_at, metadata_json, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, pid, content, "s1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    # atomic_facts
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "lifecycle, created_at, scope) VALUES (?,?,?,?,?,?,?)",
        (fid, mid, pid, content, "active", "2026-01-01T00:00:00Z", "personal"),
    )
    # bm25_tokens — real tokenization so Bm25Owner._fingerprints() recognises it
    tokens = sorted(tokenize(content))
    db.execute(
        "INSERT INTO bm25_tokens (fact_id, profile_id, tokens) VALUES (?,?,?)",
        (fid, pid, json.dumps(tokens)),
    )
    # fact_temporal_validity
    db.execute(
        "INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
        "VALUES (?,?,?)",
        (fid, pid, "2026-01-01T00:00:00Z"),
    )
    # embedding_metadata — seeded directly; VectorOwner erases via SQL DELETE
    # (no sqlite-vec; vector_store=None branch in _remove())
    db.execute(
        "INSERT INTO embedding_metadata "
        "(fact_id, profile_id, model_name, dimension, created_at) "
        "VALUES (?,?,?,?,?)",
        (fid, pid, "stub", 0, "2026-01-01T00:00:00Z"),
    )
    return [fid]


# ---------------------------------------------------------------------------
# Keep-tenant content hash (layer 6: tamper detection, not just row count)
# ---------------------------------------------------------------------------


def _tenant_content_hash(db: object, pid: str, fact_ids: tuple[str, ...]) -> str:
    """SHA-256 over deterministic serialisation of all projection rows for
    pid + fact_ids across atomic_facts, bm25_tokens, fact_temporal_validity,
    and embedding_metadata.  Identical before and after wipe-tenant erasure
    proves the keep-tenant rows were untouched (not merely still present).
    """
    ph = ",".join("?" for _ in fact_ids)
    segments: list[str] = []
    table_cols = [
        ("atomic_facts", "fact_id, content, lifecycle, created_at"),
        ("bm25_tokens", "fact_id, tokens"),
        ("fact_temporal_validity", "fact_id, valid_from, valid_until"),
        ("embedding_metadata", "fact_id, model_name, dimension"),
    ]
    for table, cols in table_cols:
        rows = db.execute(
            f"SELECT {cols} FROM {table} "
            f"WHERE profile_id = ? AND fact_id IN ({ph}) ORDER BY fact_id",
            (pid, *fact_ids),
        )
        rows_list = [dict(r) for r in rows]
        segments.append(f"{table}:{json.dumps(rows_list, sort_keys=True)}")
    payload = "\n".join(segments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _projection_present(db: object, pid: str, fact_ids: tuple[str, ...]) -> bool:
    """True only if every fact_id has a row in ALL three projection stores.

    Guards against a vacuous pass: if seeding silently failed to populate a
    store, a no-op erasure would still show zero residue. Requiring presence
    BEFORE erasure means the erasure has something real to remove.
    """
    for table in ("bm25_tokens", "fact_temporal_validity", "embedding_metadata"):
        for fid in fact_ids:
            rows = db.execute(
                f"SELECT 1 FROM {table} WHERE fact_id = ? AND profile_id = ?",
                (fid, pid),
            )
            if not rows:
                return False
    return True


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def _trial(index: int) -> TrialOutcome:
    from superlocalmemory.core.transactions.concrete_owners import (
        Bm25Owner,
        TemporalOwner,
        VectorOwner,
    )
    from superlocalmemory.core.transactions.erasure import (
        ErasureService,
        ErasureState,
        verify_receipt,
    )
    from superlocalmemory.core.transactions.owners import OperationContext

    with TempWorkspace() as ws:
        db = fresh_db(ws)
        try:
            wipe = f"wipe_{uuid.uuid4().hex[:8]}"
            keep = f"keep_{uuid.uuid4().hex[:8]}"
            add_profile(db, wipe)
            add_profile(db, keep)
            wipe_facts = tuple(_seed_projection_facts(db, wipe))
            keep_facts = tuple(_seed_projection_facts(db, keep))

            # Layer 0: the seeded rows must actually exist in every projection
            # store BEFORE erasure, so a no-op erasure cannot pass vacuously.
            seeded_present = _projection_present(db, wipe, wipe_facts)

            # Hash keep-tenant BEFORE erasure (layer 6)
            keep_hash_before = _tenant_content_hash(db, keep, keep_facts)

            # --- Real production owners from concrete_owners.py ---
            # Bm25Owner(db): erases from bm25_tokens via delete_bm25_tokens_for_fact()
            # TemporalOwner(db): erases from fact_temporal_validity via delete_temporal_validity()
            # VectorOwner(db, vector_store=None, ann_index=None):
            #   erases from embedding_metadata via SQL DELETE (no sqlite-vec branch)
            owners = {
                "bm25": Bm25Owner(db),
                "temporal": TemporalOwner(db),
                "vector": VectorOwner(db, vector_store=None, ann_index=None),
            }
            svc = ErasureService(owners)

            op_id = f"op_{uuid.uuid4().hex}"
            context = OperationContext(
                operation_id=op_id,
                profile_id=wipe,
                subject_id=wipe,
                fact_ids=wipe_facts,
            )
            receipt = svc.erase(
                db, context,
                subject_type="profile",
                subject_id=wipe,
                requested_by="exp1_test",
            )

            # --- Layer 1: receipt flags ---
            receipt_ok = receipt.all_erased and receipt.state == ErasureState.COMPLETE

            # --- Layer 2: per-proof guarantees (one proof per owner) ---
            proof_erased = all(p.erased for p in receipt.proofs)
            proof_no_residue = all(p.residue == () for p in receipt.proofs)

            # --- Layer 3: direct scan of all three projection tables ---
            wipe_residue_bm25 = 0
            wipe_residue_temporal = 0
            wipe_residue_embedding = 0
            for fid in wipe_facts:
                rows = db.execute(
                    "SELECT COUNT(*) AS c FROM bm25_tokens "
                    "WHERE profile_id = ? AND fact_id = ?",
                    (wipe, fid),
                )
                wipe_residue_bm25 += int(dict(rows[0])["c"])

                rows = db.execute(
                    "SELECT COUNT(*) AS c FROM fact_temporal_validity "
                    "WHERE profile_id = ? AND fact_id = ?",
                    (wipe, fid),
                )
                wipe_residue_temporal += int(dict(rows[0])["c"])

                rows = db.execute(
                    "SELECT COUNT(*) AS c FROM embedding_metadata "
                    "WHERE profile_id = ? AND fact_id = ?",
                    (wipe, fid),
                )
                wipe_residue_embedding += int(dict(rows[0])["c"])

            wipe_residue_total = (
                wipe_residue_bm25 + wipe_residue_temporal + wipe_residue_embedding
            )

            # --- Layer 4: tombstones written for every erased fact ---
            tombstone_count = 0
            hash_ok = False
            with db.raw_connection() as conn:
                for fid in wipe_facts:
                    row = conn.execute(
                        "SELECT 1 FROM projection_tombstones "
                        "WHERE profile_id = ? AND fact_id = ?",
                        (wipe, fid),
                    ).fetchone()
                    if row is not None:
                        tombstone_count += 1

                # --- Layer 5: audit hash re-verifies against persisted receipt ---
                hash_ok = verify_receipt(conn, op_id, profile_id=wipe)

            # --- Layer 6: keep-tenant content-hash unchanged ---
            keep_hash_after = _tenant_content_hash(db, keep, keep_facts)
            keep_intact = keep_hash_before == keep_hash_after

            held = (
                seeded_present
                and receipt_ok
                and proof_erased
                and proof_no_residue
                and wipe_residue_total == 0
                and tombstone_count == len(wipe_facts)
                and hash_ok
                and keep_intact
            )
            detail: dict = {"index": index}
            if not held:
                detail.update(
                    seeded_present=seeded_present,
                    receipt_ok=receipt_ok,
                    proof_erased=proof_erased,
                    proof_no_residue=proof_no_residue,
                    wipe_residue_bm25=wipe_residue_bm25,
                    wipe_residue_temporal=wipe_residue_temporal,
                    wipe_residue_embedding=wipe_residue_embedding,
                    tombstone_count=tombstone_count,
                    expected_tombstones=len(wipe_facts),
                    hash_ok=hash_ok,
                    keep_intact=keep_intact,
                    keep_hash_before=keep_hash_before,
                    keep_hash_after=keep_hash_after,
                )
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(n_trials: int = 200, seed: int = 0) -> object:
    return run_trials(
        name="exp1_erasure_completeness",
        guarantee=(
            "Real Bm25Owner + TemporalOwner + VectorOwner (no sqlite-vec: "
            "embedding_metadata SQL table only, no ANN index) erase all "
            "wipe-tenant projection rows from bm25_tokens, "
            "fact_temporal_validity, and embedding_metadata; tombstones and a "
            "verifiable receipt are persisted; keep-tenant content-hash is "
            "identical before and after wipe (full-row tamper detection)"
        ),
        metric_name="complete-erasure rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real production Bm25Owner(db), TemporalOwner(db), and "
            "VectorOwner(db, vector_store=None, ann_index=None) from "
            "concrete_owners.py. Each table seeded with real projection rows "
            "(bm25_tokens via tokenize(), fact_temporal_validity, "
            "embedding_metadata). Six evidence layers: receipt flags, "
            "per-proof erased+residue, direct scan of bm25_tokens + "
            "fact_temporal_validity + embedding_metadata, tombstone rows, "
            "audit_hash via verify_receipt(), keep-tenant full-row "
            "content-hash equality. All six must hold."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} held "
          f"({result.metric_value:.4f})")
