# SLM 4.0 — Reliability Evaluation Harness

Direct, reproducible measurements of SuperLocalMemory 4.0's durability,
isolation, and temporal-memory guarantees. Every experiment drives the
**installed `superlocalmemory` package** through its **real** code paths — real
SQLite databases built with the same schema and migration chain the engine runs
in production, the real backup coordinator, the real migration runner, the real
scope/authorization layer, and the real temporal machinery.

## What is real, and what is lightweight

The spine **services** are exercised with no mocking (the projection owners
vary by experiment, as noted below): ErasureService, MemoryTransactionService,
OperationPolicyRegistry, and CanonicalRememberRuntime are imported directly
from the installed package and run against real on-disk SQLite databases.

Projection-owner implementations vary by experiment:

- **exp1** uses the real production `Bm25Owner`, `TemporalOwner`, and
  `VectorOwner` from `concrete_owners.py`, seeded with real projection rows
  (bm25_tokens via `tokenize()`, fact_temporal_validity, embedding_metadata).
  `sqlite-vec` is not loadable in this environment, so `VectorOwner` runs with
  `vector_store=None`; it erases and residue-scans `embedding_metadata` only
  (the SQL path). The ANN index and the `vector_row_map` mapping are not
  exercised without `sqlite-vec`.

- **exp2** uses lightweight `_TrackingOwner` instances — a minimal but complete
  implementation of the `ProjectionOwner` protocol backed by an in-memory dict.
  This is sufficient to exercise the full service obligation-recording, apply,
  reconcile, and compensate paths against a real SQLite database, including the
  REAL `service.compensate()` call whose ledger writes are verified.

- **exp7** sets `runtime._generation` directly to model a post-rebind epoch
  without exercising the full `rebind_engine` path.  The fence behaviour under
  an epoch mismatch is what the experiment covers.

The only other synthetic input is a deterministic embedding in experiments where
a vector is structurally required; embedding quality is never the property under
measurement (plumbing correctness is).

## Why this exists

The paper's reliability section reports *measured* behaviour, not asserted
behaviour. This harness produces those numbers as JSON + a summary table so the
results are reproducible by a reviewer with nothing but the published package.

## Design principles

- **Real package, real paths.** Databases are constructed exactly as
  `MemoryEngine._init_db_layer` builds them (base schema → v3.4.3/4.6/4.7
  extensions → full forward + deferred migrations). The only synthetic input is
  a deterministic vector where a vector is *structurally* required; embedding
  quality is never the property under measurement.
- **Fail loud.** A broken harness crashes the run — it never silently fabricates
  a pass. Only the *specific* exception a guarantee promises to raise is caught.
- **Bracketed guarantees.** Each experiment fails if the mechanism does nothing
  *or* does the wrong thing. Isolation carries an explicit **positive control**
  (the requester must still see its *own* data through the same read path), so a
  zero-leak verdict cannot be an artifact of an empty query.
- **Deterministic + immutable.** Frozen result types; per-trial seeds; N
  independent trials per guarantee.

## The experiments

| File | Guarantee measured |
|---|---|
| `exp1_erasure_completeness.py` | Real Bm25Owner + TemporalOwner + VectorOwner erase all projection rows (bm25_tokens, fact_temporal_validity, embedding_metadata); receipt + tombstones verified; keep-tenant content-hash unchanged |
| `exp2_transaction_atomicity.py` | Committed op → manifest COMPLETE; faulted op → manifest DEGRADED, real service.compensate() removes successful owner's projection |
| `exp3_migration_downgrade.py` | Newer-stamped DB refused on the deferred pass with zero mutation |
| `exp4_backup_restore_atomicity.py` | Partial-restore failure rolls live data back to pre-restore bytes |
| `exp5_multitenant_isolation.py` | Personal rows never leak across tenants on any read path (with positive control) |
| `exp6_temporal_micro_eval.py` | 6a superseded-fact demotion · 6b recency-decay monotonicity · 6c time-window inference |
| `exp7_generation_fence.py` | Stale-epoch ADMISSION rejected; fresh-epoch ADMISSION from runtime.remember() committed (_generation set directly, not via full rebind_engine path) |
| `exp8_policy_registry.py` | REMEMBER allowed (reason=='allow') for OWNER/ADMIN/MEMBER; VIEWER→REMEMBER denied with insufficient_roles; unknown kind fail-open/closed; unauthenticated actor denied with authentication_required |

## Running

```bash
# From the product checkout, using the project venv (Python 3.13):
cd .backup/v4-production-audit/paper-and-research/experiments
../../../../.venv/bin/python run_all.py 200      # 200 trials per guarantee
```

Outputs land in `results/`: one JSON per experiment plus `SUMMARY.md`. The
runner exits non-zero if any guarantee failed a trial.

## Environment

Run on **Python 3.13** (the product's supported test interpreter). Each JSON
records the exact package version, interpreter, platform, and timestamp under an
`environment` block for provenance.

## Honesty notes (carried into the paper)

- This harness measures **fault-injection reliability guarantees** and
  **temporal-memory behaviour on SLM's own machinery**. It is *not* an external
  agent-task benchmark; agent-level temporal accuracy against a third-party suite
  is explicitly deferred to future work.
- 6b measures the shipped `EbbinghausCurve` decay function's mathematical
  behaviour (monotonic, bounded); it does not claim an end-to-end forgetting
  outcome.
- exp1 VectorOwner scope: embedding_metadata SQL table only — the in-process ANN
  index is excluded because sqlite-vec is not loadable in this environment.
- exp2 _TrackingOwner: a lightweight but complete ProjectionOwner implementation,
  not a mock. The guarantee covers the service, ledger, and reconciler code
  paths; it does not claim a specific production projection store was exercised.
