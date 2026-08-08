# SLM 4.0 — Reliability Evaluation Results

- Package version: **4.0.0**
- Python: 3.13.5
- Platform: macOS-26.4.1-arm64-arm-64bit-Mach-O
- Generated: 2026-08-02T18:00:04.653250+00:00

| Experiment | Guarantee | Metric | Trials | Held | Rate | Verdict |
|---|---|---|---:|---:|---:|:--:|
| exp1_erasure_completeness | Real Bm25Owner + TemporalOwner + VectorOwner (no sqlite-vec: embedding_metadata SQL table only, no ANN index) erase all wipe-tenant projection rows from bm25_tokens, fact_temporal_validity, and embedding_metadata; tombstones and a verifiable receipt are persisted; keep-tenant content-hash is identical before and after wipe (full-row tamper detection) | complete-erasure rate | 200 | 200 | 1.0000 | PASS |
| exp2_transaction_atomicity | MemoryTransactionService: committed op → manifest COMPLETE with both owners applied; faulted op → manifest DEGRADED, the successful owner's projection removed by real service.compensate() (tracking entry absent post-compensate, ledger=COMPENSATED), failed owner had no projection residue (ledger=FAILED) | manifest-correct rate | 200 | 200 | 1.0000 | PASS |
| exp3_migration_downgrade | newer-stamped DB refused on deferred pass with zero mutation | refuse-and-preserve rate | 200 | 200 | 1.0000 | PASS |
| exp4_backup_restore_atomicity | partial-restore failure rolls live data back to pre-restore bytes | rollback+clean rate | 200 | 200 | 1.0000 | PASS |
| exp5_multitenant_isolation | personal rows of one tenant never leak to another on any read path | zero-leak rate | 200 | 200 | 1.0000 | PASS |
| exp6a_superseded_demotion | superseded facts kept but demoted 0.25x and re-ranked below valid | correct-demotion rate | 200 | 200 | 1.0000 | PASS |
| exp6b_decay_monotonic | Ebbinghaus retention non-increasing in age, bounded [0,1] | monotonic-decay rate | 200 | 200 | 1.0000 | PASS |
| exp6c_time_window | date-proximate events outrank distant ones; horizon excludes far past | correct-window rate | 200 | 200 | 1.0000 | PASS |
| exp7_generation_fence | Generation fence rejects a stale-epoch ADMISSION (WriteCoordinatorError with ValueError __cause__ 'epoch is stale', writer never called) and admits a fresh-epoch ADMISSION from runtime.remember() in the same runtime | fence-correct rate | 200 | 200 | 1.0000 | PASS |
| exp8_policy_registry | _DEFAULT_REGISTRY.evaluate(): REMEMBER allowed with reason=='allow' for OWNER/ADMIN/MEMBER; VIEWER→REMEMBER denied with insufficient_roles; unknown kind fail-open in local mode (unknown_kind_allow_local), fail-closed in company mode (unknown_kind_deny_company); unauthenticated actor denied with authentication_required (auth check before role check) | policy-correct rate | 200 | 200 | 1.0000 | PASS |

**Aggregate: 2000/2000 trials upheld their guarantee (100.0000%).**
