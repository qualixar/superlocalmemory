# Next-release plan

> Historical recommendation. Release sequencing is superseded by [08-V3.7-MASTER-PLAN.md](08-V3.7-MASTER-PLAN.md): one public `3.7.0`, developed through internal alpha/beta/RC gates, with every registered defect closed before promotion.

## Release sequence

### Option 1 — one large V3.7 release

This minimizes release ceremony but combines data-path fixes, security boundaries, scoring changes, documentation corrections, packaging, and benchmarks into one high-risk cut. Rollback and root-cause isolation would be poor.

### Option 2 — corrective `3.6.24`, then V3.7 Integrity

This restores data integrity and public truth first, then changes retrieval architecture and launches benchmark evidence. It costs one additional release cycle but reduces blast radius, provides an upgrade checkpoint, and creates a reversible path.

### Option 3 — stop V3 and rebuild V4

This is the most expensive and least reversible choice. It delays revenue, discards a substantial test base, and solves architecture aesthetics before validating the market wedge.

## L99 recommendation

Ship **Option 2**. Do not merge product-code changes until the implementation scope is approved. This document is the proposed change plan, not approval to execute it.

## Change-control rule

Before editing any function/class/method, run GitNexus upstream impact analysis and record direct callers, affected execution flows, and risk. The following roots are expected to be `HIGH` or `CRITICAL` blast radius:

- `MemoryEngine.store` and `MemoryEngine.store_fast`
- the daemon `/remember` handler and materializer loop
- semantic, BM25, temporal, fusion, and final recall-result builders
- daemon discovery/health client functions
- context injection renderer and `session_init`
- cache manager construction

Split changes by contract, not file. Do not mix public-copy corrections with data migrations or retrieval scoring in the same commit.

## Release `3.6.24` — corrective integrity patch

### Workstream A — daemon ownership and test isolation

1. Introduce a daemon identity record containing instance UUID, canonical data-root hash, owner UID/SID, protocol version, package version, start nonce, and local capability identifier.
2. Return the non-secret identity fields from health/readiness; require the capability on operational routes.
3. Remove blind fixed-port adoption. A client may discover a process but cannot trust it until identity matches its requested home/data root and owner.
4. Treat legacy-port redirection as migration advice, not identity.
5. Give tests a test-owned port, data root, PID, capability, and process group. Reject real user paths.

Required tests: foreign daemon on 8765, malicious health responder, 8767 redirect, two homes, two concurrent versions, stale PID, PID reuse, data-root symlink, Windows owner model, crash/restart, and slow-suite live-data guard.

### Workstream B — canonical ingestion state machine

1. Define persisted states: `raw`, `queryable`, `enriching`, `complete`, and `failed`, with derivation version and retry metadata.
2. Make every entry point call one command service. `wait=true` controls response timing, not semantics.
3. Persist immutable raw evidence plus pending state atomically. Provide read-your-write through an explicitly partial record.
4. Make enrichment idempotent by fact ID, content/source hash, derivation version, and stage.
5. A duplicate may reuse completed derived state; it may not cause an incomplete row to be marked complete.
6. Reconcile dense, BM25, temporal, graph, provenance, trust, consolidation, and optional backend branches. Surface partial failures in health and diagnostics.
7. Backfill existing facts whose materialization state is missing or inconsistent.

Required tests: MCP/HTTP/CLI/hook parity, sync/async parity, duplicate during enrichment, crash at every stage, retry, concurrent duplicate, update during enrichment, delete during enrichment, invalid content, trust denial, migration backfill, and read-your-write latency.

### Workstream C — retrieval scope correctness

1. Create a single scope predicate/temporary allowed-ID set used by dense, FTS/BM25, temporal, graph, and associative candidate generation.
2. Apply tenant/profile/project/visibility/deletion/lifecycle policy at candidate origin, not only after top-K.
3. Test both inclusion and exclusion. A shared/global fact under another profile must be found; private facts must not leak.
4. Add a per-result trace showing the effective scope and source owner without exposing unrelated identifiers.

Required tests: personal/project/shared/global across profiles, tenants, users, deleted/archived facts, all channels, full recall, cache, and injection.

### Workstream D — context, identity, and trust boundary

1. Make one safe renderer mandatory for every injection surface.
2. Delimit memory as untrusted evidence, preserve source/provenance, strip or neutralize instruction semantics, and enforce a content/token budget.
3. Derive actor identity from authenticated daemon/MCP capability, never from a caller-selected `agent_id` alone.
4. Run trust/write policy before both raw persistence and enrichment.
5. Authenticate loopback mesh state and prohibit secret-storage examples until a formal encrypted-secret feature exists.

Required tests: stored system-prompt attacks, delimiter breaking, indirect injection, tool-call coercion, Unicode/encoding variants, oversized memory, provenance tampering, unknown actor, local hostile process, and cross-tenant injection.

### Workstream E — packaging, migration, and release truth

1. Remove npm’s `--break-system-packages` fallback. Make Python/model/hook setup explicit and consented.
2. Define and test Node 18+/npm 9+ and Python 3.11–3.14 prerequisites.
3. Make readiness fail when required migrations fail; make optional subsystem degradation explicit.
4. Normalize the AGPL/commercial-license/third-party policy across metadata and headers.
5. Generate README tool profiles/counts, integration lists, and CLI mode text from source contracts.
6. Correct all website/README/wiki/CLI/package benchmark and paper claims before package publication.
7. Publish from CI with trusted publishing, SBOM, provenance attestations, checksums, GitHub release, changelog, and matching source tag.

Required tests: clean install and uninstall, no-system-mutation assertion, fresh DB, every supported upgrade, interrupted migration, rollback, wheel/sdist/npm content parity, MCP first-run stdout purity, and package signature verification.

## Release `3.7.0` — Memory Integrity relaunch

### Retrieval architecture

1. Define a typed candidate interface used by independent dense, lexical, graph, temporal, and associative generators.
2. Let every enabled channel introduce candidates before fusion.
3. Preserve raw channel scores/ranks and declare the RRF/fusion formula.
4. Make reranker state explicit and deterministic under cold start.
5. Replace double sigmoids and confidence saturation with calibrated relevance plus an abstention threshold.
6. Make fast/fallback semantic paths rank-equivalent and publish Fisher/cosine contribution separately.
7. Separate “retrieval confidence,” “memory trust,” and “answer confidence”; they are different variables.

### Source truth and temporal integrity

1. Preserve immutable evidence spans and derivation versions.
2. Add valid-time and transaction-time fields plus explicit supersession/contradiction links.
3. Provide reviewable Markdown/JSON views, diffs, rollback, export/import, and deterministic derived-index rebuild.
4. Unify lifecycle state and prove archive/delete propagation through every index, cache, backend, and export.

### Optimize product

1. Either wire a real semantic cache into production with embeddings, calibrated thresholds, invalidation, tenant isolation, and a false-hit corpus, or remove the feature from the release.
2. Benchmark exact cache under concurrency, eviction, restart, encryption, corrupted entries, and provider-specific keying.
3. Benchmark compression per content class and surface for fidelity, reduction, latency, provider-cache interaction, and cost. Never aggregate safe and lossy modes.

### Mesh/product portability

For V3.7, market a **local coordination mesh** unless the following distributed-memory contract exists: authenticated membership, more than one peer, change log, version vectors or another declared ordering model, conflict policy, offline replay, idempotent update/delete, forget propagation, backup/restore, and partition tests. A coordination feature is valuable; calling it memory replication when it is not is not valuable.

### Benchmark and proof publication

Execute the three-track LoCoMo protocol in the competitor audit and the additional benchmark gates. Publish an SLM cross-IDE suite and an Optimize suite with raw artifacts. Build one public evidence page that links source commit, package digest, benchmark manifest, test reports, SBOM, release attestation, and known limitations.

## Cross-client integration matrix

Every named integration must pass the same contract in CI or a documented reproducible lab run:

1. Install without overwriting user configuration.
2. Start and validate the intended daemon identity.
3. Open a session and inject bounded safe context.
4. Remember an attributed fact.
5. Recall it in the same client.
6. Recall it in a second client against the same approved profile/project.
7. Update it and observe supersession rather than duplication.
8. Forget it and verify removal from dense, lexical, graph, temporal, cache, context, export, and backup views.
9. Survive client restart, daemon restart, offline period, and upgrade.
10. Disable/uninstall without deleting user-owned memory or unrelated config.

Only clients passing all ten steps count toward the public integration number.

## Release gates and artifacts

| Gate | Required result | Published artifact |
|---|---|---|
| Source suite | All default tests green | JUnit + environment manifest |
| Slow/integration | No failures and no live-data access | JUnit + isolation attestation |
| Security | Daemon, injection, tenant, mesh, installer tests green | Threat model + test report |
| Migration | Fresh/upgrade/interrupt/rollback matrix green | Migration report + fixtures |
| Packaging | Wheel, sdist, npm, clean installs, SBOM, checksums agree | Digests + provenance |
| Retrieval | Scope, channel, calibration, cold/warm, parity tests green | Trace samples + metrics |
| Cross-client | Every advertised client passes ten-step contract | Generated integration matrix |
| Benchmarks | Frozen protocol, full raw outputs, human audit | Manifest + JSONL + report |
| Public claims | Every number/entity has a live entailing source | Claim ledger generated from final text |
| Recovery | Backup/restore, corruption, crash, disk-full, and rollback green | Recovery runbook + report |

## Commands to rerun after implementation

Use test-owned environment variables and a random non-user port/data root for all integration commands.

```bash
python -m pytest -q
python -m pytest -q -m "slow and not ollama"
python -m pytest -q tests/test_benchmarks
python -m build
python -m twine check dist/*
npm pack --dry-run
```

Then install the built artifacts—not the source tree—into clean Python/Node environments on Linux, macOS, and Windows, run `pip check`, smoke the CLI/MCP/daemon, and verify package contents and digests. Do not call a release complete from source tests alone.

## Approval boundary

The next authorized execution step should be `3.6.24` Workstreams A–E only. V3.7 architecture work starts after the corrective release is measured in real use. This prevents a broad refactor from hiding whether the P0 fixes actually restored the contract.
