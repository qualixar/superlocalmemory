# Architecture and shipped-truth audit

## Source and artifact lock

| Surface | Verified state |
|---|---|
| Source | `main@631c6af` |
| Release tag | `v3.6.23@104823d`; main contains two later test commits |
| Python package | PyPI `3.6.23`; installed in an isolated environment; `pip check` clean |
| npm package | npm `3.6.23`; tarball unpacked and inspected without running post-install |
| Registry/source parity | Published source payload matched current source except expected cache files and source-only release files |
| GitHub release | Latest release remained `v3.6.22` while both registries served `3.6.23` |
| Default source suite | `6025 passed, 46 skipped, 298 deselected` |
| Synthetic benchmark tests | `18 passed, 1 skipped` |
| Slow non-Ollama integration selection | `260 passed, 4 failed, 2 skipped` |

The default suite is green. That is not a release verdict because `pyproject.toml` deselects slow, Ollama, and benchmark tests and sets no meaningful coverage floor. Runtime contract probes found P0 failures not represented by the default suite.

## Intended ingestion contract

The product needs one seven-stage state machine:

1. **Boundary:** authenticate caller; bind tenant, profile, project, agent, source, visibility, and idempotency key.
2. **Durable envelope:** validate size/type; persist immutable raw evidence and a pending operation atomically.
3. **Extraction:** normalize content; extract atomic facts, entities, dates, relations, procedures, preferences, and project state.
4. **Resolution:** canonicalize entities; detect duplicates, conflicts, supersession, and temporal validity.
5. **Trust and provenance:** attach source spans, actor, timestamps, confidence, policy, and audit events.
6. **Index fan-out:** update dense, lexical, temporal, graph, associative, and optional acceleration indexes.
7. **Commit:** expose a read-your-write representation, mark all required derived work complete, publish observability, and reconcile failed branches.

An asynchronous implementation is valid only if the state is explicit (`raw`, `queryable`, `enriching`, `complete`, `failed`) and reconciliation is idempotent.

## Actual default ingestion path

`MemoryEngine.store()` delegates to the complete pipeline in [`store_pipeline.py`](../../../src/superlocalmemory/core/store_pipeline.py). The default daemon endpoint instead calls `store_fast()` in [`engine.py`](../../../src/superlocalmemory/core/engine.py). MCP `remember` posts to that endpoint without `wait=true`, and normal CLI remember uses the same asynchronous route.

`store_fast()` writes the raw fact, regex entities, embedding/Fisher metadata, vector state, and BM25 state. It skips the complete extractor, canonical resolution, graph construction, temporal processing, provenance, trust/audit hooks, consolidation, and optional graph-backend synchronization.

The background materializer in [`unified_daemon.py`](../../../src/superlocalmemory/server/unified_daemon.py) finds the fact just written by `store_fast()`, optionally fills a missing embedding, marks the pending row complete, and continues. Runtime instrumentation observed:

```text
pending_drained           true
engine_store_calls        0
store_fact_direct_calls   0
graph_builder_calls       0
canonical_entities        []
graph_edges               0
provenance_rows           0
embedding_present         true
```

**Severity: P0, RUNTIME-PROVEN.** The default product stores a searchable note, not the fully enriched memory object described by the product.

## Intended retrieval contract

The retrieval path should also be explicit:

1. **Boundary:** bind tenant/profile/project/scope and authorize the query.
2. **Planning:** classify intent, entities, time, contradiction, and required evidence type.
3. **Independent candidate generation:** dense, lexical, temporal, graph, associative, and other channels must each introduce candidates.
4. **Policy filters:** enforce visibility, trust, lifecycle, retention, deletion, and source rules before material reaches an LLM.
5. **Fusion and rerank:** combine independent ranks, rerank under a declared budget, and resolve conflicts without hiding alternatives.
6. **Calibration:** produce an interpretable relevance/confidence score and abstain when evidence is insufficient.
7. **Safe injection:** apply provenance, token budgets, untrusted-data delimiters, and feedback telemetry consistently across every client.

## Actual retrieval path

### Candidate generation

The active candidate producers are semantic, BM25, temporal, Hopfield, and spreading activation: five channels. Entity graph is a post-fusion score enhancer and cannot introduce a graph-only fact. Product surfaces alternately call this four-, six-, or seven-channel retrieval.

### Scope failure

Database helpers correctly include global/shared facts. Candidate generation does not:

- BM25 hard-codes the active profile.
- sqlite-vec semantic KNN searches only the active profile before the later scope-aware load.
- Temporal SQL hard-codes the active profile.
- Dependent associative paths inherit the restricted candidate set.

A runtime probe stored a global fact under another profile and a current-profile decoy. The database helper saw the global fact; semantic and full recall returned only the decoy, while BM25 and temporal returned nothing. **Severity: P0, RUNTIME-PROVEN.**

### Scoring and reranking

- Positive channel scores are normalized and then passed through another sigmoid in the recall pipeline. Confidence is computed as `min(1, score * 2)`, so current non-empty results collapse to confidence `1.0` in runtime output.
- The cross-encoder reranker executes only if its worker is already warm; cold-start recall silently skips it.
- The sqlite-vec fast path adds a Fisher score floor that the full-scan fallback does not, making rank behavior dependent on the selected execution path.
- Candidate generation is cosine-based. Fisher-derived terms modify scoring; the statement “not cosine” or “every recall uses Fisher-Rao” is false.

**Severity: P1, CODE- and RUNTIME-PROVEN.**

### Injection boundary

The common renderer supports an untrusted-memory wrapper, but main MCP `session_init` calls it with `wrap=False`. Other context paths use different truncation, redaction, and wrapping rules. Stored instructions can therefore enter the agent’s context without one canonical data/instruction boundary. **Severity: P0, CODE-PROVEN.**

## Cache audit

The exact cache is the strongest Optimize component. It has tenant-separated key construction, exact replay, invalidation, TTL handling, stampede control, and persisted metrics. The checked-in synthetic harness reports 50/50 exact hits, byte-identical replay, and zero false hits across 50 unseen and 50 one-character near-miss probes.

The semantic cache is not a production feature in the current wiring:

- The singleton manager installs a no-op semantic tier.
- No production caller was found for `set_semantic_tier()`.
- Manager lookup and index paths pass `embed=None`.
- The semantic implementation immediately no-ops without an embedding.
- The benchmark injects a stub semantic tier; it proves control-flow behavior, not production semantic matching.

Optimize, proxy, semantic cache, and compression default off. The website statement “exact-match cache on by default” conflicts with the shipped configuration. **Severity: P1, CODE-PROVEN.**

## Compression audit

The checked-in harness is a useful correctness test, not a production compression benchmark:

| Payload | Safe mode | Aggressive mode | What is proved |
|---|---:|---:|---|
| JSON sample | 0% | 0% | Parse-preserving/bypassed in this corpus |
| Code sample | 0% | 0% | Byte unchanged |
| Prose sample | 0.5% | 40.68% | Safe whitespace normalization; lossy LLMLingua reduction in aggressive mode |

Proxy compression disables the lossy second layer. The cache aligner detects eligible structures but does not perform alignment mutation. No current corpus supports “60–95% prompt compression” as a general product claim. **Severity: P1, RUNTIME- and CODE-PROVEN.**

## Mesh and portability audit

Current mesh coordinates peers, messages, state, locks, and events. It does not replicate memories between machines. Remote configuration supports one peer URL rather than a federated membership model. Conflict resolution, delete propagation, offline replay, multi-writer ordering, and deterministic memory handoff are not proven.

Loopback mesh state is unauthenticated, yet tool documentation suggests storing API keys in it. The mDNS implementation appears incompatible with the byte-address representation normally returned by Zeroconf; this remains code-proven risk rather than a runtime result.

The portable installer defines 12 named targets, but one Claude Code entry writes no configuration, leaving 11 actual writers. The legacy `slm connect --list` exposes seven targets. Fifteen templates exist in the tree, but templates are not successful integrations. “17+” is not supported by an executable matrix.

## Daemon identity and installation boundary

An isolated `HOME` with no daemon PID file can still connect to any process answering health checks on default port `8765` or legacy port `8767`. The CLI then writes that foreign PID/port into the isolated home. Setting a different daemon port does not prevent the fixed legacy-port probe. An isolated recall attached to the user’s real `3.6.19` daemon. No private memory content was retained in this audit.

This is more than test contamination: different homes, installations, projects, or users on the same host can cross an identity boundary. Health must return and the client must validate a daemon instance ID, data-root fingerprint, owner UID, protocol version, and authenticated capability. **Severity: P0, RUNTIME-PROVEN.**

The npm `postinstall.js` also violates a safe installer boundary. It attempts normal pip installation, then `--user`, then `--break-system-packages`; it also installs hooks, runs setup, and may trigger a large model download. A global npm install must not mutate a protected system Python as a last resort. The audit inspected but deliberately did not execute this post-install path.

## Optional backends and lifecycle

Cozo and Lance can be initialized and migrated, but they are not primary normal-path retrieval backends. Backend synchronization is reached from `run_store_fact_direct()` rather than the dominant full/default paths. Lance is not injected into the semantic channel, and Cozo entity search is not a main recall candidate generator.

Tiering updates one lifecycle field while forgetting filters consult another. Startup tier evaluation also defaults to one profile. These are correctness risks requiring focused archive/recall and multi-profile tests.

## Release and operational audit

- The `v3.6.23` tag’s initial CI failed; two post-tag test commits fixed main.
- Tag-triggered PyPI/npm workflows failed, but both registries received `3.6.23`, indicating a manual or separate publication path.
- GitHub’s latest release is still `v3.6.22`.
- `CHANGELOG.md` begins at `3.6.17` although `README.md` contains later release notes.
- A fresh isolated `slm health --json` returned success while emitting missing-learning-table migration failures on stderr.
- First isolated synchronous remember failed after worker timeouts; after model warmup, a second remember succeeded but the reranker timed out.
- The slow integration suite used a live fixed daemon URL and inserted a canary into the user database. The exact canary and pending row were removed and verified absent. This proves the suite lacks a hard test-data boundary.
- License identifiers and headers remain inconsistent across source, npm scripts, and documentation despite the root AGPL license.

## Architecture conclusion

The dominant failure pattern is **split-brain execution paths**: full and fast ingestion, scope-aware storage and scope-blind candidate generation, exact and semantic cache claims, source and optional backends, lifecycle fields, package and release state, and multiple context-injection renderers. The required design pattern is a canonical state machine with contract tests at every external entry point. Adding more subsystems before eliminating these split paths would increase failure surface rather than product strength.
