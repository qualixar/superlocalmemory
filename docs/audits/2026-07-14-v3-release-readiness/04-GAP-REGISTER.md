# Gap register

> Audit snapshot. Execution tracking is superseded by [09-V3.7-DEFECT-LEDGER.yaml](09-V3.7-DEFECT-LEDGER.yaml), which carries the approved single-public-release V3.7 decision and requires closure of P0, P1, and P2 rows.

Status at audit close: every row is `OPEN`. Severity is based on data integrity, security boundary, product-contract impact, and credibility—not implementation effort.

## P0 — blocks corrective release or public promotion

| ID | Gap | Evidence | Owner area | Target | Acceptance criteria |
|---|---|---|---|---|---|
| SLM-P0-001 | Default MCP/HTTP/CLI remember skips full enrichment; materializer marks the partial record complete. | `RUNTIME-PROVEN` | Core ingestion | 3.6.24 | One idempotent state machine serves every entry point. Default remember becomes queryable immediately and eventually produces canonical entities, graph edges, temporal records, provenance, trust/audit events, consolidation, and backend sync. Crash/retry/duplicate tests pass. |
| SLM-P0-002 | Global/shared facts disappear in semantic, BM25, temporal, and dependent candidate generation. | `RUNTIME-PROVEN` | Retrieval/storage | 3.6.24 | A fact owned by profile B with global/shared visibility is retrievable from profile A through every enabled channel and full recall. Personal/project isolation negative tests pass. |
| SLM-P0-003 | An isolated home can attach to an unrelated daemon answering a default or legacy port. | `RUNTIME-PROVEN` | Daemon/CLI security | 3.6.24 | Client validates owner UID, canonical data-root fingerprint, daemon instance ID, protocol, version, and a local authenticated capability. Foreign health endpoints and redirecting legacy ports are rejected. |
| SLM-P0-004 | Main MCP context injection disables the untrusted-memory wrapper; renderers diverge. | `CODE-PROVEN` | Context/security | 3.6.24 | One renderer is mandatory for hooks, MCP resources, session init, AutoInvoker, and adapters. Stored instructions remain data under adversarial prompt-injection tests; provenance and truncation are consistent. |
| SLM-P0-005 | Slow integration tests can contact the live daemon and mutate the user database. | `RUNTIME-PROVEN` | Test/release | 3.6.24 | Tests fail closed unless a test-owned data root, port, instance token, and PID are present. A guard asserts no path under the real user data root is opened. |
| SLM-P0-006 | Live website/README/CLI contain materially false or unsupported benchmark, paper, scale, cache, compression, retrieval, and integration claims. | `RUNTIME-/DOC-PROVEN` | Docs/website/research | Immediate + 3.6.24 | Claim ledger corrections ship across site, README, wiki, CLI, package metadata, `llms.txt`, and generated plugin docs. A claim scanner finds no banned unsupported wording or stale LoCoMo link. |
| SLM-P0-007 | Current post-change LoCoMo performance is unknown; historical numbers are marketed as current. | `MISSING` | Evaluation | 3.7.0 relaunch | Frozen manifest, full 1,986-question retrieval/bounded-QA/agentic tracks, raw JSONL, human audit, confidence intervals, and package/container/source locks are public. |
| SLM-P0-008 | Registry `3.6.23`, tag, source, CI, publish workflows, changelog, and GitHub release do not form one reproducible release. | `RUNTIME-PROVEN` | Release engineering | 3.6.24 | Tag commit passes all release gates before publication; CI publishes both registries through provenance-enabled workflows; GitHub release, changelog, packages, SBOM, checksums, and source commit agree. |
| SLM-P0-009 | npm post-install may use `--break-system-packages`, install hooks, run setup, and acquire models without a safe explicit boundary. | `CODE-PROVEN` | Packaging/security | 3.6.24 | Post-install never overrides externally managed Python. Side effects are disclosed and opt-in. Install fails with an exact remediation command when prerequisites are absent. Clean npm/pip install tests run on all supported OS/Python/Node combinations. |
| SLM-P0-010 | Caller-supplied agent identity is not authentication; default fast ingestion bypasses trust gates. | `CODE-PROVEN` | Auth/trust | 3.6.24 | Actor identity is derived from an authenticated session/capability. Write policy runs on every path. Unknown agents cannot obtain write authority by choosing an ID. |

## P1 — blocks V3.7 integrity relaunch

| ID | Gap | Evidence | Owner area | Target | Acceptance criteria |
|---|---|---|---|---|---|
| SLM-P1-001 | Five candidate channels are described as four, six, or seven; graph cannot introduce candidates. | `CODE-PROVEN` | Retrieval | 3.7.0 | Canonical channel contract and trace schema ship. Dense, lexical, temporal, graph, and associative arms independently introduce candidates before fusion, or marketing states the smaller truthful architecture. |
| SLM-P1-002 | Double score transformation collapses confidence to 1.0 and makes calibration uninterpretable. | `RUNTIME-PROVEN` | Retrieval/evaluation | 3.7.0 | One documented score domain; calibration evaluated with reliability diagram, ECE/Brier/selective-risk metrics; runtime output spans meaningful confidence and supports abstention. |
| SLM-P1-003 | Reranking silently disappears on cold start. | `CODE-PROVEN` | Retrieval/runtime | 3.7.0 | Explicit policy: block until ready, use a declared fallback, or return `reranker_applied=false`. Cold/warm result traces and latency SLOs are tested. |
| SLM-P1-004 | Fisher contribution differs between sqlite-vec fast path and full scan. | `CODE-PROVEN` | Math/retrieval | 3.7.0 | Identical corpus/query returns rank-equivalent results across fast/fallback paths within a declared tolerance. No path-specific constant floor. |
| SLM-P1-005 | Semantic cache benchmark uses an injected stub while production manager installs a no-op and passes no embedding. | `CODE-PROVEN` | Optimize/cache | 3.7.0 | Production semantic tier is constructed, embedded, tenant-isolated, invalidated, measured on a real corpus, and fails closed at a calibrated threshold—or the feature is removed. |
| SLM-P1-006 | Compression headline exceeds measured shipped behavior; safe/proxy/aggressive paths are conflated. | `RUNTIME-PROVEN` | Optimize/compression | 3.7.0 | Corpus-specific fidelity/ratio/latency/token/cost report for each surface and mode; structured-data invariants and CCR round-trip tests; no generic ratio headline. |
| SLM-P1-007 | Cozo/Lance initialization exists, but normal ingestion/retrieval does not use them as marketed. | `CODE-PROVEN` | Storage/backends | 3.7.0 | Either wire and contract-test each backend through store/update/delete/rebuild/recall, or label them derived/experimental accelerators and remove primary-backend language. |
| SLM-P1-008 | Mesh coordinates local state/messages but is marketed as distributed memory; no replication/conflict contract. | `CODE-PROVEN` | Mesh/distributed | Post-3.7 unless narrowed | Either rename to coordination mesh or prove memory replication, identity, ordering, concurrent edits, conflict resolution, delete propagation, offline replay, recovery, and multi-peer membership. |
| SLM-P1-009 | Loopback mesh state is unauthenticated and may contain secrets. | `CODE-PROVEN` | Mesh/security | 3.6.24 | Local capability/auth on every state operation; secret-bearing values prohibited or encrypted with explicit threat model; hostile local-process test. |
| SLM-P1-010 | Lifecycle/tiering and forgetting consult different fields; startup evaluation may target only the default profile. | `CODE-PROVEN` | Lifecycle/storage | 3.7.0 | One lifecycle source of truth; archive/restore/forget tests across profiles; archived/deleted facts cannot leak through any index or cache. |
| SLM-P1-011 | Stale documented MCP profile names silently fall back to a different tool set. | `RUNTIME-PROVEN` | MCP/config | 3.6.24 | Canonical names/counts generated from source; deprecated aliases warn and map correctly for one release; unknown profiles fail closed. |
| SLM-P1-012 | “17+ integrations” is not backed by successful client contract tests. | `CODE-PROVEN` | Integrations | 3.7.0 | Public matrix generated by CI for each named client: install, start, remember, recall, update, forget, reconnect, context injection, uninstall. Only green clients are counted. |
| SLM-P1-013 | Health can return success while migration errors appear; fresh store has warmup/timeouts and stale schema warnings. | `RUNTIME-PROVEN` | Operations/migrations | 3.6.24 | Fresh install, upgrade from supported versions, downgrade/rollback, corrupted partial migration, worker cold start, and health/readiness tests. Readiness is false on required migration failure. |
| SLM-P1-014 | License claims conflict across root, source headers, npm scripts, papers, and machine-readable docs. | `CODE-PROVEN` | Legal/release | 3.6.24 | Canonical license policy approved and generated into package metadata, headers, docs, SBOM, and third-party notices. Historical paper statements are labeled by snapshot. |
| SLM-P1-015 | Recalled facts receive popularity/trust/Fisher reinforcement without proving relevance, creating feedback loops. | `CODE-PROVEN` | Learning/retrieval | 3.7.0 | Exposure is separated from positive feedback. Counterfactual/off-policy evaluation, negative signals, decay, and loop-amplification tests are present. |

## P2 — product strengthening after integrity gates

| ID | Gap | Evidence | Owner area | Target | Acceptance criteria |
|---|---|---|---|---|---|
| SLM-P2-001 | Behavioral entity access expects a missing `.keys` member and degrades to empty data. | `CODE-PROVEN` | Learning | 3.7.x | Correct typed interface and regression test with non-empty behavioral entities. |
| SLM-P2-002 | Quantization-aware search expects a `VectorStore.search_int8` API that does not exist. | `CODE-PROVEN` | Retrieval/quantization | 3.7.x | Implement and benchmark the API or remove the dead tier and associated claim. |
| SLM-P2-003 | mDNS address handling is likely incompatible with Zeroconf byte addresses. | `CODE-PROVEN` | Mesh/network | 3.7.x | Two-host IPv4/IPv6 discovery tests on supported OSes, with timeouts, duplicate services, and hostile announcements. |
| SLM-P2-004 | Raw evidence, derived facts, graph state, and human-reviewable/versioned memory are not one explicit rebuild contract. | `CODE-PROVEN` | Data architecture | 3.8 | Export/import, source spans, deterministic derivation version, git-friendly view, rollback, rebuild, and reconciliation report. |
| SLM-P2-005 | No product telemetry proves activation, retained use, cross-IDE handoff, or paid intent. | `MISSING` | Product/GTM | Immediate | Privacy-preserving local counters plus explicit opt-in aggregate reporting, or user-provided diagnostic export; activation and handoff funnels measurable without collecting memory content. |
| SLM-P2-006 | No commercial CTA, paid offer, support contract, or buyer proof on the website. | `DOC-ONLY` | GTM | Immediate after claim correction | Founding design-partner page, fixed scope and price, qualification form, calendar/email route, terms, and one measured case study. |

## Gate policy

- `3.6.24` cannot publish with an open `SLM-P0-001` through `006`, `008` through `010`, or a target-`3.6.24` P1 row.
- `3.7.0` cannot be promoted with any open P0 or P1 row.
- A row closes only with an executable test or a live, versioned evidence artifact. A code review, screenshot, or prose assertion does not close it.
