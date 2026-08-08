# SuperLocalMemory V4.0.0

> **V4.0.0 — governed local-first agent memory control plane built on the V3.8 foundation:
> local-first agent memory, 5-channel retrieval, cache, compression, trusted-peer
> coordination, team workspaces, roles, and governance controls.**

SuperLocalMemory turns conversations, observations, and connected-source
evidence into durable memory that can be recalled through a CLI, MCP, hooks,
dashboard, or documented IDE integrations. SQLite + sqlite-vec are the
canonical local store. The product also includes an explicit Scale Engine for
CozoDB graph and LanceDB vector projections, a cache/compression module, and
**SLM-Mesh** coordination controls.

> **V4.0.0 is the current release** (see [CHANGELOG](https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md) and `src/superlocalmemory/__version__.py`). Every Current page in this Wiki describes V4. Historical V3 research pages are labeled as historical and carried forward where still accurate.

## What changed in V4.0.0

V4 keeps the multi-channel retrieval and local-first store from the V3 research line, and productizes governed writes, **SLM-Mesh** peer coordination, multi-scope profiles, cache/compress, Entity Explorer and skill evolution, Modes A/B/C, and GDPR-oriented retention/audit controls. Operating mode is not legal certification under the EU AI Act — deployment context decides legal duties.

V4 hardens the full lifecycle of a memory operation — admission, canonical commit, projection to every store, migration, backup, and erasure — so each step is authorized and verifiable against its manifest and logs. Existing V4 memories and configuration are preserved; M038 (eager) and M039 (deferred) migrations are automatically applied at startup so no manual `slm db migrate` is normally required (see [CLI Reference](CLI-Reference) — forward only, no rollback). Schema downgrade is unsupported; to revert a V4 upgrade, restore a verified pre-upgrade backup of the complete data root (stop the daemon first and include WAL/SHM). V2→V3 migration is a separate `slm migrate` command (see [Migration from V2](Migration-from-V2)).

## The product in one view

```text
Sources and clients
CLI · MCP HTTP/stdio · hooks · dashboard · IDEs · adapters
                              │
                              ▼
  admission → queryable core → enrichment → brain/lifecycle
                              │
                              ▼
 semantic · BM25 · temporal · Hopfield · spreading activation
   (5 candidate producers + entity-graph score enhancement)
                              │
                              ▼
 safe bounded context with policy, provenance and trace evidence
                              │
                              ▼
 SQLite + sqlite-vec canonical ─► parity-gated graph/vector projections
```

The architecture has seven logical stages: admission, queryable durability,
enrichment, learning/lifecycle, retrieval, safe context delivery, and
operations. A specific write or recall only reports stages that actually ran;
optional enrichers and retrieval channels are dependency- and mode-aware.

## Capability map

| Area | Available capability | Important boundary |
|---|---|---|
| Memory | Facts, scenes, temporal events, entities, profiles/scopes, memory lifecycle | Recalled content is untrusted evidence, never a new instruction. |
| Ingestion | Replay-safe operation receipts; extraction, entity, graph, temporal, provenance and embedding derivations | Use `--sync` when a caller needs all declared stages, not only the immediate queryable receipt. |
| Recall | 5 candidate producers (semantic, BM25, temporal, Hopfield, spreading activation); RRF fusion, optional rerank and graph score enhancement | Runtime health determines the channels that participate; entity graph is a post-fusion enhancement, not a 6th candidate. |
| Brain | Behavioral patterns, feedback/outcomes, reward signals, consolidation, soft prompts and guarded skill evolution | Learning is not a guarantee that an outcome was correct or beneficial. |
| Graph | Canonical entities, aliases, profiles, edges, scenes, timelines and an Entity Explorer | Graph evidence is inspectable and provenance-bearing. |
| Scale Engine | CozoDB graph + LanceDB vectors with `prepare → verify → promote → rollback` and `adopt` | SQLite remains canonical; promotion is explicit and parity-gated. |
| Optimize | Exact cache, tag invalidation, safe compression, opt-in lossy prose compression and CCR originals | Only the proxy can intercept a primary provider turn. |
| **SLM-Mesh** | Authenticated peer messages, locks, inbox/outbox, queues and optional discovery | Mesh coordinates peers; it is not a replicated distributed-memory database. |
| Governance | Provenance, audit, retention, policy, export/erasure, health and diagnostics | Deployment configuration determines compliance posture. |
| Integrations | CLI, Python SDK, MCP, Claude plugin, Codex add-on, documented IDE configs, Gmail/Calendar/transcript adapters, nine framework adapters (LangGraph, Semantic Kernel, Microsoft Agent Framework, LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK, OpenAI Agents) | Connectors and hooks are opt-in and have their own data paths. |

## V4 Reliability Contract — Verified 2,200/2,200 (Scoped)

V4 states every reliability guarantee as a falsifiable invariant with an adversarial test, negative controls, and a shipped harness that regenerates the evidence. The only verified stress figure in V4 is the following; no other durability, latency, or throughput guarantee is claimed as a measured release envelope.

> **Verified source:** `benchmark/results/SUMMARY.md` (package **4.0.0**, Python **3.13.13**, `macOS-26.5.2-arm64-arm-64bit-Mach-O`, generated `2026-08-08T08:38:21Z`) and `benchmark/README.md`. Reproduce with `python benchmark/run_all.py --trials 200 --output-dir results/` — 11 experiments × 200 trials = **2,200/2,200 (100.0%)** trials upheld their guarantee. Individual trial JSON under `benchmark/results/`.

| Experiment | Guarantee | Metric |
|---|---|---|
| exp1_erasure_completeness | Real Bm25Owner + TemporalOwner + VectorOwner (embedding_metadata path; no sqlite-vec ANN in this env) erase all projection rows; tombstones + receipt persisted; keep-tenant hash unchanged | complete-erasure rate |
| exp2_transaction_atomicity | Committed → COMPLETE; faulted → DEGRADED with `compensate()` removing successful projection (ledger states verified) | manifest-correct rate |
| exp2b_real_owner_manifest | Happy path → COMPLETE with all three tables present; Bm25 fault → DEGRADED with zero residue and `compensate('temporal')` verified | manifest-correct rate |
| exp3_migration_downgrade | Newer-stamped DB refused on deferred pass with zero mutation | refuse-and-preserve rate |
| exp4_backup_restore_atomicity | Partial-restore failure rolls live data back to pre-restore bytes | rollback+clean rate |
| exp5_multitenant_isolation | Personal rows never leak across tenants (positive control: requester still sees own data) | zero-leak rate |
| exp6a/b/c | Superseded demotion, Ebbinghaus decay monotonicity, time-window inference | correct-demotion / monotonic-decay / correct-window |
| exp7_generation_fence | Stale-epoch ADMISSION rejected; fresh-epoch admitted | fence-correct rate |
| exp8_policy_registry | RBAC allow/deny with exact reason strings and unauthenticated→authentication_required | policy-correct rate |
| exp_governed_latency | Governed write envelope p50 reported alongside bypass (distinct shape) | latency_ms |

Explicit non-coverage (see `benchmark/README.md` honesty notes): fault-injection reliability and SLM's own temporal machinery — not an external agent-task benchmark; exp1 VectorOwner scope is `embedding_metadata` SQL only (ANN/`vector_row_map` excluded without sqlite-vec); exp2 uses a lightweight but complete `_TrackingOwner` (service/ledger/reconciler paths only); exp7 sets `runtime._generation` directly rather than exercising the full `rebind_engine` path; exp6b measures the shipped `EbbinghausCurve` function mathematically, not an end-to-end forgetting outcome.

> **What this is not:** Published LoCoMo scores below are historical V3 architecture evidence (`arXiv:2603.14588`) carried into V4 for continuity. They are **not** a newly rerun V4 package benchmark and are not comparable across vendors without matching protocol (conversation scope, question count, retrieval stack, answer model, judge, and release artifact).

## Operating modes

| Mode | Core behavior | Model path |
|---|---|---|
| **A — Local Guardian** | Local core memory and math-informed retrieval | No cloud model provider is required for core operations. |
| **B — Smart Local** | Mode A plus an operator-managed Ollama endpoint | Local LLM endpoint. |
| **C — Provider-assisted** | Local storage with configured provider-backed enrichment/retrieval behavior | Content sent to the configured provider follows that provider path. |

Mode A does not disable model downloads, adapters, backup, proxy providers, or
other integrations that an operator explicitly enables. Review the complete
deployment before making a privacy or compliance determination.

## What's fixed in V3.8.1

V3.8.1 hardens upgrades with bounded startup and background repair, replay-safe
ingestion, O(1) entity associations, responsive dashboard navigation, truthful
Brain telemetry, company-mode authorization across learning controls, and
repair of incomplete additive schemas such as Skill Evolution's cost ledger.

## What V3.8.0 added (carried into V4)

**Teams and enterprise memory**

- **Users and roles** — admin / member / viewer, scoped per workspace
- **Login gate** — `require_login = true` for team and enterprise deployments
- **GDPR export and erasure** — full profile data export; erasure removes data from 30+ scoped tables and is logged to the tamper-proof audit chain
- **Retention rules** — `indefinite`, `gdpr-30d`, `hipaa-7y`, `custom` policies per workspace
- **PII redaction** — configurable automatic redaction before memory content crosses trust boundaries

Personal installs are unchanged — no login required by default. See [[RBAC and Teams]] and [[GDPR Compliance]] — and [[Compliance]] for the wider deployment view.

**Bounded loops** — gate-verified iteration with a durable SLM-backed ledger. Three surfaces:
`slm loop` CLI (`demo` / `history` / `show`), the `/slm-loop` plugin command, and MCP tools `slm_loop_run` /
`slm_loop_history` / `slm_loop_show`. The gate is an independent recall query;
the agent's claim of completion is never used. See [[Bounded Loops]].

**Nine framework adapters** — LangGraph, Semantic Kernel, Microsoft Agent
Framework, LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK, and OpenAI
Agents. Each implements its framework's native memory interface and writes
through the SLM V4 ingestion contract. See [[Framework Adapters]].

**Multi-Agent Memory** — per-agent attribution via `SLM_AGENT_ID`, per-agent pane in the dashboard, and Mesh/lock coordination. See [[Multi-Agent Memory]].

**MCP profile update** — profiles now include bounded-loop tools. V4 counts:
`core` 14 / `code` 24 / `full` 42 / `power` 54 / `mesh` 8 / **`whole` 87** (all registered). See [[MCP Tools]].

## Dashboard workspaces

The local dashboard (`slm dashboard`, default `http://localhost:8765`) exposes Dashboard, Brain, Knowledge Graph, Memories, Health, Governance (Access & Users / Data Privacy / Audit / Lifecycle & Trust), Entity Explorer, Skill Evolution, Mesh Peers, MCP & Tools, Cloud Backup, Settings, and Optimize workspaces. Workspace and tab counts are illustrative — verify the installed dashboard. Use `slm health`, `slm doctor`, and `slm trace` for operational verification rather than treating a visual status or tab count as a contract.

## Quick Start

```bash
npm install -g superlocalmemory    # Primary global CLI path
slm setup                          # Choose mode A/B/C
slm warmup                         # Pre-download embedding model (optional)
```

The second primary path is Python in an activated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
slm setup
```

Then configure the client you intend to use and verify it with `slm doctor`. See [[Installation]], [[Getting Started]], and [[Quick Start Tutorial]].

> **Platform boundary (V4):** **Apple Silicon macOS, 64-bit Windows, 64-bit Linux.** Intel Mac and 32-bit Windows are not supported — packaging metadata (`package.json` `os: [darwin, linux, win32]`) does not hard-block architectures; install will fail where `cryptography==50.0.0` wheels are absent (see `pyproject.toml`).
