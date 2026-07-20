# V3.7 Architecture

SuperLocalMemory combines a local canonical memory store, multi-channel
retrieval, lifecycle and learning controls, Optimize, Mesh, and MCP/CLI/IDE
surfaces. It is an operator-controlled memory runtime rather than a single
vector index.

## Memory boundaries

Profiles are isolated workspaces. Each memory is `personal` (default),
`shared` with named profile readers, or `global`; shared/global recall is
default-deny until scope visibility is explicitly enabled. Scope sharing is a
local authorization contract, not a synonym for SLM Mesh peer coordination.

---

## Overview

```
┌──────────────────────────────────────────────────────┐
│                 SuperLocalMemory V3                    │
│                                                        │
│  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │  Product Shell     │  │  Mathematical Engine       │ │
│  │                    │  │                            │ │
│  │  CLI + Python SDK   │  │  Multi-channel Retrieval  │ │
│  │  MCP Server         │  │  Fisher-informed Scoring  │ │
│  │  Web Dashboard     │  │  Sheaf Consistency         │ │
│  │  Named MCP Clients │  │  Langevin Lifecycle        │ │
│  │  Learning (LightGBM│  │  Durable Ingestion         │ │
│  │  Trust (Bayesian)  │  │  Scene + Bridge Discovery  │ │
│  │  Compliance (ABAC) │  │  Cross-Encoder Rerank      │ │
│  │  Profiles + Mesh    │  │  3 Operating Modes         │ │
│  └──────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Hybrid Retrieval

Current recall uses five candidate producers where their dependencies are
available, then applies fusion, optional reranking, and graph-based score
enhancement:

```
Query
  │
  ├─ Strategy Classification (single-hop / multi-hop / temporal / open-domain)
  │
  ├─ Candidate Producers:
  │  ├─ Dense semantic
  │  ├─ BM25 lexical
  │  ├─ Temporal
  │  ├─ Hopfield associative
  │  └─ Spreading activation
  │
  ├─ Profile Lookup (direct SQL shortcut for entity queries)
  │
  ├─ Weighted RRF Fusion (k=60, channel weights vary by query type)
  │
  ├─ Scene Expansion (pull all facts from matched scenes)
  │
  ├─ Bridge Discovery (multi-hop only: Steiner tree + spreading activation)
  │
  ├─ Optional cross-encoder rerank when available
  ├─ Graph-derived score enhancement (cannot introduce a candidate)
  │
  └─ Top-K Results with per-channel scores
```

### Why multiple producers?

| Channel | What It Catches | What It Misses |
|---------|----------------|----------------|
| Semantic | Meaning similarity | Exact keywords, entity names |
| BM25 | Exact terms, rare words | Paraphrases, synonyms |
| Hopfield associative | Distributed associative similarity | Exact-term-only relationships |
| Temporal | Time-relevant facts | Atemporal knowledge |
| Spreading activation | Linked graph neighborhoods | Isolated facts |

No single channel handles all query types. The fusion combines their strengths.

## Safe context injection

Recalled memory is untrusted evidence, not a new instruction source. Hooks, MCP
`session_init`, CLI session context, and chat use one renderer for provenance,
budgets, recognized-secret redaction, canonical delimiters, and forged-marker
neutralization. IDE rule files contain only static product protocol; dynamic
memory is retrieved through MCP at runtime.

---

## Three Operating Modes

| Mode | Description | LLM | Compliance status |
|:----:|:-----------|:---:|:------------------|
| **A: Local Guardian** | Core memory operations use the local data root. Optional integrations have separate network behavior. | None | Deployment assessment required |
| **B: Smart Local** | Mode A + operator-configured Ollama extraction. | Local | Deployment assessment required |
| **C: Full Power** | Mode B + cloud-provider calls and agentic retrieval. | Cloud | Deployment and provider assessment required |

Mode A runs the core memory path without a cloud model provider. Retrieval quality depends on the configured embedding model, corpus, and enabled indexes; this page makes no market-uniqueness claim.

---

## Seven-stage execution model

SLM uses seven logical stages to make the product inspectable from a write to
an agent-facing recall. They are not seven services and they do not imply that
every optional backend is active for every request.

| Stage | Responsibility | Inspect with |
|---|---|---|
| 1. Admission | Actor, profile/scope, source identity, idempotency, raw evidence | `remember --json`, MCP write receipt |
| 2. Queryable core | Transactional SQLite fact + FTS checkpoint | receipt `materialization_state` |
| 3. Enrichment | Fact/entity/scene/graph/temporal/provenance/embedding derivations | operation status and daemon logs |
| 4. Brain and lifecycle | Consolidation, feedback/outcomes, patterns, rewards, forgetting and retention | Brain, Health and Operations workspaces |
| 5. Retrieval | Candidate producers, fusion, optional rerank and graph scoring | `slm trace` |
| 6. Context safety | Scope/policy, secret redaction, provenance, budgeted untrusted renderer | session/MCP context output |
| 7. Operations | Backups, audit, daemon, Scale Engine, cache/compression and Mesh | `slm doctor`, dashboard, diagnostics |

## Durable Ingestion Pipeline

Every accepted write owns an M018 operation and immutable replay identity.

| State | Contract |
|:----:|:------------|
| `raw` | Raw content, metadata, scope, session context, and trusted actor are durable. |
| `queryable` | Relational facts and SQLite FTS are committed before the receipt returns. |
| `enriching` | A lease-owning worker runs the configured extraction, entity, consolidation, graph, temporal, provenance, and embedding stages. |
| checkpoint | Final fact IDs and completed derivation stages commit before optional projectors. |
| `complete` | Every declared derivation and configured projector succeeded. |
| `failed` | Raw evidence, error, attempt count, and retry timing remain inspectable. |

---

## Database Schema

V3 uses an additive SQLite schema with FTS5 full-text search. The exact table count changes with forward migrations and is not a compatibility contract.

**Core:** `profiles`, `memories`, `atomic_facts`, `atomic_facts_fts` (FTS5)
**Entities:** `canonical_entities`, `entity_aliases`, `entity_profiles`
**Graph:** `graph_edges`, `memory_scenes`, `temporal_events`
**Quality:** `consolidation_log`, `trust_scores`, `provenance`
**Learning:** `feedback_records`, `behavioral_patterns`, `action_outcomes`
**Compliance:** `compliance_audit`
**Infrastructure:** `ingestion_operations`, `config`, `schema_version`, migration records

Core data tables are profile-scoped. M017 adds scope to CCQ blocks; M018 owns canonical ingestion operations. The legacy `pending.db` spool is an offline compatibility input whose replay submits through M018.

### Canonical store and Scale Engine

SQLite and sqlite-vec are the canonical store. CozoDB (graph) and LanceDB
(vector) are derived Scale Engine projections; they do not replace the
canonical database on installation. The supported activation sequence is:

```bash
slm db scale prepare
slm db scale verify <stage-id>
slm db scale promote <stage-id>
# If needed:
slm db scale rollback <backup-id>
```

Prepare builds private projections, verify compares them against the active
SQLite source, promote takes an explicit rollback point, and rollback restores
that named point. This is a correctness control, not a published throughput or
memory-count guarantee.

---

## Code Structure

```
superlocalmemory/src/superlocalmemory/
├── core/           Engine, config, modes, profiles, embeddings
├── retrieval/      Candidate producers, fusion, reranking, graph score enhancement
├── math/           Fisher-Rao metric, sheaf cohomology
├── dynamics/       Langevin lifecycle, Fisher-Langevin coupling
├── encoding/       Materialization stages (entity resolver, fact extractor, graph...)
├── storage/        Database, schema, migrations, V2 migrator
├── compliance/     EU AI Act, GDPR, ABAC
├── learning/       Adaptive learning, behavioral tracking, outcomes
├── trust/          Trust scoring, provenance tracking, gates
├── llm/            LLM backbone (Ollama / Azure / OpenAI)
├── mcp/            MCP server with profile-selected tools and resources
├── cli/            CLI, setup wizard, diagnostics, and maintenance commands
├── server/         Dashboard API + UI server
└── tests/          Unit, contract, artifact, and integration tests
```

---

## Dashboard and operational workspaces

```bash
slm dashboard    # Opens at http://localhost:8765
```

<details open>
<summary><strong>V3 Dashboard</strong></summary>

![Dashboard](https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/01-dashboard-main.png)

<table><tr>
<td><img src="https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/04-recall-lab.png" width="280"/></td>
<td><img src="https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/03-math-health.png" width="280"/></td>
</tr></table>

</details>

The dashboard exposes Dashboard, Brain, Knowledge Graph, Memories, Health,
Operations, Entity Explorer, Skill Evolution, Mesh Peers, Settings, and
Optimize workspaces. Some data is mode- or configuration-dependent. A visible
panel means the API surface is available; use CLI/MCP traces for deployment
verification.

## Benchmarks

Evaluated on the [LoCoMo benchmark](https://arxiv.org/abs/2402.09714) — 10 multi-session conversations, 1,986 total questions.

These are published V3 architecture experiments carried into V3.7 messaging:

- **60.4%** Mode A Raw across 10 conversations / 1,276 questions with zero-LLM answer construction.
- **74.8%** Mode A Retrieval across the same scope with local retrieval and GPT-4.1-mini answer synthesis.
- **87.7%** Mode C on Conv-30 / 81 questions with cloud embeddings and GPT-4.1-mini answer generation and judge.
- The raw zero-LLM and category results remain documented in the paper and must not be compared with other vendors without matching protocols.

These scores retain their evaluated protocol scope; they are not a newly rerun 3.7 package benchmark. See the paper for ablations and full results.
Those historical experiments are not performance claims for the current V3.7
runtime or a cross-vendor comparison.

---

*Part of [Qualixar](https://qualixar.com) · Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
