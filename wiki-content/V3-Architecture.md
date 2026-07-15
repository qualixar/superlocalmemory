# V3 Architecture

SuperLocalMemory V3 combines local-first storage, multi-channel retrieval,
optional mathematical scoring layers, lifecycle controls, and MCP clients.

---

## Overview

```
┌──────────────────────────────────────────────────────┐
│                 SuperLocalMemory V3                    │
│                                                        │
│  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │  Product Shell     │  │  Mathematical Engine       │ │
│  │                    │  │                            │ │
│  │  CLI                │  │  Multi-channel Retrieval  │ │
│  │  MCP Server         │  │  Fisher-informed Scoring  │ │
│  │  Web Dashboard     │  │  Sheaf Consistency         │ │
│  │  Named MCP Clients │  │  Langevin Lifecycle        │ │
│  │  Learning (LightGBM│  │  11-Step Ingestion         │ │
│  │  Trust (Bayesian)  │  │  Scene + Bridge Discovery  │ │
│  │  Compliance (ABAC) │  │  Cross-Encoder Rerank      │ │
│  │  Profiles           │  │  3 Operating Modes         │ │
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

## Dashboard

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

17 tabs: Dashboard, Recall Lab, Knowledge Graph, Memories, Trust, Math Health, Compliance, Learning, IDE Connections, Settings, and more.

## Benchmarks

Evaluated on the [LoCoMo benchmark](https://arxiv.org/abs/2402.09714) — 10 multi-session conversations, 1,986 total questions.

These are historical paper experiments, not current V3.7 measurements:

- Historical V3 research result: **74.8%** used local retrieval with GPT-4.1-mini answer construction.
- Historical Mode C result: **87.7%** on **81 questions from one conversation** with cloud-assisted components.
- The raw zero-LLM and category results remain documented in the versioned paper and must not be compared with other vendors without matching protocols.

### Ablation (conv-30, 81 questions)

| Removed | Impact |
|:--------|:------:|
| Cross-encoder reranking | **-30.7pp** |
| Fisher-Rao metric | **-10.8pp** |
| All math layers | **-7.6pp** |
| BM25 channel | **-6.5pp** |
| Sheaf consistency | -1.7pp |
| Entity graph | -1.0pp |

Mathematical layers contribute **+12.7pp average** across 6 conversations (n=832), with up to **+19.9pp** on the most challenging dialogues.

Full methodology and results in the [V3 paper](https://arxiv.org/abs/2603.14588) ([Zenodo](https://zenodo.org/records/19038659)).

---

*Part of [Qualixar](https://qualixar.com) · Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
