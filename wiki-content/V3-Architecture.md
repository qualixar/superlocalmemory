# V3 Architecture

SuperLocalMemory V3 introduces a mathematical retrieval engine alongside the existing V2 product infrastructure. This page describes the V3 architecture.

---

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                   SuperLocalMemory                       │
│                                                          │
│  ┌────────────────────┐    ┌──────────────────────────┐ │
│  │   V2 Product Shell  │    │   V3 Mathematical Engine │ │
│  │                     │    │                          │ │
│  │  CLI (slm command)  │    │  4-Channel Retrieval     │ │
│  │  MCP Server         │    │  Fisher-Rao Similarity   │ │
│  │  Web Dashboard      │    │  Sheaf Consistency       │ │
│  │  17+ IDE Configs    │    │  Langevin Lifecycle      │ │
│  │  Hooks & Skills     │    │  11-Step Ingestion       │ │
│  │  Learning (LightGBM)│    │  Scene + Bridge          │ │
│  │  Trust (Bayesian)   │    │  Cross-Encoder Rerank    │ │
│  │  Compliance (ABAC)  │    │  3 Operating Modes       │ │
│  └────────────────────┘    └──────────────────────────┘ │
│              Full integration in progress                 │
└─────────────────────────────────────────────────────────┘
```

---

## 4-Channel Hybrid Retrieval

V3 retrieves memories through four parallel channels, each capturing different aspects of relevance:

```
Query
  │
  ├─ Strategy Classification (single-hop / multi-hop / temporal / open-domain)
  │
  ├─ 4 Parallel Channels:
  │  ├─ Semantic Channel (Fisher-Rao weighted embedding similarity)
  │  ├─ BM25 Channel (keyword matching, Robertson 2009)
  │  ├─ Entity Graph Channel (spreading activation, 3 hops, decay 0.7)
  │  └─ Temporal Channel (3-date model: authored, valid-from, valid-until)
  │
  ├─ Profile Lookup (direct SQL shortcut for entity queries)
  │
  ├─ Weighted RRF Fusion (k=60, channel weights vary by query type)
  │
  ├─ Scene Expansion (pull all facts from matched scenes)
  │
  ├─ Bridge Discovery (multi-hop only: Steiner tree + spreading activation)
  │
  ├─ Cross-Encoder Rerank (energy-weighted: α·sigmoid(CE) + (1-α)·RRF)
  │
  └─ Top-K Results
```

### Why Four Channels?

| Channel | What It Catches | What It Misses |
|---------|----------------|----------------|
| Semantic | Meaning similarity | Exact keywords, entity names |
| BM25 | Exact terms, rare words | Paraphrases, synonyms |
| Entity Graph | Relational connections | Unconnected memories |
| Temporal | Time-relevant facts | Atemporal knowledge |

No single channel handles all query types. The fusion combines their strengths.

---

## Three Operating Modes

| Mode | Description | LLM | EU AI Act |
|:----:|:-----------|:---:|:---------:|
| **A: Local Guardian** | Pure mathematical retrieval. Zero cloud calls. | None | Compliant |
| **B: Smart Local** | Mode A + local LLM (Ollama) for extraction. | Local | Compliant |
| **C: Full Power** | Mode B + cloud LLM + agentic retrieval. | Cloud | Partial |

**Mode A** is architecturally unique: no other memory system achieves meaningful accuracy without LLM calls. This is possible because the 4-channel retrieval + cross-encoder reranking provides high-quality results without generative AI.

---

## 11-Step Ingestion Pipeline

Every memory is processed through structured encoding before storage:

| Step | What Happens |
|:----:|:------------|
| 1 | **Metadata extraction** — timestamps, source, importance |
| 2 | **Entity resolution** — canonical names with alias tracking |
| 3 | **Fact extraction** — atomic, typed facts (world/experience/opinion/temporal) |
| 4 | **Knowledge graph construction** — entities as nodes, relationships as edges |
| 5 | **Temporal parsing** — 3-date model (authored, valid-from, valid-until) |
| 6 | **Emotional signal extraction** — sentiment and emotional context |
| 7 | **Scene clustering** — group facts by temporal-semantic coherence |
| 8 | **Observation building** — structured observations for profiles |
| 9 | **Foresight generation** — anticipatory indexing for future queries |
| 10 | **Entropy gating** — information-theoretic filtering (low-entropy = skip) |
| 11 | **Compression and storage** — write to 21-table SQLite schema |

---

## Database Schema

V3 uses a 21-table SQLite schema (extending V2's 8 tables):

**Core tables:** memories, semantic_facts, kg_nodes, memory_edges, canonical_entities
**Retrieval tables:** memory_metadata, bm25_tokens, memory_scenes
**Intelligence tables:** memory_observations, memory_depth, contradictions, belief_history
**Math tables:** langevin_state, sheaf_sections
**Infrastructure tables:** trust_scores, provenance, compliance_events, profiles, v3_config

All tables are partitioned by `profile_id` for multi-context isolation (16+ profiles).

---

## Code Structure

```
superlocalmemory/
├── core/           Engine, config, modes, profiles, embeddings
├── retrieval/      4-channel engine, semantic, BM25, entity, temporal, fusion, reranker
├── math/           Fisher-Rao metric, sheaf cohomology, Langevin dynamics
├── encoding/       11-step pipeline (entity resolver, fact extractor, scene builder...)
├── storage/        Database, schema, migrations, access control
├── compliance/     EU AI Act, GDPR, lifecycle management
├── learning/       Adaptive learning, behavioral tracking, outcomes
├── trust/          Trust scoring, provenance tracking
├── attribution/    Mathematical DNA, cryptographic signing, watermarking
├── dynamics/       Fisher-Langevin coupling
├── llm/            LLM backbone (Ollama / Azure / OpenAI)
└── tests/          140+ tests
```

---

## Dashboard

![V3 Dashboard](https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/01-dashboard-main.png)

| Recall Lab | Math Health |
|:---:|:---:|
| ![Recall Lab](https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/04-recall-lab.png) | ![Math Health](https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/03-math-health.png) |

## Benchmarks

Evaluated on the LoCoMo conversational memory benchmark:

| Configuration | Micro Avg | Multi-Hop | Open Domain |
|:-------|:--:|:--:|:--:|
| **Mode A (Zero-LLM, all layers)** | **62.3%** | **50%** | **78%** |
| **Mode A (math layers off)** | 59.3% | 38% | 70% |
| **Mode C (Cloud, full power)** | **87.7%** | — | — |

Mathematical layers contribute +12.7pp average improvement over the engineering baseline across six conversations. Multi-hop sees the largest gain: 50% vs 38% (+12pp).

Full methodology and results in the [V3 paper](https://zenodo.org/records/19038659).

---

*Part of [Qualixar](https://qualixar.com) · Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
