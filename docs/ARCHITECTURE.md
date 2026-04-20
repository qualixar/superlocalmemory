# Architecture
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

A high-level overview of how SuperLocalMemory V3 stores, organizes, and retrieves your memories.

---

## Design Principles

1. **Local-first.** Your data stays on your machine by default. Cloud is opt-in.
2. **Zero-configuration.** Install and forget. Memory capture and recall happen automatically.
3. **Multi-channel retrieval.** No single search method is best for every query. V3 combines four channels and picks the best results.
4. **Mathematically grounded.** Retrieval quality, consistency, and lifecycle management are backed by information geometry rather than heuristics.

## System Overview

```
Your IDE (Claude, Cursor, VS Code, ...)
       |
       | MCP Protocol
       v
+------------------+
| MCP Server       |  24 tools, 6 resources
+------------------+
       |
       v
+------------------+
| Memory Engine    |  Ingestion + Retrieval + Lifecycle
+------------------+
       |
       v
+------------------+
| SQLite Database  |  ~/.superlocalmemory/memory.db
+------------------+
```

## How Memories Are Stored (Ingestion)

When you store a memory (or one is auto-captured), it passes through an 11-step pipeline:

| Step | What it does |
|------|-------------|
| 1. Metadata extraction | Timestamps, session context, source identification |
| 2. Entity extraction | People, projects, technologies, organizations mentioned |
| 3. Fact extraction | Discrete facts pulled from the text |
| 4. Emotion detection | Sentiment and emotional tone (frustration, confidence, etc.) |
| 5. Belief extraction | Opinions, preferences, and stated beliefs |
| 6. Entity resolution | Links mentions to canonical entities (e.g., "React" and "ReactJS" become one) |
| 7. Graph wiring | Connects entities and facts into a relationship graph |
| 8. Foresight tagging | Predicts what future queries this memory might answer |
| 9. Observation building | Creates structured observations for pattern learning |
| 10. Entropy gating | Filters out redundant or low-value information |
| 11. Storage | Writes to all indexes: semantic vectors, BM25 tokens, graph edges, temporal events |

In Mode A, all steps run locally without any LLM calls. In Modes B and C, the LLM enhances entity extraction and fact decomposition.

## How Memories Are Retrieved (Recall)

Every recall query runs through four independent retrieval channels, then fuses the results:

### The Four Channels

| Channel | How it works | Best for |
|---------|-------------|----------|
| **Semantic** | Vector similarity using sentence embeddings, enhanced by Fisher-Rao geometry | "Queries that mean the same thing but use different words" |
| **BM25** | Classic keyword matching with term frequency scoring | "Queries with specific names, codes, or exact terms" |
| **Entity Graph** | Traverses the relationship graph via spreading activation | "Who works with Maria?" or "What connects service A to service B?" |
| **Temporal** | Matches based on time references and event ordering | "What did we decide last Friday?" or "Changes since the sprint started" |

### Fusion and Ranking

1. Each channel returns its top candidates with scores
2. **Reciprocal Rank Fusion** (RRF) combines the four ranked lists into one
3. In Mode C, a **cross-encoder** reranks the top results for final precision
4. The top results are returned to your AI assistant

This multi-channel approach means V3 finds memories that any single search method would miss. A keyword search alone might miss paraphrased content. A vector search alone might miss exact names. The combination catches both.

## Three Operating Modes

| Mode | Retrieval | LLM Usage | Data Location |
|------|-----------|-----------|---------------|
| **A: Zero-Cloud** | 4-channel + math scoring | None | 100% local |
| **B: Local LLM** | 4-channel + local LLM reranking | Ollama (local) | 100% local |
| **C: Cloud LLM** | 4-channel + cross-encoder + agentic retrieval | Cloud provider | Queries sent to cloud |

Mode A is the default. It delivers strong recall quality using mathematical scoring without any network calls.

## Mathematical Foundations

V3 uses three mathematical layers. These are not academic additions — they solve specific practical problems.

### Fisher-Rao Similarity

**Problem:** Standard vector similarity treats all memories equally, regardless of how much evidence backs them.

**Solution:** Fisher-Rao geometry accounts for the statistical confidence of each memory's embedding. A memory accessed and confirmed many times gets a tighter confidence region. A memory stored once and never validated has wider uncertainty. Similarity scoring respects this difference.

**Effect:** Frequently validated memories rank higher. Uncertain memories are flagged for verification.

### Sheaf Consistency

**Problem:** Over time, you store contradictory memories. "We use PostgreSQL" and later "We migrated to MySQL." Simple retrieval returns both without flagging the conflict.

**Solution:** Sheaf cohomology detects when memories attached to the same entity or topic contradict each other. When a contradiction is found, the system marks the older memory as superseded and surfaces the newer one.

**Effect:** Recall returns consistent information. Contradictions are flagged for your review.

### Langevin Lifecycle

**Problem:** Memory databases grow endlessly. Old memories dilute retrieval quality.

**Solution:** Langevin dynamics models each memory's lifecycle — from Active (frequently accessed) through Warm, Cold, and eventually Archived. The transition is not based on simple time rules but on a self-organizing dynamic that balances recency, access frequency, and information value.

**Effect:** Active memories stay prominent. Stale memories fade gracefully. Storage stays efficient.

## EU AI Act Compliance

Mode A satisfies data sovereignty requirements under the EU AI Act by design:

- **No cloud dependency.** All memory operations run locally. No data leaves your machine.
- **Right to erasure.** `slm forget` deletes data locally. No cloud logs to purge.
- **Transparency.** The retrieval pipeline is auditable. No black-box LLM decisions in Mode A.
- **Risk classification.** A local retrieval system with no AI decision-making qualifies as minimal risk.

Mode C sends queries to a cloud LLM provider. In that mode, the cloud provider's compliance posture applies to those queries.

## Database

All data is stored in a single SQLite database:

```
~/.superlocalmemory/memory.db
```

The database uses WAL (Write-Ahead Logging) mode for safe concurrent access from multiple IDE connections.

Key table groups:

- **Core:** memories, sessions, profiles
- **Knowledge:** semantic_facts, kg_nodes, memory_edges, canonical_entities
- **Retrieval indexes:** bm25_tokens, memory_metadata, temporal_events
- **Math layers:** fisher_state, sheaf_sections, langevin_state
- **Compliance:** trust_scores, provenance, audit_trail, retention_policies

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
