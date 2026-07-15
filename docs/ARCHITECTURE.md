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

An accepted write is first recorded in the M018 `ingestion_operations` table.
The source type and idempotency key own replay identity; reuse with different
immutable evidence is rejected.

| Durable phase | Contract |
|---|---|
| `raw` | Raw content, metadata, scope, session context, and trusted actor are durable. |
| `queryable` | A relational fact and its SQLite FTS projection are committed in the same transaction. |
| `enriching` | A lease-owning worker performs extraction, entity resolution, consolidation, graph, temporal, provenance, and embedding work. |
| checkpoint | Final fact IDs and completed derivation stages are committed before optional external projection. |
| `complete` | Every declared derivation and projector stage reports success. |
| `failed` | Evidence and checkpoint data remain inspectable and retryable with bounded backoff. |

Mode and configured dependencies determine how individual enrichment stages are
implemented. The state record reports what the released runtime actually
completed; documentation does not imply that an unavailable optional backend
participated.

## How Memories Are Retrieved (Recall)

Recall uses five candidate producers where their dependencies are available,
then fusion, optional reranking, and graph-based score enhancement.

### Candidate Producers

| Channel | How it works | Best for |
|---------|-------------|----------|
| **Semantic** | Vector similarity using sentence embeddings, enhanced by Fisher-Rao geometry | "Queries that mean the same thing but use different words" |
| **BM25** | Classic keyword matching with term frequency scoring | "Queries with specific names, codes, or exact terms" |
| **Profile** | Retrieves profile-scoped facts and accumulated entity context | "What does this profile know about Maria?" |
| **Temporal** | Matches based on time references and event ordering | "What did we decide last Friday?" or "Changes since the sprint started" |
| **Associative** | Uses stored associations to extend candidate evidence | Related decisions and linked technical context |

### Fusion and Ranking

1. Available producers return candidates with scores and provenance.
2. Reciprocal Rank Fusion combines the ranked lists.
3. Optional rerankers refine the leading candidates when enabled.
4. Graph-derived evidence can modify later scoring without becoming a sixth producer.
5. The response returns the final ranked evidence and measured runtime fields.

No single producer is guaranteed to run in every mode. The runtime degrades to
the indexes that are healthy and reports its result provenance.

## Safe context injection

Stored text is data, even when it came from a local or first-party source. Every
runtime injection surface maps recalled results into one renderer that applies
configured budgets, redacts recognized secrets, neutralizes attempts to forge
the canonical boundary, and carries fact/source provenance. The rendered block
is explicitly reference-only evidence; instructions found inside do not gain
authority to call tools, change roles, or request secrets.

Cursor, Copilot, and Antigravity instruction files contain only
product-authored static protocol. Dynamic memories are fetched through MCP at
runtime and are not persisted into these high-trust files. See
[`docs/adr/0001-untrusted-memory-boundary.md`](adr/0001-untrusted-memory-boundary.md)
for the decision and its limits.

## Three Operating Modes

| Mode | Retrieval | LLM Usage | Data Location |
|------|-----------|-----------|---------------|
| **A: Local** | Candidate retrieval + math-informed scoring | None for core memory operations | Local data root; optional integrations may use the network |
| **B: Local LLM** | Candidate retrieval + local LLM enrichment | Ollama (local) | Local data root; optional integrations may use the network |
| **C: Cloud LLM** | 4-channel + cross-encoder + agentic retrieval | Cloud provider | Queries sent to cloud |

Mode A is the default. Core memory operations can run without a cloud model provider, but model and dependency downloads, connectors, cloud backup, and explicitly enabled integrations may use the network.

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

## Privacy and compliance controls

SuperLocalMemory provides controls that may support a deployment's privacy and compliance program. It is not a legal certification. Operators must assess the configured system, use case, data flows, and surrounding services.

- **Local core path.** Mode A can keep memory content in the configured local data root during core operations.
- **Erasure command.** `slm forget` removes selected local records; backups, exports, caches, derived indexes, and provider logs require separate verification.
- **Auditability.** Retrieval and lifecycle surfaces expose local records and diagnostics, subject to release-specific verification.
- **Policy controls.** Provenance, retention, and access-policy features are available for operator configuration.

Mode C sends queries to a cloud LLM provider. In that mode, the cloud provider's compliance posture applies to those queries.

## Database

Core memory is SQLite-backed:

```
~/.superlocalmemory/memory.db
```

The data root also contains configuration, logs, models, queues, derived indexes, and optional backend state. Back up or migrate the documented data root rather than copying only `memory.db`. The core database uses WAL (Write-Ahead Logging) mode for concurrent access.

Key table groups:

- **Core:** memories, sessions, profiles
- **Knowledge:** atomic_facts, graph_edges, canonical_entities, temporal_events
- **Durable ingestion:** ingestion_operations (M018 operation state and raw evidence)
- **Retrieval indexes:** SQLite FTS plus configured derived indexes
- **Math layers:** fisher_state, sheaf_sections, langevin_state
- **Compliance and provenance:** trust scores, provenance records, audit and retention controls

M017 additively adds scope to CCQ consolidation blocks. M018 is the canonical
expand-phase ingestion contract. The legacy `pending.db` spool remains only as
an offline compatibility input; replay submits its evidence into M018 before it
is marked done.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
