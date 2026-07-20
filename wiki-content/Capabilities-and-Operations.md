# Capabilities and Operations

This page is the product-level map for SuperLocalMemory V3.7. It separates
what the runtime provides from what an operator must explicitly configure or
verify.

## Memory and retrieval

SLM stores local memory as atomic facts, episodic scenes, temporal events,
canonical entities, profiles and scopes. A write creates a durable operation:
`raw → queryable → enriching → complete`, with retryable failures retaining
their evidence. `slm remember --sync` waits for declared materialization work;
the default write receipt is intentionally queryable-first.

Recall can combine dense semantic, BM25 lexical, temporal, Hopfield
associative, and spreading-activation candidate channels. It then applies
fusion, optional reranking, and graph-derived score enhancement. Use `slm
trace "query"` to inspect channel participation and evidence. SLM does not
represent its internal ranking utility as calibrated answer confidence.

## Brain, entities and evolution

The local Brain contains consolidation, behavioral patterns, feedback and
outcome records, rewards, soft prompts, and ranking-related learning
components. The Entity Explorer surfaces compiled entity summaries and
timelines. Skill Evolution maintains lineage, budgets and verification
outcomes for opt-in evolution workflows.

These components learn from qualified evidence and configured feedback. They
do not imply that every observed result was good, that a skill will improve,
or that an agent may execute a generated change without review.

## Databases and Scale Engine

SQLite plus sqlite-vec are canonical. CozoDB and LanceDB are packaged derived
projections for graph and vector work. They must be activated deliberately:

```bash
slm db scale status
slm db scale prepare
slm db scale verify <stage-id>
slm db scale promote <stage-id>
slm db scale rollback <backup-id>
```

The lifecycle uses the **expand–verify–promote–rollback** pattern. It prevents
an installed backend from silently becoming authoritative and gives the
operator a named rollback point. SLM does not publish an unqualified memory
count, latency, or throughput guarantee for this feature.

## Optimize: cache and compression

Optimize has three surfaces:

| Surface | What it can affect |
|---|---|
| Proxy | Intercepts configured provider calls and can cache a primary provider turn. |
| MCP tools | Caches/compresses outputs that an agent explicitly routes through SLM. |
| Skill | Guides an agent to route selected content through the same controls. |

Exact cache, tags and invalidation are the stable cache path. Safe compression
uses conservative normalization; aggressive prose compression is opt-in and
lossy. MCP and skill use do not intercept the primary conversation turn unless
a proxy is in the path.

## Mesh and integrations

SLM Mesh provides authenticated messages, queues, locks, inbox/outbox, and
optional peer discovery. It coordinates configured peers; it is not a
replicated, conflict-resolving memory database.

SLM exposes CLI, Python, MCP HTTP/stdio, dashboard, a Claude Code plugin, and
an additive Codex add-on. Named IDE configurations are available through `slm
connect`; hooks, IDE configuration, connectors, cloud backup, and Gmail,
Calendar, or transcript adapters must be enabled explicitly. Dynamic memories
are retrieved at runtime as bounded untrusted evidence, not copied into IDE
instruction files.

## Governance and verification

SLM includes provenance, policy, audit, lifecycle, export/erasure, health,
diagnostic and backup surfaces. These controls can support a deployment’s
governance work; they are not legal certification.

Before relying on a deployment, run:

```bash
slm doctor
slm health
slm status
slm trace "a representative production query"
```

Then verify the selected MCP client, adapters, provider and network boundary in
the environment that will actually run them.
