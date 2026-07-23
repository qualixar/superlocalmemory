# SuperLocalMemory V3.8.0

> **Local-first agent memory, retrieval, cache, compression, and trusted-peer coordination in one operator-controlled runtime. V3.8.0 adds team workspaces, user roles, and GDPR-ready data governance.**

SuperLocalMemory turns conversations, observations, and connected-source
evidence into durable memory that can be recalled through a CLI, MCP, hooks,
dashboard, or documented IDE integrations. SQLite + sqlite-vec are the
canonical local store. The product also includes an explicit Scale Engine for
CozoDB graph and LanceDB vector projections, a cache/compression module, and
SLM Mesh coordination controls.

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
| Recall | Semantic, BM25, temporal, Hopfield and spreading-activation candidates; fusion, optional rerank and graph score enhancement | Runtime health determines the channels that participate. |
| Brain | Behavioral patterns, feedback/outcomes, reward signals, consolidation, soft prompts and guarded skill evolution | Learning is not a guarantee that an outcome was correct or beneficial. |
| Graph | Canonical entities, aliases, profiles, edges, scenes, timelines and an Entity Explorer | Graph evidence is inspectable and provenance-bearing. |
| Scale Engine | CozoDB graph + LanceDB vectors with prepare → verify → promote → rollback | SQLite remains canonical; promotion is explicit and parity-gated. |
| Optimize | Exact cache, tag invalidation, safe compression, opt-in lossy prose compression and CCR originals | Only the proxy can intercept a primary provider turn. |
| Mesh | Authenticated peer messages, locks, inbox/outbox, queues and optional discovery | Mesh coordinates peers; it is not a replicated distributed-memory database. |
| Governance | Provenance, audit, retention, policy, export/erasure, health and diagnostics | Deployment configuration determines compliance posture. |
| Integrations | CLI, Python SDK, MCP, Claude plugin, Codex add-on, documented IDE configs, Gmail/Calendar/transcript adapters, nine framework adapters (LangGraph, Semantic Kernel, Microsoft Agent Framework, LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK, OpenAI Agents) | Connectors and hooks are opt-in and have their own data paths. |

## Operating modes

| Mode | Core behavior | Model path |
|---|---|---|
| **A — Local Guardian** | Local core memory and math-informed retrieval | No cloud model provider is required for core operations. |
| **B — Smart Local** | Mode A plus an operator-managed Ollama endpoint | Local LLM endpoint. |
| **C — Provider-assisted** | Local storage with configured provider-backed enrichment/retrieval behavior | Content sent to the configured provider follows that provider path. |

Mode A does not disable model downloads, adapters, backup, proxy providers, or
other integrations that an operator explicitly enables. Review the complete
deployment before making a privacy or compliance determination.

## What's new in V3.8.0

**Teams and enterprise memory**

- **Users and roles** — admin / member / viewer, scoped per workspace
- **Login gate** — `require_login = true` for team and enterprise deployments
- **GDPR export and erasure** — full profile data export; erasure removes data from 30+ scoped tables and is logged to the tamper-proof audit chain
- **Retention rules** — `indefinite`, `gdpr-30d`, `hipaa-7y`, `custom` policies per workspace
- **PII redaction** — configurable automatic redaction before memory content crosses trust boundaries

Personal installs are unchanged — no login required by default. See [[RBAC and Teams]] and [[GDPR Compliance]].

**Bounded loops** — gate-verified iteration with a durable SLM-backed ledger. Three surfaces:
`slm loop` CLI, `/slm-loop` plugin command, and MCP tools `slm_loop_run` /
`slm_loop_history` / `slm_loop_show`. The gate is an independent recall query;
the agent's claim of completion is never used. See [[Bounded Loops]].

**Nine framework adapters** — LangGraph, Semantic Kernel, Microsoft Agent
Framework, LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK, and OpenAI
Agents. Each implements its framework's native memory interface and writes
through the SLM V3 ingestion contract. See [[Framework Adapters]].

**MCP profile update** — profiles now include bounded-loop tools. Counts:
core 14 / code 24 / full 42 / power 54 / whole 84. See [[MCP Tools]].

## Dashboard workspaces

V3.8.0 reorganized the dashboard navigation. The former "Operations" section is
now **Governance**, with sub-tabs for Access & Users, Data Privacy, and Audit.
A new **Integrations** group adds the **MCP & Tools** pane for profile management.

The local dashboard includes Dashboard, Brain, Knowledge Graph, Memories,
Health, Governance (Access & Users / Data Privacy / Audit / Lifecycle & Trust),
Entity Explorer, Skill Evolution, Mesh Peers, MCP & Tools, Cloud Backup,
Settings, and Optimize workspaces. Use it with `slm health`, `slm doctor`, and
`slm trace` for operational verification rather than treating a visual status as
a guarantee.

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

Then configure the client you intend to use and verify it with `slm doctor`.
