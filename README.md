<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/slm-wordmark-dark.svg">
    <img src="assets/branding/slm-wordmark-light.svg" alt="SuperLocalMemory" width="390">
  </picture>
</p>

<h1 align="center">SuperLocalMemory V3.8.3</h1>
<p align="center"><strong>Enterprise-grade, local-first memory for AI agents and teams.</strong><br/>
<em>A persistent, auditable long-term brain for your agents that runs on your own infrastructure — with multi-workspace isolation, role-based access, and GDPR + EU AI Act governance controls built in.</em></p>
<p align="center"><code>v3.8.3</code> — one control plane: auditable retrieval · multi-scope memory (personal / shared / global) · Cache · Compress · trusted-peer Mesh · bounded loops — across CLI, MCP, dashboard, the <strong>Claude plugin</strong>, the <strong>Codex add-on</strong>, and documented IDE integrations.<br/>
Proxy: <code>slm wrap claude</code> &nbsp;·&nbsp; MCP: add <code>slm_compress</code> to your config &nbsp;·&nbsp; Skill: zero-config</p>
<p align="center"><strong>3 public research preprints</strong> (arXiv + Zenodo archives) · <a href="https://arxiv.org/abs/2603.02240">arXiv:2603.02240</a> · <a href="https://arxiv.org/abs/2603.14588">arXiv:2603.14588</a> · <a href="https://arxiv.org/abs/2604.04514">arXiv:2604.04514</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14588"><img src="https://img.shields.io/badge/arXiv-2603.14588-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="#three-surfaces-proxy--mcp-tools--skill"><img src="https://img.shields.io/badge/Proxy_|_MCP_|_Skill-22c55e?style=for-the-badge" alt="Three Surfaces: Proxy, MCP Tools, Skill"/></a>
  <a href="https://pypi.org/project/superlocalmemory/"><img src="https://img.shields.io/pypi/v/superlocalmemory?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/superlocalmemory"><img src="https://img.shields.io/npm/v/superlocalmemory?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=for-the-badge" alt="AGPL v3"/></a>
  <a href="#privacy-controls-and-operating-modes"><img src="https://img.shields.io/badge/Privacy-Deployment_Assessed-brightgreen?style=for-the-badge" alt="Privacy controls require deployment assessment"/></a>
  <a href="#teams-and-enterprise-memory-v380"><img src="https://img.shields.io/badge/Enterprise-GDPR_%7C_EU_AI_Act-0b5394?style=for-the-badge" alt="Enterprise governance: GDPR and EU AI Act controls"/></a>
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Web-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/MCP-Native-blue?style=for-the-badge" alt="MCP Native"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/CLI-Agent--Native-green?style=for-the-badge" alt="CLI Agent-Native"/></a>
  <a href="#multilingual-embedding-support"><img src="https://img.shields.io/badge/Multilingual-30%2B_Languages-ff69b4?style=for-the-badge" alt="Multilingual 30+ Languages"/></a>
</p>

---

## Why SuperLocalMemory?

SuperLocalMemory is an enterprise-grade, local-first memory control plane for AI agents. Your team's agent memory lives on infrastructure you control, with per-workspace isolation, role-based access, and GDPR / EU AI Act governance controls — built for organizations, and for EU data-residency obligations where agent context must not leave your environment by default.

Agent-memory systems make different storage, model-provider, and deployment trade-offs. SuperLocalMemory starts with a local runtime and makes provider-backed enrichment, cloud backup, connectors, and proxy use explicit choices.

Different products solve different boundaries. The published benchmark evidence carried into V3.8.0 is protocol-scoped evidence from the published V3 research, not a claim of a newly rerun V3.8.0 package benchmark.

SuperLocalMemory V3 combines conventional dense and lexical retrieval with graph, temporal, associative, and statistical relevance scoring. The default local runtime does not require Docker, a separately operated graph database, or an API key.

**Memory with a sense of time.** SLM does not only store *what* an agent learned — it records *when*. Every fact carries ingestion timing and provenance; recall runs a dedicated temporal candidate channel alongside semantic, lexical, and associative retrieval; scenes and entity timelines reconstruct sequence; and the lifecycle lets neglected memory decay and self-archive instead of growing without bound. Time is a first-class ranking and lifecycle signal rather than a timestamp column an agent never reads — which is what lets a long-lived agent reason about how its context changed, not only what it currently holds.

**What V3.8.0 added.** The 3.8.0 capability release introduced the following
foundation; 3.8.1 is the existing-install stability patch for it:

- **Temporal depth** — the time-aware retrieval and lifecycle described above.
- **Governance & EU compliance** — [team roles, workspace isolation, a login gate, multi-scope memory, GDPR access/erasure/portability rights, a hash-chained audit trail, and per-mode EU AI Act self-assessment](#teams-and-enterprise-memory-v380).
- **Framework adapters** — [drop-in, engine-backed memory for nine agent frameworks](#framework-adapters-v380).
- **Bounded loops** — [gate-verified agent loops where an independent check, not the agent's own claim, decides when a task is done](#bounded-loops-v380).
- **Stronger cache and compression** — exact-match caching with tagged invalidation plus opt-in reversible compression, across proxy, MCP, and skill surfaces.
- **Stability** — a long defect-and-audit sweep across ingestion, retrieval, mesh, and the dashboard hardens the everyday path.

SLM is one strand of Qualixar's work on AI reliability engineering: making agent behavior observable, bounded, and reproducible instead of best-effort.

The architecture evaluated in the V3 paper remains the foundation of this release. The figures below keep their original LoCoMo protocol, answer-construction, model, and sample scope.

### How SLM fits beside other memory systems

Different products solve different boundaries. SLM is for developers who want
one local-first operating control plane—not only an SDK, managed context API,
or agent runtime. It combines dated evidence, graph-aware retrieval, cache and
compression controls, trusted-peer Mesh, and MCP/CLI/hooks/dashboard/IDE
surfaces in one install.

| If your primary need is… | Product boundary to evaluate |
|---|---|
| Local-first agent memory plus operations, optimization, and IDE-agent surfaces | **SuperLocalMemory** — Mode A local core; Modes B/C by explicit choice. |
| A memory SDK, self-hosted server, or managed platform | [Mem0](https://github.com/mem0ai/mem0) |
| A temporal context-graph service or graph engine | [Zep / Graphiti](https://github.com/getzep/graphiti) |
| A stateful agent runtime with memory blocks and archival memory | [Letta](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy) |
| LangGraph-native memory primitives and managers | [LangMem](https://github.com/langchain-ai/langmem) |
| A context API/app with profiles, connectors, and RAG | [Supermemory](https://github.com/supermemoryai/supermemory) |
| User profiles and event-timeline memory | [Memobase](https://github.com/memodb-io/memobase) |

See the [source-linked market comparison](https://superlocalmemory.com/comparison)
for current primary sources and protocol-scoped benchmark evidence. A LoCoMo
percentage is comparable only when the dataset scope, answer model, judge,
retrieval stack, and release artifact match.

### The V3.8.0 capability architecture

SuperLocalMemory is one local control plane for persistent agent context. It is
not just a vector store: the same runtime can accept evidence, build and govern
memory, retrieve bounded evidence for an agent, and expose cache, compression,
and peer-coordination controls through a CLI, MCP, dashboard, and supported
IDE integrations.

![SuperLocalMemory V3 capability architecture: modes, seven operating layers, Scale Engine, Mesh, delivery surfaces, and opt-in adapters](docs/assets/slm-v37-capability-architecture.png)

*Architecture boundary: SQLite + sqlite-vec remain canonical; CozoDB and
LanceDB are parity-gated projections; Mesh coordinates trusted peers rather
than replicating a distributed memory database; connectors are opt-in.*

**Memory boundaries:** profiles isolate workspaces by default. Every memory is
`personal`, `shared` with named profile readers, or `global`; cross-profile
recall is default-deny and must be explicitly enabled. This scoped sharing is
local authorization, not SLM Mesh synchronization. See
[shared-memory.md](docs/shared-memory.md).

```text
 IDEs, agents, scripts, connectors, and humans
             │  CLI · MCP (HTTP/stdio) · hooks · dashboard
             ▼
 ┌────────────────────────── SLM CONTROL PLANE ──────────────────────────┐
 │  1. Admission       identity, scope, idempotency, raw evidence         │
 │  2. Queryable core  SQLite facts + FTS durable receipt                  │
 │  3. Enrichment      facts, entities, scenes, time, provenance, graph   │
 │  4. Memory brain    feedback, patterns, rewards, consolidation          │
 │  5. Retrieval       semantic · BM25 · temporal · Hopfield · activation │
 │  6. Context safety  policy, trust, provenance, redaction, budgets      │
 │  7. Operations      lifecycle, audit, cache/compress, mesh, backups    │
 └───────────────────────────────────────────────────────────────────────┘
             │
             ▼
 SQLite + sqlite-vec canonical store  ──► optional graph/vector projections
```

The seven stages are an execution model, not a promise that every optional
enricher or retrieval channel runs for every request. The receipt, trace, and
health surfaces expose the stages actually completed by the installed runtime.

| Capability | What ships today | Operator boundary |
|---|---|---|
| **Memory types and lifecycle** | Atomic facts, episodic scenes, temporal events, canonical entities, profiles/scopes, consolidation, forgetting and retention controls | Lifecycle policies and retention decisions remain operator-configured. |
| **Memory boundaries** | Profile-isolated workspaces plus `personal`, `shared`, and `global` memory scopes | Personal is the default; shared/global recall requires explicit scope policy or per-call opt-in. |
| **Ingestion** | Durable raw-to-complete operation state, fact extraction, entity resolution, graph/temporal/provenance derivations, and replay-safe identity | `--sync` waits for declared stages; dependencies and mode determine which enrichers are available. |
| **Retrieval and recall** | Semantic, lexical, temporal, Hopfield and spreading-activation candidate channels; RRF fusion, optional reranking and graph score enhancement | Healthy channels participate; response provenance states the evidence used. |
| **Brain and learning** | Behavioral patterns, feedback/outcome records, rewards, consolidation, LightGBM-related ranking components, soft prompts, and guarded skill-evolution workflows | Learning is evidence-driven; it does not claim autonomous correctness or guaranteed improvement. |
| **Knowledge graph and entities** | Canonical entities, aliases, entity profiles, graph edges, scenes, timelines, explorer and graph APIs | Stored/derived graph data is evidence, not an instruction authority. |
| **Scale Engine** | SQLite + sqlite-vec are canonical. CozoDB graph and LanceDB vector projections are managed with prepare → verify → promote → rollback; a structurally detected pre-v3.7 projection can be explicitly adopted. | Promotion is parity-gated and crash-recoverable. Legacy adoption preserves the prior projection as a rollback backup; repeated physical edge rows normalize to one logical edge with the strongest weight. |
| **Optimize** | Exact cache, tagged invalidation, safe compression, opt-in aggressive prose compression, CCR originals, proxy/MCP/skill surfaces | Only proxy intercepts a primary provider turn. MCP/skill cache results explicitly routed through SLM. |
| **Mesh** | Authenticated peer messages, inbox/outbox, locks, offline queue, optional discovery and mesh MCP tools | Mesh is coordination, not automatic replicated memory or conflict resolution. |
| **Governance and operations** | Provenance, audit/retention/policy surfaces, export/erasure controls, diagnostics, health, backups and daemon lifecycle | These are engineering controls, not a legal certification. |
| **Integrations** | CLI, Python SDK, MCP HTTP/stdio, Claude plugin, Codex add-on, supported IDE configurations, Gmail/Calendar/transcript adapters | Hooks, IDE edits, connectors, and networked adapters require explicit operator activation. |

### What the dashboard exposes

`slm dashboard` opens a local operational view of the same control plane:

| Workspace | Use it to inspect or control |
|---|---|
| Dashboard and Health | daemon identity, storage/runtime health, diagnostics and recent activity |
| Brain | consolidation, behavioral patterns, outcomes/rewards, learning state and soft prompts |
| Knowledge Graph and Memories | graph neighborhoods, entities, scenes, temporal evidence, memory inspection and mutation |
| Operations | ingestion-operation state, traces, maintenance and lifecycle work |
| Entity Explorer and Skill Evolution | compiled entity summaries/timelines; opt-in skill lineage, budgets and verification outcomes |
| Multi-Agent Memory | per-agent write activity and attribution; memories stamped by `SLM_AGENT_ID`, agent write counts, and trust signals |
| Mesh Peers | configured peers, inbox/outbox, pending coordination and locks |
| Settings and Optimize | mode/provider/configuration; cache, compression and savings telemetry |

Dashboard visibility is not a substitute for runtime proof: use `slm doctor`,
`slm health`, `slm trace`, and the relevant CLI/MCP operation to validate a
deployment.

### Watch the product walkthrough

[![Watch the SuperLocalMemory demo](https://img.youtube.com/vi/PMWW_ypsL60/hqdefault.jpg)](https://www.youtube.com/watch?v=PMWW_ypsL60)

**[Watch the SuperLocalMemory demo on YouTube](https://www.youtube.com/watch?v=PMWW_ypsL60)** — a five-minute walkthrough of installation, setup, recall, cache, and compression. The video shows a product walkthrough; use the commands and release notes in this README as the current release contract.

### Published LoCoMo evidence (V3 architecture, carried into V3.8.0)

The V3 paper evaluates the architecture carried into V3.8.0. Every figure below
is protocol-scoped, so a reader can distinguish local retrieval, answer
construction, and cloud-assisted evaluation rather than treating unlike runs as
one score.

| Published configuration | LoCoMo aggregate | Protocol scope | What the result establishes |
|---|---:|---|---|
| **Mode A Raw** | **60.4%** | 10 conversations; 1,276 scored questions; local embeddings, local retrieval, and zero-LLM answer construction | End-to-end local answer construction under the published V3 protocol. |
| **Mode A Retrieval** | **74.8%** | 10 conversations; 1,276 scored questions; local retrieval, then GPT-4.1-mini answer synthesis | Retrieval evidence: local retrieval contributes the evidence, while the disclosed external model constructs the final answer. |
| **Mode C** | **87.7%** | Conv-30 only; 81 scored questions; text-embedding-3-large plus GPT-4.1-mini answer generation and judge | Cloud-assisted configuration on one fully disclosed conversation; not a full-dataset result. |

Published category results: Mode A Retrieval scored **72.0%** single-hop,
**70.3%** multi-hop, **80.0%** temporal, and **85.0%** open-domain. Mode C
scored **64.0%** single-hop, **100.0%** multi-hop, and **86.0%** open-domain
on its 81-question Conv-30 scope (no temporal category was reported for that
run). Across six LoCoMo conversations, the paper reports **71.7%** with the
information-geometric layers versus **58.9%** without them: **+12.7pp**.

See [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) and the [official
LoCoMo paper](https://arxiv.org/abs/2402.17753) for the full protocol,
ablation table, and limitations. These are published V3 architecture results
carried into V3.8.0—not a substitute for a newly rerun release-artifact benchmark.

---

## Quick Start

```bash
# Primary path 1 — npm global CLI (Node 18+)
# Creates a package-owned virtual environment. It does not modify system Python.
npm install -g superlocalmemory
slm setup       # Choose mode (A/B/C)
slm doctor      # Verify everything is working
```

```bash
# Primary path 2 — Python CLI + SDK in an activated virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
slm setup
slm doctor
```

```bash
# First use
slm remember "Alice works at Google as a Staff Engineer" --json
slm recall "What does Alice do?"
slm status
```

The default daemon write commits raw evidence plus a relational/FTS projection
and returns a durable receipt in `queryable` state. Enrichment then advances the
same operation through `enriching` to `complete`, or records a retryable
`failed` state. Use `slm remember "..." --sync` when the caller must wait for
all declared derivation and projector stages. JSON output includes the opaque
`operation_id`, current `materialization_state`, and fact IDs.

```bash
# Wrap your agent — starts proxy + sets environment + launches agent
slm wrap claude
# Your first repeat prompt → CACHE HIT → $0.00
# See savings: slm optimize savings --since 1
```

**Upgrading:** use the owner of the installation: `npm update -g superlocalmemory`
or, while the Python virtual environment is active,
`python -m pip install --upgrade superlocalmemory`. Then run
`slm restart && slm doctor`. Repository-clone users use the matching `upgrade`
action in `scripts/install.sh` or `scripts/install.ps1`. Installers never move
or delete memory data.

---

## Three Pillars

### Memory

<a id="dual-interface-mcp--cli"></a>

Current recall has five candidate producers—dense semantic, BM25 lexical,
temporal, Hopfield associative, and spreading activation—followed by fusion,
optional reranking, and entity-graph score enhancement. The entity graph does
not create an independent candidate in the current implementation. Core memory
is SQLite-backed. SQLite and sqlite-vec remain the canonical source of truth.
The packaged Scale Engine can maintain CozoDB graph and LanceDB vector
projections, and it remains outside active retrieval paths until a staged
parity witness proves it matches the canonical store. New installations remain
on Local Core. During upgrade, `slm db scale status` can identify a positive
pre-v3.7 layout candidate; the operator confirms it with `slm db scale adopt`.
SLM then rebuilds from canonical SQLite, verifies it, and promotes it with a
durable recovery journal while retaining the prior directories as a rollback
backup. `adopt` reports `restart_required: true`; run `slm restart` before
checking daemon health. If proof fails, recall remains on SQLite and status
retains the rejected manifest for inspection, retires its replaceable derived
payload, and allows a corrected retry.

Canonical ingestion is a durable state machine: `raw → queryable → enriching →
complete`, with `failed` retaining raw evidence, error details, attempt count,
and retry timing. SQLite relational facts and FTS are the queryable checkpoint;
optional ANN/vector projectors are verified before `complete` is granted.

Recalled text is treated as untrusted evidence. Hooks, MCP `session_init`, CLI
session context, and chat use one bounded renderer that redacts recognized
secrets, neutralizes forged boundary markers, and attaches provenance. Trusted
IDE instruction files contain only the static SLM protocol; fresh memory is
retrieved at runtime rather than copied into those files.

**Score Contract v2:** `relevance_score` is query-relative relevance;
`ranking_score` is internal ranking utility; `memory_confidence` belongs to the
stored assertion; and `trust_score` is an evidence-policy signal. Legacy
`score` and `confidence` remain aliases for one compatibility release. V3.8.0 is
explicitly uncalibrated: `calibration_status` is `uncalibrated` and
`answer_confidence` is `null`. See
[the retrieval score contract](docs/retrieval-score-contract.md).

The retrieval/lifecycle implementation includes three mathematical layers that
can run without a cloud LLM:

1. **Fisher-informed scoring** — dense candidate generation uses cosine similarity; Fisher-derived terms can modify later scoring when their state is available.
2. **Sheaf Cohomology for Consistency** — algebraic topology detects contradictions via coboundary norms on the knowledge graph.
3. **Riemannian Langevin Lifecycle** — memory positions evolve on the Poincare ball; neglected memories self-archive, no hardcoded thresholds.

Auto-capture hooks are installed explicitly with `slm hooks install` (Claude
Code) or `slm hooks install --agent codex` (Codex). Hook latency and capture
quality must be evaluated for the target client and workload; V3.8.0 publishes no universal p99 claim.

**Multi-scope memory (v3.6.15, opt-in):** keep memories `personal` (default), `shared` with named profiles, or `global` across the machine. Off by default — recall only ever returns your own facts until you turn sharing on, per call or in config. See **[docs/shared-memory.md](docs/shared-memory.md)**.

<a id="multilingual-embedding-support"></a>

**Multilingual models:** configure an OpenAI-compatible embedding endpoint such as Ollama, vLLM, LiteLLM, `bge-m3`, `multilingual-e5`, or `Qwen3-Embedding`. Language coverage and retrieval quality depend on the selected model and should be evaluated for the deployment corpus.

### Cache + Compress

<a id="three-surfaces-proxy--mcp-tools--skill"></a>

One engine, three ways in — choose the surface that fits your setup:

| Surface | How you use it | Requires proxy? | Window effect | Cache scope |
|---------|---------------|:---------------:|:-------------:|-------------|
| **A — Proxy** | `slm wrap claude` or `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` | **Yes** | Shrinks | Full-turn cache — every call |
| **B — MCP tools** | Add 5 tools to MCP config; call `slm_compress`, `slm_cache_set/get` | **No** | **Preserved (1M)** | Results you explicitly route through SLM |
| **C — Skill** | Copy `skills/slm-optimize/SKILL.md` → `~/.claude/skills/` | **No** | **Preserved (1M)** | Auto-applied by the agent per skill rules |

**The hard constraint:** The primary Claude conversation turn cannot be cached without a proxy. The MCP/skill path caches results you explicitly route through SLM (tool outputs, file reads, sub-model calls) — without a proxy the main conversation turn is not intercepted.

**How to choose:**
- Metered API (pay-per-token), want every call cached → **Proxy (A)**
- Pro/Max/Team subscription or any plan where you won't run a proxy → **MCP tools (B)** or **Skill (C)**
- Zero configuration → **Skill (C)**: install once, auto-compresses CLAUDE.md and large outputs
- Agent-controlled caching of repeated file reads → **MCP tools (B)**

**Cache:** exact-match SQLite lookup is the stable cache path. Semantic cache
controls are experimental until release-linked precision, invalidation, and
tenant-isolation evidence exists. A cache hit can avoid a provider request, but
actual cost and latency savings depend on the intercepted surface and provider.

**Compress:** safe mode uses conservative normalization and preserves JSON and code; measured reduction varies by content and can be zero. Aggressive prose compression is opt-in and lossy. CCR can retain an original for later byte-exact retrieval when reversible storage is enabled.

**Savings dashboard:** `slm optimize savings --since 7` — live USD/INR/tokens saved. Hot-reload config, fail-open.

### Mesh

<a id="multi-machine-mesh-coordination"></a>

Mesh provides authenticated coordination messages between configured peers, with an offline queue and optional mDNS discovery (`SLM_MESH_DISCOVERY=on`). It is not a replicated or conflict-resolving distributed-memory database.

```bash
# Machine A (broker)
export SLM_MESH_HOST=192.168.1.100
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init

# Machine B (client)
export SLM_MESH_PEER_URL=http://192.168.1.100:8765
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init
```

8 mesh MCP tools: `mesh_summary`, `mesh_peers`, `mesh_send`, `mesh_inbox`, `mesh_state`, `mesh_lock`, `mesh_events`, `mesh_status`.

Full docs: [docs/multi-machine.md](docs/multi-machine.md) · [docs/distributed-deployment.md](docs/distributed-deployment.md)

---

## Install Paths

| Path | Command | When |
|:-----|:--------|:-----|
| **npm global CLI** (primary) | `npm install -g superlocalmemory` | Node 18+; package-owned virtual environment; system Python is not modified; run `slm setup` explicitly afterward |
| **Python CLI + SDK** (primary) | Activate a Python virtual environment, then `python -m pip install superlocalmemory` | Python 3.11+; the `slm` CLI and importable SDK stay inside that environment |
| **Repository clone — macOS/Linux** | `./scripts/install.sh install` | Research/contributor path; delegates to an existing uv or pipx installation |
| **Repository clone — Windows** | `.\scripts\install.ps1 -Action Install` | Research/contributor path; delegates to an existing uv or pipx installation |
| **Claude Code Plugin** (WP-06) | `/plugin marketplace add qualixar/superlocalmemory` then `/plugin install superlocalmemory@qualixar` | Self-bootstraps venv, isolated SLM_DATA_DIR, additive — 14-tool core. Ships the skills/agents/hooks/commands |
| **Portable / IDE connect** (WP-08) | `slm connect <ide> [--here]` | Wire any IDE without reinstalling; `slm connect claude-code` → plugin pointer |

After any install path: `slm setup` → `slm doctor` → `slm warmup` (optional, pre-downloads ~500MB embedding model).

| Component | Size | When |
|:----------|:-----|:-----|
| Core libraries (numpy, scipy, networkx) | ~50MB | During install |
| Dashboard & MCP server (fastapi, uvicorn) | ~20MB | During install |
| Learning engine (lightgbm) | ~10MB | During install |
| Search engine (sentence-transformers, torch) | ~200MB | During install |
| Embedding model (nomic-embed-text-v1.5, 768d) | ~500MB | First use or `slm warmup` |
| **Mode B** requires [Ollama](https://ollama.com) + a model (`ollama pull llama3.2`) | ~2GB | Manual |

---

## MCP + Profiles

SLM supports two MCP transports:

**HTTP (recommended, v3.6.7+):**
```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```
Or: `claude mcp add --transport http superlocalmemory http://127.0.0.1:8765/mcp/`

**stdio (universal fallback):**
```json
{ "mcpServers": { "superlocalmemory": { "command": "slm", "args": ["mcp"] } } }
```

### MCP Profiles (WP-01)

Control tool surface via `SLM_MCP_PROFILE`:

| Profile | Tools | Use case |
|:--------|:-----:|:---------|
| `core` | 14 | Memory, session, and optimize core |
| `code` | 24 | Core + code-graph tools + profile switching + bounded loops |
| `mesh` | 8 | Mesh-only — multi-machine coordination |
| `full` | 42 | Memory + optimize + evolution + mesh + bounded loops |
| `power` | 54 | Full + administration, lifecycle, and diagnostics |
| `whole` | all registered | Every registered MCP tool |

**Precedence:** `ALL` > `TOOLS` > `PROFILE` > `default`

```bash
export SLM_MCP_PROFILE=full   # or core / code / mesh / power / whole
slm mcp
```

For a predictable small surface, set `core` explicitly. Leaving the variable
unset retains the compatibility default, whose mesh tools follow the local
mesh setting. Count-suffixed aliases remain for backward compatibility and emit a migration warning: `core14`, `code20`, `code21`, `code24`, `mesh8`, `full38`, `full39`, `full42`, `power50`, `power51`, `power54`, `whole81`, `whole84`. Unknown names stop startup instead of silently selecting another tool set.

Per-IDE configs available for Claude Code, Cursor, Windsurf, VS Code Copilot, Continue, Gemini CLI, JetBrains, Zed, and more (15 configs in `ide/configs/`). See [docs/ide-setup.md](docs/ide-setup.md).

---

## Claude Code Plugin

Install directly in Claude Code — no system-level npm/pip needed. This is how you
get the **skills, agents, hooks, commands, and rules** (the MCP server is
bootstrapped automatically). It is a two-step flow — add the marketplace once,
then install:

```bash
# 1. Add the Qualixar marketplace (one-time — the repo IS the marketplace)
/plugin marketplace add qualixar/superlocalmemory

# 2. Install the plugin
/plugin install superlocalmemory@qualixar
```

- Self-bootstraps a Python venv, installs all deps in an isolated `SLM_DATA_DIR`
- Registers the 14-tool core MCP surface (`core14` profile by default)
- Ships the SLM skills / agents / hooks / commands / rules
- Additive — does not replace an existing SLM install
- `slm connect claude-code` detects an existing plugin install and links them

> **Plugin vs Python/npm:** `python -m pip install superlocalmemory` inside an
> activated virtual environment, or `npm i -g superlocalmemory`,
> give you the `slm` CLI + the MCP server (the *tools*). The **skills/agents/hooks/
> commands** come only through the plugin above. Use the plugin for Claude Code; use
> pip/npm for the CLI or other IDEs.

To update later: `/plugin marketplace update qualixar` then `/plugin install superlocalmemory@qualixar`.

## Codex add-on

For Codex, install the SLM-owned skills, two focused subagents, and four
lifecycle hooks explicitly:

```bash
slm codex install
```

This adds only SLM-owned files under `~/.agents/skills`, `~/.codex/agents`, and
`~/.codex/hooks.json`; it does not replace another agent's hooks or rewrite
`~/.codex/config.toml`. Codex requires review and trust for new command hooks:
open `/hooks` after installation. MCP wiring remains a separate explicit step:

```bash
slm connect codex
```

`slm connect codex` semantically merges the `superlocalmemory` MCP server into
`~/.codex/config.toml`, preserving unrelated configuration keys and writing
atomically. TOML serializers can normalize whitespace and comments, so it is
not a byte-preserving operation; use it only when you want the MCP server
configured. Check the result with `slm codex status`; undo SLM-owned add-ons
with `slm codex remove`.

## GitHub Copilot integration

The shipped installer configures the SuperLocalMemory MCP server and additive
agent instructions for VS Code with GitHub Copilot:

```bash
slm connect vscode-copilot --here
```

Run it from the project root. It semantically merges the SLM server into
`.vscode/mcp.json` and adds SLM-owned guidance inside
`.github/copilot-instructions.md`, preserving unrelated servers and existing
instructions. The generated `copilot-plugin/` source bundle is maintained for
parity checks, but v3.8.1 does not claim that `slm connect` installs its prompt,
agent, or hook files.

---

## Privacy controls and operating modes

<a id="privacy-controls-and-operating-modes"></a>

| Mode | What | Core memory path | Optional network behavior |
|:----:|:-----|:-----------------|:--------------------------|
| **A** | Local Guardian | Local processing | Model/dependency downloads, connectors, backup, and other enabled integrations may use the network |
| **B** | Smart Local | Local Ollama enrichment | Same optional integrations as Mode A |
| **C** | Provider-assisted | Local storage with provider calls | Query or enrichment content is sent to the configured provider |

```bash
slm mode a   # Zero-cloud (default)
slm mode b   # Local Ollama
slm mode c   # Cloud LLM
```

Mode A can run core memory operations without sending memory content to a cloud model provider. This does not disable optional connectors, cloud backup, proxy providers, dependency acquisition, or model downloads; review configuration and network policy for the deployment.

SuperLocalMemory provides local storage, export/erasure commands, provenance, policy, and audit features that can support a compliance program. The software is not a legal certification, and compliance depends on the use case, operator, configuration, and surrounding systems.

Available controls include local export and erasure commands, hash-chained audit records, provenance tracking, and ABAC policy enforcement. Verify their behavior and retention boundaries for your deployment; see [docs/compliance.md](docs/compliance.md).

---

## Teams and Enterprise Memory (v3.8.0)

V3.8.0 adds multi-user, multi-workspace controls for teams and organizations. These are opt-in — personal single-user installs work exactly as before with no required login.

### Users and roles

SLM supports three role tiers within a workspace: **admin**, **member**, and **viewer**.

| Role | Can read memory | Can write memory | Can manage users/config |
|------|:---------------:|:----------------:|:-----------------------:|
| admin | yes | yes | yes |
| member | yes | yes | no |
| viewer | yes | no | no |

Roles are scoped per workspace (profile). A user may have different roles in different workspaces.

### Workspace isolation

Each workspace (profile) is a fully isolated memory namespace. One workspace cannot read another's personal memories. Shared and global scopes are opt-in and still profile-bounded at the authorization layer.

### Login gate

Enterprise deployments set `require_login = true` in configuration. With login enabled:
- Every dashboard and API request requires an authenticated session.
- First-run creates an admin account with a user-chosen password (no default credentials are shipped).
- Session cookies use `HttpOnly` with optional `Secure` enforcement.
- Personal installs run with `require_login = false` (loopback owner is trusted).

```bash
slm config set security.require_login true   # Enable for team/enterprise use
```

### Memory scopes

| Scope | Who can recall | Set with |
|-------|---------------|----------|
| `personal` | Owner profile only (default) | `slm remember "..." --scope personal` |
| `shared` | Named profiles the owner grants | `slm remember "..." --scope shared --shared-with profile-a,profile-b` |
| `global` | Any authorized user on this machine | `slm remember "..." --scope global` |

Recall is default-deny: shared and global facts are never returned unless the caller explicitly opts in (`--include-shared`, `--include-global`) or the scope policy allows it. See [docs/shared-memory.md](docs/shared-memory.md).

### GDPR and data governance

SLM ships built-in controls that support GDPR compliance programs:

- **Export** — full profile data export as a structured JSONL bundle
- **Erasure** — profile deletion removes data from 30+ scoped tables; erasure is logged to the tamper-proof audit chain before any data is deleted
- **Retention rules** — time-based policies (`indefinite`, `gdpr-30d`, `hipaa-7y`, `custom`) applied per profile
- **Audit trail** — every store, recall, mutation, and erasure produces a hash-chained audit record
- **PII redaction** — configurable automatic redaction before memory content crosses trust boundaries

These are engineering controls. Compliance depends on deployment configuration, use case, and operator responsibility. See [docs/compliance.md](docs/compliance.md).

### EU AI Act mode verification

SLM includes a per-mode EU AI Act self-assessment. The `EUAIActChecker` produces a compliance report for the active operating mode — risk category, whether data stays local, whether generative AI is used, and transparency / human-oversight signals:

- **Mode A (Local Guardian)** and **Mode B (Smart Local)** — assessed as compliant: memory processing stays local and uses no generative AI.
- **Mode C (Provider-assisted)** — flagged non-compliant, because query or enrichment content is sent to a cloud model provider.

This is operator self-assessment tooling, not a legal certification or conformity assessment; actual EU AI Act obligations depend on your system, deployment, and role. See [docs/compliance.md](docs/compliance.md).

### Deployment tiers

SLM ships one binary and is configured for the appropriate tier at install or post-install time.

| Tier | Login gate | PII redaction | Retention | Audit |
|------|:---------:|:-------------:|:---------:|:-----:|
| **Personal** | off | off | off | on |
| **Enterprise** | on | on | on | on |

The installer or `slm reconfigure` sets the tier. Each setting is independently overridable at runtime. Full tier documentation: [docs/deployment-tiers.md](docs/deployment-tiers.md).

### RBAC and teams docs

Full reference: [docs/rbac-teams.md](docs/rbac-teams.md) · [docs/deployment-tiers.md](docs/deployment-tiers.md)

---

## Bounded Loops (v3.8.0)

A bounded loop terminates only when an **independent gate** passes — a test
suite exit code, a linter, a JSON-schema check, or an SLM-recall condition.
The agent's own "I finished" message is recorded as advisory context and never
used as the termination signal. Every lap is persisted to SLM memory under the
tag `loop:<name>`, so runs are auditable and resumable across sessions.

Three surfaces ship together:

| Surface | How you use it |
|---------|---------------|
| **CLI** | `slm loop demo` · `slm loop history [--name <n>]` · `slm loop show <run_id>` |
| **Skill + agent** | `/slm-loop` skill with the `slm-loop-runner` agent — delegate a task that has a checkable acceptance condition |
| **MCP tools** | `slm_loop_run` · `slm_loop_history` · `slm_loop_show` — call from any IDE or agent (available in the `code` and `full` MCP profiles) |

```bash
# Run the built-in convergence demo (no API key needed)
slm loop demo

# Inspect recorded runs
slm loop history --name convergence-demo
slm loop show <run_id>
```

Loop laps are stored as ordinary SLM memories and are visible in the dashboard
under Knowledge Graph and Memories (filter by tag `loop:<name>`) and in the
Multi-Agent Memory workspace.

---

## Framework Adapters (v3.8.0)

SLM ships nine framework adapters under `ide/integrations/`. Each adapter
wires SLM as the memory and history provider for the respective framework
without replacing the framework's own agent runtime.

| Framework | Directory |
|-----------|-----------|
| LangGraph | `ide/integrations/langgraph/` |
| Semantic Kernel | `ide/integrations/semantic-kernel/` |
| Microsoft Agent Framework | `ide/integrations/agent-framework/` |
| LangChain | `ide/integrations/langchain/` |
| LlamaIndex | `ide/integrations/llamaindex/` |
| CrewAI | `ide/integrations/crewai/` |
| AutoGen | `ide/integrations/autogen/` |
| Google ADK | `ide/integrations/google-adk/` |
| OpenAI Agents | `ide/integrations/openai-agents/` |

Pydantic AI is not included — it does not expose a formal memory interface for
external providers. Each adapter's `README.md` covers installation and
configuration for that framework.

---

## Advanced

| Topic | Link |
|:------|:-----|
| Full optimize docs | [docs/optimize-overview.md](docs/optimize-overview.md) · [docs/optimize-cli.md](docs/optimize-cli.md) · [docs/optimize-config.md](docs/optimize-config.md) |
| Distributed deployment | [docs/distributed-deployment.md](docs/distributed-deployment.md) |
| Multi-machine mesh | [docs/multi-machine.md](docs/multi-machine.md) |
| Auto-memory hooks | [docs/auto-memory.md](docs/auto-memory.md) |
| Architecture + math | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Published benchmark evidence | [docs/benchmarks.md](docs/benchmarks.md) |
| CLI reference | [docs/cli-reference.md](docs/cli-reference.md) |
| MCP tools reference | [docs/mcp-tools.md](docs/mcp-tools.md) |
| Getting started | [docs/getting-started.md](docs/getting-started.md) |
| IDE setup (15 configs) | [docs/ide-setup.md](docs/ide-setup.md) |
| Teams, users, and RBAC | [docs/rbac-teams.md](docs/rbac-teams.md) |
| Deployment tiers | [docs/deployment-tiers.md](docs/deployment-tiers.md) |
| pi.dev integration | [docs/pi-dev-integration.md](docs/pi-dev-integration.md) |
| Skill evolution | [docs/skill-evolution.md](docs/skill-evolution.md) |
| V2 migration | [docs/migration-from-v2.md](docs/migration-from-v2.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Retrieval score contract | [docs/retrieval-score-contract.md](docs/retrieval-score-contract.md) |
| Wiki | [github.com/qualixar/superlocalmemory/wiki](https://github.com/qualixar/superlocalmemory/wiki) |

Open the web dashboard with `slm dashboard`; workspaces appear only when their
runtime capability is enabled and healthy. See [CHANGELOG.md](CHANGELOG.md) for
the complete release history.

## Research Papers

SuperLocalMemory is backed by three preprints by Varun Pratap Bhardwaj (2026):

- **The Living Brain (V3.3):** [arXiv:2604.04514](https://arxiv.org/abs/2604.04514) · [Zenodo 19435120](https://zenodo.org/records/19435120)
- **Information-Geometric Foundations (V3):** [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) · [Zenodo 19038659](https://zenodo.org/records/19038659)
- **Trust & Behavioral Foundations (V2):** [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) · [Zenodo 18709670](https://zenodo.org/records/18709670)

Use the citation metadata on the linked arXiv or Zenodo records.

## Support / License / Qualixar

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. [Wiki](https://github.com/qualixar/superlocalmemory/wiki) for detailed documentation.

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

For commercial licensing (closed-source, proprietary, or hosted use), see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or contact varun.pratap.bhardwaj@gmail.com.

Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar.

Part of [Qualixar](https://qualixar.com) · Author: [Varun Pratap Bhardwaj](https://varunpratap.com)

### Acknowledgments

- **[Everything Claude Code (ECC)](https://github.com/affaan-m/everything-claude-code)** inspired SLM's skill-observation patterns; SLM can ingest ECC observations with `slm ingest --source ecc`.
- **[HKUDS/OpenSpace](https://github.com/HKUDS/OpenSpace)** informed the skill-evolution verification design (arXiv:2604.01687).

### Qualixar AI Agent Reliability Platform

Qualixar builds open-source infrastructure for AI reliability engineering.
Start at **[qualixar.com](https://qualixar.com)** or browse the
[Qualixar research archive](https://huggingface.co/Qualixar).

## Star This Project

If this project solves a real problem for you, **please star the repo** — it helps other developers discover Qualixar and signals that the AI agent reliability community is growing.

[![Star SuperLocalMemory on GitHub](https://img.shields.io/github/stars/qualixar/superlocalmemory?style=for-the-badge&logo=github&label=Star%20on%20GitHub)](https://github.com/qualixar/superlocalmemory)

The live Star History chart is intentionally not embedded: its upstream service timed out during release validation. The link above is the stable, direct way to star and follow the repository.
