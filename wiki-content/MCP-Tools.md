# MCP Tools — V4.0.4 (92 whole)

SuperLocalMemory exposes profile-selected tools and resources through the Model
Context Protocol (MCP). The installed profile registry (`src/superlocalmemory/mcp/profiles.py` and `src/superlocalmemory/mcp/server.py`) is the source of truth
for names and counts. An MCP-compatible client still decides when to call a
tool.

> **Current V4 profile counts (from `CHANGELOG.md` and the MCP exposure contract `tests/test_mcp/test_mcp_exposure_contract.py` / `tests/mcp/test_profile_selector.py`):**
> `core` **14**, `code` **29** (installed coding agents), `full` **47**, `power` **59**, `whole` **92** (all registered), plus `mesh` **8**. The unrestricted default surface is **47** with mesh enabled. See also `src/superlocalmemory/mcp/profiles.py`.

> **Optimize tools:** `slm_compress`, `slm_retrieve`, `slm_cache_set`,
> `slm_cache_get`, and `slm_optimize_stats` provide explicit compression and
> routed-result caching. They do not intercept the primary conversation turn
> without a proxy.

> **V3.1 New (carried into V4):** 3 Active Memory tools (`session_init`, `observe`, `report_feedback`) and 1 resource (`slm://context`) for automatic learning and context injection.

## Starting the MCP Server

```bash
slm mcp    # Starts stdio transport — your IDE calls this automatically
```

Preferred HTTP transport (V3.6.7+):

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

Or stdio fallback:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

## Core Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `remember` | `content`, `tags?`, `project?`, `importance?`, `session_id?`, `agent_id?`, `scope?`, `shared_with?`, `idempotency_key?` | Submit durable evidence and return an operation receipt |
| `recall` | `query`, `limit?` | Retrieve relevant memories (5 producers + graph enhancement) |
| `search` | `query`, `limit?` | Search across all memories |
| `forget` | `query` | Delete matching memories |
| `fetch` | `id` | Get a specific memory by ID |
| `list_recent` | `limit?` | List recent memories |
| `get_status` | — | System status (mode, DB, count, math health) |
| `health` | — | Math layer health (Fisher, Sheaf, Langevin) |
| `build_graph` | — | Rebuild the knowledge graph |
| `get_attribution` | — | Return system attribution metadata: product name, author, organization, license, and URLs. No parameters. |
| `compact_memories` | — | Compress and optimize storage |
| `memory_used` | — | Storage usage statistics |
| `backup_status` | — | Backup and database health |
| `audit_trail` | `limit?` | Recent operations log |

`remember` returns `operation_id`, `fact_ids`, `materialization_state`, and
`pending`. The default daemon path returns after SQLite relational/FTS is
`queryable`; enrichment continues on the same durable operation. Offline replay
preserves the original source and idempotency identity.

`recall`, `search`, `recall_trace`, and session context follow [Score Contract
v2](Retrieval-Score-Contract). `relevance_score` is query relevance,
`ranking_score` is diagnostic ranking utility, and `memory_confidence` belongs
to the stored assertion. V4 (like V3.8.0) declares `calibration_status: "uncalibrated"`
and `answer_confidence: null`.

## Active Memory Tools (V3.1, carried into V4)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `session_init` | `project_path?`, `query?` | Auto-recall project context at session start. Returns relevant memories + learning status. Call once at the beginning of every session. |
| `observe` | `content` | Send conversation content for auto-capture. Detects decisions, bug fixes, and preferences. Stores automatically when confidence > 0.5. |
| `report_feedback` | `fact_id`, `feedback`, `query?` | Report whether a recalled memory was useful. Feedback: "relevant", "irrelevant", or "partial". Trains the adaptive ranker. |

## Management Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `switch_profile` | `name` | Switch to a different memory profile |
| `set_retention_policy` | `days`, `categories?` | Set data retention period |
| `report_outcome` | `memory_id`, `outcome` | Report whether a recalled memory was helpful |
| `correct_pattern` | `pattern_id`, `correction` | Correct a learned behavioral pattern |
| `get_behavioral_patterns` | `limit?` | View learned patterns |
| `get_learned_patterns` | `limit?` | View ML-learned recall patterns |

## V3 Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `recall_trace` | `query`, `limit?` | Recall with per-channel score breakdown (5 producers) |
| `get_lifecycle_status` | `limit?`, `status?` | Memory lifecycle health (active/warm/cold counts) |
| `consistency_check` | — | Run sheaf consistency verification |
| `set_mode` | `mode` | Switch operating mode (a/b/c) |
| `get_mode` | — | Current operating mode |

## Resources

MCP resources provide read-only data streams that IDEs can subscribe to.

| Resource | URI | Description |
|----------|-----|-------------|
| Active Context | `slm://context` | Active session context auto-injected on MCP connect. Returns relevant memories + learning status. |
| Recent Memories | `slm://recent` | The 20 most recently stored memories |
| Memory Stats | `slm://stats` | Memory count, database size, mode, profile |
| Topic Clusters | `slm://clusters` | Topic clusters detected across memories |
| Identity | `slm://identity` | Learned user preferences and patterns |
| Learning State | `slm://learning` | Current state of the adaptive learning system |
| Engagement | `slm://engagement` | Usage statistics and interaction patterns |

## Optimize Tools (v3.6.11, carried into V4)

Proxy-free compression and routed-result caching. The tools are designed to
return `ok:False` with the original content on handled optimization failures;
verify the frozen artifact before relying on that as a fault-containment
boundary.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `slm_compress` | `content`, `mode?`, `reversible?`, `ttl_seconds?` | Compress text. `mode`: `normalize` (lossless), `auto`, `aggressive`. Returns `ccr_id` when lossy+reversible. |
| `slm_retrieve` | `ccr_id` | Recover exact original from a lossy compress. |
| `slm_cache_set` | `key`, `value`, `ttl_seconds?` | Cache any string result (file read, bash output, search). Namespaced per agent. |
| `slm_cache_get` | `key` | Retrieve cached result. Returns `hit:True/False`. |
| `slm_optimize_stats` | — | Compression + cache statistics for the current session. |

> **Hard constraint:** Surfaces B and C cache results you explicitly route through SLM — not the Claude conversation turn. Full-turn caching requires Surface A (proxy).

## MCP Profiles (V4.0.4 — whole is 92)

A profile is a named, fixed subset of tools exposed to the connecting client.
Set the active profile via the `SLM_MCP_PROFILE` environment variable (or `SLM_MCP_ALL_TOOLS`/`SLM_MCP_TOOLS` overrides — see `src/superlocalmemory/mcp/server.py` precedence: `ALL > TOOLS > PROFILE > default`). `whole` exposes the raw server with all registered tools; `switch_profile` tool switches the active workspace profile (separate concept).

| Profile | Tool count | Included surfaces |
|---|---|---|
| `core` | **14** | Store, recall, search, sessions, optimize (5 tools) |
| `code` | **29** | Core + portable Brain evidence + code graph + `switch_profile` + 3 bounded-loop tools |
| `full` | **47** | All everyday memory, portable Brain evidence, optimize, and mesh tools |
| `power` | **59** | Full + governance and behavioral analysis tools |
| `mesh` | **8** | SLM-Mesh coordination only |
| `whole` | **92** | All registered tools (raw server) — verified by `tests/test_mcp/test_mcp_exposure_contract.py` `whole == 92` |

> **Why 92 not 91:** V4.0.4 adds one explicit Bounded Loops observation tool to the portable Brain surface. Earlier `whole81`/`whole84`/`whole91` aliases still resolve to `whole`; `whole92` names the current registered surface.

Legacy count-suffixed aliases (`core14`, `code20`/`code21`/`code24`/`code28`/`code29`, `full38`/`full39`/`full42`/`full46`/`full47`, `power50`/`power51`/`power54`/`power58`/`power59`, `mesh8`, and `whole81`/`whole84`/`whole91`/`whole92`) resolve to their canonical name for backward compatibility and emit a migration warning.

### Optional Bounded Loops evidence bridge

`observe_bounded_loop_evidence(workspace)` is available in the default,
`code`, `full`, `power`, and `whole` surfaces when the separate
`bounded-loops-mcp` executable is installed. It negotiates the producer's
`bounded-loops.dev/slm-bridge/v1` capability, lists terminal runs, and uses
each returned `run_ref` to fetch one sanitized receipt. It never accepts an
agent-supplied command or shell arguments.

Receipts enter SLM's active profile in `learning.db` and appear in Living Brain
as Bounded Loop observations. They are observation-only: compatible v1
evidence has `eligible_for_learning: false`, so it cannot alter recall,
ranking, routing, reward, or automatic behavioural learning.

The three bounded-loop tools (`slm_loop_run`, `slm_loop_history`,
`slm_loop_show`) are included in `code`, `full`, `power`, and `whole`.
See [Bounded Loops](Bounded-Loops) for the tool reference.

**Retrieval note:** Current recall in V4 has **five candidate producers** — dense semantic, BM25 lexical, temporal, Hopfield associative, and spreading activation — followed by RRF fusion, optional reranking, and entity-graph score enhancement (the graph does not create a separate candidate). Some pre-V4 prose described “four-channel” retrieval; the current implementation is five producers (see [Retrieval Score Contract](Retrieval-Score-Contract) and `src/superlocalmemory/retrieval/`).

Switch profiles without restarting the daemon:

```bash
slm profile switch code    # CLI — switches active workspace profile
```

Or via MCP tool:

```python
await switch_profile(name="full")
```

The active MCP profile (`SLM_MCP_PROFILE`) and the active workspace profile (`slm profile ...`) are distinct controls. The dashboard **MCP & Tools** pane shows the active MCP profile.

## How MCP Integration Works

1. Your IDE connects to the SuperLocalMemory MCP server via `slm mcp` (stdio) or HTTP `http://127.0.0.1:8765/mcp/`
2. When you chat with your AI, the IDE calls `recall` with relevant context
3. SuperLocalMemory runs the healthy subset of its 5 candidate producers, then applies fusion and optional score enhancements
4. The IDE injects those memories into the AI's context
5. Your AI responds with knowledge of your past work

Whether this happens automatically depends on the client and its configured
instructions or hooks. SLM does not control an IDE's tool-selection policy.

See [IDE Setup](IDE-Setup) for per-IDE configuration paths and [Framework Adapters](Framework-Adapters) for framework-native memory.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
