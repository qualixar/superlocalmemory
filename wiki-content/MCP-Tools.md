# MCP Tools

SuperLocalMemory exposes 32 tools and 7 resources via the Model Context Protocol (MCP). These are what your IDE uses to interact with the memory system.

> **v3.6.11 New:** 5 Optimize tools — `slm_compress`, `slm_retrieve`, `slm_cache_set`, `slm_cache_get`, `slm_optimize_stats`. Proxy-free compression + routed-result caching. Full 1M context window preserved.

> **V3.1 New:** 3 Active Memory tools (`session_init`, `observe`, `report_feedback`) and 1 resource (`slm://context`) for automatic learning and context injection.

## Starting the MCP Server

```bash
slm mcp    # Starts stdio transport — your IDE calls this automatically
```

Your IDE config should look like:

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
| `recall` | `query`, `limit?` | Retrieve relevant memories |
| `search` | `query`, `limit?` | Search across all memories |
| `forget` | `query` | Delete matching memories |
| `fetch` | `id` | Get a specific memory by ID |
| `list_recent` | `limit?` | List recent memories |
| `get_status` | — | System status (mode, DB, count, math health) |
| `health` | — | Math layer health (Fisher, Sheaf, Langevin) |
| `build_graph` | — | Rebuild the knowledge graph |
| `get_attribution` | `memory_id` | Get provenance chain for a memory |
| `compact_memories` | — | Compress and optimize storage |
| `memory_used` | — | Storage usage statistics |
| `backup_status` | — | Backup and database health |
| `audit_trail` | `limit?` | Recent operations log |

`remember` returns `operation_id`, `fact_ids`, `materialization_state`, and
`pending`. The default daemon path returns after SQLite relational/FTS is
`queryable`; enrichment continues on the same durable operation. Offline replay
preserves the original source and idempotency identity.

## Active Memory Tools (V3.1)

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
| `recall_trace` | `query` | Recall with per-channel score breakdown |
| `get_lifecycle_status` | — | Memory lifecycle health (active/warm/cold counts) |
| `consistency_check` | — | Run sheaf consistency verification |
| `set_mode` | `mode` | Switch operating mode (a/b/c) |
| `get_mode` | — | Current operating mode |

## Resources (6)

MCP resources provide read-only data streams that IDEs can subscribe to.

| Resource | URI | Description |
|----------|-----|-------------|
| Memory Stats | `memory://stats` | Total memories, storage size, profile count |
| Recent Memories | `memory://recent` | Last 10 memories stored |
| Active Profile | `memory://profile` | Current profile name and settings |
| System Health | `memory://health` | Database status, math layer scores |
| Knowledge Graph | `memory://graph` | Graph summary (nodes, edges, communities) |
| Learning State | `memory://learning` | ML model state and learned patterns |

## Optimize Tools (v3.6.11)

Proxy-free compression and routed-result caching. All five are **fail-open** — any internal error returns `ok:False` with the original content unchanged; the agent always continues.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `slm_compress` | `content`, `mode?`, `reversible?`, `ttl_seconds?` | Compress text. `mode`: `normalize` (lossless), `auto`, `aggressive`. Returns `ccr_id` when lossy+reversible. |
| `slm_retrieve` | `ccr_id` | Recover exact original from a lossy compress. |
| `slm_cache_set` | `key`, `value`, `ttl_seconds?` | Cache any string result (file read, bash output, search). Namespaced per agent. |
| `slm_cache_get` | `key` | Retrieve cached result. Returns `hit:True/False`. |
| `slm_optimize_stats` | — | Compression + cache statistics for the current session. |

> **Hard constraint:** Surfaces B and C cache results you explicitly route through SLM — not the Claude conversation turn. Full-turn caching requires Surface A (proxy).

## How MCP Integration Works

1. Your IDE connects to the SuperLocalMemory MCP server via `slm mcp`
2. When you chat with your AI, the IDE calls `recall` with relevant context
3. SuperLocalMemory runs 4-channel retrieval and returns matching memories
4. The IDE injects those memories into the AI's context
5. Your AI responds with knowledge of your past work

This happens automatically — you do not need to manually call tools.

See [IDE Setup](IDE-Setup) for per-IDE configuration paths.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
