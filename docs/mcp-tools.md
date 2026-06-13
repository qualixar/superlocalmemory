# MCP Tools Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SuperLocalMemory exposes 38 tools and 7 resources via the Model Context Protocol (MCP). Any MCP-compatible AI assistant can use these automatically.

> **v3.6.11:** Five new **Optimize tools** — `slm_compress`, `slm_retrieve`, `slm_cache_set`, `slm_cache_get`, `slm_optimize_stats`. Proxy-free compression + routed-result caching. Full 1M context window preserved. [See Three Surfaces →](optimize-overview.md)

## Connecting

SLM supports two transports — both expose the exact same tools.

**HTTP (v3.6.7+, recommended):** one shared daemon process, flat RAM.

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

**stdio (universal fallback):** one subprocess per connection.

```json
{ "mcpServers": { "superlocalmemory": { "command": "slm", "args": ["mcp"] } } }
```

See [`docs/ide-setup.md`](ide-setup.md) for per-IDE configs and the `mcp-remote` bridge option for stdio-only tools.

---

## Core Tools

### `remember`

Store a new memory.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `content` | string | Yes | The text to remember |
| `tags` | string | No | Comma-separated tags |
| `metadata` | object | No | Additional key-value metadata |

### `recall`

Search memories by natural language query.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Natural language search query |
| `limit` | number | No | Max results (default: 10) |

### `search`

Search memories with filters. More control than `recall`.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Search query |
| `limit` | number | No | Max results (default: 10) |
| `tags` | string | No | Filter by tags |
| `before` | string | No | Filter by date (ISO format) |
| `after` | string | No | Filter by date (ISO format) |

### `fetch`

Retrieve a specific memory by ID.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `id` | number | Yes | Memory ID |

### `list_recent`

List the most recently stored memories.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `limit` | number | No | Max results (default: 20) |

### `get_status`

Returns system status: mode, profile, memory count, health, database path.

*No parameters.*

### `build_graph`

Rebuild the entity relationship graph from stored memories. Useful after bulk imports.

*No parameters.*

### `switch_profile`

Switch to a different memory profile.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `profile` | string | Yes | Profile name to switch to |

### `backup_status`

Check the status of automatic backups.

*No parameters.*

### `memory_used`

Return memory usage statistics: total memories, database size, per-profile counts.

*No parameters.*

### `get_learned_patterns`

Return patterns the system has learned from your usage (e.g., preferred technologies, coding conventions).

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `limit` | number | No | Max patterns to return (default: 10) |

### `correct_pattern`

Correct a learned pattern that is wrong or outdated.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `pattern_id` | number | Yes | ID of the pattern to correct |
| `correction` | string | Yes | The corrected pattern text |

### `get_attribution`

Return attribution and provenance information for a specific memory.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `id` | number | Yes | Memory ID |

## V2.8 Tools

### `report_outcome`

Report the outcome of using a memory (was it helpful?). Feeds the learning system.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `memory_id` | number | Yes | ID of the memory used |
| `outcome` | string | Yes | `"helpful"`, `"wrong"`, or `"outdated"` |
| `context` | string | No | Additional context about the outcome |

### `get_lifecycle_status`

Return lifecycle status for memories (Active, Warm, Cold, Archived).

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `limit` | number | No | Max results (default: 20) |
| `status` | string | No | Filter by lifecycle stage |

### `set_retention_policy`

Apply a retention policy to the current profile.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `policy` | string | Yes | Policy name: `"indefinite"`, `"gdpr-30d"`, `"hipaa-7y"`, or `"custom"` |
| `days` | number | No | Days for custom policy |

### `compact_memories`

Merge redundant memories and optimize storage.

*No parameters.*

### `get_behavioral_patterns`

Return behavioral patterns observed across your usage (e.g., you always check docs before coding).

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `limit` | number | No | Max patterns to return (default: 10) |

### `audit_trail`

Return audit log entries. Each entry is hash-chained for tamper detection.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `limit` | number | No | Max entries (default: 50) |
| `action` | string | No | Filter by action type: `"store"`, `"recall"`, `"delete"` |

## V3 Tools

### `set_mode`

Switch the operating mode.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `mode` | string | Yes | `"a"`, `"b"`, or `"c"` |

### `get_mode`

Return the current operating mode and its configuration.

*No parameters.*

### `health`

Return health diagnostics for mathematical layers (Fisher-Rao, Sheaf, Langevin), embedding model, and database.

*No parameters.*

### `consistency_check`

Run contradiction detection across stored memories. Returns pairs of memories that may conflict.

*No parameters.*

### `recall_trace`

Recall with a full breakdown of how each retrieval channel scored each result.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Search query |
| `limit` | number | No | Max results (default: 10) |

## Resources

MCP resources provide read-only data that AI assistants can access passively.

| Resource URI | Description |
|-------------|-------------|
| `slm://recent` | The 20 most recently stored memories |
| `slm://stats` | Memory count, database size, mode, profile |
| `slm://clusters` | Topic clusters detected across memories |
| `slm://identity` | Learned user preferences and patterns |
| `slm://learning` | Current state of the adaptive learning system |
| `slm://engagement` | Usage statistics and interaction patterns |

---

## Optimize Tools (v3.6.11 — Surface B)

Proxy-free compression and routed-result caching. All five are **fail-open** — any internal error returns `ok:False` with the original content unchanged.

### `slm_compress`

Compress text or tool output to reduce context window usage.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | required | Text to compress (max 1MB) |
| `mode` | string | `"auto"` | `"normalize"` (lossless whitespace) · `"auto"` · `"aggressive"` |
| `reversible` | boolean | `true` | If lossy, store original in CCR and return `ccr_id` |
| `ttl_seconds` | integer | `86400` | CCR original lifetime (seconds) |

**Returns:** `{ok, compressed, strategy, tokens_before, tokens_after, ratio, lossy, ccr_id, note}`

### `slm_retrieve`

Recover the exact original bytes stored during a lossy `slm_compress` call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ccr_id` | string | UUID4 returned by `slm_compress` when `reversible=True` |

**Returns:** `{ok, content, size_bytes, error}`

### `slm_cache_set`

Cache a string result the agent wants to reuse (file reads, bash output, search results, sub-model calls). Does **not** cache the Claude conversation turn.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | string | required | Cache key (max 512 chars, namespaced per agent) |
| `value` | string | required | Value to store (max 1MB) |
| `ttl_seconds` | integer | `86400` | Time-to-live in seconds |

**Returns:** `{ok, stored, note}`

### `slm_cache_get`

Retrieve a previously cached result.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | string | Same key used in `slm_cache_set` |

**Returns:** `{ok, hit, value, note}`

### `slm_optimize_stats`

Return compression and cache statistics for the current session.

*No parameters.*

**Returns:** `{ok, compress_runs, tokens_saved_compress, cache_proxy_hits, cache_proxy_misses, cache_kv_hits, cache_kv_misses, ccr_note, note}`

> **Note:** Proxy stats are daemon-persisted (accurate across restarts). KV stats (`cache_kv_hits`, `cache_kv_misses`) are in-module counters for this MCP process session only.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
