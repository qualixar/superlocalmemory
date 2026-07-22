# MCP Tools Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SuperLocalMemory exposes profile-selected tools and resources through the Model
Context Protocol (MCP). The installed profile registry is the source of truth
for names and counts; a client still decides when to call a tool.

> **Optimize tools:** `slm_compress`, `slm_retrieve`, `slm_cache_set`,
> `slm_cache_get`, and `slm_optimize_stats` provide explicit compression and
> routed-result caching. They do not intercept or cache the primary
> conversation turn without a proxy. [See Three Surfaces →](optimize-overview.md)

> **Profile exposure:** The active `SLM_MCP_PROFILE` determines which tools are
> visible to the connected client. Core tools are in every profile. Code-graph
> tools require `code`, `full`, or `power`. Mesh tools require `full`, `power`, or
> `mesh`. See [MCP Profiles →](../README.md#mcp--profiles) and
> [docs/profiles.md](profiles.md).

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
| `project` | string | No | Project classification |
| `importance` | integer | No | Importance hint (default: 5) |
| `session_id` | string | No | Session attribution and stable retry input |
| `agent_id` | string | No | Calling-agent attribution |
| `scope` | string | No | `personal`, `shared`, or `global` |
| `shared_with` | string | No | Comma-separated profile IDs for shared scope |
| `idempotency_key` | string | No | Stable identity for safe retries |

The result is a durable receipt containing `operation_id`, `fact_ids`,
`materialization_state`, and `pending`. A daemon-backed default call normally
returns after the SQLite relational/FTS projection is `queryable`; enrichment
continues on the same operation. The offline compatibility spool preserves the
same source and idempotency identity when it is replayed.

### `recall`

Search memories by natural language query.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Natural language search query |
| `limit` | number | No | Max results (default: 20) |

Recall results follow [Score Contract v2](retrieval-score-contract.md):
`relevance_score` is query relevance, `ranking_score` is diagnostic ranking
utility, and `memory_confidence` belongs to the stored assertion. Canonical
responses declare `calibration_status: "uncalibrated"` and
`answer_confidence: null`; retrieval scores are not answer probabilities.

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

Return system attribution metadata — product name, author, organization, license, and URLs.

*No parameters.*

Returns: `{success, product, author, organization, license, urls:{product, author, organization}}`

### `forget`

Run Ebbinghaus forgetting decay cycle. Computes retention scores and transitions memories between lifecycle zones (active → warm → cold → archive → forgotten) based on access patterns and importance. Run with `dry_run=true` first to preview changes.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `profile_id` | string | No | Profile to process (default: active profile) |
| `dry_run` | boolean | No | If true, compute stats but do not apply transitions (default: true) |

### `delete_memory`

Delete a specific memory by exact fact ID. All deletions are logged with `agent_id` for audit.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `fact_id` | string | Yes | Exact fact ID to delete (from `recall` or `list_recent`) |
| `agent_id` | string | No | Calling-agent identifier for audit log |

### `update_memory`

Update the content of a specific memory by exact fact ID. The fact must belong to the active profile.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `fact_id` | string | Yes | Exact fact ID to update |
| `content` | string | Yes | New content (cannot be empty) |
| `agent_id` | string | No | Calling-agent identifier for audit log |

## Active Memory Tools (V3.1)

Three tools for automatic context injection and learning. Call `session_init` once at the start of every session.

### `session_init`

Initialize session context from stored memories. Returns relevant memories for the project path or query, plus current learning status.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `project_path` | string | No | Working directory path — used to derive the search query |
| `query` | string | No | Override the search query (overrides `project_path` when set) |
| `max_results` | number | No | Maximum memories to return (default: 10) |
| `max_age_days` | number | No | Suppress memories older than N days unless relevance ≥ 0.70 (default: 30; set to 0 to disable) |

### `observe`

Send conversation content for automatic memory capture. The system evaluates whether the content contains decisions, bug fixes, or preferences worth storing and captures them when confidence > 0.5.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `content` | string | Yes | Conversation snippet to evaluate |
| `agent_id` | string | No | Calling-agent attribution (defaults to `SLM_AGENT_ID` env var) |

### `report_feedback`

Report whether a recalled memory was useful. Trains the adaptive ranker over time.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `fact_id` | string | Yes | ID of the recalled memory |
| `feedback` | string | No | `"relevant"`, `"irrelevant"`, or `"partial"` (default: `"relevant"`) |
| `query` | string | No | The original query that surfaced this memory |

### `close_session`

Close the current session and create temporal summary events. Aggregates session facts into per-entity summaries.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `session_id` | string | No | Session to close (default: most recent session) |

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

## V3 Utility Tools

### `get_version`

Return the installed SLM version and build information.

*No parameters.*

**Returns:** `{success, version, build_info}`

### `prestage_context`

Pre-stage a recall query so the result is immediately available when `recall` or `session_init` is called next. Reduces perceived latency for known upcoming queries.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Query to pre-stage |
| `max_results` | integer | No | Maximum results to warm (default: 10) |

### `get_retention_stats`

Return retention statistics for the active profile: memory counts by lifecycle zone (active/warm/cold/archived) and current retention policy.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `profile_id` | string | No | Profile to query (default: active profile) |

### `consolidate_cognitive`

Run the CCQ cognitive consolidation pipeline. Identifies redundant or contradictory memories and proposes consolidations. Does not apply changes unless `dry_run=false`.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `profile_id` | string | No | Profile to consolidate (default: active profile) |
| `dry_run` | boolean | No | If true, propose without applying (default: true) |

### `get_soft_prompts`

Return the auto-learned soft prompts for the active profile. Soft prompts capture recurring patterns and preferences observed across sessions.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `profile_id` | string | No | Profile to query (default: active profile) |
| `limit` | integer | No | Max results (default: 20) |

### `reap_processes`

Find and optionally terminate orphaned SLM daemon or MCP processes. Safe to call from inside an agent: uses process metadata to identify orphans rather than killing all matching processes.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `force` | boolean | No | If true, terminate found orphans (default: false — dry run) |

---

## Code-Graph Tools

Available in profiles `code` (21 tools), `full` (39 tools), and `power` (51 tools). Not available in `core` or `mesh` profiles.

These tools build and query a structural code graph over a local repository. The graph maps functions, classes, modules, call sites, imports, and dependencies. It is built on demand from the repository path and persisted in SLM's database.

### `build_code_graph`

Build or rebuild the code graph for a repository.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `repo_path` | string | Yes | Absolute path to the repository root |
| `include_patterns` | string | No | Glob patterns to include (default: `**/*.py,**/*.ts,**/*.js`) |
| `changed_files` | array | No | If set, incremental update for only these files |

### `update_code_graph`

Incrementally update the code graph for changed files without a full rebuild.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `repo_path` | string | Yes | Repository root |
| `changed_files` | array | Yes | List of changed file paths (relative to repo root) |

### `get_blast_radius`

Given a symbol (function, class, or module), return all code that would be affected by changing it — direct callers, transitive dependents, and reachable test files.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `target` | string | Yes | Symbol name or file path |
| `depth` | integer | No | Maximum traversal depth (default: 3) |

### `get_review_context`

Return the code-graph context relevant to a code review: callers, callees, related tests, and recent change history for the changed symbols.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `changed_files` | array | Yes | Files changed in the review |
| `repo_path` | string | Yes | Repository root |

### `query_graph`

Query the code graph with a free-form description or a structural predicate.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Natural language or structural query |
| `limit` | integer | No | Max results (default: 20) |

### `semantic_search_code`

Semantic search across code symbols using the same embedding model as memory recall.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Natural language description of the code to find |
| `limit` | integer | No | Max results (default: 10) |

### `list_graph_stats`

Return summary statistics for the built code graph: node counts by type, edge counts, last build time.

*No parameters.*

### `find_large_functions`

Return functions or methods exceeding a size threshold.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `min_lines` | integer | No | Minimum line count (default: 50) |
| `limit` | integer | No | Max results (default: 20) |

### `detect_changes`

Compare the current on-disk state of files against the last-built graph and report which symbols have changed, been added, or been removed.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `repo_path` | string | Yes | Repository root |
| `files` | array | No | Specific files to check (default: all tracked) |

### `get_architecture_overview`

Return a high-level summary of the repository's module structure, dependency layers, and largest components.

*No parameters.*

### `list_flows` / `get_flow` / `get_affected_flows`

Inspect call flows through the codebase.

| Tool | Description |
|------|-------------|
| `list_flows` | List all named or detected flows in the graph |
| `get_flow(name)` | Return the full call chain for a named flow |
| `get_affected_flows(target)` | Return flows that pass through a given symbol |

### `list_communities` / `get_community`

Inspect logical module communities detected by graph clustering.

| Tool | Description |
|------|-------------|
| `list_communities` | List detected communities with size and cohesion score |
| `get_community(id)` | Return members and edges for a specific community |

### `code_memory_search`

Search memory facts that are linked to specific code symbols or file paths.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `query` | string | Yes | Query (may include symbol names) |
| `limit` | integer | No | Max results (default: 10) |

### `code_entity_history`

Return the memory history for a code entity: all decisions, bugs, and observations linked to a given symbol.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `target` | string | Yes | Symbol name or file path |

### `link_memory_to_code`

Explicitly link a stored memory to a code symbol. Strengthens code-memory retrieval for that symbol.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `fact_id` | string | Yes | Memory fact ID to link |
| `target` | string | Yes | Symbol name or file path to link to |

### `enrich_blast_radius`

Enrich a blast-radius result with relevant memories: decisions, known bugs, and past observations about the affected symbols.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `target` | string | Yes | Symbol to analyze |
| `depth` | integer | No | Traversal depth (default: 3) |

### `code_stale_check`

Check whether stored memories about a symbol are stale relative to recent code changes.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `target` | string | Yes | Symbol or file path to check |

### `refactor_preview` / `apply_refactor`

Preview or apply a code refactor using graph-aware rename and dependency tracking.

| Tool | Key Parameters | Description |
|------|---------------|-------------|
| `refactor_preview` | `target`, `new_name`, `repo_path` | Show all rename sites and impact before applying |
| `apply_refactor` | `target`, `new_name`, `repo_path` | Apply the rename across the graph (writes files) |

> `apply_refactor` writes files. Review the `refactor_preview` output before applying.

---

## Resources

MCP resources provide read-only data that AI assistants can access passively.

| Resource URI | Description |
|-------------|-------------|
| `slm://context` | Active session context auto-injected on MCP connect. Returns relevant memories + learning status. |
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
