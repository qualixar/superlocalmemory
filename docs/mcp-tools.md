# MCP Tools Reference
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SuperLocalMemory exposes 24 tools and 6 resources via the Model Context Protocol (MCP). Any MCP-compatible AI assistant can use these automatically.

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

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
