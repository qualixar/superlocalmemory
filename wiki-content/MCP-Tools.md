# MCP Tools

SuperLocalMemory exposes 27 tools and 7 resources via the Model Context Protocol (MCP). These are what your IDE uses to interact with the memory system.

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
| `remember` | `content`, `tags?` | Store a new memory |
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
