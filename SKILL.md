---
name: superlocalmemory
description: "AI agent memory with mathematical foundations. Store, recall, search, and manage memories locally with zero cloud dependency."
version: "3.4.22"
author: "Varun Pratap Bhardwaj"
license: Elastic-2.0
homepage: https://superlocalmemory.com
repository: https://github.com/qualixar/superlocalmemory
triggers:
  - remember something
  - recall memory
  - search memories
  - memory status
  - store fact
  - agent memory
  - local memory
  - memory health
---

# SuperLocalMemory

AI agent memory that runs 100% locally. Four-channel retrieval (semantic, graph, BM25, temporal) with mathematical similarity scoring. No cloud, no API keys, EU AI Act compliant.

## Installation

```bash
pip install superlocalmemory
# or
npm install -g superlocalmemory
```

## Quick Start

```bash
slm remember "Alice works at Google as a Staff Engineer" --json
slm recall "Who is Alice?" --json
slm status --json
```

## Commands

All data-returning commands support `--json` for structured agent-native output.

### Memory Operations

```bash
slm remember "<content>" --json           # Store a memory
slm remember "<content>" --tags "a,b" --json
slm recall "<query>" --json               # Semantic search
slm recall "<query>" --limit 5 --json
slm list --json -n 20                     # List recent memories
slm forget "<query>" --json               # Preview matches (add --yes to delete)
slm forget "<query>" --json --yes         # Delete matching memories
slm delete <fact_id> --json --yes         # Delete specific memory by ID
slm update <fact_id> "<content>" --json   # Update a memory
```

### Diagnostics

```bash
slm status --json                         # System status (mode, profile, DB)
slm health --json                         # Math layer health
slm trace "<query>" --json                # Recall with per-channel breakdown
```

### Configuration

```bash
slm mode --json                           # Get current mode
slm mode a --json                         # Set mode (a=local, b=ollama, c=cloud)
slm profile list --json                   # List profiles
slm profile switch <name> --json          # Switch profile
slm profile create <name> --json          # Create profile
slm connect --json                        # Auto-configure IDEs
slm connect --list --json                 # List supported IDEs
```

### Services (no --json)

```bash
slm setup                                 # Interactive setup wizard
slm mcp                                   # Start MCP server (for IDE integration)
slm dashboard                             # Open web dashboard
slm warmup                                # Pre-download embedding model
```

## JSON Envelope

Every `--json` response follows a consistent envelope:

```json
{
  "success": true,
  "command": "recall",
  "version": "3.0.22",
  "data": {
    "results": [
      {"fact_id": "abc123", "score": 0.87, "content": "Alice works at Google"}
    ],
    "count": 1,
    "query_type": "semantic"
  },
  "next_actions": [
    {"command": "slm list --json", "description": "List recent memories"}
  ]
}
```

Error responses:

```json
{
  "success": false,
  "command": "recall",
  "version": "3.0.22",
  "error": {"code": "ENGINE_ERROR", "message": "Description of what went wrong"}
}
```

## Operating Modes

| Mode | Description | Cloud Required |
|------|-------------|----------------|
| A | Local Guardian -- zero cloud, zero LLM, EU AI Act compliant | None |
| B | Smart Local -- local Ollama LLM, data stays on your machine | Local only |
| C | Full Power -- cloud LLM for maximum accuracy | Yes |

## Dual Interface

SuperLocalMemory works via both MCP and CLI:

- **MCP**: 24 tools for IDE integration (Claude Code, Cursor, Windsurf, VS Code, JetBrains, Zed)
- **CLI**: 18 commands with `--json` for scripts, CI/CD, agent frameworks (OpenClaw, Codex, Goose)

---

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
