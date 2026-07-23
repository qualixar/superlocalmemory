---
name: superlocalmemory
description: "AI agent memory with mathematical foundations. Store, recall, search, and manage memories locally. Local data root; optional networked features have separate behavior."
version: "3.8.0"
author: "Varun Pratap Bhardwaj"
license: AGPL-3.0-or-later
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

AI agent memory with a local data root. Five candidate producers (semantic, BM25, temporal, spreading-activation, Hopfield) fuse via RRF, with an entity-graph post-fusion score enhancement — all with mathematical similarity scoring. Mode A operates without sending memory content to a cloud model provider; optional connectors, backup, and proxy providers are explicit choices with separate behavior.

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

### Bounded Loops

```bash
slm loop demo                             # Run built-in convergence demo (no API key needed)
slm loop history [--name <loop-name>]     # List recorded runs from SLM memory
slm loop show <run_id>                    # Show every lap of one run
```

Loop laps are persisted to SLM memory under the tag `loop:<name>`. MCP tools
`slm_loop_run`, `slm_loop_history`, and `slm_loop_show` are available in the
`code` and `full` profiles.

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
| A | Local Guardian -- core memory runs without a cloud model provider; optional connectors and model downloads may use the network | None (for core memory) |
| B | Smart Local -- local Ollama LLM, data stays on your machine | Local only |
| C | Full Power -- cloud LLM for maximum accuracy | Yes |

## Dual Interface

SuperLocalMemory works via both MCP and CLI:

- **MCP**: 24 tools (`code` profile) for IDE integration (Claude Code, Cursor, Windsurf, VS Code, JetBrains, Zed); includes bounded-loop tools `slm_loop_run/history/show`
- **CLI**: commands with `--json` for scripts, CI/CD, and agent frameworks; includes `slm loop demo/history/show`

---

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
