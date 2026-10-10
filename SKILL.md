---
name: superlocalmemory
description: "AI agent memory with mathematical foundations. Store, recall, search, and manage memories locally. Local data root; optional networked features have separate behavior."
version: "4.1.25"
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
  - save a picture
  - save a PDF
  - message another bot
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

### Pictures, documents and folders (4.1.25, optional)

Off by default. Needs a computer with 16 GB of memory and downloads about 1.5 GB.
Ask the user before turning it on.

```bash
slm media status --json                   # is it on, and how is set-up going
slm media enable                          # asks first; --yes skips the question (only with the user's yes)
slm media disable                         # memories are kept
slm media repair --dry-run                # count pictures that cannot be found by what they show yet
slm sources add ~/Notes --kind obsidian   # or --kind folder; read-only; shows what would be read, then asks
slm sources list --json
```

When it is on, a normal `slm recall` also returns saved pictures and PDF pages (a
`media` block per result). The MCP tools are `remember_media`, `remember_document`,
`get_media`, `media_status` and, for web apps, `media_upload_link`; they are listed
only in the `full`, `power` and `whole` tool sets. A picture is up to 25 MB, a PDF up
to 100 MB and 500 pages. Do not save a file that shows a secret. The `slm-media` skill
has the details.

### Company mode and the dashboard key

```bash
slm team status --json                    # Require login: on|off, user count
slm team policy --require-login on|off    # on the SLM computer; only when the user asks
slm token show                            # the key the dashboard asks for in strict mode; a secret, never paste it into chat
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
  "version": "4.0.0",
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
  "version": "4.0.0",
  "error": {"code": "ENGINE_ERROR", "message": "Description of what went wrong"}
}
```

## Operating Modes

| Mode | Description | Cloud Required |
|------|-------------|----------------|
| A | Local Guardian -- core memory runs without a cloud model provider; optional connectors and model downloads may use the network | None (for core memory) |
| B | Smart Local -- a model on this machine (Ollama by default, any local server works), data stays on your machine | Local only |
| C | Full Power -- your own endpoint, or a cloud provider, for maximum accuracy | Yes, unless using a keyless custom endpoint |

## Dual Interface

SuperLocalMemory works via both MCP and CLI:

- **MCP**: 38 tools (`code` profile), 57 (`full`, the default; 62 with pictures and documents on) for IDE integration (Claude Code, Cursor, Windsurf, VS Code, JetBrains, Zed); includes bounded-loop tools `slm_loop_run/history/show`, bot-message tools `mesh_peers`, `mesh_send`, `mesh_inbox`, `mesh_wait` (a connected web app passes each reply's `ack_ids` back as `ack`) and `mesh_state`
- **CLI**: commands with `--json` for scripts, CI/CD, and agent frameworks; includes `slm loop demo/history/show`

---

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
