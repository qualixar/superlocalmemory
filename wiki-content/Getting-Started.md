# Getting Started

Get SuperLocalMemory running in under 5 minutes.

## Prerequisites

- **Python** 3.11+ (`python3 --version`)
- **Node.js** 18+ (only if installing via npm)
- Any supported IDE (Claude Code, Cursor, VS Code, Windsurf, etc.)

## Install

**npm (recommended):**
```bash
npm install -g superlocalmemory
```

**Python CLI + SDK (primary activated-venv path):**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
```

See [Installation](Installation) for git clone, platform-specific notes, and troubleshooting.

## Setup

```bash
slm setup     # Choose mode A/B/C, configure provider
slm warmup    # Pre-download embedding model (~500MB, optional)
```

**Modes:**
- **A** — Core memory operations use the local data root without a cloud model provider. Optional integrations have separate network behavior. **(default)**
- **B** — Local LLM via Ollama. Still fully private.
- **C** — Cloud LLM for provider-assisted enrichment; configured content is sent to that provider.

Switch anytime: `slm mode a`, `slm mode b`, `slm mode c`.

## Your First Memory

Store something:

```bash
slm remember "Our API uses JWT tokens with 24-hour expiry. Refresh tokens last 30 days." --json
```

The default receipt reports an opaque `operation_id`, fact IDs, and
`materialization_state: queryable`. Enrichment continues durably in the
background. Add `--sync` when the next step requires `complete`.

Recall it later:

```bash
slm recall "JWT token expiry"
```

You should see the stored memory returned with a relevance score.

## Verify Installation

```bash
slm status    # System info — mode, DB path, size
slm health    # Math layer health — Fisher-Rao, Sheaf, Langevin stats
```

## Connect Your IDE

Auto-configure all detected IDEs:

```bash
slm connect        # Configure all detected IDEs
slm connect --list # See which IDEs are configured
```

Or add manually to your IDE's MCP config:

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

See [IDE Setup](IDE-Setup) for per-IDE instructions.

## Web Dashboard

```bash
slm dashboard    # Opens at http://localhost:8765
```

17 tabs: memory browser, knowledge graph, recall lab, trust scores, math health, compliance, and more. Runs locally — no data leaves your machine.

## Next Steps

- [Quick Start Tutorial](Quick-Start-Tutorial) — Step-by-step for new users and V2 upgraders
- [Installation](Installation) — Full guide with platform notes and troubleshooting
- [Modes Explained](Modes-Explained) — Understand the three operating modes
- [CLI Reference](CLI-Reference) — Full command reference
- [IDE Setup](IDE-Setup) — Configure additional IDEs
- [Auto-Memory](Auto-Memory) — How auto-capture works
- [Migration from V2](Migration-from-V2) — Upgrade guide for existing V2 users

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
