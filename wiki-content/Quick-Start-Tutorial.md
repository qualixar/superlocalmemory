# Quick Start Tutorial

Get SuperLocalMemory working in under 5 minutes — whether you're a new user or upgrading from V2.

---

## New Users

### 1. Install

```bash
npm install -g superlocalmemory
```

Python alternative: create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
```

### 2. Setup

```bash
slm setup
```

The wizard asks you to pick a mode:
- **A (Local Guardian)** — Core memory operations use the local data root without a cloud model provider. Optional integrations have separate network behavior. Default.
- **B (Smart Local)** — Local LLM via Ollama for answer synthesis.
- **C (Full Power)** — Cloud LLM for maximum accuracy. Requires API key.

Most users should start with **Mode A** — you can switch anytime with `slm mode b` or `slm mode c`.

### 3. Pre-download the embedding model (optional)

```bash
slm warmup
```

Downloads the nomic-embed-text-v1.5 model (~500MB). If you skip this, it downloads automatically on first use.

### 4. Store your first memory

```bash
slm remember "Our API uses JWT tokens with 24-hour expiry. Refresh tokens last 30 days." --json
```

Output includes `operation_id`, fact IDs, and `materialization_state:
queryable`. This means the SQLite relational/FTS projection is recallable and
enrichment is pending. Use `--sync` to wait for `complete`.

### 5. Recall it

```bash
slm recall "token expiry"
```

Output shows the stored memory with a relevance score:
```
  1. [0.82] Our API uses JWT tokens with 24-hour expiry. Refresh tokens last 30 days.
```

### 6. Check system status

```bash
slm status
```

```
SuperLocalMemory V3
  Mode: A
  Provider: none
  Base dir: ~/.superlocalmemory
  Database: ~/.superlocalmemory/memory.db
  DB size: 0.12 MB
```

### 7. Check math layer health

```bash
slm health
```

```
Math Layer Health:
  Total facts: 1
  Fisher-Rao indexed: 1/1
  Langevin positioned: 1/1
  Mode: A
```

### 8. Connect to your IDE

```bash
slm connect        # Auto-configure all detected IDEs
slm connect --list # See what's configured
```

Or manually add to your IDE's MCP config:

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

Works with: Claude Code, Cursor, VS Code Copilot, Windsurf, Continue, Cody, ChatGPT Desktop, Gemini CLI, JetBrains, Zed, and more.

### 9. Open the dashboard

```bash
slm dashboard
```

Opens at http://localhost:8765. 17 tabs: memory browser, knowledge graph, recall lab, trust scores, math health, compliance, and more.

---

## Upgrading from V2

If you already have SuperLocalMemory V2 (2.8.6 or earlier):

### 1. Install V3

```bash
npm install -g superlocalmemory
```

V3 installs alongside V2. Your V2 data is untouched until you migrate.

### 2. Migrate your data

```bash
slm migrate
```

This will:
- Show your V2 stats (memory count, DB size)
- Ask for confirmation
- Back up your V2 database automatically
- Copy data to the V3 location (`~/.superlocalmemory/`)
- Convert V2 memories to V3 atomic facts
- Create a symlink so old tools still find the data

### 3. Setup V3

```bash
slm setup     # Choose mode (A/B/C)
slm warmup    # Pre-download embedding model
```

### 4. Verify

```bash
slm status    # Check V3 is running
slm health    # Check math layers are active
slm recall "something you stored in V2"   # Verify old memories are accessible
```

### What changed from V2 to V3

| Feature | V2 | V3 |
|:--------|:---|:---|
| Retrieval | Cosine similarity only | Five candidate producers plus fusion and optional score enhancements |
| Similarity | Cosine distance | Dense cosine relevance with optional Fisher-informed later scoring |
| Consistency | None | Sheaf cohomology (algebraic topology) |
| Lifecycle | Hardcoded thresholds | Self-organizing Langevin dynamics |
| Modes | Single mode | A (zero-cloud), B (local LLM), C (cloud LLM) |
| Privacy and compliance controls | Not addressed | Deployment-specific controls and assessment |
| Dashboard | 5 tabs | 17 tabs |
| MCP Tools | 6 | Profile-selected V3 tool surfaces |
| Tests | Historical V2 suite | V3 unit, contract, artifact, and integration suites |

### Rollback if needed

```bash
slm migrate --rollback
```

This restores your V2 installation. No data is lost.

---

## Key Commands Reference

| Command | What It Does |
|:--------|:-------------|
| `slm remember "..."` | Store a memory |
| `slm recall "..."` | Search memories (semantic + keyword + entity + temporal) |
| `slm forget "..."` | Delete matching memories (with confirmation) |
| `slm trace "..."` | Recall with per-channel score breakdown |
| `slm status` | System status (mode, DB size, path) |
| `slm health` | Math layer health (Fisher, Sheaf, Langevin stats) |
| `slm mode a/b/c` | Switch operating mode |
| `slm dashboard` | Launch web dashboard (http://localhost:8765) |
| `slm mcp` | Start MCP server (for IDE integration) |
| `slm connect` | Auto-configure IDE integrations |
| `slm profile list` | List memory profiles |
| `slm profile create work` | Create isolated memory space |
| `slm profile switch work` | Switch to a different profile |

Full reference: [CLI Reference](CLI-Reference)

---

## Next Steps

- [Modes Explained](Modes-Explained) — Understand A vs B vs C
- [MCP Tools](MCP-Tools) — Profile-selected tool and resource contracts
- [IDE Setup](IDE-Setup) — Per-IDE configuration guides
- [Auto-Memory](Auto-Memory) — How auto-capture and auto-recall work
- [Architecture Overview](V3-Architecture) — How the system works under the hood

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
