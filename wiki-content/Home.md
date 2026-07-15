# SuperLocalMemory V3

> **Local-first memory, cache, and compression controls for AI-agent workflows.**
> *Documented clients include Claude Code, Cursor, Windsurf, Codex, and other named MCP configurations.*

SuperLocalMemory gives AI assistants persistent memory across sessions. Optimize adds an opt-in exact cache, content-dependent compression, and proxy or agent-routed surfaces. Core memory state uses a local data root; optional providers, connectors, backup, and downloads have separate network behavior.

### v3.6.12 remote coordination controls
**`SLM_REMOTE=1`** (default off) exposes selected HTTP surfaces to configured LAN clients. This is not replicated memory or automatic federation. Review authentication, host allowlists, transport security, and recovery behavior in the [Distributed Deployment guide](https://github.com/qualixar/superlocalmemory/blob/main/docs/distributed-deployment.md).

### v3.6.11 Optimize surfaces
SLM exposes a proxy for intercepted provider calls plus MCP tools and skills for
content explicitly routed through SLM. MCP/skill use does not cache the primary
conversation turn without a proxy. Compression reduction is content- and
mode-dependent; safe mode can return the original unchanged.

### v3.4.5 historical scale work
This release introduced tiering, pruning, and optional experimental CozoDB/LanceDB paths. V3.7 does not yet publish a release-linked memory-count, quality-at-scale, or latency envelope; do not treat the historical scale page as current proof.

### V3.3.6 hook lineage
Hooks can provide session recall, observation, and close-time checkpoints after
the operator explicitly installs them. Current installers do not add hooks or
edit IDE configuration implicitly; run `slm setup` or `slm hooks install` at
the activation boundary.

### V3.1 active-memory lineage
SLM includes local feedback, outcome, and adaptive-ranking components. Exposure
to a result is not proof that it was helpful, and V3.7 does not make a market
uniqueness or guaranteed-improvement claim. [Read the limitations →](Active-Memory)

## Quick Start

```bash
npm install -g superlocalmemory    # Primary global CLI path
slm setup                          # Choose mode A/B/C
slm warmup                         # Pre-download embedding model (optional)
```

The second primary path is Python in an activated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install superlocalmemory
slm setup
```

Then configure the client you intend to use and verify it with `slm doctor`.
