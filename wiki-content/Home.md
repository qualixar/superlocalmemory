# SuperLocalMemory V3

> **Local-first memory, cache, and compression controls for AI-agent workflows.**
> *Documented clients include Claude Code, Cursor, Windsurf, Codex, and other named MCP configurations.*

SuperLocalMemory gives AI assistants persistent memory across sessions. Optimize adds an opt-in exact cache, content-dependent compression, and proxy or agent-routed surfaces. Core memory state uses a local data root; optional providers, connectors, backup, and downloads have separate network behavior.

### v3.6.12 remote coordination controls
**`SLM_REMOTE=1`** (default off) exposes selected HTTP surfaces to configured LAN clients. This is not replicated memory or automatic federation. Review authentication, host allowlists, transport security, and recovery behavior in the [Distributed Deployment guide](https://github.com/qualixar/superlocalmemory/blob/main/docs/distributed-deployment.md).

### v3.6.11 "Optimize Everywhere" — Three Surfaces. No Proxy Required.
**Cache. Compress. Remember.** SLM v3.6.11 delivers compression + caching across three surfaces: **proxy** (full-turn caching via `slm wrap claude`), **MCP tools** (5 new tools in `slm mcp`, full 1M window preserved), or **skill** (`~/.claude/skills/slm-optimize/`, zero-config auto-compress). Every setup covered — with or without a proxy. Install once with `pip install -U superlocalmemory`. [View details →](https://superlocalmemory.com/optimize-everywhere)

### v3.4.5 historical scale work
This release introduced tiering, pruning, and optional experimental CozoDB/LanceDB paths. V3.7 does not yet publish a release-linked memory-count, quality-at-scale, or latency envelope; do not treat the historical scale page as current proof.

### V3.3.6: Zero-Friction Hooks — Install Once, Forget Forever
One `npm install` and your AI memory is fully automatic:
- **Auto-recall** at session start — your context is there before you ask
- **Auto-observe** during coding — decisions and changes captured silently
- **Auto-save** at session end — full summary with git context
- **Zero setup** — hooks install themselves, no config needed
- **Zero risk** — every hook fails silently, never blocks your workflow

### V3.1: Active Memory — Memory That Learns
SLM **learns from your usage patterns** and gets smarter over time — at zero token cost. Every recall generates learning signals. After 20+ signals, the system starts optimizing retrieval for YOUR specific patterns. After 200+, a full ML model trains on your data. No other memory system learns without spending LLM tokens. [Read more →](Active-Memory)

## Quick Start

```bash
npm install -g superlocalmemory    # or: pip install superlocalmemory
slm setup                          # Choose mode A/B/C
slm warmup                         # Pre-download embedding model (optional)
```

That's it. Your AI now remembers you.
