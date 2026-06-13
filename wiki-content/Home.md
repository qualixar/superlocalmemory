# SuperLocalMemory V3

> **Five years of daily AI use. Your system won't feel it.**
> *Infinite memory for Claude Code, Cursor, Windsurf & 17+ AI tools.*

SuperLocalMemory gives AI assistants persistent memory across sessions. **v3.6.11 "Optimize Everywhere"** — cache, compress, and remember on any plan, no proxy required. No cloud. No APIs. No data leaves your machine.

### v3.6.11 "Optimize Everywhere" — Three Surfaces. No Proxy Required.
**Cache. Compress. Remember.** SLM v3.6.11 delivers compression + caching across three surfaces: **proxy** (full-turn caching via `slm wrap claude`), **MCP tools** (5 new tools in `slm mcp`, full 1M window preserved), or **skill** (`~/.claude/skills/slm-optimize/`, zero-config auto-compress). Every setup covered — with or without a proxy. Install once with `pip install -U superlocalmemory`. [View details →](https://superlocalmemory.com/optimize-everywhere)

### v3.4.5 "Scale-Ready" — 1 Million Memories. Zero Slowdown.
**Tiered storage auto-classifies every memory as active, warm, cold, or archived.** Graph pruning removes redundant connections. Optional acceleration backends (CozoDB, LanceDB) for graph + vector operations. Tested on **1.18 million real graph edges** with under 2-second recall. Migration is automatic: `pip install -U superlocalmemory && slm restart`. [View details →](https://superlocalmemory.com/scale-ready)

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
