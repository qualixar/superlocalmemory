# Getting Started
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Get your AI's memory system running in under 5 minutes. **V3.1: Now with Active Memory — your memory learns from your usage and gets smarter over time, at zero token cost.**

<!-- MKT-M4: orient new Claude users who already have Anthropic's free
     built-in memory (shipped March 2026) on why SLM still earns a
     `pip install` + daemon. Three-bullet contrast. -->
<!-- MKT-M5: one-line framing so expectations match the product shape. -->
<!-- MKT-L1: reframe the integration surface as MCP-native so every MCP
     client (LangChain-MCP, LlamaIndex MCP, CrewAI-via-MCP, etc.) is
     covered, not just the 5 named IDEs. -->

### Why not just use Claude's built-in memory?

Anthropic's free Claude Memory (March 2026) and Claude Code's Auto-Memory
are fine defaults. SLM earns the daemon in three places:

- **Local-only by default.** Your memory never leaves your laptop — no
  cloud sync, no vendor lock-in. (Opt-in skill evolution is the only
  outbound path, and it is off by default.)
- **Shared across tools, not just one chat.** One memory, consumed by
  Claude Code + Cursor + Antigravity + VS Code + Claude Desktop —
  anything that speaks MCP.
- **Learns from your outcomes.** Implicit reward signals (dwell,
  re-query, edit, cite) retrain the ranker against how you actually
  work, not just summarised chat transcripts.

SuperLocalMemory is built for **one developer, one laptop, many tools.**
Team / multi-user memory is a different product (SLM-Mesh).

**Integration surface: MCP-native.** The five IDEs listed below are
explicit wirings, but any MCP-compatible client (LangChain-MCP adapters,
LlamaIndex MCP, CrewAI-via-MCP, etc.) can use SLM without a custom
adapter.

---

## Prerequisites

- **Node.js** 18 or later
- **Python** 3.10 or later (installed automatically on most systems)
- An AI coding tool (Claude Code, Cursor, VS Code, Windsurf, or any MCP-compatible IDE)

## Install

```bash
npm install -g superlocalmemory
```

This installs the `slm` command globally.

## Run the Setup Wizard

```bash
slm setup
```

The wizard walks you through three choices:

1. **Pick your mode**
   - **Mode A** (default) — Zero cloud. All memory stays on your machine. No API key needed.
   - **Mode B** — Local LLM. Uses Ollama on your machine for smarter recall.
   - **Mode C** — Cloud LLM. Uses OpenAI, Anthropic, or another provider for maximum power.

2. **Connect your IDE** — The wizard detects installed IDEs and configures them automatically.

3. **Verify installation** — A quick self-test confirms everything works.

> **Tip:** Start with Mode A. You can switch to B or C anytime with `slm mode b` or `slm mode c`.

## Store Your First Memory

```bash
slm remember "The project uses PostgreSQL 16 on port 5433, not the default 5432"
```

You should see:

```
Stored memory #1 (Mode A, profile: default)
```

## Recall a Memory

```bash
slm recall "what database port do we use"
```

Output:

```
[1] The project uses PostgreSQL 16 on port 5433, not the default 5432
    Score: 0.94 | Stored: 2 minutes ago | Profile: default
```

## Check System Status

```bash
slm status
```

This shows:

- Current mode (A, B, or C)
- Active profile
- Total memories stored
- Database location
- Health of math layers (Fisher, Sheaf, Langevin)

## How It Works With Your IDE

Once connected, SuperLocalMemory works automatically:

- **Auto-recall** — When your AI assistant responds, relevant memories are injected as context. No manual queries needed.
- **Auto-capture** — Decisions, bug fixes, architecture choices, and preferences are stored as you work. No manual tagging needed.

You can still use `slm remember` and `slm recall` from the terminal whenever you want explicit control.

## Next Steps

| What you want to do | Guide |
|---------------------|-------|
| Set up a specific IDE | [IDE Setup](ide-setup.md) |
| Switch modes or providers | [Configuration](configuration.md) |
| Learn all CLI commands | [CLI Reference](cli-reference.md) |
| Migrate from V2 | [Migration from V2](migration-from-v2.md) |
| Understand how it works | [Architecture](architecture.md) |

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
