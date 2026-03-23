# Getting Started
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Get your AI's memory system running in under 5 minutes. **V3.1: Now with Active Memory — your memory learns from your usage and gets smarter over time, at zero token cost.**

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

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. MIT License. Part of Qualixar.*
