# Getting Started
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Install the CLI, activate the product explicitly, and verify one store/recall
round trip.

<!-- MKT-M4: orient new Claude users who already have Anthropic's free
     built-in memory (shipped March 2026) on why SLM still earns a
     `pip install` + daemon. Three-bullet contrast. -->
<!-- MKT-M5: one-line framing so expectations match the product shape. -->
<!-- MKT-L1: reframe the integration surface as MCP-native so every MCP
     client (LangChain-MCP, LlamaIndex MCP, CrewAI-via-MCP, etc.) is
     covered, not just the 5 named IDEs. -->

### Product boundary

SLM is useful when you need a user-operated memory service across configured
tools:

- **Local core path by default.** Core memory state uses the configured local
  data root. Optional providers, connectors, backup, model downloads, and
  skill evolution have separate network behavior and must be enabled or configured.
- **Named client configurations.** MCP and CLI surfaces can point multiple
  configured tools at one approved data root. Treat a client as verified only
  when it passes the release integration matrix.
- **Outcome-aware ranking components.** Explicit feedback and qualified
  outcomes can inform local ranking. Exposure alone is not a positive signal.

SuperLocalMemory is built for **one developer, one laptop, many tools.**
Team / multi-user memory is a different product (SLM-Mesh).

**Integration surface:** SLM exposes MCP and CLI contracts. Protocol
compatibility does not by itself prove install, lifecycle, identity, and
cross-client behavior for every product that implements MCP.

---

## Prerequisites

- **Node.js** 18 or later
- **Python** 3.11 or later — macOS ships 3.9; use `brew install python@3.11` or a version manager.
  Ubuntu 22.04 users: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv`
- An AI coding tool (Claude Code, Cursor, VS Code, Windsurf, or any MCP-compatible IDE)

> **Linux / Ubuntu 22.04:** Install in a venv to avoid system-Python conflicts:
> ```bash
> python3.11 -m venv ~/.slm-venv && source ~/.slm-venv/bin/activate
> python -m pip install superlocalmemory
> ```
> Then set `SLM_PYTHON=~/.slm-venv/bin/python` so `slm` uses that interpreter.

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
   - **Mode A** (default) — Local core memory path. Optional downloads, connectors, backups, and explicitly enabled integrations can use the network.
   - **Mode B** — Local LLM. Uses Ollama on your machine for smarter recall.
   - **Mode C** — Cloud LLM. Uses OpenAI, Anthropic, or another provider for maximum power.

2. **Connect your IDE** — The wizard detects installed IDEs and configures them automatically.

3. **Verify installation** — A quick self-test confirms everything works.

> **Tip:** Start with Mode A. You can switch to B or C anytime with `slm mode b` or `slm mode c`.

## Store Your First Memory

```bash
slm remember "The project uses PostgreSQL 16 on port 5433, not the default 5432" --json
```

You should see:

```
{"success":true,"command":"remember","data":{"operation_id":"<opaque-operation-id>","materialization_state":"queryable","fact_ids":["<queryable-fact-id>"],"note":"queryable now; canonical enrichment pending"}}
```

The exact identifiers differ on every installation. Use `--sync` if your next
step requires `complete` rather than the default queryable-first receipt.

## Recall a Memory

```bash
slm recall "what database port do we use"
```

Output:

```
[1] The project uses PostgreSQL 16 on port 5433, not the default 5432
    Relevance: 0.94 | Stored: 2 minutes ago | Profile: default
```

The value is query-relative relevance, not answer confidence. V3.7 declares
`calibration_status: "uncalibrated"` and `answer_confidence: null`; see the
[retrieval score contract](retrieval-score-contract.md).

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

Automation depends on the client plus the hooks/instructions you explicitly
enable:

- **Auto-recall** — Supported session hooks can request bounded, untrusted
  evidence context.
- **Auto-capture** — Supported observe hooks can submit content to configured
  admission rules.

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
