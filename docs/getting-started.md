# Getting Started

Install the CLI, activate the product explicitly, and verify one store/recall
round trip.

### Product boundary

SLM is useful when you need a user-operated memory service across configured
tools:

- **Local core path by default.** Core memory state uses the configured local
  data root. Optional providers, connectors, backup, model downloads, skill
  evolution and [Web access](remote-access/README.md) have separate network
  behavior and must be enabled or configured.
- **Named client configurations.** MCP and CLI surfaces can point multiple
  configured tools at one approved data root. Treat a client as verified only
  when it passes the release integration matrix.
- **Outcome-aware ranking components.** Explicit feedback and qualified
  outcomes can inform local ranking, which is off unless you enable it with
  `SLM_RANKING`. Exposure alone is not a positive signal.

SuperLocalMemory is built for **one person's computer and many tools**. On that
computer it can also serve a small team: roles, company mode with sign-in and scoped
sharing are built in, and SLM-Mesh lets agents and connected bots message each other.
It is not a hosted, multi-tenant service.

**Integration surface:** SLM exposes MCP and CLI contracts. Protocol
compatibility does not by itself prove install, lifecycle, identity, and
cross-client behavior for every product that implements MCP.

---

## Prerequisites

> **Supported platforms:** Apple Silicon macOS, 64-bit Windows, and 64-bit Linux.
> Intel Mac and 32-bit Windows are not supported by the patched cryptographic
> runtime SLM requires.

- **Node.js** 18 or later
- **Python** 3.12 to 3.14. macOS ships an older Python; use
  `brew install python@3.12` or a version manager. Ubuntu 22.04 users:
  `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv`
- An AI coding tool (Claude Code, Cursor, VS Code, Windsurf, or any MCP-compatible IDE)

> **Linux / Ubuntu 22.04:** Install in a venv to avoid system-Python conflicts:
> ```bash
> python3.12 -m venv ~/.slm-venv && source ~/.slm-venv/bin/activate
> python -m pip install superlocalmemory
> ```
> Then set `SLM_PYTHON=~/.slm-venv/bin/python` so `slm` uses that interpreter.

## Install

```bash
npm install -g superlocalmemory
```

This installs the `slm` command globally and gives it a private Python
environment inside the package. The package carries every adapter folder
(Claude Code, Codex, Copilot for VS Code, Hermes and Antigravity) together with
all the skills, agents, commands and rules, so connecting a host does not need a
second download. See [IDE Setup](ide-setup.md). Installing does not create a data folder, edit an IDE, start a daemon or
download a model.

## Run the Setup Wizard

```bash
slm setup
```

For an existing installation that was updated through npm, pip, or a repository
checkout, use `slm upgrade-hosts` first. It previews existing SLM integrations
without changing host configuration; apply only reviewed targets with
`slm upgrade-hosts --host <host> --apply`. See [Host Integration
Upgrades](host-upgrades.md).

The wizard walks you through these choices:

1. **Pick your mode**
   - **Mode A** (Local Guardian, the default) — no language model; the core
     memory path makes no model-provider call. Optional downloads, connectors,
     backups and explicitly enabled integrations can still use the network.
   - **Mode B** (Smart Local) — a model on your machine improves recall:
     Ollama by default, or any local OpenAI-compatible server.
   - **Mode C** (Full Power) — your own endpoint or a cloud provider such as
     OpenAI or Anthropic, for maximum accuracy. A cloud provider needs a key.

2. **Optional features and models** — the code knowledge graph, the embedding
   and reranker models, an optional compression model, mesh, ingestion
   adapters, entity compilation, and skill evolution (off unless you turn it on).

3. **Verification** — a quick self-test confirms recall works.

4. **Integrations, with your consent** — the wizard asks before installing the
   Claude Code plugin and hooks, before connecting any other IDE it detects, and
   before turning on auto-start after login. Run without a terminal, it skips
   all three; you can do each later with `slm connect <ide>` and `slm serve
   install`.

On a Mac with Apple Silicon, the wizard also offers to set up [Answer
check](answer-check.md), which lets recall say "I don't have that" instead of
guessing. Say no and set it up later from **Settings → Answer check** in the
dashboard. On other platforms, the wizard points you to that same settings page
for the online option instead.

> **Tip:** Start with Mode A. You can switch anytime with `slm mode b` or `slm mode c`,
> then run `slm restart`.

## Store Your First Memory

```bash
slm remember "The project uses PostgreSQL 16 on port 5433, not the default 5432" --json
```

You should see:

```
{"success":true,"command":"remember","data":{"fact_ids":["<fact-id>"],"count":1,"operation_id":"<opaque-operation-id>","status":"queryable","materialization_state":"queryable","searchable_by":"wording","note":"stored and searchable by wording; searchable by meaning shortly", ...}}
```

The exact identifiers and a few extra fields differ on every installation. Use `--sync` if your next
step requires `complete` rather than the default queryable-first receipt. Add
`--kind rule` or `--kind decision` for something you want loaded at the start of
later sessions (see [Memory kinds](memory-kinds.md)).

## Recall a Memory

```bash
slm recall "what database port do we use"
```

Output:

```
  1. [0.94] The project uses PostgreSQL 16 on port 5433, not the default 5432
```

The bracketed number is query-relative relevance, not answer confidence. By
default, SLM declares `calibration_status: "uncalibrated"` and
`answer_confidence: null`; see the [retrieval score
contract](retrieval-score-contract.md). Turn on [Answer check](answer-check.md)
and recall additionally tells you whether the top result actually answers your
question, not just whether it is related. Narrow a recall by project, kind, tag
or time with the flags in [Recall](recall.md).

## Check System Status

```bash
slm status
```

This shows the current mode, provider, active profile, data folder, database
path and size, and any saves still being indexed. `slm status --verbose` adds
the daemon port and the last booted version, `slm health` reports the
mathematical layers, and `slm doctor` checks dependencies and connectivity. SLM
also checks your memory store once after each upgrade; see [Troubleshooting](troubleshooting.md#memory-store-check).

## How It Works With Your IDE

Automation depends on the client plus the hooks and instructions you explicitly
enable:

- **Auto-recall** — Supported session hooks can request bounded, untrusted
  evidence context.
- **Auto-capture** — Supported observe hooks can submit content to configured
  admission rules.

You can still use `slm remember` and `slm recall` from the terminal whenever you
want explicit control.

## Use Your Memory From a Web App

If you want an AI app on the internet (ChatGPT, Claude on the web, Composio,
Muse) to use your memory, open **Connected apps** in the dashboard. It is off
until you set it up, needs a GitHub sign-in, and works only while this computer
is on and online. See [Web access](remote-access/README.md).

## Try Bounded Loops

A bounded loop runs laps until an independent gate passes, not until the agent
claims it is done. Every lap is persisted to your SLM data root, queryable via
`slm recall`, and visible on the dashboard.

Verify the engine end to end with the built-in demo:

```bash
slm loop demo
```

Expected output:

```
✓ [DONE] gate passed on lap 3 (laps: 3)
   lap 1: changed  gate-fail  demo gate: lap 1
   lap 2: changed  gate-fail  demo gate: lap 2
   lap 3: changed  gate-pass  demo gate: lap 3
run_id: <id>  (recall with tag loop:convergence-demo)
```

Then recall the stored lap history:

```bash
slm recall "convergence-demo loop"
```

For the full parameter set, see [CLI Reference → Bounded Loops](cli-reference.md#bounded-loops) and [MCP Tools Reference → Bounded-loop tools](mcp-tools.md#bounded-loop-tools).

## Next Steps

| What you want to do | Guide |
|---------------------|-------|
| Let recall say "I don't have that" instead of guessing | [Answer Check](answer-check.md) |
| Set up a specific IDE | [IDE Setup](ide-setup.md) |
| Use your memory from a web app | [Web access](remote-access/README.md) |
| Switch modes or providers | [Configuration](configuration.md) |
| Learn all CLI commands | [CLI Reference](cli-reference.md) |
| Something is not working | [Troubleshooting](troubleshooting.md) |
| Migrate from V2 | [Migration from V2](migration-from-v2.md) |
| Understand how it works | [Architecture](ARCHITECTURE.md) |
| Use SLM from a Python framework | [Framework Adapters](framework-adapters.md) |

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
