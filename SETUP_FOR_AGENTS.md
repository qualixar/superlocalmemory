# SETUP_FOR_AGENTS.md — SuperLocalMemory First-Install Guide

> **Written for AI agents.** Execute the numbered steps top-to-bottom.
> No human input required unless a step explicitly asks for it.

## What SLM Is (and why wire it)

SuperLocalMemory (SLM) is a local-first memory layer for AI coding agents.
It stores facts, project decisions, and session context in a local
SQLite/graph store and surfaces them via semantic recall — no cloud required
in Mode A. Once wired, call `session_init` at session start to load context
automatically, then `remember`/`recall` throughout to persist and retrieve
facts across sessions. SLM is Qualixar's core tool in the AI Reliability
Engineering category: it eliminates repeated context loss and keeps
multi-session projects coherent. Cost at Mode A: zero.

---

## Setup Checklist

### Step 1 — Confirm install

```bash
slm --version
```
Expected: `superlocalmemory 3.x.x`

On failure: `pip install superlocalmemory` then re-run.

---

### Step 2 — First-run init (non-interactive)

```bash
slm init --auto
```

Writes Mode A config and installs Claude Code hooks. Does **not** connect
IDEs and does **not** download the embedding model — both are separate
required steps below.

Expected on stderr: `[OK] slm init --auto: setup complete (mode A, non-interactive)`

On failure: check write permissions on `~/.superlocalmemory/` (or `$SLM_HOME`).

---

### Step 3 — Pick an operating mode

Decision rule — check for Ollama first:
```bash
ollama list 2>/dev/null && echo "use mode B" || echo "use mode A"
```

| Mode | When | Command |
|------|------|---------|
| A — Local Guardian | Default. No LLM calls. Zero-cost. | `slm mode a` |
| B — Smart Local | Ollama installed and a model pulled. | `slm mode b` |
| C — Full Power | Cloud provider (OpenAI / Anthropic / Azure). | `slm mode c` |

```bash
slm mode a        # use this if unsure — always works
```

Verify:
```bash
slm mode
```
Expected: `Current mode: A`

Mode C requires `OPENAI_API_KEY` (or equivalent); run
`slm provider set openai` to configure.

---

### Step 4 — Start the daemon

```bash
slm serve start
```

Verify:
```bash
slm serve status
```
Expected: `Daemon: RUNNING (PID …, mode=A, …)`

On failure:
```bash
slm reap --force   # kill orphaned slm processes
slm serve start    # retry
slm doctor         # pre-flight: deps, embedding worker, connectivity
```
---

### Step 5 — Download the embedding model (one-time, ~500 MB)

Run **after** the daemon is up. Warmup is daemon-aware: if the daemon already
loaded the model it reports success immediately.

```bash
slm warmup
```

Expected when daemon is healthy:
`[PASS] Daemon is running with embedding model loaded.`

On restricted networks: model fetches from HuggingFace. If blocked, Mode A
still works via keyword search; only semantic recall needs the model.

---

### Step 6 — Connect your IDE / agent

Auto-detect and configure all present IDEs:
```bash
slm connect
```

Target a specific one:
```bash
slm connect claude-code       # Claude Code (hooks + MCP pointer)
slm connect cursor            # Cursor
slm connect vscode-copilot    # VS Code / GitHub Copilot
slm connect codex             # Codex CLI
slm connect claude-desktop    # Claude Desktop
slm connect continue          # Continue.dev
slm connect windsurf          # Windsurf
slm connect gemini-cli        # Gemini CLI
slm connect jetbrains         # JetBrains IDEs
```

List all supported IDs: `slm connect --list`

Verify:
```bash
slm status                # system status: mode, profile, DB size
slm hooks status          # for Claude Code: confirms hooks wired
```

On failure:
```bash
slm hooks install                  # additive; does not overwrite other hooks
slm connect claude-code --here     # project-scope wiring
```

---

### Step 7 — Smoke test

```bash
slm remember "SLM setup complete — smoke test fact"
```

Wait 2 seconds, then:
```bash
slm recall "smoke test"
```

Expected: the stored fact returns with a relevance score.

If recall returns 0 results:
1. `slm serve status` — confirm daemon running.
2. `slm doctor` — full pre-flight check.
3. `slm warmup` — confirm model loaded.
4. `slm list -n 5` — if fact appears here, indexing is lagging; wait 10 s
   and retry recall.

---

## Using SLM Every Session

Hooks handle `session_init` automatically on each Claude Code session start.
For other agents or manual use:

| When | MCP tool | CLI equivalent |
|------|----------|----------------|
| Session start | `session_init(project_path, topic)` | `slm session open --project-path .` |
| Save a decision | `remember(content, tags)` | `slm remember "…" --tags "t1,t2"` |
| Recall before acting | `recall(query)` | `slm recall "query"` |
| List recent facts | `list_recent()` | `slm list -n 20` |
| Session end | `close_session()` | `slm session close` |

Discipline:
- Call `session_init` **before** your first tool call.
- Call `recall` **before** claiming something is unknown.
- Call `remember` after significant decisions, file paths, or blockers.
- Do not recall every turn — `session_init` already loaded context at start.
- Refine on low confidence: if `no_confident_match` is `true` (or `answer_confidence` is low / `abstained` is `true`), rewrite the query into 1–3 more specific sub-queries (split multi-hop questions; try entity names, synonyms, or broader phrasing) and call `recall` again before concluding nothing was found. SLM returns fast local results (~1–2s, no server-side LLM round on the hot path) — you, the calling model, drive refinement.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Daemon won't start | `slm reap --force` → `slm serve start` → `slm doctor` |
| Recall returns nothing | `slm serve status` (start if down); `slm warmup` (verify model) |
| Mode change not applied | `slm mode a` (or b/c) → `slm restart` |
| IDE not connecting | `slm connect --list` → `slm connect <exact-id> [--here]` |
| Hooks missing after upgrade | `slm hooks install` |
| Scoring / consistency errors | `slm health` → `slm doctor` |

---

## Quick Reference

```
slm --version              Confirm install
slm init --auto            Non-interactive first-run setup
slm mode [a|b|c]           Get or set operating mode
slm serve start            Start daemon
slm serve status           Check daemon health
slm warmup                 Download / verify embedding model
slm connect                Auto-configure all detected IDEs
slm connect <ide>          Configure a specific IDE
slm remember "…"           Store a fact
slm recall "…"             Retrieve by semantic query
slm list -n 20             Show 20 most recent facts
slm status                 System status
slm doctor                 Full pre-flight diagnostics
slm hooks status           Check Claude Code hook wiring
slm health                 Math layer health
slm restart                Nuclear restart (kill orphans, start fresh)
```
