# SuperLocalMemory — Codex Plugin

Local-first agent memory + reversible context compression for OpenAI Codex CLI.

SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later

---

## What this gives Codex

| Capability         | How                                            |
|--------------------|------------------------------------------------|
| Agent rules        | `AGENTS.md` — loaded automatically per project |
| MCP memory tools   | `.codex/config.toml` — the default 57-tool profile (no profile forced) |
| Lifecycle hooks    | `hooks/hooks.json` — session start/stop/prompt |
| Slash skills       | `skills/*/SKILL.md` — the SLM skills via `/skills` |
| Venv launcher      | `scripts/slm-launch` — optional isolated mode  |

---

## Prerequisites

- **Python 3.12+** on PATH
- **SuperLocalMemory installed:**

  ```bash
  pip install superlocalmemory   # global
  # or
  pipx install superlocalmemory  # isolated (recommended)
  ```

- Verify: `slm --version`
- **Windows:** install with pipx or pip (they put `slm.exe` on PATH). On
  native Windows the MCP server always runs that installed `slm`; with no
  `slm` on PATH it does not start, and its log says `slm is not on PATH` with
  the install command.

---

## Install — recommended: the SLM commands

```bash
slm connect codex --dry-run   # show what would be written, change nothing
slm connect codex             # wire the superlocalmemory MCP server into your Codex config
slm codex install             # SLM-owned skills, subagents and lifecycle hooks (preview with --dry-run)
slm codex status              # what is installed
slm codex remove              # remove only the SLM-owned add-ons
```

`slm codex install` leaves MCP wiring to `slm connect codex` and leaves your own
settings alone; `slm codex remove` removes only what SLM installed. Review and
trust newly installed hooks in Codex with `/hooks`. The agent rules in
`codex-plugin/AGENTS.md` are not written for you: append them to your `AGENTS.md`
as described below.

---

## Manual install (if you prefer to edit by hand)

> **Do NOT `cp` these files over your existing ones** — that would delete your other MCP
> servers, rules, and hooks. Merge each block into your existing files.

### 1 — Append the SLM rules to your `AGENTS.md`

Codex reads `AGENTS.md` from the repo root. Append the contents of `codex-plugin/AGENTS.md`
to your existing `AGENTS.md` (or create one if you have none). Wrap what you add in
`<!-- BEGIN SuperLocalMemory -->` … `<!-- END SuperLocalMemory -->` comments of your own so
you can find and update it later without touching your other rules.

### 2 — Add the MCP server to your Codex config

Add ONLY this section to your existing `.codex/config.toml` (project) or
`~/.codex/config.toml` (all projects) — keep every other `[mcp_servers.*]` you already have:

```toml
[mcp_servers.superlocalmemory]
command = "slm"
args = ["mcp"]
env = { SLM_AGENT_ID = "codex" }
```

This registers the `superlocalmemory` MCP server with only the agent id. No profile is
forced, so it serves the default 57-tool profile; add `SLM_MCP_PROFILE = "code"` yourself
for the 38-tool code profile (memory, portable Brain evidence, code graph, profile
switching, memory-kind management, and bounded loops).

### 3 — Append lifecycle hooks (optional)

Merge the hook entries from `codex-plugin/hooks/hooks.json` into your existing
`hooks.json` (append to the `SessionStart` / `UserPromptSubmit` / `Stop` arrays — do not
replace the file). Every SLM hook is fail-open. Review the trust flow with `/hooks` before
accepting.

---

## Skills

To use SLM skills in Codex, type `/skills` in the Codex chat and select the skill you want:

| Skill          | Purpose                                          |
|----------------|--------------------------------------------------|
| slm-recall     | Search memories, decisions, past context         |
| slm-remember   | Save durable facts and decisions                 |
| slm-session    | Session lifecycle (init + close)                 |
| slm-status     | Health check, optimize stats                     |
| slm-cache      | KV cache for repeated reads                      |
| slm-compress   | Reversible context compression                   |
| slm-graph      | Code graph: blast radius, callers, search        |
| slm-scope      | Personal / shared / global memory scoping        |
| slm-profile    | Memory profiles versus MCP tool sets             |
| slm-governance | Enterprise roles, retention, audit, GDPR         |
| slm-mesh       | Cross-session peer coordination                  |
| slm-loop       | Bounded, gate-verified agent loops               |
| slm-bot-memory | Sharing one computer with other bots             |
| slm-getting-started-bot | First session on a headless bot host    |

To make these available as project skills, copy `codex-plugin/skills/` to your project:

```bash
cp -r codex-plugin/skills ./codex-skills
```

Then reference them from your Codex settings or use `/skills` to load them directly.

---

## Advanced: venv-isolated launcher

If you need an isolated Python environment (prevents `slm` from sharing packages with other
tools), use the provided launcher script instead of the bare `slm` binary.

1. Install SLM into a venv at `~/.superlocalmemory/venv/`:

   ```bash
   python3 -m venv ~/.superlocalmemory/venv
   ~/.superlocalmemory/venv/bin/pip install superlocalmemory
   ```

2. Update `.codex/config.toml` to use the launcher:

   ```toml
   [mcp_servers.superlocalmemory]
   command = "/absolute/path/to/codex-plugin/scripts/slm-launch"
   args = []
env = { SLM_MCP_PROFILE = "code", SLM_AGENT_ID = "codex", SLM_DATA_DIR = "~/.superlocalmemory" }
   ```

   Replace `/absolute/path/to/codex-plugin/scripts/slm-launch` with the actual path.

3. Bootstrap the venv on first use:

   ```bash
   SLM_DATA_DIR=~/.superlocalmemory codex-plugin/scripts/ensure-venv.sh
   ```

---

## Hooks: session lifecycle

The provided `hooks/hooks.json` wires three Codex lifecycle events:

| Event             | Command                  | Purpose                         |
|-------------------|--------------------------|---------------------------------|
| `SessionStart`    | `slm hook mandate`       | Load SLM mandate into context   |
| `SessionStart`    | `slm hook start`         | Open session, load memories     |
| `UserPromptSubmit`| `slm hook topic_shift`   | Detect topic changes mid-session|
| `Stop`            | `slm hook stop`          | Close session, save summary     |
| `Stop`            | `slm hook stop_outcome`  | Record session outcome          |

All hook commands use `slm` from PATH and are fail-open (`|| true`). If `slm` is not on
PATH they silently no-op without breaking Codex.

**PostToolUse/PreToolUse hooks** are not included because Codex's internal tool names
differ from Claude Code's (`Write`, `Edit`, `Bash`, etc.) and are not publicly documented.
Add them manually once you know your Codex version's tool names.

---

## Difference from the Claude Code plugin

| Aspect             | Claude Code plugin      | Codex plugin                     |
|--------------------|-------------------------|----------------------------------|
| Rules file         | `CLAUDE.md`             | `AGENTS.md`                      |
| MCP config         | `.mcp.json` (JSON)      | `.codex/config.toml` (TOML)      |
| Runtime vars       | `${CLAUDE_PLUGIN_ROOT}` | Script-relative path resolution  |
| Hook tool matchers | `Write\|Edit\|Bash`     | Omitted (Codex names unconfirmed)|
| Plugin manifest    | `.claude-plugin/plugin.json` | Not applicable             |
| Skills             | The same skills (identical) | The same skills (identical)  |

---

## Troubleshooting

**`slm: command not found`** — SLM is not on PATH. Run `pip install superlocalmemory` or
use the venv launcher.

**MCP server not appearing in Codex** — Confirm `.codex/config.toml` is in your project
root (not inside a subdirectory), then restart Codex.

**Session hooks not firing** — Run `slm status` to confirm the daemon is healthy. Check
`/hooks` in Codex to see whether the hooks are loaded and trusted.

---

SuperLocalMemory v4.1.25 · Qualixar · https://github.com/qualixar/superlocalmemory
