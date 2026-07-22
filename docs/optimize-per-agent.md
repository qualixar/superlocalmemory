# Using SLM cache & compression per coding agent

> SuperLocalMemory V3 · https://superlocalmemory.com · Part of Qualixar

SuperLocalMemory's token optimization is **proxy-free** by default. The MCP tools
`slm_compress`, `slm_retrieve`, `slm_cache_set`, `slm_cache_get`, and
`slm_optimize_stats` are available to any agent connected to the SLM MCP server,
and safe **lossless** compression (whitespace normalization + JSON minification)
is **on by default**. What differs per front-end is the *instruction layer* that
prompts the agent to actually use those tools.

## What each front-end ships

| Front-end | MCP tools | Skill / rules | Optimize advisor | Auto hook |
|---|---|---|---|---|
| **Claude Code** | ✅ | ✅ CLAUDE.md + `slm-cache`/`slm-compress` skills | ✅ full | checkpoint on Write/Edit |
| **Codex** | ✅ | ✅ AGENTS.md + skills | ✅ full | — (see limitation below) |
| **VS Code / Copilot** | ✅ | ✅ copilot-instructions + prompts | ✅ full | checkpoint (Copilot CLI GA; VS Code Preview) |
| **Cursor** | ✅ | ✅ `.cursor` rules (memory + optimize) | — | — |
| **Antigravity** | ✅ | ✅ `.agent` skill (memory + optimize) | — | — |
| **JetBrains (Junie)** | ✅ | ✅ `.junie/AGENTS.md` | — | — |
| **Windsurf / Gemini CLI / Zed** | ✅ | ✅ rules file (memory + optimize) | — | — |

## Enabling and using it

- **Default (every agent):** run `slm connect <agent>` to install the MCP server
  and instruction layer. The agent's rules then tell it to `slm_compress` large
  tool output and `slm_cache_*` repeated reads. Safe lossless compression needs
  no configuration.
- **Turn it off:** `slm optimize off` (fully disables compression).
- **Check savings:** `slm optimize savings` / `slm optimize status`.

## Aggressive tier (opt-in)

Lossy prose compression (LLMLingua-2) and semantic caching are opt-in and run
through `slm proxy` — intended for apps that make **API-key / SDK** calls you
control, not for subscription coding agents (whose OAuth traffic can't be
proxied). The live proxy applies **lossless compression only**; lossy Layer-2
prose runs via the explicit `slm_compress` tool, which returns a `ccr_id` you can
pass to `slm_retrieve` to recover the exact original.

## Codex hook limitation

Codex CLI fires lifecycle hooks **only for its `shell` tool** — file
edit/read/write and MCP calls do not fire hooks, and PostToolUse cannot modify
tool output. So Codex cannot auto-compress on file edits the way Claude Code
does. On Codex, cache/compress works through the MCP tools plus the
`slm-optimize-advisor` sub-agent and the AGENTS.md rules instead.

---

SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later
