---
name: slm-getting-started-bot
description: First-session orientation for SuperLocalMemory on a headless bot host (Grok Bot, or any Cursor-format plugin install with no hooks, no dashboard, and no interactive setup wizard). What is different here vs. Claude Code/Codex, which 18 tools are actually available, and the first three calls to make.
when_to_use: |
  - The very first session after SLM is installed as a Grok Bot / Cursor plugin
  - "What can I do with SuperLocalMemory here?"
  - A recall or remember call fails and you are not sure why on this host
  - Comparing what works here vs. what the other skills describe for Claude Code
allowed-tools: session_init, remember, recall, switch_profile
---

# slm-getting-started-bot — First Session on a Headless Bot Host

You are talking to SuperLocalMemory through its MCP server, started as
`uvx --from superlocalmemory==<version> slm mcp` by the plugin manifest. This
skill is for the ways that is different from a Claude Code or Codex install —
read it once per new host, not once per session.

---

## What is NOT available here

Grok Bot (and Cursor's plugin host generally) reads only two parts of a
plugin: `mcpServers` and `skills`. It does not run:

- **Hooks** — no automatic session-start context load, no automatic
  checkpoint on session end. Call `session_init` yourself at the start of a
  session if you want prior context; nothing injects it for you.
- **Slash commands or sub-agents** — the `commands/` and `agents/` components
  other SLM plugins ship are Claude-Code-shaped and are not part of this
  install. Everything you can do here, you do through the MCP tools
  directly.
- **A dashboard.** There is no browser UI to check on this host. `slm status`
  style information comes back from the tools themselves (`session_init`'s
  response, or asking `recall` for recent memories) — there is nothing to
  click.
- **`ANTHROPIC_BASE_URL` or any Claude-specific proxy behavior.** SLM does not
  depend on it anywhere; nothing to configure here either way.

None of this is a degraded mode — it is the whole surface. Do not tell the
user a feature is "missing" when it was simply never wired to this host
format in the first place.

---

## What IS available: the core profile (18 tools)

This plugin sets `SLM_MCP_PROFILE=core`, the smallest tool tier, on purpose —
a shared, memory-tight computer should not load tool descriptions or RAM for
tiers it will not use. The 18 tools:

`remember`, `recall`, `search`, `fetch`, `list_recent`, `update_memory`,
`forget`, `session_init`, `close_session`, `slm_compress`, `slm_retrieve`,
`slm_cache_set`, `slm_cache_get`, `slm_optimize_stats`, `review_correction`,
`list_corrections`, `get_memory_summary`, `switch_profile`.

If you are looking at another skill that mentions a tool NOT in this list
(`build_code_graph`, `mesh_send`, `report_outcome`, `delete_memory`,
`set_memory_kind`, anything governance/compliance-shaped), that skill describes
a different, larger profile (`code`, `full`, `power`) — it is accurate for Claude
Code or Codex installs, not for this one, unless someone has deliberately
reconfigured `SLM_MCP_PROFILE`. Skip those steps rather than reporting an error.

Two things the core set does cover that other skills explain: `remember` takes
`kind` and `replaces` (a newer version of a memory should be saved with
`replaces=<fact_id>`; see `slm-remember`), and `list_corrections` /
`review_correction` handle the review of an `update_memory` edit. A bot with
no one to review should prefer `replaces` over `update_memory`.

---

## The first three calls on a new host

1. **`session_init`** — nothing runs this automatically here (no hooks), so
   call it yourself at the start of the first real session:
   ```
   session_init(project_path="<where you are working>", query="")
   ```
   Returns recent relevant memories and a session id to pass to later calls.

2. **`remember`** one real fact, before doing anything else, as a smoke test:
   ```
   remember(content="First session on this host, verifying setup.", tags="setup", session_id="<sid>")
   ```

3. **`recall`** it back:
   ```
   recall(query="first session verifying setup", session_id="<sid>")
   ```
   If this returns the fact, the daemon, the embedding path, and the
   CPU-only torch install all worked. If it does not, the result still tells
   you something — see Troubleshooting below.

---

## Cold start is slower here than on a warm laptop

The first `remember`/`recall` on a fresh `SLM_DATA_DIR` has to: resolve
`uvx`'s pinned `superlocalmemory` package, start the daemon, and warm an
embedding model (and, unless this is the lite profile, a reranker model) —
tens of seconds, not milliseconds. A `recall` whose `channel_status` shows
`warming` (or `no_embedding`) for some channels in that window is not a bug; it
means the embedding model had not finished loading yet, and the answer is
incomplete rather than empty. Retry once rather than assuming something is
broken.

---

## This host already runs the lite bot-host profile

Unless something overrode it, `mcp.cursor.json` sets `SLM_RERANKER_ENABLED=
false`, `SLM_RERANKER_IDLE_TIMEOUT=120`, and `SLM_MAX_EMBEDDING_WORKERS=1` —
the RAM-conscious default for a computer shared by every bot on it. Recall
still runs every other channel (BM25, semantic, entity graph, temporal,
spreading activation, Hopfield) and fuses them; it just skips the
cross-encoder re-ranking pass. If recall quality seems to matter more than
RAM headroom for your use case, the user can set `SLM_RERANKER_ENABLED=true`
in the manifest's env block — ask before changing it, since it trades RAM for
quality on a shared box.

---

## Troubleshooting

- **A tool call returns a "profile" or "tool not available" style error** —
  you are probably calling a tool outside the 18-tool core set. Check the
  list above before assuming the server is broken.
- **Everything times out on the very first call of a session** — likely the
  cold-start warmup described above. Wait and retry once before reporting a
  failure.
- **You are not sure whether another bot on this computer can see what you
  just stored** — read `slm-bot-memory` before storing anything you would
  not want another bot on this machine to read.

---

## Related skills

- `slm-bot-memory` — cross-bot namespacing and what must never be stored
- `slm-remember` / `slm-recall` — full parameter reference for the two tools
  you will use the most
- `slm-session` — what `session_init`/`close_session` actually do

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
