---
name: slm-bot-memory
description: Cross-bot memory on a shared computer (Grok Bot, Cursor plugin, any host where several agents share one machine). Explains agent_id attribution vs. profile/scope access, how to namespace by agent_id + profile + scope, and what must never be written to memory. Read this before the first remember/recall on a new Grok-Bot-style host.
version: "4.1.25"
agent: agent
tools:
  - remember
  - recall
  - search
  - session_init
  - switch_profile
---

# slm-bot-memory — Cross-Bot Memory on a Shared Host

SuperLocalMemory ships as an MCP server. On a single-user laptop, one bot is
the only thing talking to it. On a **shared-bot host** — Grok Bot's "one
computer, shared by all your Bots" model, several Cursor windows, or any
setup where more than one agent process points at the same install — several
agents can reach the same memory store at once. This skill is about that
case: what is actually isolated, what is not, and what must never be written
regardless.

---

## Three things that sound like isolation and are not the same

| Setting | What it controls | Does it hide memories from another bot? |
|---|---|---|
| `SLM_AGENT_ID` | **Attribution** — who gets credit for a memory in logs and summaries | **No.** It is metadata, never an authenticated principal (verified in `mcp/agent_context.py`'s own docstring). Any bot that can query the same store can read a memory written under a different `agent_id`. |
| `SLM_DATA_DIR` (profile root) | **Which database file** the daemon opens | **Yes.** Two bots pointed at different `SLM_DATA_DIR` values have fully separate, independent memory stores — nothing is shared unless you explicitly configure sharing. |
| `scope` (`personal` / `shared` / `global`, see `slm-scope`) | **Visibility inside one store**, between profiles that share that store | Only the way you set it. `personal` (the default) stays within the writing profile; `shared`/`global` are explicit, user-initiated opt-ins. |

A memory profile (the namespace `profile_id` and `switch_profile` select) is not
a fourth kind of isolation between bots. Any local caller that can name a profile
can read it, and `switch_profile` moves the active profile for every bot on the
computer. Treat profiles as organisation, not privacy.

The practical rule: **`SLM_AGENT_ID` tells you who wrote something; `SLM_DATA_DIR`
and `scope` decide who can read it.** If two bots on the same Grok Bot computer
use the same `SLM_DATA_DIR` (the common case — it is a shared filesystem, and
nothing in Grok Bot's plugin model points different bots at different data
directories by default), they see each other's `personal`-scope memories too,
the same way two terminal sessions on one laptop would. Giving two bots
different `SLM_AGENT_ID` values does not change that — it only changes whose
name shows up on the memory.

If you need one bot's memories genuinely unreachable from another, that bot
needs its **own `SLM_DATA_DIR`** (or a store where company mode with roles is
configured — see `slm-governance`), not just its own `SLM_AGENT_ID` or its own
profile.

---

## Namespacing that actually works

Use all three dimensions together, each for what it is good at:

1. **`agent_id`** (`SLM_AGENT_ID` env, or the `agent_id` parameter on
   `remember`/`recall` directly) — so a human or another agent reading
   `get_memory_summary` or the stored facts later can tell which bot said
   what. Set it to something stable and specific: `grok_bot_support`,
   `grok_bot_release_notes`, not a generic `bot`.
2. **`SLM_DATA_DIR`** — the real isolation boundary. One store per bot that
   genuinely needs privacy from the others. A `profile_id` inside a shared store
   groups memories but does not hide them.
3. **`scope`** (see `slm-scope`) — inside a store two or more bots
   legitimately share, keep writes `personal` by default and only promote to
   `shared`/`global` when a fact is meant for every bot on that store.

```
remember(
  content="Deploy window for the support bot is Tue/Thu 14:00-16:00 UTC",
  tags="ops,schedule",
  agent_id="grok_bot_support",
  session_id="<sid>",
)
```

```
recall(
  query="deploy window",
  agent_id="grok_bot_support",
  session_id="<sid>",
)
```

Recall is not filtered by `agent_id` automatically — it is attribution on the
written record. If you need "only what this bot wrote", pass
`saved_by="grok_bot_support"` to `recall` (or `--saved-by` on `slm recall`),
which keeps only memories saved by that agent id. It filters; it does not
protect, because another bot can leave it out.

---

## What never goes into memory, on any host

This is not Grok-Bot-specific — it is the baseline for every SLM host — but
it matters more here because a shared computer means a shared blast radius if
it is violated:

- **Secrets, API keys, tokens, passwords, certificates.** Never pass them to
  `remember`, even "just this once" or "temporarily." If a fact needs to
  reference that a credential exists, store the *fact of its existence and
  where to rotate it*, never the value.
- **Personal data about identifiable people** that was not already something
  the user explicitly asked to be remembered about themselves (names,
  emails, addresses, health, financial details of third parties).
- **Anything from Grok Bot's shared filesystem you did not write yourself**
  and were not asked to remember. "Files are visible to every Bot" on that
  computer (Grok Bot's own docs) — a file being readable is not the same as
  having permission to copy its contents into long-term memory that other
  bots and later sessions will see.
- **Command-line credentials.** Grok Bot's own docs note these are shared
  across bots on the computer; SLM must not become a second, more durable
  copy of them.

If you are unsure whether something is a secret, treat it as one and ask the
user before storing anything derived from it.

---

## Checklist for a new bot on an existing shared host

1. Confirm `SLM_DATA_DIR` — is this bot meant to share the existing store, or
   does it need its own? (`slm status --json` on the computer shows `data.base_dir`
   and `data.db_path`.)
2. Set a specific `SLM_AGENT_ID` for this bot before the first `remember`.
3. Read `slm-scope` before writing anything with `scope="shared"` or
   `scope="global"` — those are machine-wide, not just cross-bot.
4. Read the "never goes into memory" list above once, out loud if you are
   unsure, before the first write on this host.

---

## Related skills

- `slm-getting-started-bot` — first-session setup for a headless bot host
- `slm-scope` — the personal/shared/global visibility model this skill
  namespaces on top of
- `slm-profile` — memory profiles versus tool sets, and what switching does
- `slm-governance` — role-based access when a store has multiple human/bot
  members and real access control, not just convention

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
