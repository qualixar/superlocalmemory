---
name: slm-remember
description: Capture durable facts, decisions, constraints, and gotchas into SuperLocalMemory. Use when the user says "remember that", "save this decision", "note this constraint", or when a session produces a conclusion worth persisting across sessions. Always recall first to avoid duplicates.
version: "4.2.34"
agent: agent
tools:
  - remember
  - recall
  - update_memory
  - set_memory_kind
  - review_memory_kinds
  - confirm_memory_kinds
  - list_corrections
  - review_correction
  - delete_memory
  - Bash(slm:*)
---

# slm-remember — Capture Durable Facts

Store atomic, durable facts into SuperLocalMemory for retrieval in future
sessions. One fact per call. Recall before you remember.

---

## What to store (and what not to)

**Store:**
- Architectural decisions ("Decided to use Postgres not MySQL — reason: JSONB support")
- Project conventions ("All API routes follow /api/v1/resource/{id} pattern")
- Hard constraints ("Never expose raw SQL errors to the HTTP response")
- Resolved gotchas ("Ollama needs keep_alive=-1 or it unloads the model between calls")
- Security rules ("Rate limit all public endpoints at 100 req/min")

**Do not store:**
- Pictures or PDFs through `remember`. It stores text. A screenshot, photo or
  PDF goes through `remember_media` or `remember_document`; see `slm-media`.
  Add the reason it matters in the text of a normal memory if you also want a
  fact about it.
- Transient context that is only relevant within this conversation
- Large blobs of code or full file contents (those belong in the project, not memory)
- Facts the project README already captures

---

## Recall-before-remember (mandatory discipline)

Before calling `remember`, always call `recall` first with the core terms of
what you are about to store. If a near-duplicate exists:

- If the new text **replaces** the old one (a status that moved on, a decision
  that changed, a rule that was reworded), save it with
  `remember(..., replaces="<fact_id>")`. It takes effect at once and can be
  undone.
- If you are **correcting** the old memory and want a person to approve the
  change, use `update_memory(fact_id, content)`. This opens a correction for
  review; it does not overwrite anything (see "Updating a memory" below).
- Only call `remember` with neither when no sufficiently similar fact is found.

Duplicates degrade retrieval quality for every future session.

---

## MCP-first workflow

### 1. Check for duplicates first

```
recall(query="JWT token expiry auth", limit=5, session_id="<sid>")
```

If a near-duplicate is returned:

```
remember(
  content="JWT tokens use 1h expiry for API access tokens; refresh tokens 30d",
  tags="auth,security,decision",
  project="superlocalmemory",
  kind="decision",
  replaces="f8a2bc91",
)
```

The old memory is retired, not deleted: recall stops returning it and it is no
longer loaded at session start. The response's `replaced` says what was retired,
or why nothing was (the new memory is saved either way). An unknown id, or an id
from another profile, is refused before anything is saved.

### 2. Store a new fact

```
remember(
  content="Decided to use JWT with 1h expiry for API auth; refresh tokens persist 30 days",
  tags="auth,security,decision",
  project="superlocalmemory",
  importance=8,
  session_id="<sid>",
)
```

Real response shape:
```json
{
  "success": true,
  "fact_ids": ["c9d4e112"],
  "count": 1,
  "pending": false,
  "pending_id": null,
  "operation_id": "op-7c21",
  "materialization_state": "complete",
  "message": "Stored through canonical daemon ingestion."
}
```

`pending: true` with `materialization_state` `queryable` or `enriching` means the
memory is already stored and findable by its words while enrichment finishes;
do not re-save. A reply with `materialization_state: "accepted"` and `count: 0`
means the memory is durable but the writer was busy; it becomes searchable
within seconds, and resending the same `idempotency_key` returns the final
receipt. When the call used `replaces` the reply carries `replaced`; when the
same words were already saved under another confirmed kind it carries
`kind_conflict` and keeps that kind.

Failures come back as `success: false` with a `code` and `retryable`:
`DAEMON_UNAVAILABLE` (`retryable: true`), `INVALID_KIND`, `NOT_AUTHORIZED`, or an
unknown-profile / unknown-`replaces` refusal (`retryable: false`).

**Never claim "saved" unless `success: true` is in the response.**

### 3. Parameter reference

```
remember(
  content: str,       # required — the atomic fact to store
  tags: str = "",     # comma-separated tags, e.g. "auth,security,gotcha"
  project: str = "",  # project scope, e.g. "superlocalmemory"
  importance: int = 5,# 1–10; see scale below
  session_id: str = "",# from session_init; attributes the write to this session
  session_date: str = "",# when the memory is ABOUT, if not today
  scope: str = None,   # "personal" (default) | "shared" | "global"
  shared_with: str = "",# comma-separated profile_ids for scope="shared"
  idempotency_key: str = "",# replaying the same key will not store a second copy
  kind: str = "",      # what sort of memory this is; see "Say what kind it is"
  replaces: str = None,# fact_id (or memory_id) of the earlier memory this one replaces
  slot: str = "",      # "subject/relation[/qualifier]": the one thing this states a value of
  slot_value: str = "",# that value in a few words; the same value later changes nothing
  source_file: str = "",# absolute path of the file this was learned from (recall flags changes)
  cues: list = None,   # when to bring it back: ["path:src/**/schema.py", "tool:Bash:git push"]
  profile_id: str = "",# write to another existing profile; "" = the active one
  agent_id: str = "mcp_client", # attribution; resolved from the connection when left alone
)
```

> **Multi-scope (opt-in):** leave `scope` unset for `personal` (private to
> this profile — the default, the same as a single-profile install). `"global"` is visible to every
> profile on the machine; `"shared"` is visible to the profiles in `shared_with`.
> See [docs/shared-memory.md](../../../docs/shared-memory.md).

**importance scale:**
- 1–3: Low — passing notes, ideas, soft preferences
- 4–6: Normal — patterns, conventions, standard decisions (default: 5)
- 7–8: High — architectural decisions, integration contracts, known gotchas
- 9–10: Critical — security rules, blockers, irreversible decisions

Use 7–10 only for facts that would cause real damage if forgotten.

### 4. Date a memory to when it happened

`session_date` says **when the memory is about**, as distinct from when you
wrote it. Omit it and the memory is dated today.

```
remember(
  content="The outage on the payments queue was caused by a stale DNS entry",
  tags="incident,payments,postmortem",
  project="platform",
  session_date="2026-08-14",       # YYYY-MM-DD, or a full ISO 8601 timestamp
  session_id="<sid>",
)
```

Use it whenever you are writing something down after the fact — a postmortem, a
decision taken in a meeting last week, a migration that ran on a known date.
Time-filtered recall (`window="7d"`, `window="2026-07-01..2026-07-31"`) reads
event time, so a mis-dated memory is one a time-scoped question cannot find.

`session_date` does not change what **kind** of memory it is. A memory that
describes something planned — "the migration is scheduled for Tuesday", "the
certificate expires on 2026-09-01" — is stored as a **prospective** memory, and
recall reports it as `"fact_type": "prospective"`. That is inferred from how the
content reads, not from the date you pass. Older stores spelled this type
`"temporal"`; that value still reads correctly and needs nothing from you.

---

### 5. Say what kind it is

`kind` is one of nine values: `rule` (a standing instruction: "always/never
..."), `decision` (a choice that settles a question), `status` (the current
state of something, which a later update will replace), `procedure` (steps or
commands), `prospective` (a plan or to-do), `opinion` (a preference or view),
`correction` (says an earlier memory was wrong), `episodic` (something that
happened) or `semantic` (a lasting fact). An unknown value is refused with
`INVALID_KIND` before anything is saved.

Set it whenever you know. A kind you declare is confirmed, and confirmed rules
(up to 10) and confirmed decisions (up to 5) are loaded by `session_init` at
the start of later sessions. A rule counts once however many sentences it has;
with more than ten, rules saved in the session's project come first, then those
the opening recall found relevant, then the newest. Leave `kind` empty when unsure: SLM may suggest
one later, and a suggestion changes nothing until someone confirms it. Never
set a kind just to get a memory loaded at session start.

Manage kinds after the fact:

| Tool | Use |
|---|---|
| `set_memory_kind(fact_id, kind)` | Set and confirm the kind of one memory you know the type of |
| `review_memory_kinds(kind="", limit=20)` | List suggestions waiting for confirmation (limit 1–100) |
| `confirm_memory_kinds(items)` | Confirm up to 200 at once; each item is `{"fact_id": ..., "kind": ...}`, and omitting `kind` accepts the suggestion |
| `memory_kinds_status()` | Counts per kind, the classifying backend, any run in progress |

CLI: `slm remember "<text>" --kind rule`, `slm kinds status|set|review|confirm`,
`slm kinds backfill start|pause|resume|cancel|revert|status`, and
`slm kinds settings` (backend, standing rules on or off). Classifying with Jev
sends memory text online and needs its own consent (`--jev-consent yes`).

### 6. Updating a memory: `replaces` versus `update_memory`

`update_memory(fact_id, content)` and `slm update <fact_id> "<new text>"` do not
overwrite a memory. They keep the original and propose a correction: the reply
has `predecessor_fact_id`, `successor_fact_id`, `correction_case` (with
`case_id`, `status`, `version`) and `review_required: true`. Until a person
applies the case, **the original stays current and the new text is not
returned by recall**. A memory can have only one open correction at a time; a
second edit is refused with a message naming the open case.

Review cases with `list_corrections(limit=100)` and
`review_correction(case_id, action, expected_version)`, where `action` is
`apply`, `reject` or `rollback` and `expected_version` is the case's current
`version`. `apply` retires the original and admits the new text; `rollback`
undoes an applied case (including one made by `replaces`). CLI:
`slm review-correction <case_id> apply|reject|rollback <expected_version>`.

Two rules about SLM's own guesses. When it saves a memory that looks like an
update to an older one, it may file a correction case of its own. That guess
never hides the memory you saved: both stay findable until a person reviews it,
approving retires the older one, and rejecting keeps both. And if you edit,
replace or delete a memory that only such an unreviewed machine guess names,
your action wins and the guess is closed as overtaken (`slm corrections
overtaken` lists them, `slm corrections restore-overtaken <case_id>` puts one
back). A correction a person proposed, or one already decided, is kept
permanently and protects the memories in it from being deleted.

When thousands of unreviewed guesses pile up (a Mode B/C store can file one per
save), close them in bulk with `slm corrections dismiss`. It is a dry run until
`--apply`, which copies `memory.db` first. Filters: `--reason R` (repeatable:
`temporal_contradiction`, `consolidation_update`, `consolidation_supersede`),
`--older-than DAYS`, `--before DATE`, `--profile P`, `--json`. It touches only
guesses nobody reviewed, never a correction a person made, applied, rejected or
rolled back. Each dismissed case can be put back with
`slm corrections restore-overtaken <case_id>`. There is deliberately no bulk
reject: a rejected case counts as decided, and a memory in a decided case can
never be deleted, so rejecting thousands would lock their memories for good.
Never reject cases in bulk to tidy the list. To cut the guesses at the source,
see `slm-status` ("Too many model calls").

Since 4.2.30 the background judges also read these guesses. A judge may close
one only when its answer is that the older memory stays true (the two say the
same thing, both hold, they are separate events, or they are unrelated), and
only once it has earned a certificate from people's audit answers on this
store. Closed guesses show in `slm corrections overtaken` as `judge_closed` and
come back with `slm corrections restore-overtaken <case_id>`. A guess that the
newer memory replaces the older one always waits for a person.

### 6b. Values that change: name the slot, and the file it came from

For a value a later save may change (a port, a version, a release cadence, a
preference), name the slot and the value:

```
remember(content="SLM now ships one big release a month",
         slot="SLM/release cadence", slot_value="monthly")
```

A later memory with the same slot and the same value is a restatement and
changes nothing. A different value does not silently replace the old one: it
opens a dispute, and a decision (by you on evidence, the user, or a checked rule) settles it; `settle_dispute` records who decided and `keep="reopen"` undoes it. A slot
with no value only groups memories and never retires anything.

When the fact was read from a local file, pass `source_file` with its absolute
path. SuperLocalMemory fingerprints the file at once; a later recall marks the
memory `source: changed` or `missing` when the file changes or goes, so the
next agent reads it again before relying on it. CLI: `--slot`, `--slot-value`,
`--source-file`.

### 6c. Cues: bring a memory back at the step where it matters

Most memories are best found by asking. A plan or a warning that matters at one
exact step is different: nobody will ask for it before the step, so attach a cue
and SuperLocalMemory shows it when the step arrives.

```
remember(content="Run the schema migration check before pushing",
         kind="prospective", project="superlocalmemory",
         cues=["path:src/**/schema.py", "tool:Bash:git push"])
```

Add a cue when the memory is a to-do or a warning tied to a file (`path:`), a
command (`tool:Bash:git push`), an error you have seen before (`error:database
is locked`), a repository (`repo:superlocalmemory`) or a date (`date:2026-11-20`,
from that day on). Pick the narrowest cue that still catches the step: a broad
one such as `repo:` or an early `date:` brings the memory back in every session,
which is noise. Do not add cues to ordinary facts, preferences or rules; those are
already loaded or recalled. At most 8 cues, `kind:pattern` each; a malformed one
is refused before anything is saved. Each memory comes back once per session, and
only while the user has the `delivery.cues` setting on. CLI: `--cue KIND:PATTERN`
(repeatable). `slm cues list` shows them.

### 7. One fact per call

Store one atomic fact per `remember` call. Do not concatenate multiple unrelated
points into a single content string — they will be hard to update individually
and harder to retrieve cleanly. If you have three separate decisions, make three
calls.

### 8. Always set tags and project

Untagged, unscoped facts are harder to retrieve and harder to manage. Minimum:
set `tags` to one or two relevant terms and `project` to the repo/product name.

---

## Credentials are never stored (the secret screen, added in 4.2.30)

Never put an API key, token, password or private key in a memory. With `trust.secret_screen`
on, SLM replaces anything shaped like a credential by a typed placeholder before it is
stored, in every door (`remember`, `observe`, the hooks, imports): `[REDACTED:GITHUB:ab12]`
keeps the type and the last four characters of a long key, nothing else, and the original is
kept nowhere (not the database, the search index, the journal or a log). A sentence such as
"the wifi password is ..." is not a credential shape and is stored as written, so do not
rely on the screen: leave the secret out and say where it lives ("in the keychain").
`slm health --memory` counts the placeholders by type, never a value.

Who saved it also decides what the memory may become. A save by an agent (MCP, or a shell with
no terminal) is the agent's own words (`agent_inferred`), not the user's statement
(`user_direct`): with `trust.taint` on, saving it with `kind="rule"` or `kind="decision"` (or
confirming that kind later) is its recorded promotion, so it reaches session start and appears
under Recent decisions with Undo, while a pin, a classifier suggestion and anything from a web
page, a tool's output or an import still wait for the user's OK (`slm trust pending`). Do not
save text copied from a web page or a tool's output as a rule.

## Importing a user's export from another app (WS-9)

When the user hands you an export from ChatGPT, Claude, Mem0 or Letta, or a folder of Markdown
notes (`AGENTS.md`, `CLAUDE.md`, `MEMORY.md`), do not loop over it with `remember`. Use the
importer, which is one run that can be undone: there is no MCP tool for it (tool caps), so run it
in the shell.

```bash
slm import preview PATH            # what the file holds, in plain words; saves nothing
slm import run PATH --json         # background job; prints the run to watch
slm import status [RUN_ID]         # progress
slm import undo RUN_ID             # takes the whole run back exactly
```

It is on by default and runs only when someone starts an import; if a command
says it is off (the user or the kill switch turned it off), tell the user and ask before you switch it on (`slm flags set portability.import on`). Preview first and read the user the
sentence it prints. Imported text is marked `imported_file`, the lowest origin: it never replaces
a memory the user already has, never becomes a rule or a pin by itself, and a value that disagrees
with what they said waits as a dispute (`slm disputes`) for a decision. Every import is recorded
with who started it, and Undo is `slm import undo`. ChatGPT's saved-memory list is not part of its
export: ask the user to copy it into a text file and import that. A Letta agent file does not carry
archival memory; say so.

## Deleting stale facts

The MCP `forget(dry_run=True)` tool runs an Ebbinghaus decay cycle over the
whole profile — it does NOT delete by query. To delete one memory you have the
id of, call `delete_memory(fact_id)`. To delete by query, use the CLI:

```bash
# Preview what would be deleted (always do this first)
slm forget "<query>" --dry-run [--json]

# Execute deletion after confirming the preview
slm forget "<query>" --yes [--json]

# Delete a specific fact by exact ID (use when you have the fact_id)
slm delete <fact_id> --yes [--json]
```

Flags verified in source (main.py):
- `slm forget`: positional `query`, `--dry-run`, `--yes` / `-y`, `--json`
- `slm delete`: positional `fact_id`, `--yes` / `-y`, `--json`

Always run `--dry-run` first and review the preview before passing `--yes`.
A memory that belongs to a reviewed correction cannot be deleted (it is refused
with the case ids and nothing is changed). Treat every other deletion as permanent.

---

## CLI fallback (when MCP is unavailable)

```bash
# Store a fact
slm remember "<content>" [--tags a,b,c] [--kind rule|decision|...] [--json]

# Replace an earlier memory (takes effect at once; undo with review-correction)
slm remember "<content>" --replaces <fact_id> --kind status

# Store a shared/global fact (opt-in)
slm remember "<content>" --scope global
slm remember "<content>" --scope shared --shared-with alice,bob

# Propose a reviewed correction instead of replacing
slm update <fact_id> "<new text>"
slm review-correction <case_id> apply|reject|rollback <expected_version>

# Flags: --tags, --kind, --replaces, --json, --sync, --scope, --shared-with
# --sync: wait for full enrichment before returning (default is async)
# --scope: personal (default) | shared | global ; --shared-with: profile ids for shared
```

**Flags that do NOT exist** on `slm remember`:
`--importance`, `--project`, `--format` — these are MCP-only params or fabricated.

---

## Update vs forget discipline

| Scenario | Action |
|----------|--------|
| Fact has changed and the new version should win now | `remember(..., replaces="<fact_id>")` |
| Fact needs a correction a person should approve | `update_memory(fact_id, new_content)`, then `review_correction` |
| Duplicate found that matches recall result | `replaces` the existing one, or leave it |
| Fact is wrong and should disappear | `slm forget "<query>" --dry-run` then `--yes` |
| Fact has a known ID and is clearly obsolete | `delete_memory(fact_id)` or `slm delete <fact_id> --yes` |

---

## Sharing across profiles (opt-in)

Every `remember` call defaults to `personal` scope — private to the active profile.
To share a fact with other profiles on the same machine, set the `scope` parameter:

```
# Share with every profile on this machine
remember(
  content="API rate limit is 100 req/min per tenant",
  tags="api,limits,shared",
  project="platform",
  scope="global",      # visible to all profiles
  session_id="<sid>",
)

# Share with specific profiles only
remember(
  content="Staging DB migration runs Fridays 22:00 UTC",
  tags="db,ops",
  scope="shared",
  shared_with="work-profile,devops-profile",
  session_id="<sid>",
)
```

**Only set scope when the user explicitly asks to share.** The default
`personal` scope is identical to single-profile SLM. See `slm-scope` for the
complete sharing model and when to use each scope.

---

## Profile-aware storage

`remember` stores in the active profile. To write one memory to another existing
profile, pass `profile_id="<name>"`; an unknown name is rejected, never created.
`switch_profile` moves the active profile for every later call and for other
sessions on this machine, so use it only when the user asks to move. See
`slm-profile`.

---

## Related skills

- `slm-recall` — retrieve what was remembered
- `slm-session` — session lifecycle; session_id is required for attribution
- `slm-scope` — complete guide to personal / shared / global scopes
- `slm-profile` — workspace isolation and profile switching
- `slm-media` — saving pictures and PDFs (`remember_media`, `remember_document`)

<!-- slm:no-mcp-cli:start -->
## When SLM's MCP tools are not in your tool list

Some hosts block local MCP servers by policy, and many agents, bots and agent frameworks never had them. SuperLocalMemory works the same from any shell, so this skill still applies:

- Run `slm status --json` first. Never report "SLM unavailable" only because an MCP tool is missing.
- Every MCP tool this skill names runs as `slm tool <name> --json '{...}'`, with the same arguments, checks and results: a successful call prints exactly what the MCP tool returns, and a failed one prints `"success": false` with `error.code` and `error.message`. `slm tool describe <name>` shows one tool's arguments; `slm tool list` lists the tools available right now (pictures and documents appear once they are turned on, `slm media status`).
- Every other `slm` command answers in one JSON document on stdout: add `--json` after it, put `slm --json` before it, or set `SLM_JSON=1` in the environment. `slm mcp`, `slm session` and `slm session-context` ignore `SLM_JSON` so servers and hooks keep working; give them `--json` directly. Read `success` first; a failure carries `error.code` and `error.message`. Exit code 0 means success, 1 that the command ran and failed (`error.exit_code` keeps a more specific code), 2 that it could not run, 130 that it was interrupted. Under JSON a command never asks a question: one that needs a confirmation answers `confirmation_required` and names the flag. `slm gdpr export` streams its document over several lines.
- Shortcuts for everyday work: `slm recall "<question>" --json` and `slm remember "<text>" --json`. `slm help --json` lists every command with its options.

The slm-cli skill has the full command reference.
<!-- slm:no-mcp-cli:end -->

---

*SuperLocalMemory v4.2.34 · Qualixar · AGPL-3.0-or-later*
