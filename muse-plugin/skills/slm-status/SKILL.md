---
name: slm-status
description: Health and optimization stats for SuperLocalMemory — call slm_optimize_stats() for live compression and cache counters (compress_runs, tokens_saved_compress, cache_proxy_hits, cache_proxy_misses, cache_kv_hits, cache_kv_misses); run slm status [--json] for system state (mode, profile, DB size, fact/entity/edge counts) and slm doctor [--json] for preflight including the "Optimize (Surface B)" health line; use together to confirm optimization is actually saving tokens.
when_to_use: "check slm status, health check, is slm working, optimize stats, tokens saved, cache hits, compress runs, slm doctor, preflight, db size, slm info, diagnose slm, slm media status, are pictures on, slm features, picture search not working, slm media repair, embedder status, memory health, slm enrich, repair memories, undo a repair, failed saves, repair saves, redacted memory, blanked memory, mode b slow, too many model calls, lower llm cost, pile of correction proposals, dismiss corrections, truth checks, slm doctor --truth, judge certificates, second opinions, slm health --memory, does memory help, which memories are used, memory use rate, withholding, audit questions waiting, slm audit --count, slm proposals, proposed changes, undo a change, roll back a change, previous version of a summary, slm consent, what leaves my computer, slm measurement, turn off withholding, does slm experiment on me, stop measuring"
---

# slm-status — Health and Optimize Stats

## Purpose

Use this skill to answer: "Is SLM healthy?", "Is compression/caching actually saving tokens?", and "What does the system look like right now?" It covers the MCP stats and status tools, the `slm status` CLI, the `slm doctor` preflight, and the store-health commands.

## Primary MCP Tool: slm_optimize_stats

```
slm_optimize_stats() -> dict
```

No arguments. Returns the counters the daemon has persisted.

### Return dict (all keys always present)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on success; `False` on internal error |
| `compress_runs` | int | Total compress calls recorded by the daemon (persisted across restarts) |
| `tokens_saved_compress` | int | Cumulative tokens saved by compression (daemon-persisted) |
| `cache_proxy_hits` | int | Proxy-layer cache hits (daemon-persisted) |
| `cache_proxy_misses` | int | Proxy-layer cache misses (daemon-persisted) |
| `cache_kv_hits` | int | Hits on the `slm_cache_get` key-value cache (daemon-persisted) |
| `cache_kv_misses` | int | Misses on the `slm_cache_get` key-value cache (daemon-persisted) |
| `ccr_note` | str \| None | Note about CCR entry count (not tracked per-session; see daemon `/api/v1/metrics`) |
| `note` | str \| None | Scope clarification or error detail |

### Important scope distinction

All six counters are **daemon-persisted**: they survive MCP restarts and accumulate over the install's lifetime, so they are totals, not per-session figures. Only if the persisted KV counters cannot be read does the tool fall back to this process's own tally (the `note` field says so). To judge one stretch of work, call `slm_optimize_stats` before and after and subtract.

### Reading whether optimization is saving tokens

```python
stats = await slm_optimize_stats()
if stats["ok"]:
    savings = stats["tokens_saved_compress"]
    kv_hit_rate = (
        stats["cache_kv_hits"] / max(stats["cache_kv_hits"] + stats["cache_kv_misses"], 1)
    )
    # savings > 0 and kv_hit_rate > 0.5 means Surface B is actively reducing costs
```

If `compress_runs` is 0 after several sessions, compression is not being triggered — check daemon config and whether `slm_compress` is being called.

If `cache_kv_hits` does not grow after repeated work, verify key naming consistency (the same key string must be used for set and get, by the same agent).

## Secondary CLI: slm status

```bash
slm status [--json] [--verbose]
```

Reports system-level state — not optimization counters. Canonical fields:

- **mode** — active operation mode (`a`, `b` or `c`)
- **provider** — the LLM provider for modes B and C, or `none`
- **profile** — current memory profile name
- **db_size_mb**, **db_path**, **base_dir** — where the store is and how big
- **fact_count**, **entity_count**, **edge_count** — counts for the active profile
- **version**, **profile_generation**, **projection_queue_depth**, and (daemon running) **saves_waiting** and **unreadable_saves**

`--verbose` / `-v` adds: the disabled marker, last booted version, and daemon port.

`--json` prints the standard envelope `{"success", "command", "version", "data": {...}}`; read the fields above from `data`. Prefer it for agent consumption:

```bash
slm status --json
```

```json
{"success":true,"command":"status","version":"...","data":{"mode":"a","provider":"none","profile":"default","db_size_mb":12.4,"fact_count":384,"entity_count":201,"edge_count":519,"profile_generation":0,"projection_queue_depth":0,"saves_waiting":0,"unreadable_saves":0}}
```

The MCP equivalent is `get_status(profile_id="")`, which returns the same fields at the top level (it is not part of the smallest `core` tool set). `saves_waiting` above zero means saved memories are durable but still being indexed (searchable within seconds). `unreadable_saves` above zero means that many saves could not be read back with this computer's key and were kept unchanged in the admission journal (the daemon log has their ids); a negative value means the journal did not answer. Status does not report a user role.

Do not rely on the human-readable format for parsing — always use `--json` when the output feeds another tool.

## Secondary CLI: slm doctor

```bash
slm doctor [--json] [--quick] [--deep] [--fix]
```

Preflight check covering dependencies, embedding worker, daemon connectivity, and Surface B health. The **"Optimize (Surface B)"** line confirms whether the compression and cache subsystem initialised correctly.

`--quick` skips the daemon and embedding probes — runs only dependency and config checks; faster but incomplete.

`--deep` reads every database page (`PRAGMA integrity_check`) instead of the structural check; slow on a large store. `--fix` repairs what it can (re-downloads missing models, installs sqlite-vec) before checking, then reports.

`--json` outputs structured results per check — use this in automated health pipelines.

A passing doctor output confirms:
- Python deps present
- Embedding worker reachable
- Daemon responding
- Surface B (Optimize) initialised

A failing "Optimize (Surface B)" line means `slm_compress`, `slm_cache_set`, and `slm_cache_get` may not function correctly — investigate daemon config before relying on those tools.

## Secondary CLI: slm optimize status

```bash
slm optimize status [--json]
```

Shows whether the Optimize module (cache + compress) is currently enabled or disabled at the daemon level. Available subcommands also include `optimize on`, `optimize off`, and `optimize savings`.

The `optimize savings` subcommand accepts:

```bash
slm optimize savings [--since <days>] [--provider anthropic|openai|gemini] [--json]
```

`--since` defaults to 7 days. `--provider` filters by the target AI provider.

`slm_optimize_stats()` via MCP is the same data as `slm optimize savings`; use whichever surface you have.

## Store health and recovery

```bash
slm db integrity [--pages] [--json]   # read-only, safe while SLM runs
slm db repair [--json]                # preview of what a repair would do (read-only)
slm db repair --apply --root <data folder> [--batch-size N] [--pause-ms MS] [--max-seconds S]
slm db repair --undo <run_id> --root <data folder>
slm ops list | status | resolve <operation_id> --action retry|force_reconcile|cancel
slm ops repair-saves [--apply] [--json]   # re-file saves that failed for good; dry run unless --apply
slm brain status [--json]             # observation-only Living Brain evidence totals
slm embedder status                   # progress of an embedding-model switch
slm models                            # recommended and installed local models
```

`slm db integrity` answers five separate questions so one cannot hide another:
page integrity (only with `--pages`), relational integrity (orphan rows, erased
words still stored), source fidelity (facts withheld from answers or no longer
saying what their memory said), projection readiness (keyword, vector and date
search work still owed), and any repair running or last run. It prints counts
only, never memory text.

`slm db repair` fixes leftover rows, erased-word leftovers, unfinished deletes
and memories that lost their searchable fact, with receipts and an undo. It
previews by default; `--apply` and `--undo` insist on `--root` naming the data
folder you mean, and refuse any other. A repair never brings back anything that
was erased, deleted or withheld. `slm ops` lists failed, stuck or degraded
operations and, for an owner or admin, resolves them. The MCP counterpart for
Living Brain totals is `get_brain_evidence_status(profile_id="")`; it only
observes and does not change recall, ranking or review.

### Saves that failed for good: `slm ops repair-saves`

A save that failed ten times is parked, and `slm ops list` shows it as failed.
`slm ops repair-saves` re-files those saves from the save itself. A plain run
is a dry run: it lists what it would do and writes nothing. `--apply` copies
`memory.db` first (the path is printed) and then repairs. Run it with `--json`
for structured output. It cannot be undone, so the copy is the way back.

Since 4.2.14 it also handles saves from 4.1.1x builds whose text was blanked
with `[REDACTED:...]` markers (#308). Those show as `restore_then_add_fact` or
`restore_then_repoint`: it puts the original text back, with the same guarded
writes the one-time store repair used, and files the fact again. An original
that looks like it holds a secret, or that the ingest gate rejects, is never
restored automatically; it is listed under "left for review" with the reason.
When the stored text differs from the save by more than redaction markers, the
review line says so. If that save is not wanted, close it with
`slm ops resolve <operation_id> --action cancel`. Never restore or cancel on
the user's behalf without showing the dry run first.

### Too many model calls in Mode B or C

In Mode B and C, each stored fact costs one model call for its context sentence
and up to five for contradiction checks. Two settings trade that for speed. Both
default to the earlier behaviour (5 and true), so nothing changes until the user
chooses, and each mode keeps its own value:

```bash
slm config set temporal_validator.max_llm_checks 0     # 0 = embedding check only; default 5
slm config set encoding.context_generation_llm false   # context sentences by rule; default true
slm restart                                            # needed to apply
```

Contradiction checks only file correction proposals for a person to review;
they never change recall. If a pile of unreviewed proposals has built up, use
`slm corrections dismiss` (see `slm-remember`). `slm config set` now stops and
changes nothing when `config.json` cannot be read, rather than starting from an
empty file. A setting changed in the dashboard no longer undoes one made with
`slm config set`.

## Memory health: SLM looks after older memories

While the computer is idle, Memory health describes older memories (event dates,
names, projects, tags and kinds) by adding only — it never changes or deletes what
was written. Repairs that move or delete anything wait for the owner's **Fix**, and
every run that changed something can be undone.

```bash
slm enrich status              # progress, repairs waiting, the id of every run that changed something
slm enrich pause <run>         # recall and remember keep working
slm enrich resume <run>
slm enrich fix entity-links    # apply a suggested repair (asks first)
slm enrich undo                # undo the latest run that changed something (asks first)
```

`repair-saves` keeps no undo record; it says so and takes a safety copy before it
runs. An undone repair is not suggested again until you ask with
`--include-undone`. The dashboard shows the same state on the **Memory health**
card at the top of the Health pane. Guide: `docs/memory-health.md`.

## Pictures and documents: status and repair

Pictures and documents are on by default (automatic on 16 GB or more, one click below). Check them with
the CLI (the MCP `get_status` does not report them):

```bash
slm media status [--json]     # off | on, setting up (state, percent) | on | failed, with the step
slm features [--json]         # what is on and what can be turned on: pictures and documents, folder sources, bot messages
slm media repair --dry-run    # count pictures that cannot be found by what they show yet (owner or admin)
slm media repair              # re-embed them; up to four minutes per run, run again if it says some are left
slm media gc                  # report picture records without a memory, files without a record (removes nothing)
slm embedder status           # progress of an embedding-model switch or the memory-engine upgrade
```

If `slm media status` says "set-up failed", run `slm doctor` for the step that
failed. The five picture tools (`remember_media`, `get_media`,
`remember_document`, `media_status`, `media_upload_link`) are listed only while
the feature is on and only in the `full`, `power` and `whole` tool sets; a
restart (`slm restart`) starts the picture worker. A document's own progress
is the `media_status` MCP tool, not this command. Details, limits and the
the 16 GB rule (automatic above, one click below): `slm-media`. `slm status` also prints an "Embedding switch" line while a model switch or
the memory-engine upgrade is pending.

## Truth checks, judges and audit questions (added in 4.2.30)

```bash
slm doctor --truth [--json] [--db <path>]   # read-only; exits 1 when an invariant is violated
slm audit --count [--json]                  # how many blind audit questions wait for a label
slm signing status [--all] [--json]         # signed memories and the keyed audit chain (trust.signing)
slm proposals list [--json]                 # changes SLM proposes to memory (any caller may apply or reject)
slm wiki status [--json]                    # compiled pages, the Markdown folder, edits waiting
```

`slm doctor --truth` reports five things, counts and ids only:

- **Invariants** (I-01 ... I-16): whether the current values and their history
  are consistent. `FAIL` lines name the invariant and a few ids. I-10 (D-49) says no
  memory changed without a certificate id, a rule-certificate id or a recorded decision
  (who and channel): it checks judge receipts that acted, settled disputes, applied or
  rolled-back proposal sets, promotions and approvals; an id is prefixed with its kind
  (`dispute:`, `proposal_set:`, `promotion:`, `approval:`, `receipt:`).
- **Judge certificates**: which judges may act on their own, from how many
  audited answers, and which were stopped (`labels`: new answers no longer
  support it; `drift`: its recent scores changed). None is normal on a new store.
- **Second opinions**: how often a stronger judge, asked in the background
  about 1 in 20 confident answers, was sure of something else, with 95% bounds.
- **Signatures** (`signatures` in `--json`; `trust.signing`): how many memories
  are signed, how many not yet, and a random sample of 500 verified (`--signatures-all`
  checks every one): `modified` (by id), `signed_under_another_key` (a lost key; cannot
  be checked, nothing deleted) and `moved_by_a_fold`. The key is in the user's OS
  keychain; with none, signing is "unavailable" and counts are all that is reported.
  It detects a change made by something that has the files but not the user's login
  session; it does not stop a process running as the user.
- **Audit chain** (`audit_chain`): the audit trail's links verified, including HMAC
  links keyed with the same key; `plain_after_keyed` counts plain links after keying
  began (a command run while the daemon was down, or a downgrade). Exit 1 on a modified
  memory or a broken link. `slm signing resign --yes` (any caller, recorded as a decision, daemon stopped)
  signs again what a lost key signed: look at the listed memories with `slm why` first.

```bash
slm health --memory [--json] [--db <path>]   # read-only Memory Health report
```

(MCP: `health(memory=true)`.) It reports what was delivered (and whether
withholding is on), how often each delivered memory is used, with a 95% range,
and whether memory measurably helps. Read it honestly: a use rate is how often
the agent acted on a memory, not proof it helped; "helps"/"hurts" appears only
when the anytime-valid range excludes 0, and each line says how many
deliveries it has against how many its plan needs (months of use on one store;
a single retrieval channel's value is usually out of reach). Never report "no
clear effect yet" as "memory does not help". Its learning section lists each
learner as learning, stalled, off, waiting or not wired: report a stalled learner
as stalled (with its reason), never as healthy.
  A disagreement is a warning, not a proven error.

The dashboard shows the same on the Health page's **Truth checks** card.

**Audit labels come from a different model (D-49).** A blind audit question is two
memories and the question a judge was asked, never the judge's answer. Any caller may
label. `slm audit --count --json` says how many wait (`data.count`); `slm audit --list
--model <your model id> --json` (or `list_disputes(audit=true, model=...)`) lists them,
leaving out pairs only your own model judged; `slm audit --answer ITEM LABEL --model <id>`
(or `settle_dispute(dispute_id="audit:<item>", keep=LABEL, model=<id>)`) gives one label,
and `--retract ITEM` takes it back. A label is recorded with who gave it and counts toward
a judge's certificate only when your model differs from the judged tier's model (a person
on the dashboard always counts); give your real model id, never a guess, and `cannot_tell`
when you cannot tell. A person at a terminal can still run `slm audit` for one question at
a time.

**Proposed changes: any caller decides, every decision is recorded.** SLM can propose a unit of changes to
memory (retire an older memory, merge copies, rewrite a derived summary). Each
set is applied all-or-nothing, undone exactly with `rollback`, and every
derived summary, persona, entity profile or core block keeps its previous
version before it is rewritten. `slm proposals list --json` shows what waits
(`data.count`, and per set its plain-words proposals); `slm proposals show
SET_ID --json` shows one. `apply`, `reject` and `rollback` work from any caller (no
terminal, no confirmation); each is recorded with who, the channel and the time, and
the dashboard's **Proposed changes** card has **Apply**, **Reject** and **Undo** while
**Recent decisions by your agents** lists every decision with an Undo. Read what a set
changes before you apply it, and report anything that looks wrong.

**The living wiki** (added in 4.2.30) is pages compiled from memory: current state per
subject, the story of a family of decisions, a topic. Every sentence is a
word-for-word piece of a memory it cites or a fixed template over slot values,
checked again whenever the page is read; no model writes any of it, and each page
says how many sentences it withheld. A page is never a memory. `slm wiki status
--json` shows the pages and the Markdown folder; `slm wiki diff --json` shows what
a build would change and which files hold the user's edits; `slm wiki build` writes
the approved pages (`compiler.wiki` is on by default; `shadow` only reports).
`slm wiki rebuild` (compile and apply the pages, recorded and undone with `slm proposals
rollback`) and `slm wiki sync` (read the user's edits to the Markdown back as proposals)
work from any caller, and an edit is only ever a proposal, decided like any set with
`slm proposals`.
A deleted line in a file forgets nothing, and erasing a memory removes it from the
pages and the files. `compiler.dream` (off) lets the background cycle queue pages
and exact-duplicate merges for review; it applies nothing. The dashboard's **Living
wiki** card lists the pages, shows each sentence with the memory it comes from, and
has **Rebuild pages** and **Read my edits**.

`slm consent [--json]` lists what may leave this computer. The background
judges may ask the online Jev judge about pairs of memories only with the
user's consent (`judge_ladder_hosted`), given in the dashboard or with
`slm consent give` at a terminal. Never give it for the user; withdrawing
(`slm consent withdraw judge_ladder_hosted`) is fine when they ask.

`slm measurement status [--json]` shows what SLM measures about its own
deliveries and each measure's state. By default it writes a receipt for every
delivery (ids, ranks, scores, the logged chance of each choice; this computer
only), in about 1 in 10 session starts leaves one eligible memory out or
switches off one way of searching (by a logged random draw, never pinned rules,
standing rules, core blocks, explicit recall, cues or lease warnings), and in
about 1 in 20 of an agent's recalls swaps two neighbouring results. Usage
detection is on too: when a reply ends, SLM compares it with the memories the
agent was given, to see which were used, and keeps ids and scores only (on this
computer). That is how `slm health --memory` can say whether memory helps. The JSON carries
`measuring`, `kill_switch`, each flag's `state`, `pending` (the user has not yet
seen the notice) and the `notice` text; reading it never marks the notice as
seen. Turning it off is the user's choice: `slm measurement off` (withholding,
swapping and usage detection; add `--receipts` to stop the receipts too) or the dashboard's
Settings card "Help SLM measure what works". There is no plugin setting for it:
this is an instruction to you, and the dashboard card is the user's own switch.
Never switch it off on your own
initiative, even to make a test or a measurement tidier. Run it only when the
user asks, and when `pending` is true, pass the notice on to the user.

**The kill switch (added in 4.2.30).** `get_status` and `health` carry `kill_switch`
(`on`, `source`: file or environment, and a sentence in words). `slm flags`
shows it too. When it is on, everything added in 4.2.30 is off and recall reads the
way it did before those features: no dispute, source, standing or origin notes, no `why`
block, no background pass writing anything. Say so when results look plainer than
usual. Turning it on or off is the user's choice (`slm flags kill on|off`, the
kill-switch card in the dashboard settings, or `POST /api/flags/kill`);
never flip it on your own. It does not undo retirements: when the user wants
back the memories retired by the features added in 4.2.30, run `slm flags rollback-retirements --dry-run`,
tell them the count, and run it without `--dry-run` only on their yes. It is
recorded as a decision and **Recent decisions** can undo it.

## Trust: the instruction firewall and the secret screen (added in 4.2.30)

```bash
slm trust status [--json]     # is the firewall acting; how many memories wait; how many unclassified
slm trust pending [--json]    # the memories held back until promoted, with where each came from
slm health --memory --json    # `trust`: origins, waiting_items, credentials held as placeholders (counts by type)
```

SLM never hides a memory from you: recall and the session start show every memory. A memory is data
until SLM puts it where an agent obeys it: a pin, a standing rule or decision, core memory. With
`trust.taint` on, a memory whose taint is below `agent_inferred` (a web page, a tool's output, a
peer's message, an import, a model's text, or one whose source was not recorded) is held back from
those roles until it is promoted on purpose. Your own saved rules (`agent_inferred`) and memories
saved before trust classes were added in 4.2.30 (`legacy`) are trusted as they are. See what is held and why (`slm trust pending --json` →
`data.items[].origin`, `wanted_words`). One explicit call promotes a memory: `slm trust promote
FACT_ID`, `core_memory(action="promote")`, or the dashboard's **Promote** button. Any caller may
promote; the promotion is recorded (who, channel, time), listed under **Recent decisions by your
agents**, and `slm trust withdraw FACT_ID` takes it back. Promote only what you have checked the
user wants as an instruction. `data.unclassified` is the number of memories not yet given an
origin class; they cannot be an instruction until the background batch runs. Memories saved before trust classes were
added in 4.2.30 are `legacy` and keep working. Saving or confirming your own memory as a rule or decision
needs no promotion, while the same act on text from a web page, a tool, a peer or an import only
puts it on the waiting list, and the background classifier never promotes.
`trust.secret_screen` is separate: it stores a detected credential as `[REDACTED:TYPE:LAST4]`,
and `data.trust.secrets_held_back.by_type` counts them (never a value).

## Recommended Health Workflow

1. Run `slm doctor --json` at session start to confirm all subsystems are up.
2. Call `slm_optimize_stats()` after a batch of work to check token savings.
3. Run `slm status --json` when you need DB size or memory counts.
4. If `ok: false` on any MCP tool — check `note` field, then run `slm doctor` to isolate the failure.
5. If recall seems to miss memories that were saved, run `slm db integrity` before concluding anything.
6. If a picture will not come back for a question about what it shows, run `slm media status`, then `slm media repair --dry-run`.
7. If remembered values look wrong or contradictory, run `slm doctor --truth`, then `slm disputes list`.

## Fail-Open

`slm_optimize_stats()` never raises. On internal error it returns `ok: false` with all counters at 0. Continue the session — stats unavailability does not affect compression or caching operations.

---

## Profile-aware status

`slm status --json` reports the active profile in `data.profile`. Use it to confirm which workspace is active before starting work on a multi-profile setup. `get_status(profile_id="<name>")` counts another profile without moving the active one. To change the active profile, see `slm-profile`.

---

## Related skills

- `slm-session` — session lifecycle (session_init/close_session)
- `slm-profile` — workspace isolation and profile switching
- `slm-cache` — KV cache performance metrics
- `slm-compress` — reversible context compression
- `slm-media` — pictures, PDFs and folders
- `slm-web-access` — the Connected apps page and web-app connection states

---

SuperLocalMemory v4.2.31 · Qualixar · AGPL-3.0-or-later
