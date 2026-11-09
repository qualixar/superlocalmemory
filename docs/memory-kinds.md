# Memory Kinds and Standing Rules

Every memory can say what sort of thing it is. Recall uses that to answer
"what did we decide" with decisions and "how do I" with how-tos, and confirmed
rules and decisions are handed to every new agent session.

## The nine kinds

| Kind | Use it for |
|---|---|
| `semantic` | A fact: "Staging DB is Postgres 16." |
| `episodic` | Something that happened: "The deploy failed on Tuesday." |
| `status` | The current state of something: "Release 2.3 is in QA." |
| `opinion` | A view someone holds: "Alice prefers Terraform." |
| `rule` | A standing rule: "Always run the migration dry-run before a release." |
| `decision` | A decision: "We deploy the API blue-green." |
| `procedure` | A how-to: "To rotate the key, run ..." |
| `prospective` | A plan or to-do: "Upgrade the runner next sprint." |
| `correction` | A fix to something remembered earlier |

## Saying the kind when you save

```bash
slm remember "Always run the migration dry-run before a release." --kind rule
slm remember "We deploy the API blue-green; rollback is a DNS flip." --kind decision
```

Over MCP, pass `kind` to `remember`. If you give no kind, SLM suggests one. A
suggestion changes nothing until someone confirms it, and background work
never overwrites a kind you confirmed.

## Replacing an old memory

```bash
slm remember "Staging DB moved to Postgres 17 on port 5434." --replaces <fact_id>
```

The old memory is retired, not deleted, and recall returns the new one. Undo it
with `review_correction` (action `rollback`); the old memory comes back exactly.

## Standing rules at session start

Confirmed rules (up to 10) and confirmed decisions (up to 5) are given to every
new agent session through MCP `session_init`. Proposed, rejected, archived,
quarantined and replaced items are never included. Turn it off with
`slm kinds settings --standing-rules off`.

Each saved rule counts once, however many sentences it has. When there are more
rules than slots, rules saved in the session's project come first, then the
ones the session's opening recall found relevant, then the newest. Rules and
decisions never take the places of the session's recall results.

## Asking for a kind

```bash
slm recall "what did we decide about rollback" --kind decision
```

Without `--kind`, questions shaped like "what did we decide", "how do I" or
"what is the current state" favour the matching kind. Questions that ask for no
particular kind are ranked exactly as before.

## When a suggested kind is shown as "not sure"

A kind a model suggested (as opposed to one you set with `--kind` or
confirmed) carries a confidence. Below **`display_min_confidence`**
(default **0.20**), the suggestion is never shown as the kind itself —
instead the memory is shown under its older type (`fact_type`, mapped to the
nearest kind) if it has one, or as untyped if it does not. A
suggestion that clears the threshold is still shown as a *suggestion*, not a
fact, until someone confirms it:

| State | Shown as | When |
|---|---|---|
| `confirmed` | the kind, plain | you set it, or confirmed a suggestion |
| `suggested` | the kind, marked "(suggested)" | a model's confidence ≥ `display_min_confidence` |
| `legacy` | the nearest kind from the old `fact_type`, plain | no kind, or confidence < `display_min_confidence` |
| `untyped` | nothing | no kind and no mappable `fact_type` |

`display_min_confidence` is one setting, read by every surface that shows a
kind (`slm list`, `slm recall`, `slm trace`, the MCP tools, the HTTP API and the
dashboard), so a memory shows the same state everywhere.

It is stored in `memory_kinds.json` in the data folder, beside `backend` and
`jev_consent`, rather than as a CLI flag. Change it with the daemon's settings
endpoint, `POST /api/memory-kinds/settings` with
`{"display_min_confidence": 0.35}` (0.0 to 1.0), which needs the same
credential as other settings changes.

## Managing kinds

| Command | What it does |
|---|---|
| `slm kinds status` | Counts per kind, the backend in use, any run in progress |
| `slm kinds settings` | Turn kinds on or off, choose the backend (`auto`, `rules`, `laya`, `jev`, `llm`, `off`), standing rules on or off |
| `slm kinds backfill start\|pause\|resume\|cancel\|revert\|status` | Classify existing memories in the background, with an exact undo |
| `slm kinds review` | Suggestions waiting for confirmation |
| `slm kinds confirm FACT_ID[=KIND] ...` | Confirm 1–200 kinds at once |
| `slm kinds set FACT_ID KIND` | Set and confirm one memory's kind |

MCP tools: `set_memory_kind`, `memory_kinds_status`, `review_memory_kinds`,
`confirm_memory_kinds` (in the `code`, `full` and `power` tool sets, in `whole`,
and in the default set; not in `core`).

Classifying with Jev sends memory text online, so it needs its own consent
(`--jev-consent yes`) on top of the answer-check consent. See
[Answer check](answer-check.md).
