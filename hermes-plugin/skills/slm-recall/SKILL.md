---
name: slm-recall
description: Search and retrieve facts, decisions, and past context from SuperLocalMemory. Use when the user asks to recall, find, search, or "what did we decide/say about X". Triggers multi-channel semantic retrieval with reranking; always call before storing anything new.
when_to_use: |
  - "What did we decide about X?"
  - "Recall anything about Y"
  - "Do we have context on the Z feature?"
  - "Find stored information about authentication / the database / error handling"
  - "Search for what I said about Y"
  - Automatically before any non-trivial task, to surface prior context
allowed-tools: recall, search, fetch, list_recent, get_memory_summary, run_view, report_outcome, report_feedback, Bash
---

# slm-recall — Search & Retrieve Memory

Retrieve stored facts, decisions, and past context from SuperLocalMemory using
multi-channel retrieval. The golden rule: **recall before you remember**.

---

## When to use recall vs search vs fetch vs list_recent

| Situation | Tool |
|-----------|------|
| Conceptual or paraphrase query ("what did we agree on for auth?") | `recall` — full multi-channel retrieval + rerank |
| Exact keyword match needed ("find facts containing BM25") | `search` — FTS5 BM25 only, lower latency |
| You have a specific `fact_id` from a prior result | `fetch` — exact lookup, full detail |
| Browse newest entries without a query | `list_recent` |
| A day, a project or a session in readable form | `get_memory_summary` |
| A question the user saved under a name | `run_view` |

Use `recall` as the default. `search` is a fallback for zero-result recall on a
known exact term. `fetch` is for when you already know the ID.

---

## Recall-before-remember discipline

Before storing anything new, always call `recall` first. If a near-duplicate
fact already exists, call `update_memory(fact_id, content)` to refine it
rather than creating a duplicate. Duplicates degrade retrieval quality for
every future session.

---

## MCP-first workflow

### 1. Standard recall

```
recall(
  query="authentication strategy decision",
  limit=20,            # default 20; reduce to 5 for quick pre-task checks
  session_id="<sid>",  # pass the session_id returned by session_init
  fast=None,           # leave unset; see "Fast mode" below for what it controls
)
```

Real response shape (trimmed to the fields that matter; `slm recall --json`
wraps the same data in a `data` envelope):
```json
{
  "success": true,
  "results": [
    {
      "fact_id": "f8a2bc91",
      "memory_id": "m7e19ca0",
      "content": "Decided to use JWT with 1h expiry for API auth (2026-06-10)",
      "score": 0.87,
      "confidence": 0.91,
      "trust_score": 0.84,
      "fact_type": "semantic",
      "memory_kind": "decision",
      "memory_kind_state": "confirmed",
      "age_label": "4 months ago",
      "channel_scores": {
        "semantic": 0.88,
        "bm25": 0.61,
        "temporal": 0.72,
        "hopfield": 0.55
      }
    }
  ],
  "count": 1,
  "query_type": "semantic",
  "channel_weights": {
    "semantic": 0.4,
    "bm25": 0.2,
    "temporal": 0.2,
    "hopfield": 0.2
  },
  "channel_status": {
    "semantic": "ok",
    "bm25": "ok",
    "temporal": "empty",
    "hopfield": "ok",
    "spreading_activation": "no_candidates",
    "entity_graph": "no_embedding",
    "profile": "disabled"
  },
  "incomplete_channels": [],
  "retrieval_time_ms": 134,
  "query_id": "q-3f9a",
  "no_confident_match": false,
  "calibration_status": "uncalibrated",
  "calibration_id": null,
  "answer_confidence": null,
  "abstained": false,
  "abstention_reason": null,
  "answer_check_status": "off",
  "answerability": "unjudged",
  "answerability_reason": "disabled"
}
```

Other fields that can appear: `profile` (the namespace that answered),
`project_scope` (what a `project` filter did), `tag_scope` (what a `tags`
filter did), `temporal_frame`, `reranker_status`, and `thematic_context`.
Long contents are clamped and later results can come back as short stubs
(`truncated` / `stub` on the result); call `fetch` for the full text.

**Read `channel_status` before concluding that nothing is stored.** It reports
what each retrieval channel did on this query. `channel_weights` says how much
each channel counts; `channel_status` says whether it ran at all.

| status | meaning |
|---|---|
| `ok` | the channel ran and contributed candidates |
| `empty` | it ran and there was genuinely nothing to return |
| `no_candidates` | it ran but nothing survived fusion |
| `error` | it raised — **its results are missing from this answer** |
| `timeout` | it exceeded its guard — **results missing** |
| `no_embedding` | the query could not be embedded, so it could not run |
| `warming` | the embedding model, or the entity graph, was still loading right after a start — **results missing**, and the same question a little later usually works |
| `disabled` | switched off by configuration |
| `not_configured` | the backing service is not set up |

`semantic`, `bm25`, `temporal`, `hopfield` and `spreading_activation` each
search and return their own candidates. `profile` is a shortcut that runs before
them and can answer directly. `entity_graph` produces nothing of its own — it
re-scores what the others found, by how well each result connects to the
entities in your question, which is why it reports `no_candidates` when the
rest come back empty.

`empty`, `no_candidates`, `disabled` and `not_configured` are normal. `error`,
`timeout`, `no_embedding` and `warming` mean the answer is **incomplete, not
negative** — say so to the user rather than reporting "no memories found".
`incomplete_channels` lists the channels abandoned at the time limit, so two
runs of the same question that differ only there are an incomplete answer, not
a changed memory.

**Refine on low confidence.** `recall` returns confidence signals with every result. If `no_confident_match` is `true` (or `answer_confidence` is low / `abstained` is `true`), do NOT invent a memory — rewrite the query into 1–3 more specific sub-queries (split multi-hop questions; try entity names, synonyms, or broader phrasing) and call `recall` again before concluding nothing was found. A confident match → use it directly. SLM answers from this machine, typically in a second or two, with no server-side LLM round — unless the user turned on the online answer check, which adds one request to that service per recall — and lets you, the calling model, drive this refinement.

**`abstained` means the results shown do not answer the question.** If `abstained` is `true`, say you don't have it, or ask — never present the returned memories as the answer anyway. `abstention_reason` distinguishes why: `"judged_insufficient"` means candidates were found and scored, but none of them actually answers this question; `"evidence_floor"` / `"no_candidates"` means nothing was found at all. `answer_confidence` is a measurement, not a guarantee — treat a low number the same way you'd treat `abstained: true`. `calibration_status: "uncalibrated"` means no judge is configured for this recall; in that case `abstained` only ever reflects the older "nothing found" signal.

**`abstained: false` does not mean the answer was checked.** `answerability` says
whether it was: `supported` (the answer check ran and judged the shown memories
sufficient), `unsupported` (it ran and judged them insufficient) or `unjudged`
(it produced no verdict). `answerability_reason` gives the cause: `judged_fresh`
or `judged_from_memo` for a verdict, and for `unjudged` one of `disabled`,
`warming`, `unavailable`, `busy`, `budget_exhausted`, `no_results`,
`not_a_question` or `other_profile`. An empty result set is `unjudged` /
`no_results`, which is not evidence that no answer exists (check
`channel_status`). `answer_check_note` carries a sentence you can show a person.

### 2. Passing session_id

Pass the `session_id` returned by `session_init`, on **every** recall in that
session. It does two things.

1. **It carries the conversation forward.** Each recall offers its five
   best-ranked results to a small per-session working set of seven slots. A
   memory that keeps coming back is reinforced rather than duplicated, and the
   least-activated slot is the one evicted, so something referenced across
   several turns is hard to lose. Later recalls in the same session rank the
   held memories higher, and turn three is not as cold as turn one. The bias is
   deliberately small — it nudges the order, it never overrides an exact match.
2. **It attributes engagement to the session**, so a later `report_outcome`
   can close the loop on the right recall.

If you omit it, SLM looks for `SLM_SESSION_ID`, `CLAUDE_SESSION_ID` or
`CLAUDE_CODE_SESSION_ID` in the environment, then for the host's registered
session, and only then falls back to a per-agent label that is never credited
to a conversation. Recall still returns correct results either way, but a
fallback label means turns start cold and feedback cannot be attributed.

**Use the real id, not a made-up one.** An id beginning `http:`, `mcp:`, `cli:`,
`probe:`, `engine:`, `agent:`, `api:` or `view:` is treated as a synthetic
per-request label, not a conversation, and is excluded from the working set —
inventing one per call would otherwise fill the registry and evict genuine
conversations.

### 3. Fast mode

`fast` controls **one** thing: whether the server runs its own internal LLM
reformulation round. It does **not** disable any retrieval channel — every
channel and the reranker run either way. Four channels register always: meaning,
keyword, entity graph and time. Spreading activation and Hopfield register as a
fifth and sixth when their prerequisites are present, so a store sees up to six
(`profile` in `channel_status` is a shortcut ahead of them, not a search).

Leave it unset. Unset resolves to "skip the internal round", because you are the
reasoner: you refine the query yourself using the confidence signals above, and
you do it better than a local model would. Pass `fast=False` only when SLM is
deployed with no capable client in front of it.

```
recall(query="rate limiting approach", limit=5, session_id="<sid>")
```

### 4. Keyword fallback via search

When `recall` returns zero results on a specific term, try `search`:

```
search(query="BM25 indexing", limit=20)
```

Full-text FTS5 with BM25 ranking, no semantic channel. It takes the same
`kind`, `tags` / `tags_match` and `profile_id` arguments as `recall` (below).
The response has `success`, `results` (each with `fact_id`, `content`,
`fact_type`, `confidence`, `date` and the memory-kind fields) and `count`, but
no `channel_scores`, `query_type` or answer-check fields.

### 5. Pull full detail for a known fact

```
fetch(fact_ids="f8a2bc91,d4c1e203")
```

Returns the full record for each ID: `entities`, `lifecycle`, `access_count`,
`importance`, `observation_date`, `referenced_date`, `project`. `fact_ids` may be
a comma-separated string or a list. Ids it could not resolve come back in
`not_found`; if none resolve, `success` is `false`. Use this when the recall
summary (120-char truncation in `list_recent`) is not enough.

### 6. Browse recent memories

```
list_recent(limit=20, kind="", tags="", tags_match="all", profile_id="")
```

Returns facts newest-first. Content is truncated to 120 chars. Use `fetch`
once you have the `fact_id` for full content.

### 7. Narrow a recall

Every filter below is optional and composes with the others as AND.

| Argument | Effect |
|---|---|
| `project` | Only memories saved under that project (a name or a path; case ignored). If none of the memories found were saved under it, the unfiltered results come back and `project_scope.filter.applied` is `false` with a `note` — never a silent empty answer. |
| `project_strict=True` | With `project`: no fall-back, only that project's memories even if there are none. |
| `prefer_project` | Ranks memories saved under that project above others of similar relevance and removes nothing. Pass your working directory on every recall in a project. |
| `saved_by` | Only memories saved by that agent id (for example `claude-desktop`). |
| `about` | Only memories that mention that person, project or tool. |
| `kind` | Only memories of one kind: `semantic`, `episodic`, `status`, `opinion`, `rule`, `decision`, `procedure`, `prospective` or `correction`. An unknown value is refused with `INVALID_KIND`. |
| `tags`, `tags_match` | Only memories saved with these exact labels (a comma-separated string, or a list when a label contains a comma). `tags_match` is `"all"` (default) or `"any"`. Case and spacing do not matter. Unlike `project`, a tag filter never falls back; an empty answer says why in `tag_scope`. |
| `window` | Event-time range: `"24h"`, `"7d"`, `"30d"`, `"1y"` or `"2026-07-01..2026-07-31"`. An unreadable value is refused. |
| `as_of`, `known_as_of`, `valid_at` | Point-in-time recall (ISO 8601). `known_as_of` is what SLM knew by then; `valid_at` is what was true then. `include_unknown=True` also admits older memories with no recorded time provenance. |
| `profile_id` | Serve this one call from another existing profile without moving the active one. |

Questions phrased like "what did we decide", "how do I" or "what is the current
status of" are answered with decisions, how-tos and the newest current-state
memory first, without any filter.

```
recall(query="rollback procedure", kind="procedure", tags="ops", prefer_project="/work/api", session_id="<sid>")
```

### 8. Summaries and saved views

`get_memory_summary(kind="day", target="")` returns a readable summary of a
`day` (an ISO date, `today` or `yesterday`), a `project` (a directory path), a
`session` (a session id; leave `target` empty to get `recent_sessions` to pick
from) or a `community` (the `community_id` a recall gave in `thematic_context`).
It comes with `coverage` and `source_fact_ids`; session data is sparse, so do
not present a partial summary as a complete record.

`run_view(name="")` lists the user's saved views; with a name it runs that
saved question through ordinary recall, and the same `no_confident_match` rule
applies. Views are created from the dashboard, with `slm view create`, or with
`manage_view`.

---

### 9. Close the loop — say which memories helped

Whether a memory has been useful is evidence SLM records only if you supply it.
It is stored as learning signals (the count shows in `session_init`'s `learning`
block and on the dashboard). Signals reorder results only where adaptive ranking
is switched on by the operator (`SLM_RANKING`); it is off unless set, so do not
promise the user that a report changes the next answer.

```
report_outcome(
  memory_ids="f8a2bc91,c31d0f77",   # the ids you actually used
  outcome="success",                # "success" | "failure" | "partial"
  context="used the JWT expiry decision to write the refresh handler",
  recall_query_id="q-3f9a",         # the query_id of the recall; ties the report to that answer
)
```

`report_outcome` and `report_feedback` are not in the smallest (`core`) tool set.
If your host does not list them, skip this step; nothing else depends on it.

Call it when a recall visibly changed what you did: you applied the decision,
followed the convention, or avoided the gotcha. Report `failure` when a
confidently-returned memory turned out to be wrong or stale — a negative signal
is recorded the same as a positive one. To retire a stale memory outright, save
the new version with `replaces` (see `slm-remember`).

Report only ids you genuinely used. Reporting every returned id marks the
irrelevant ones useful and records noise as signal. Without `recall_query_id`
the report is matched to a recall by overlapping memory ids within a time window.

`report_feedback(fact_id, feedback, query)` is the finer-grained form for a
single fact and the query that surfaced it; `feedback` is `"relevant"` (the
default), `"irrelevant"` or `"partial"`. If the signal could not be written to
the learning store it answers `success: false` with `durable: false`: do not
tell the user it was recorded.

---

## How multi-channel retrieval works

`recall` runs multiple candidate producers in parallel — semantic vector similarity,
keyword matching, temporal recency weighting, and contextual graph channels — then
fuses and reranks the combined results, with an optional entity-graph score
enhancement. The `channel_weights` field in the response shows how each channel
contributed for that query. Adaptive re-weighting from engagement signals is an
operator opt-in (`SLM_RANKING`) and is off by default.

To inspect per-channel scores for a real query against your own data:

```bash
slm trace "<query>" [--limit N] [--json]
```

No benchmark numbers are cited here; performance is workload-dependent.

---

## CLI fallback (when MCP is unavailable)

```bash
# Multi-channel recall
slm recall "<query>" [--limit N] [--json]

# Narrow it (all optional, combined as AND)
slm recall "<query>" --project <name-or-path> [--project-strict] [--prefer-project <path>]
slm recall "<query>" --kind decision --tag auth --tag security [--tags-match any]
slm recall "<query>" --saved-by claude-desktop --about "Alice" --window 7d
slm recall "<query>" --as-of 2026-01-01T00:00:00+00:00 [--known-as-of ...] [--valid-at ...] [--include-unknown]

# Opt into shared/global facts for one query (off by default)
slm recall "<query>" --include-global --include-shared

# Per-channel score breakdown
slm trace "<query>" [--limit N] [--json]

# Browse recent memories; each line shows the memory's kind and its id
slm list [--limit N] [--kind KIND] [--tag LABEL ...] [--tags-match all|any] [--json]

# Summaries and saved views
slm summary day [today|yesterday|YYYY-MM-DD] | project <path> | session <id> | sessions  [--json]
slm view list | create <name> "<query>" [--kind K --window W --limit N] | run <name> | show <name> | rename <name> <new> | delete <name>
```

`slm search` is an alias of `slm recall`: it runs the same multi-channel
retrieval, not the FTS5-only keyword search the MCP `search` tool performs.
`--fast` forces the quick path; it is already the default for agents.

Flags that do **not** exist on `slm recall`: `--min-score`, `--format`,
`--tags` (the flag is the repeatable `--tag`).

Recall returns only this profile's own memories unless you pass
`--include-global` / `--include-shared` (or the MCP `include_global` /
`include_shared` arguments), or the user has set the defaults in their
configuration. See [docs/shared-memory.md](../../../docs/shared-memory.md).

---

## Never fabricate a memory

After re-querying with refined sub-queries (see **Refine on low confidence** above), if `no_confident_match` is still `true` or results are empty, report it plainly.
Never construct a response as if a memory was found when it was not. The user
trusts that what you surface came from the store.

---

## Shared and global memories (opt-in)

By default `recall` returns only memories in the active profile (personal scope).
To also surface memories shared from other profiles, pass the scope flags:

```
recall(
  query="...",
  include_global=True,   # include global-scope memories (visible to all profiles)
  include_shared=True,   # include shared-scope memories (shared with this profile)
  session_id="<sid>",
)
```

Scope flags are **off by default**. Only enable them when the user explicitly asks
to see shared or global facts. See `slm-scope` for the full sharing model.

---

## Profile-aware retrieval

`recall` queries the active profile. To read another existing profile for one
call, pass `profile_id="<name>"` (the active profile is not moved). The
`switch_profile` tool changes the active profile for every later call and for
other sessions on this machine, so reserve it for when the user asks to move.
See `slm-profile`.

---

## Related skills

- `slm-remember` — store the decisions and facts that recall surfaces later
- `slm-session` — session lifecycle (must call before first recall)
- `slm-scope` — multi-scope sharing model (personal / shared / global)
- `slm-profile` — workspace isolation and profile switching

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
