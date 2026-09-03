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
allowed-tools: recall, search, fetch, list_recent, Bash
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

Real response shape (`--json` equivalent):
```json
{
  "success": true,
  "results": [
    {
      "fact_id": "f8a2bc91",
      "content": "Decided to use JWT with 1h expiry for API auth (2026-06-10)",
      "score": 0.87,
      "confidence": 0.91,
      "trust_score": 0.84,
      "fact_type": "semantic",
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
  "no_confident_match": false
}
```

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
| `disabled` | switched off by configuration |
| `not_configured` | the backing service is not set up |

`semantic`, `bm25`, `temporal`, `hopfield` and `spreading_activation` each
search and return their own candidates. `profile` is a shortcut that runs before
them and can answer directly. `entity_graph` produces nothing of its own — it
re-scores what the others found, by how well each result connects to the
entities in your question, which is why it reports `no_candidates` when the
rest come back empty.

`empty`, `no_candidates`, `disabled` and `not_configured` are normal. `error`,
`timeout` and `no_embedding` mean the answer is **incomplete, not negative** —
say so to the user rather than reporting "no memories found". `incomplete_channels`
carries the same warning as a plain list.

**Refine on low confidence.** `recall` returns confidence signals with every result. If `no_confident_match` is `true` (or `answer_confidence` is low / `abstained` is `true`), do NOT invent a memory — rewrite the query into 1–3 more specific sub-queries (split multi-hop questions; try entity names, synonyms, or broader phrasing) and call `recall` again before concluding nothing was found. A confident match → use it directly. SLM returns fast local results (~1–2s, no server-side LLM round on the hot path) and lets you, the calling model, drive this refinement.

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

Omitting it costs both: recall still returns correct results, but every turn
starts cold and no feedback is attributable.

**Use the real id, not a made-up one.** An id beginning `http:`, `mcp:`, `cli:`
or `probe:` is treated as a synthetic per-request label, not a conversation, and
is excluded from the working set — inventing one per call would otherwise fill
the registry and evict genuine conversations.

### 3. Fast mode

`fast` controls **one** thing: whether the server runs its own internal LLM
reformulation round. It does **not** disable any retrieval channel — every
channel and the reranker run either way. There are four always: meaning,
keyword, entity graph and time. Spreading activation and Hopfield register as a
fifth and sixth when their prerequisites are present, so a store sees up to six.

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
search(query="BM25 indexing", limit=10, profile_id="")
```

`profile_id=""` uses the active profile. Response has `success`, `results`,
and `count` but no `channel_scores` or `query_type`.

### 5. Pull full detail for a known fact

```
fetch(fact_ids="f8a2bc91,d4c1e203")
```

Returns the full record for each ID: `entities`, `lifecycle`, `access_count`,
`importance`, `observation_date`, `referenced_date`. Use this when the recall
summary (120-char truncation in `list_recent`) is not enough.

### 6. Browse recent memories

```
list_recent(limit=20, profile_id="")
```

Returns facts newest-first. Content is truncated to 120 chars. Use `fetch`
once you have the `fact_id` for full content.

---

### 7. Close the loop — say which memories helped

Retrieval ranks a memory partly on whether it has actually been useful before.
That evidence only exists if you supply it.

```
report_outcome(
  memory_ids="f8a2bc91,c31d0f77",   # the ids you actually used
  outcome="success",                # "success" | "failure" | "partial"
  context="used the JWT expiry decision to write the refresh handler",
)
```

Call it when a recall visibly changed what you did: you applied the decision,
followed the convention, or avoided the gotcha. Report `failure` when a
confidently-returned memory turned out to be wrong or stale — a negative signal
is worth as much as a positive one, and it is the only way a stale memory stops
being promoted.

Report only ids you genuinely used. Reporting every returned id marks the
irrelevant ones useful and trains the ranker toward noise.

`report_feedback(fact_id, feedback, query)` is the finer-grained form for a
single fact and the query that surfaced it.

---

## How multi-channel retrieval works

`recall` runs multiple candidate producers in parallel — semantic vector similarity,
keyword matching, temporal recency weighting, and contextual graph channels — then
fuses and reranks the combined results, with an optional entity-graph score
enhancement. The `channel_weights` field in the response shows how each channel
contributed for that query. Weights adapt over time based on engagement signals
attributed via `session_id`.

To inspect per-channel scores for a real query against your own data:

```bash
slm trace "<query>" [--limit N] [--json]
```

No benchmark numbers are cited here; performance is workload-dependent.

---

## CLI fallback (when MCP is unavailable)

```bash
# Multi-channel semantic recall
slm recall "<query>" [--limit N] [--fast] [--json]

# Opt into shared/global facts for one query (v3.6.15 — off by default)
slm recall "<query>" --include-global --include-shared

# Keyword/FTS5 search (alias: slm search)
slm search "<query>" [--limit N] [--json]

# Per-channel score breakdown
slm trace "<query>" [--limit N] [--json]

# Browse recent memories
slm list [--limit N] [--json]
```

Flags verified in source (main.py):
- `slm recall`: `--limit`, `--fast`, `--json`, `--include-global` / `--no-global`, `--include-shared` / `--no-shared`
- `slm search`: `--limit`, `--json`
- `slm trace`: `--limit`, `--json`
- `slm list`: `--limit` / `-n`, `--json`

> **Multi-scope (v3.6.15, opt-in):** recall is shared-OFF by default — it returns only
> this profile's facts. Pass `--include-global` / `--include-shared` (or the MCP
> `include_global` / `include_shared` args) to opt in for a query, or set the defaults in
> your `mode_a/b/c.json` config. See [docs/shared-memory.md](../../../docs/shared-memory.md).

**Flags that do NOT exist** (fabricated in old skills — never write these):
`--min-score`, `--format`, `--project`, `--tags` on recall or search.

---

## Never fabricate a memory

After re-querying with refined sub-queries (see **Refine on low confidence** above), if `no_confident_match` is still `true` or results are empty, report it plainly.
Never construct a response as if a memory was found when it was not. The user
trusts that what you surface came from the store.

---

## Multi-scope retrieval (v3.6.15+, opt-in)

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

## Profile-aware retrieval (v3.8.0+)

`recall` always queries the currently active profile. To query a different
workspace, use `switch_profile` (requires `code`, `full`, or `power` MCP profile)
before recalling, then switch back. See `slm-profile` for workspace switching.

---

## Related skills

- `slm-remember` — store the decisions and facts that recall surfaces later
- `slm-session` — session lifecycle (must call before first recall)
- `slm-scope` — multi-scope sharing model (personal / shared / global)
- `slm-profile` — workspace isolation and profile switching

---

*SuperLocalMemory v4.1.12 · Qualixar · AGPL-3.0-or-later*
