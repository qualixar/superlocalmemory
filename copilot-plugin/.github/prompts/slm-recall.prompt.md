---
name: slm-recall
description: Search and retrieve facts, decisions, and past context from SuperLocalMemory. Use when the user asks to recall, find, search, or "what did we decide/say about X". Triggers multi-channel semantic retrieval with reranking; always call before storing anything new.
version: "3.8.2"
agent: agent
tools:
  - recall
  - search
  - fetch
  - list_recent
  - Bash
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
  fast=False,          # default False; True enables faster, reduced-channel retrieval
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
      "fact_type": "decision",
      "channel_scores": {
        "semantic": 0.88,
        "lexical": 0.61,
        "temporal": 0.72,
        "structural": 0.55
      }
    }
  ],
  "count": 1,
  "query_type": "semantic",
  "channel_weights": {
    "semantic": 0.4,
    "lexical": 0.2,
    "temporal": 0.2,
    "structural": 0.2
  },
  "retrieval_time_ms": 134,
  "no_confident_match": false
}
```

**Refine on low confidence.** `recall` returns confidence signals with every result. If `no_confident_match` is `true` (or `answer_confidence` is low / `abstained` is `true`), do NOT invent a memory — rewrite the query into 1–3 more specific sub-queries (split multi-hop questions; try entity names, synonyms, or broader phrasing) and call `recall` again before concluding nothing was found. A confident match → use it directly. SLM returns fast local results (~1–2s, no server-side LLM round on the hot path) and lets you, the calling model, drive this refinement.

### 2. Passing session_id

Pass the `session_id` returned by `session_init`. It threads engagement signals
through to the ranker so each recall contributes to improving retrieval for
your project over time. Omitting it degrades the learning loop — recall works
correctly, but feedback is not attributed to the session.

### 3. Fast mode

Use `fast=True` for pre-tool-call checks where sub-second response matters.
This enables a faster, reduced-channel mode. Core semantic and keyword channels
always run; additional graph and contextual channels are skipped.

```
recall(query="rate limiting approach", limit=5, session_id="<sid>", fast=True)
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

*SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later*
