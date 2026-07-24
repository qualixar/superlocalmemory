---
name: slm-session
description: Manage SuperLocalMemory session lifecycle — call session_init once at the start of every fresh session to load relevant project context and get a session_id; call close_session when work is meaningfully complete to commit temporal summaries. Correct lifecycle hygiene is what makes SLM's learning loop work.
when_to_use: |
  - At the start of every session (auto-trigger on first user message in a project context)
  - When the user says "start a new session" or "initialize memory"
  - When meaningful work completes and context should be committed
  - When the user says "close session" or "end session"
allowed-tools: session_init, close_session, Bash
---

# slm-session — Session Lifecycle Hygiene

Session lifecycle is the mechanism that makes SuperLocalMemory's learning loop
work. Without it, recall signals are not attributed and temporal summaries are
not written. This is not optional housekeeping — it is load-bearing.

---

## The lifecycle in one diagram

```
Session starts
     |
     v
session_init(project_path, query)
     |--- returns session_id, context, memories
     |
     v
Use session_id in every recall() and remember() call
     |
     v
Work completes
     |
     v
close_session(session_id)
     |--- writes temporal summaries to DB
```

---

## session_init — call once per fresh session

### When to call

Call `session_init` exactly once at the start of every fresh session, before
any `recall` or `remember`. Never call it twice in a session — the second call
would generate a new `session_id` and break signal attribution for any prior
recalls or remembers in that session.

### Signature

```
session_init(
  project_path: str = "",  # working directory path, e.g. "/Users/me/projects/foo"
  query: str = "",         # topic override; if omitted, derived from project_path
  max_results: int = 10,   # max memories to return (default: 10)
  max_age_days: int = 30,  # suppress memories older than N days unless score >= 0.7
                           # set to 0 to disable the age gate entirely
)
```

### What it does

1. Derives a search query from `project_path` (or uses your explicit `query`).
2. Runs a 2-tier recall: full daemon retrieval (primary) or FTS5 BM25
   (emergency fallback if daemon is unreachable).
3. Merges any pinned "core memory" facts with the recall results.
4. Applies an age gate — memories older than `max_age_days` are suppressed
   unless their relevance score is 0.70 or above (architectural decisions that
   remain permanently relevant still surface).
5. Returns a pre-formatted `context` block and a structured `memories` array
   for your session.
6. Generates a stable `session_id` (`slm-YYYYMMDD-<8hex>`) and returns it.

### Real response shape

```json
{
  "success": true,
  "session_id": "slm-20260616-a3f8c1d2",
  "context": "# Relevant Memory Context\n\n- JWT tokens use 1h expiry ...",
  "memories": [
    {
      "fact_id": "f8a2bc91",
      "content": "JWT tokens use 1h expiry for API auth (2026-06-10)",
      "score": 0.87,
      "is_core": false
    }
  ],
  "memory_count": 3,
  "core_memory": [],
  "degraded_mode": false,
  "retrieval_mode": "full_6_channel",
  "learning": {
    "feedback_signals": 37,
    "phase": 1,
    "status": "collecting"
  }
}
```

**Check `degraded_mode`.** When `true`, the daemon was unreachable and only
FTS5 BM25 was used — semantic, graph, temporal, and structural channels were
unavailable. The context is still usable; note the degradation if relevant.

**Check `learning.phase`:**
- Phase 1 (< 50 signals): collecting baseline feedback
- Phase 2 (50–199 signals): active learning
- Phase 3 (≥ 200 signals): full ML-driven ranking

### How to use the returned session_id

Store it and thread it into every `recall` and `remember` call in this session:

```
session_id = "<value from session_init>"

recall(query="auth strategy", session_id=session_id, limit=10)
remember(content="...", session_id=session_id, tags="auth,decision", project="myapp")
```

This attribution is what allows the ranker to learn which recalls led to useful
outcomes for this project.

---

## close_session — call when work is meaningfully complete

### When to call

Call `close_session` when a meaningful unit of work is done — end of a coding
session, after shipping a feature, after a design review. You do not need to
call it after every small interaction. The signal is "this session's work is
committed and should be summarised."

Do not call it at the start of a new session as a cleanup step — `session_init`
is the correct opener and it does not require a prior close.

### Signature

```
close_session(
  session_id: str = "",  # the session_id from session_init; if omitted,
                         # the system queries the DB for the most recent session
)
```

### What it does

Aggregates facts written during the session into per-entity temporal summary
events. These summaries enable future queries like "what happened during session
X?" and contribute to the temporal channel in retrieval.

### Real response shape

```json
{
  "success": true,
  "session_id": "slm-20260616-a3f8c1d2",
  "summary_events_created": 4
}
```

`summary_events_created: 0` is normal for short sessions where no new facts
were written. It is not an error.

---

## Why this matters

Every `recall` call with a `session_id` enqueues engagement signals — which
results were shown, which were acted on. The learning ranker processes these
signals to gradually up-weight channels and facts that prove useful for your
project. Without `session_id`, signals land on a fallback identifier and are
never attributed to a project or agent. Over many sessions this compounds:
projects where lifecycle is respected have measurably better retrieval quality
than projects where session_init is skipped.

---

## CLI fallback (when MCP is unavailable)

There are no direct `session_init` or `close_session` CLI subcommands.
When MCP is unavailable, use `slm status` to check system health and
`slm recall` / `slm remember` directly. Session attribution will not be
available in degraded CLI-only mode.

```bash
slm status [--json]    # check mode, profile, DB size, fact count
slm doctor [--json]    # preflight check including daemon and embedding worker
```

---

## Common mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Calling `session_init` twice in one session | Two session IDs; signals split across them | Call once; store the returned ID |
| Omitting `session_id` from `recall` / `remember` | No learning attribution | Always pass the stored `session_id` |
| Never calling `close_session` | Temporal summaries not written | Call at end of each meaningful work unit |
| Calling `close_session` without a `session_id` when no prior writes exist | Returns error "No session_id found" | Pass the explicit `session_id` from `session_init` |

---

## Profile context (v3.8.0+)

`session_init` operates on the currently active profile. If you need to start
a session on a different workspace, call `switch_profile` (requires `code`,
`full`, or `power` MCP profile) before `session_init`, then call `session_init`
for that workspace. See `slm-profile` for workspace switching.

The returned `memories` array reflects facts stored in the active profile only.
To also surface shared or global facts in the initial context, pass the query
explicitly and call `recall` with `include_global`/`include_shared` after
`session_init`. See `slm-scope` for the sharing model.

---

## Related skills

- `slm-recall` — multi-channel retrieval during the session
- `slm-remember` — store durable facts during the session
- `slm-profile` — workspace isolation and profile switching
- `slm-scope` — multi-scope sharing model

---

*SuperLocalMemory v3.8.3 · Qualixar · AGPL-3.0-or-later*
