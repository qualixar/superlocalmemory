# SuperLocalMemory — Agent Rules

> Drop into any IDE agent config or AGENTS.md to give agents disciplined SLM usage.
> SLM is local-first: MCP memory tools use the configured local runtime. Optional providers, connectors, backup, and downloads have separate network behavior.
> Prefer MCP tools when the server is running; use the CLI fallback table when not.

---

## Session start

Call `session_init(project_path, query, max_results, max_age_days)` **once** at the start of every new session before any `recall` or `remember` calls. Never call it twice in the same session. `session_init` loads recent decisions and relevant memories into context automatically.

---

## Remember discipline

- **Atomic durable facts only** — decisions, conventions, constraints, gotchas, stable preferences. One fact per `remember` call.
- **Recall before remember** — call `recall(query, 5)` first. If a near-identical memory exists, use `update_memory(fact_id, content)` to correct it; never duplicate.
- **Tags + project + importance** — always supply meaningful tags and the project name. Use importance 7–10 for blockers, security findings, and architecture decisions; 5 for general facts.
- **Never dump a whole file** — extract only the durable decision or constraint; store the path reference if needed.
- **Scope is opt-in (v3.6.15)** — memories are `personal` (private to this profile) by default. Only pass `scope="shared"/"global"` (or recall's `include_global`/`include_shared`) when the user EXPLICITLY asks to share across local profiles. Never opt in on your own; the default is identical to single-profile SLM.

---

## Recall discipline

- **recall for concepts, search for keywords** — use `recall(query, limit)` for conceptual / semantic queries; use `search(query, limit)` for exact keyword / FTS5 matches.
- **Concept phrasing** — phrase queries as concepts, not vague words ("database migration strategy" not "database").
- **Pass session_id** when available to narrow results to the current session.
- **Empty / low results** — broaden the query, try `search`, or fall back to `list_recent`. Never fabricate a memory.

---

## Optimize rules

Apply automatically when context pressure builds:

1. Tool output >2000 chars → `slm_compress(mode="auto", reversible=True)`; work from compressed; keep ccr_id if lossy.
2. Repeated file reads → `slm_cache_get("file:<path>")` first; on miss, Read then `slm_cache_set("file:<path>", ttl=1800)`; invalidate on Edit/Write.
3. Repeated bash/web → `slm_cache_set("bash:<cmd[:64]>", ttl=600)`; web ttl=3600.
4. **NEVER** compress/cache: code for Edit/Write; JSON to parse; secrets/keys/tokens; ccr_ids; content <500 chars.
5. **FAIL-OPEN** — if any optimize call returns ok:false, continue with the original content, no retry, don't surface the error.

---

## Session end

Call `close_session(session_id)` when the work in the session is meaningfully complete. This finalises session metadata and allows the next session to find this one via `session_init`.

---

## CLI fallback table

When the SLM MCP server is unavailable, use these CLI equivalents:

| MCP tool               | CLI fallback                                              |
|------------------------|-----------------------------------------------------------|
| `recall`               | `slm recall "<query>" --limit N`                          |
| `search`               | `slm search "<query>"`                                    |
| `remember`             | `slm remember "<content>" --tags a,b` (project/importance are MCP-only) |
| `list_recent`          | `slm list --limit N`                                      |
| `forget`               | `slm forget` (always preview first)                       |
| `slm_optimize_stats`   | `slm optimize status` / `slm optimize savings`            |
| `slm_compress`         | `slm compress`                                            |
| `slm_cache_*`          | `slm cache ...`                                           |
| `slm status` (health)  | `slm status`                                              |
| `session_init`         | daemon-implicit — skip when MCP is down                   |
| `close_session`        | daemon-implicit — skip when MCP is down                   |

---

## Tool reference (core profile — 14 tools)

> The MCP config ships `SLM_MCP_PROFILE=code` (21 tools): the 14 core memory tools below
> **plus** 6 code-graph tools (`build_code_graph`, `get_blast_radius`, `query_graph`,
> `semantic_search_code`, `get_review_context`, `detect_changes`) and `switch_profile`.
> Use `full` (39 tools) to add mesh coordination. Use `power` (51 tools) for governance
> and audit tools. See slm-profile for profile switching.

| Tool               | Signature (key params)                                                                       | Notes                                  |
|--------------------|----------------------------------------------------------------------------------------------|----------------------------------------|
| `remember`         | `content, tags="", project="", importance=5, session_id="", scope="personal", shared_with=""` | Store atomic fact. `scope` opt-in (personal default). See slm-scope. |
| `recall`           | `query, limit=10, agent_id, session_id="", fast=False, include_global, include_shared`       | Multi-channel retrieval. Scope flags off by default. See slm-scope. |
| `search`           | `query, limit=10, profile_id=""`                                                             | FTS5 BM25 keyword search               |
| `fetch`            | `url, ...`                                                                                   | Fetch remote content                   |
| `list_recent`      | `limit=20, profile_id=""`                                                                    | Newest memories first                  |
| `update_memory`    | `fact_id, content, agent_id`                                                                 | Correct an existing memory by id       |
| `forget`           | `profile_id="", dry_run=True`                                                                | Decay cycle; always dry_run first      |
| `session_init`     | `project_path="", query="", max_results=10, max_age_days=30`                                 | Once per session; loads context        |
| `close_session`    | `session_id=""`                                                                              | Finalise session                       |
| `slm_compress`     | `content, mode="auto", reversible=True, ttl_seconds=86400`                                   | Returns compressed, lossy, ccr_id      |
| `slm_retrieve`     | `ccr_id`                                                                                     | Retrieve original from ccr_id          |
| `slm_cache_set`    | `key, value, ttl_seconds=86400`                                                              | KV cache set                           |
| `slm_cache_get`    | `key`                                                                                        | KV cache get; returns hit, value       |
| `slm_optimize_stats` | `()`                                                                                       | Returns compress_runs, tokens_saved_compress, cache_kv_hits |

## Skills

| Skill | Purpose |
|-------|---------|
| slm-recall | Multi-channel memory retrieval |
| slm-remember | Store durable facts and decisions |
| slm-session | Session lifecycle (init + close) |
| slm-status | Health check, optimize stats |
| slm-cache | KV cache for repeated reads |
| slm-compress | Reversible context compression |
| slm-graph | Code graph: blast radius, callers, search (code profile) |
| slm-scope | Personal / shared / global memory scoping |
| slm-profile | Workspace isolation and profile switching |
| slm-governance | Enterprise roles, retention, audit, GDPR |
| slm-mesh | Cross-session peer coordination (full/mesh profile) |

## Subagents

- **slm-memory-advisor** — memory decisions, session hygiene, scope and profile guidance
- **slm-optimize-advisor** — context compression and KV cache
- **slm-governance-advisor** — scope/role compliance, retention policies, GDPR

SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later
