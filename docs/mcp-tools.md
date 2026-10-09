# MCP Tools Reference

SuperLocalMemory exposes 104 tools and 7 resources through the Model Context
Protocol (MCP). A client sees only the tools its tool set allows (see
[Which tools a client sees](#which-tools-a-client-sees)); the registered
function signatures are the source of truth for names and parameters, and a
client still decides when to call a tool.

Every tool that reads or writes memory takes an optional `profile_id`. Empty
means the active memory profile. A non-empty value sends that one call to the
named profile, which must already exist, and never moves the active profile.
`profile_id` is a namespace, not a security boundary; see
[Profiles](profiles.md).

> **Optimize tools:** `slm_compress`, `slm_retrieve`, `slm_cache_set`,
> `slm_cache_get` and `slm_optimize_stats` give an agent explicit compression
> and routed-result caching. They do not intercept or cache the primary
> conversation turn without a proxy. See [Three surfaces](optimize-overview.md).

## Connecting

SLM supports two transports. Both expose the same tools.

**HTTP (recommended):** one shared daemon process, flat RAM.

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

**stdio (universal fallback):** one subprocess per connection.

```json
{ "mcpServers": { "superlocalmemory": { "command": "slm", "args": ["mcp"] } } }
```

See [IDE setup](ide-setup.md) for per-IDE configuration. An AI app on the
internet reaches a small subset of these tools through [Web
access](remote-access/README.md).

## Which tools a client sees

`SLM_MCP_PROFILE` picks a tool set. It controls which tools are visible to the
client and has nothing to do with the memory profile that `switch_profile`
changes. The tool set is fixed when the MCP server starts.

| `SLM_MCP_PROFILE` | Tools | What it adds |
|---|---:|---|
| `core` | 18 | remember, recall, search, fetch, list_recent, update_memory, forget, session_init, close_session, the five optimize tools, review_correction, list_corrections, get_memory_summary, switch_profile |
| `code` | 38 | `core` plus the code-graph tools build_code_graph, get_blast_radius, query_graph, semantic_search_code, get_review_context, detect_changes; Brain evidence; memory-kind tools; the three bounded-loop tools; report_outcome, report_feedback |
| `full` | 57 | everyday memory, sessions, learning, skills, optimize, bounded loops, saved views, memory kinds, Brain evidence, and the 9 mesh tools |
| `power` | 69 | `full` plus get_version, get_mode, health, consistency_check, recall_trace, get_lifecycle_status, set_retention_policy, compact_memories, get_behavioral_patterns, audit_trail, quantize, get_retention_stats |
| `mesh` | 9 | the mesh tools only |
| `whole` | 104 | every registered tool |

With no `SLM_MCP_PROFILE`, a client gets the same 57 tools as `full` (the mesh
tools are included while mesh is enabled, which is the default).
`SLM_MCP_ALL_TOOLS=1` exposes all 104. `SLM_MCP_TOOLS=name1,name2` exposes
exactly the names listed. An unknown `SLM_MCP_PROFILE` value is an error, not a
silent fallback.

The code-graph tools other than the six listed above, and tools such as
`prestage_context`, `get_attribution`, `memory_used`, `backup_status`,
`build_graph` and `list_failed_operations`, appear only with `whole`,
`SLM_MCP_ALL_TOOLS=1` or an explicit `SLM_MCP_TOOLS` list.

---

## Memory tools

### `remember`

Store a memory. SLM extracts atomic facts, resolves entities and indexes the
result for retrieval.

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `content` | string | Yes | The text to remember |
| `tags` | string | No | Comma-separated tags |
| `project` | string | No | Project the memory belongs to |
| `importance` | integer | No | Importance hint (default 5) |
| `session_id` | string | No | Session attribution and stable retry input |
| `agent_id` | string | No | Calling-agent attribution (default `mcp_client`) |
| `scope` | string | No | `personal`, `shared` or `global`; unset means the configured default, which is `personal` |
| `shared_with` | string | No | Comma-separated profile ids, for shared scope |
| `idempotency_key` | string | No | Stable identity so a retry stores nothing twice |
| `session_date` | string | No | When the memory is about, as `YYYY-MM-DD` or ISO 8601. Omitted means today |
| `profile_id` | string | No | Write to this profile instead of the active one |
| `kind` | string | No | One of the nine [memory kinds](memory-kinds.md). A declared kind is confirmed; leave it empty when unsure |
| `replaces` | string | No | `fact_id` of an earlier memory this one replaces |

The result is a receipt with `fact_ids`, `count`, `operation_id`,
`materialization_state` and `pending`. A normal call returns once the memory is
`queryable` by its words; enrichment and the meaning-based index follow on the
same operation. When the writer is busy the state is `accepted`: the memory is
saved durably, `fact_ids` is empty, and resending with the same
`idempotency_key` returns the final receipt. If the daemon cannot be reached
the call returns `DAEMON_UNAVAILABLE` with `retryable: true`.

`replaces` retires the earlier memory at once: recall and session start stop
returning it, nothing is deleted, and the response's `replaced` says what was
retired or why nothing was. The new memory is saved either way. An unknown id,
or another profile's memory, is refused before anything is saved. Undo with
`review_correction(case_id, "rollback", version)`.

### `recall`

Search memories with hybrid retrieval, rank fusion and reranking. See
[Recall](recall.md) for the filters in depth.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string, required | Natural language question |
| `limit` | integer | Max results (default 20) |
| `agent_id`, `session_id` | string | Attribution; `session_id` lets hooks tie later outcomes to this recall |
| `fast` | boolean | Skip the slower steps |
| `include_global`, `include_shared` | boolean | Opt in to global or shared memories for this call. Unset uses the configured default, which is off |
| `window` | string | Event-time span: `24h`, `7d`, `30d`, `1y`, or a range `2026-07-01..2026-07-31` |
| `as_of`, `known_as_of`, `valid_at` | string | ISO 8601 point-in-time filters |
| `include_unknown` | boolean | With the time filters, also return memories whose time is unknown |
| `project` | string | Keep only memories saved under this project |
| `project_strict` | boolean | With `project`, never fall back to unfiltered results |
| `prefer_project` | string | Rank this project's memories higher; removes nothing |
| `saved_by` | string | Keep only memories saved by this agent |
| `about` | string | Keep only memories that mention this name |
| `kind` | string | Keep only one [memory kind](memory-kinds.md) |
| `tags`, `tags_match` | string or list, `all`/`any` | Exact tag filter; `tags_match` defaults to `all` |
| `profile_id` | string | Recall from this profile |

Results follow the [score contract](retrieval-score-contract.md):
`relevance_score` is query relevance, `ranking_score` is diagnostic ranking
utility, and `memory_confidence` belongs to the stored assertion. Without a
configured check the response says `calibration_status: "uncalibrated"` and
`answer_confidence: null`; retrieval scores are not answer probabilities.

The response also carries `query_id` (pass it to `report_outcome`),
`no_confident_match`, `abstained`, `answerability`, `answer_check_status`,
`channel_status`, `incomplete_channels`, `project_scope`, `tag_scope` and
`reranker_status`. When [Answer check](answer-check.md) is on, `answer_confidence`
and `abstention_reason` (for example `judged_insufficient`) carry its verdict.
`no_confident_match: true` means nothing cleared the evidence floor: do not
invent a memory, rewrite the query more specifically and try once more. An
invalid `kind` returns `INVALID_KIND` before anything is retrieved.

### `search`

Full-text search (FTS5 with BM25 ranking).

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string, required | Words to find |
| `limit` | integer | Max results (default 20) |
| `kind` | string | Keep only one memory kind |
| `tags`, `tags_match` | string or list, `all`/`any` | Exact tag filter |
| `profile_id` | string | Search this profile |

### `fetch`

Full details for specific memories.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fact_ids` | string or list, required | Comma-separated string or a list of fact ids |
| `profile_id` | string | Read this profile; an unknown profile is refused |

Ids that do not resolve are listed in `not_found`. When none resolve,
`success` is `false`, so a fetch is a reliable check that a write landed.

### `list_recent`

Newest memories first.

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Max results (default 20) |
| `kind` | string | Keep only one memory kind |
| `tags`, `tags_match` | string or list, `all`/`any` | Exact tag filter |
| `profile_id` | string | List this profile |

### `update_memory`

Propose a correction to one memory by exact fact id. The old text stays
current, and the new text is not recalled, until someone applies the proposal
with `review_correction`. See [Reviewed corrections](reviewed-corrections.md).

| Parameter | Type | Description |
|-----------|------|-------------|
| `fact_id` | string, required | Memory to correct |
| `content` | string, required | New text (not empty) |
| `agent_id` | string | Calling agent, logged for audit |
| `profile_id` | string | Profile the memory belongs to |

Returns `predecessor_fact_id`, `successor_fact_id`, `correction_case` and
`review_required`.

### `review_correction`

Apply, reject or roll back a correction case. The daemon derives the reviewer
from the authenticated local connection; the client supplies the case, the
expected version (compare-and-swap) and, optionally, an event-time boundary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `case_id` | string, required | Case from `update_memory` or `remember(..., replaces=...)` |
| `action` | string, required | `apply`, `reject` or `rollback` |
| `expected_version` | integer, required | Current case version |
| `event_valid_until` | string | Reviewer-approved event-time boundary; `apply` only |
| `profile_id` | string | Profile the case belongs to. To undo a `remember(..., profile_id=..., replaces=...)`, pass the same profile |

### `list_corrections`

Correction cases awaiting review: identifiers, status and timing, never memory
text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Max cases (default 100) |
| `profile_id` | string | List this profile's cases |

### `delete_memory`

Delete one memory by exact fact id. Destructive; every deletion is logged with
the calling `agent_id`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fact_id` | string, required | Exact id from `recall` or `list_recent` |
| `agent_id` | string | Calling agent, logged for audit |
| `profile_id` | string | Profile the memory belongs to |

### `forget`

Run the Ebbinghaus forgetting decay cycle. It computes retention scores and
moves memories between lifecycle zones (active, warm, cold, archive,
forgotten). It does not delete memories matching a query; the CLI command
`slm forget <query>` does that and is a different operation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dry_run` | boolean | Compute statistics without applying transitions (default `true`) |

### `get_memory_summary`

Summarise a day, project, session or community. Summaries are extractive
unless the profile runs a local or cloud model. The response carries
`coverage` and `source_fact_ids`; session data is sparse, so a session summary
is usually partial and should not be presented as a complete record.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kind` | string | `day` (default), `project`, `session` or `community` |
| `target` | string | Day: an ISO date, `today` or `yesterday`. Project: a directory path. Session: the session id (empty lists `recent_sessions`). Community: the `community_id` from a recall's `thematic_context` |
| `profile_id` | string | Profile to summarise |
| `limit`, `offset` | integer | `community` only: page through `source_fact_ids` |

### `run_view` and `manage_view`

A saved view is a named recall query. `run_view(name, profile_id)` runs one
through the normal recall path; leave `name` empty to list the views.
`manage_view(action, name, query, filters, limit, new_name, profile_id)`
creates (`action="create"`), renames or deletes a view and never changes a
memory. `filters` may hold `kind`, `window` and `as_of`; `limit` is 1-50
(default 10); a name is at most 80 characters and a query at most 1000.

### Memory kinds

See [Memory kinds](memory-kinds.md) for what the nine kinds mean.

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `set_memory_kind` | `fact_id`, `kind`, `profile_id` | Set (confirm) one memory's kind; returns its kind fields |
| `memory_kinds_status` | `profile_id` | Counts per kind, the classifying backend, and any run in progress |
| `review_memory_kinds` | `kind`, `limit` (20), `profile_id` | Suggestions waiting for confirmation |
| `confirm_memory_kinds` | `items` (1-200 of `{fact_id, kind}`), `profile_id` | Confirm many at once; omit `kind` to accept the suggestion. Each item reports its own `ok` or `error` |

### `core_memory`

Manage the explicit pin set: `core_memory(action, fact_id, profile_id)` with
`action` of `pin`, `unpin` or `list`. A pinned memory is always injected.

---

## Session and learning tools

Call `session_init` once at the start of a session.

### `session_init`

Load relevant context: recent decisions and patterns for the project, the top
memories for the query, confirmed standing rules and decisions, and the
learning status.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_path` | string | Working directory. Memories saved under it rank higher; its name builds the query when none is given |
| `query` | string | Override the search query |
| `max_results` | integer | Max memories (default 10) |
| `max_age_days` | integer | Hide memories older than this unless relevance is at least 0.70 (default 30; 0 disables the age gate) |
| `session_id` | string | Use this id instead of a generated one |
| `agent_id` | string | Attribution |
| `profile_id` | string | Load another profile's context |

`session_init` does not run the [Answer check](answer-check.md), so
`answerability` is `unjudged` in its response.

### `observe`

Send conversation text for automatic capture. SLM keeps decisions, bug fixes
and preferences it judges worth storing and ignores low-confidence content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | string, required | Text to evaluate |
| `agent_id` | string | Attribution (defaults to `SLM_AGENT_ID`) |
| `session_id` | string | Session |
| `profile_id` | string | Capture into this profile, always as `personal` |

### `close_session`

Close a session and write temporal summaries. `close_session(session_id,
profile_id)`; an empty `session_id` means the most recent session.

### `report_feedback`

Say whether a recalled memory helped. This feeds the adaptive ranker, which
is off unless `SLM_RANKING` is set (for example `v2-ensemble`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `fact_id` | string, required | The recalled memory |
| `feedback` | string | `relevant` (default), `irrelevant` or `partial` |
| `query` | string | The query that surfaced it |
| `profile_id` | string | Profile it was recalled from |

If `query` is given, only a pseudonymized grouping key is stored, not the text:
a per-install keyed HMAC of the query, truncated to 16 hex characters. It is
not encryption. The same query gives the same key within one install and a
different key on another install. The 32-byte key lives in `.feedback-hash-key`
beside the database with mode `0600`.

### `report_outcome`

Report how using recalled memories turned out.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_ids` | string, required | Comma-separated fact ids |
| `outcome` | string, required | `success`, `failure` or `partial` |
| `context` | string | Free-text context |
| `recall_query_id` | string | The `query_id` from the recall; ties the report to that exact answer. Without it SLM matches by overlapping memories within a time window |
| `profile_id` | string | Profile the recall was made in |

### Learning and telemetry

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `log_tool_event` | `tool_name`, `event_type` (`invoke`, `complete`, `error`, `correction`), `input_summary`, `output_summary` (each cut to 500 characters and scrubbed), `duration_ms`, `metadata` (JSON string), `session_id`, `agent_id`, `project_path`, `profile_id` | Passive telemetry for behavioural learning |
| `get_assertions` | `min_confidence`, `category`, `project_path`, `limit` (50), `profile_id` | Learned behavioural assertions |
| `reinforce_assertion` | `assertion_id`, `profile_id` | Raise an assertion's confidence |
| `contradict_assertion` | `assertion_id`, `profile_id` | Lower confidence by 30%; below 0.2 the assertion is deleted |
| `settle_session_outcomes` | `session_id` (required), `agent_id`, `finalize`, `profile_id` | Settle pending recall outcomes for one host session; used by host hooks |
| `get_behavioral_patterns` | `limit` (20), `profile_id` | Detected patterns |
| `get_learned_patterns` | `pattern_type`, `limit` (20), `profile_id` | Interests, refinements, archival habits |
| `correct_pattern` | `pattern_id`, `correction`, `profile_id` | Correct or annotate a learned pattern |
| `get_soft_prompts` | `profile_id` | Auto-learned soft prompts |
| `skill_health` | `skill_name`, `include_history`, `profile_id` | Per-skill invocation counts and status |
| `skill_lineage` | `skill_name`, `profile_id` | How a skill evolved |
| `evolve_skill` | `skill_name`, `evolution_type` (`fix`, `derived`, `captured`), `reason` | Run the evolution pipeline for one skill; evolution must be enabled |

---

## Status, lifecycle and maintenance

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `get_status` | `profile_id` | Fact, entity and edge counts, mode, profile, database size |
| `get_version` | none | SLM, Python and platform versions |
| `get_attribution` | none | Author, version, license and provenance |
| `health` | `profile_id` | Math-layer, database and component health |
| `get_mode` | none | Current mode and its capabilities |
| `set_mode` | `mode` (`a`, `b`, `c`) | Switch mode and reset the engine |
| `switch_profile` | `profile_id` | Switch the active memory profile for this computer |
| `memory_used` | `profile_id` | Usage by fact type and lifecycle state |
| `backup_status` | none | Backup state and available backup files |
| `build_graph` | none | Rebuild knowledge-graph edges for the active profile |
| `audit_trail` | `limit` (50) | Compliance audit entries for the active profile |
| `consistency_check` | `limit` (100) | Pairs of contradicting facts, with severity |
| `recall_trace` | as `recall`: `query`, `limit` (10), `as_of`, `known_as_of`, `valid_at`, `include_unknown`, `project`, `prefer_project`, `project_strict`, `tags`, `tags_match`, `profile_id` | Recall with per-channel score breakdown |
| `prestage_context` | `query`, `limit` (5), `profile_id`, `as_of` | Return the top memories for a query ahead of time |
| `get_lifecycle_status` | `limit` (50), `profile_id` | Counts and samples per lifecycle state |
| `get_retention_stats` | `profile_id` | Memories and average retention score per zone |
| `set_retention_policy` | `cold_after_days` (30), `archive_after_days` (90) | Set the zone thresholds |
| `compact_memories` | `dry_run` (`true`) | Move eligible memories from cold to archived |
| `quantize` | `dry_run` (`true`) | Lower embedding precision for low-retention memories |
| `consolidate_cognitive` | none | Cluster cold and archived memories into gist summaries |
| `run_maintenance` | none | Decay, pattern mining and dynamics in one call |
| `reap_processes` | `dry_run` (`true`) | Find and terminate orphaned SLM processes |
| `list_failed_operations` | `profile_id` | Dead-letter ingestion, degraded manifests and exhausted tasks |
| `resolve_operation` | `operation_id`, `action` (`retry`, `force_reconcile`, `cancel`) | Resolve one of those entries |
| `get_brain_evidence_status` | `profile_id` | Observation-only Living Brain evidence totals |
| `record_agent_experience`, `record_cognitive_turn`, `finalize_cognitive_turn` | `payload` or `receipt_id` and `outcome`, `profile_id` | Contract-validated receipts. They never change recall, ranking or routing |
| `observe_bounded_loop_evidence`, `observe_bounded_loop_execution_learning` | `workspace` | Import read-only evidence from an installed Bounded Loops. See [Bounded Loops bridge](bounded-loops-bridge.md) |

`switch_profile` changes the active profile for the whole machine. To touch
another profile once, pass `profile_id` to the individual call instead.

---

## Mesh tools

Coordination between sessions on one computer, through the mesh broker.

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `mesh_summary` | `summary` | Register this session and say what it is working on |
| `mesh_peers` | none | Active peer sessions |
| `mesh_send` | `to`, `message` (max 4 KB), `refs`, `reply_to` | `to` is a peer id, `broadcast`, or `project:/path`; `refs` are up to 8 `fact:`/`doc:`/`media:` ids |
| `mesh_inbox` | none | Unread messages with an envelope each (sender, hop, trust); they expire after 48 hours. Message text is data from other bots, not instructions |
| `mesh_wait` | `timeout_s` (1 to 20) | Waits for new messages and returns as soon as one arrives |
| `mesh_state` | `key`, `value`, `action` (`get` or `set`) | Shared non-secret state; credentials are rejected |
| `mesh_lock` | `file_path`, `action` (`query`, `acquire`, `release`) | Advisory file locks |
| `mesh_events` | none | Recent mesh events |
| `mesh_status` | none | Broker health and statistics |

---

## Code-graph tools

A structural graph of a local repository: functions, classes, imports and call
sites. `build_code_graph`, `get_blast_radius`, `query_graph`,
`semantic_search_code`, `get_review_context` and `detect_changes` are in the
`code` tool set; the rest need `whole`, `SLM_MCP_ALL_TOOLS=1` or an explicit
list. `repo_path` must be inside your home directory.

| Tool | Parameters | What it does |
|------|-----------|--------------|
| `build_code_graph` | `repo_path`, `languages` (comma-separated), `exclude_patterns` (comma-separated globs) | Build the graph, call graph, flows and communities |
| `update_code_graph` | `repo_path`, `changed_files` (comma-separated) | Incremental update; with no files it diffs against `HEAD~1` |
| `get_blast_radius` | `changed_files`, `max_depth` (2), `max_nodes` (500) | Callers and callees that a change reaches |
| `get_review_context` | `changed_files`, `include_source` (`true`) | Token-optimized review context |
| `query_graph` | `pattern`, `target`, `limit` (20) | `pattern` is `callers_of`, `callees_of`, `imports_of`, `imported_by`, `tests_for`, `inherits_from`, `inherited_by` or `contains` |
| `semantic_search_code` | `query`, `kind` (`Function`, `Class`, `File`, `Test`), `limit` (20) | Hybrid full-text and vector search over code entities |
| `list_graph_stats` | none | Graph size and health |
| `find_large_functions` | `threshold` (50), `limit` (20) | Functions over a line count |
| `detect_changes` | `base` (`HEAD~1`) | Changed code with risk scores |
| `get_architecture_overview` | none | Communities and how they relate |
| `list_flows`, `get_flow`, `get_affected_flows` | `sort_by` (`criticality` or `size`) and `limit`; `flow_name`; `changed_files` | Execution flows |
| `list_communities`, `get_community` | `sort_by` (`cohesion` or `size`) and `limit`; `community_id` | Code communities |
| `code_memory_search` | `code_entity`, `link_type`, `limit` (10) | Memories linked to a code entity |
| `code_entity_history` | `code_entity` | Memory timeline for one entity |
| `link_memory_to_code` | `fact_id`, `code_entity`, `link_type` (`mentions`, `decision_about`, `bug_fix`, `refactor`, `design_rationale`) | Link a memory to a code node |
| `enrich_blast_radius` | `changed_files`, `max_depth` (2) | Blast radius plus related memories |
| `code_stale_check` | `scope` (`all` or a file path) | Memories that reference deleted or changed code |
| `refactor_preview` | `action` (`rename`, `find_dead_code`, `find_duplicates`), `target`, `new_name` | Preview only |
| `apply_refactor` | `action`, `target`, `new_name`, `dry_run` | A stub: it returns the preview and never writes files |

---

## Resources

MCP resources are read-only data a client can read passively.

| Resource URI | Description |
|--------------|-------------|
| `slm://context` | Session context: relevant memories and learning status |
| `slm://recent` | The most recently stored memories |
| `slm://stats` | Memory count, database size, mode, profile |
| `slm://clusters` | Topic clusters |
| `slm://identity` | Learned preferences and patterns |
| `slm://learning` | State of the adaptive learning system |
| `slm://engagement` | Usage statistics |

---

## Optimize tools

Compression and routed-result caching without a proxy. Each call returns `ok`;
a failure leaves the original content unchanged.

| Tool | Parameters | Notes |
|------|-----------|-------|
| `slm_compress` | `content` (max 1 MB), `mode` (`normalize`, `auto` default, `aggressive`), `reversible` (`true`), `ttl_seconds` (86400) | A lossy result with `reversible` returns a `ccr_id` |
| `slm_retrieve` | `ccr_id` | Recover the original of a lossy compression. Do not log or share a `ccr_id` |
| `slm_cache_set` | `key` (max 512 characters), `value` (max 1 MB), `ttl_seconds` (86400) | Namespaced per agent. Do not cache secrets |
| `slm_cache_get` | `key` | `hit: false` on a miss, expiry or error |
| `slm_optimize_stats` | none | Compression and cache statistics. Proxy and key-value counters are persisted by the daemon and survive restarts |

---

## Bounded-loop tools

A bounded loop ends only when an independent gate passes, never because the
agent says it is done. Over MCP the gate is an SLM recall: the loop converges
on the first lap where a recall of `gate_query` returns a confident match.
The `slm loop` CLI, the `/slm-loop` command and these tools share one durable
ledger.

### `slm_loop_run`

Runs one loop and blocks, polling the gate, until it passes or a bound trips.
Every lap is written to memory with the tag `loop:<name>` and shows in the
dashboard. Each lap is one recall of at most three memories; with the answer
check on, each lap also asks it once, which with the online check is a request
to the provider that may be billed. It is host-only for Web access.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string, required | Loop name and tag, 1-128 characters |
| `gate_query` | string, required | Recall query checked each lap |
| `gate_min_score` | number | Minimum top-result score (default 0.0) |
| `max_iterations` | integer | Lap cap, 1-200 (default 20) |
| `max_wallclock_s` | number | Wall-clock cap in seconds (default 15, maximum 120, 0 disables) |
| `poll_interval_s` | number | Seconds between laps (default 1.0, minimum 0.25) |
| `max_tokens` | integer | Token budget (0 disables) |
| `no_progress_window` | integer | Halt after this many no-change laps (0 disables) |
| `require_support` | boolean | Pass only if the answer check ran and judged the memories sufficient (`answerability == "supported"`) |

The result reports `status` (`DONE`, `HALT`, `PAUSE`, `KILLED` or `ERROR`),
`reason`, `passed`, `laps`, `run_id` and a per-lap `ledger`. Setting the
environment variable `SLM_LOOP_KILL` to any non-empty value stops loops.

### `slm_loop_history` and `slm_loop_show`

`slm_loop_history(name, limit, profile_id)` lists recorded runs for a loop name
(limit 1-200, default 20). `slm_loop_show(run_id, limit, profile_id)` shows
every lap of one run in order (limit 1-1000, default 200). Both are read-only.
Each lap records the agent's own done-claim for audit; it never ends the loop.

---

## Tools over Web access

An AI app connected through [Web access](remote-access/README.md) reaches only
`recall`, `search`, `fetch` and `get_status`, plus `remember` if saving was
allowed and `session_init`, `close_session`, `report_feedback` and
`report_outcome` if session tools were allowed. Everything else stays on your
computer.

API keys created with `slm remote keys add` for [remote access over
TLS](distributed-deployment.md#remote-access-over-tls) are a separate path.
A read-only key can call the read tools; any other key can also call the write
tools. Tools that manage the machine itself
(switching profile, maintenance, retention, mesh, the code graph, `forget`,
`slm_loop_run`) are host-only on both paths.
