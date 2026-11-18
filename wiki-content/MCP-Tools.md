# MCP Tools

SLM exposes 112 tools and 9 resources through the Model Context Protocol (MCP), but a client never sees all of them by default. A named tool profile (`core`, `code`, `full`, `power`, `mesh`, `whole`) picks a fixed subset when the MCP server starts, and an AI app on another computer or on the internet gets a smaller, default-deny subset again. This page lists every tool with its parameters, the profiles that include it and whether a remote caller may use it. The registered function signatures in `src/superlocalmemory/mcp/` are the source of truth, and the profile sets are computed from `mcp/profiles.py`.

## Connecting

SLM speaks two transports and both expose the same tools. With stdio, the IDE runs `slm mcp` as a subprocess, which costs one process per open tool. With HTTP, the IDE connects to the running daemon, which shares one process.

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

```json
{ "mcpServers": { "superlocalmemory": { "command": "slm", "args": ["mcp"] } } }
```

`slm connect IDE [--transport stdio, http or http-mcp-remote]` writes either form for you ([IDE Setup](IDE-Setup)). Over HTTP the path `/mcp/AGENT_ID` labels every write with that agent; over stdio set `SLM_AGENT_ID` in the entry's environment. The id is cleaned to letters, digits, dot, underscore and hyphen, at most 64 characters. It is attribution, not authentication. See [Multi-Agent Memory](Multi-Agent-Memory).

Every tool that reads or writes memory takes an optional `profile_id`. Empty means the active memory profile; a name routes that one call to that profile without moving the active one, and an unknown name is refused. A memory profile is a namespace, not a security boundary ([Profiles](Profiles)).

## Choosing which tools a client sees

> Current profile counts: `core` **20**, `code` **40**, `full` **60**, `power` **72**, `mesh` **9**, `whole` **112**. While pictures and documents are on, `full` and `power` each gain five tools (65 and 77).

Three environment variables decide the surface, and the first one that applies wins: `SLM_MCP_ALL_TOOLS=1` exposes all 110, then `SLM_MCP_TOOLS=a,b,c` exposes exactly the names listed, then `SLM_MCP_PROFILE` picks a named profile, and with none of them set the default applies. The default is the same set as `full`. An unknown `SLM_MCP_PROFILE` value is an error, not a silent fallback. `slm connect IDE --profile NAME` writes the variable into an IDE's entry. The choice is fixed when the server starts. It has nothing to do with the memory profile that `switch_profile` changes.

| `SLM_MCP_PROFILE` | Tools | What it holds |
|---|---:|---|
| `core` | **20** | remember, recall, search, fetch, expand, list_recent, update_memory, forget, session_init, close_session, the five Optimize tools, review_correction, settle_dispute, list_corrections, get_memory_summary, switch_profile |
| `code` | **40** | `core`, plus the five Brain tools, the four memory-kind tools, six code-graph tools (build_code_graph, get_blast_radius, query_graph, semantic_search_code, get_review_context, detect_changes), the three bounded-loop tools, report_outcome and report_feedback |
| `full` | **60** | Everyday memory, sessions, learning, skills, Optimize, bounded loops, saved views, memory kinds, Brain tools and the nine mesh tools. This is also the default |
| `power` | **72** | `full`, plus get_version, get_mode, health, consistency_check, recall_trace, get_lifecycle_status, set_retention_policy, compact_memories, get_behavioral_patterns, audit_trail, quantize, get_retention_stats |
| `mesh` | **9** | The nine `mesh_*` tools only |
| `whole` | **112** | Every registered tool (the raw server) |

The five picture and document tools join `full` and `power` only while images and documents are turned on ([Pictures and Documents](Pictures-and-Documents)). That is read once when the server starts. The mesh tools join the default while mesh is enabled, which is the default (`SLM_MCP_MESH_TOOLS=1` forces them on). Old count-suffixed names such as `core14`, `code34`, `full50`, `power61`, `mesh8` and `whole94` still resolve to their current profile and log a deprecation warning.

Twenty-nine tools are in no named profile at all: the code-graph tools beyond the six in `code` (sixteen of them), and others such as `build_graph`, `core_memory`, `prestage_context`, `reap_processes`, `list_failed_operations` and `resolve_operation`. They appear only under `whole`, `SLM_MCP_ALL_TOOLS=1` or an explicit `SLM_MCP_TOOLS` list, and the tables below mark them "whole only". The five picture and document tools are a separate case: they are in `full` and `power` only while images are on.

## The tools

In the tables, a parameter followed by `?` is optional. The Profiles column says which named profiles include the tool (`whole` includes every tool). The Remote column says what an AI app on another computer may do: `read` means any remote key, `write` means a key that is not read-only, `mesh key` and `media key` mean the key must also be opted in, `host only` means the tool is one of the 45 tools classified as host-only, and `refused` means the tool is not on the read, write, mesh or media lists, so a remote caller cannot run it. The refusal is the generic denial, not the host-only message. Tools that only read declare MCP's `readOnlyHint`; `delete_memory`, `forget`, `resolve_operation` and `manage_view` declare `destructiveHint`.

**The `forget` naming collision.** The MCP tool `forget` runs the Ebbinghaus decay cycle and is a dry run by default. It moves memories between lifecycle zones and never deletes by query. The CLI command `slm forget QUERY` deletes memories whose text matches the query. To delete one memory over MCP, use `delete_memory`, which is not in the `core` or `code` profiles.

### Memory

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `remember` | `content`, `tags`?, `project`?, `importance`?, `session_id`?, `agent_id`?, `scope`?, `shared_with`?, `idempotency_key`?, `session_date`?, `profile_id`?, `kind`?, `replaces`?, `slot`?, `slot_value`?, `source_file`? | Store a memory and return an operation receipt (`operation_id`, `fact_ids`, `materialization_state`). `scope` and `shared_with` set visibility, `kind` and `replaces` classify or supersede, `idempotency_key` makes a retry safe | core, code, full, power | write |
| `recall` | `query`, `limit`?, `agent_id`?, `session_id`?, `fast`?, `include_global`?, `include_shared`?, `window`?, `as_of`?, `known_as_of`?, `valid_at`?, `include_unknown`?, `profile_id`?, `project`?, `saved_by`?, `about`?, `kind`?, `prefer_project`?, `tags`?, `tags_match`?, `project_strict`?, `show_weak`?, `include_related`?, `depth`?, `max_tokens`?, `budget_s`? | Multi-channel retrieval with the same filters as `slm recall` (project, kind, tags, time windows, scope). Returns Score Contract results plus a `tier` (`supported`, `related` or `none`), `abstention_note`, `warming_up` and the answer-check fields. When nothing supports the question it returns no results and says so; `include_related=true` adds up to three labelled close memories, `show_weak=true` shows every weak match, `depth` (0 glance, 1 full, 2 with context) and `max_tokens` shape and cap the reply. A question waits for its full answer. `budget_s` is for a caller that blocks a turn on recall (a per-message prefetch): recall then answers inside that many seconds, with unchecked keyword matches when the full search cannot finish. When every recall slot is busy the tool returns `code: RECALL_BUSY`; wait a few seconds and ask again, it says nothing about the memory. See [How Recall Works](How-Recall-Works) | core, code, full, power | read |
| `search` | `query`, `limit`?, `kind`?, `profile_id`?, `tags`?, `tags_match`?, `depth`?, `max_tokens`? | Full-text search over memories (FTS5, BM25 ranking), with kind and tag filters. Each result carries a `tier`: `supported`, or `related` (at most three) when none is supported, with `abstention_note` saying so. `depth` and `max_tokens` work as in `recall` | core, code, full, power | read |
| `fetch` | `fact_ids`, `profile_id`?, `why`? | Full details for specific fact ids, comma-separated or as a list. `why=true` adds to each result (up to 10 per call) the same report as `slm why`: where the memory came from, whether the instruction firewall holds it back, its signature, its disputes, and the last deliveries that showed it, with the reason for each | core, code, full, power | read |
| `expand` | `memory_id`, `depth`? (default 2), `max_tokens`?, `profile_id`? | Drill into one recall result: the full memory plus, at depth 2, its source memory, up to three memories from the same session, up to three about the same entities and the correction that replaced it. Takes a result's `fact_id` or `memory_id` and applies the same profile and visibility checks as `fetch`. `max_tokens` is never exceeded | core, code, full, power | refused |
| `list_recent` | `limit`?, `kind`?, `profile_id`?, `tags`?, `tags_match`? | Most recent memories, newest first, with optional kind and tag filters | core, code, full, power | read |
| `update_memory` | `fact_id`, `content`, `agent_id`?, `profile_id`? | Replace the text of one memory by exact fact id | core, code, full, power | write |
| `delete_memory` | `fact_id`, `agent_id`?, `profile_id`? | Delete one memory by exact fact id | full, power | write |
| `review_correction` | `case_id`, `action`, `expected_version`, `event_valid_until`?, `profile_id`? | Apply, reject or roll back a review-gated correction case; `expected_version` guards against stale reviews | core, code, full, power | write |
| `list_corrections` | `limit`?, `profile_id`? | List correction cases for a human or host reviewer | core, code, full, power | read |
| `list_disputes` | `limit`?, `profile_id`?, `audit`?, `model`? | What needs an answer: open disputes (both sides, newest first) and memories whose source file changed; `audit=true` lists the blind audit queue | full, power | read |
| `settle_dispute` | `dispute_id`, `keep`, `profile_id`?, `model`? | Settle a dispute (recorded with your agent id, undone with `keep="reopen"`): `keep` is the true side's `fact_id`, `both` or `unrelated`; `dispute_id="audit:<item>"` gives a blind audit label | core, code, full, power | write |
| `switch_profile` | `profile_id` | Change the active memory profile. This is a different concept from the MCP tool profile that decides which tools you see | core, code, full, power | host only |
| `core_memory` | `action`, `fact_id`?, `profile_id`?, `promote`? | Manage the Core Memory pin set: `action` is `pin`, `unpin`, `list`, `promote` or `withdraw`. A pinned fact is always injected. With the instruction firewall on, pinning a memory that did not come from the user answers `needs_approval` and is held back (`slm trust pending`); one explicit `promote` (or `pin` with `promote=true`) releases it, recorded with your agent id, and `withdraw` takes it back | whole only | write |

### Sessions and learning

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `session_init` | `project_path`?, `query`?, `max_results`?, `max_age_days`?, `session_id`?, `agent_id`?, `profile_id`? | Load relevant memory context at the start of a session. Call it once; `project_path` ranks that project's memories higher, `max_age_days` (default 30) hides stale ones unless strongly relevant | core, code, full, power | write |
| `observe` | `content`, `agent_id`?, `session_id`?, `profile_id`? | Offer conversation content for automatic capture of decisions, fixes and preferences. Low-confidence content is not stored | full, power | write |
| `close_session` | `session_id`?, `profile_id`? | Close a session and create its temporal summaries (default: the most recent session) | core, code, full, power | write |
| `report_feedback` | `fact_id`, `feedback`?, `query`?, `profile_id`?, `recall_query_id`? | Say whether a recalled memory helped: `feedback` is `relevant` (default), `irrelevant` or `partial`. Recorded in the learning store; it reorders recall only when adaptive ranking is switched on. With `recall_query_id` it is also stored as the outcome of that exact recall ([Active Memory](Active-Memory)) | code, full, power | write |
| `report_outcome` | `memory_ids`, `outcome`, `context`?, `recall_query_id`?, `profile_id`?, `agent_id`?, `report_id`? | Report how using recalled memories went: `outcome` is `success`, `failure` or `partial`; `memory_ids` is comma-separated; `recall_query_id` ties the report to one recall; `agent_id` names the reporter; `report_id` makes a retry record nothing new | code, full, power | write |
| `log_tool_event` | `tool_name`, `event_type`?, `input_summary`?, `output_summary`?, `duration_ms`?, `metadata`?, `session_id`?, `agent_id`?, `project_path`?, `profile_id`? | Log a tool-use event for behavioural learning (`event_type`: invoke, complete, error or correction). Passive; no model call | full, power | write |
| `settle_session_outcomes` | `session_id`, `agent_id`?, `finalize`?, `profile_id`? | Settle pending recall outcomes for one exact host session; `session_id` must be explicit | whole only | write |
| `get_assertions` | `min_confidence`?, `category`?, `project_path`?, `limit`?, `profile_id`? | Learned behavioural assertions, filtered by confidence, category and project | full, power | read |
| `reinforce_assertion` | `assertion_id`, `profile_id`? | Raise the confidence of a behavioural assertion | full, power | write |
| `contradict_assertion` | `assertion_id`, `profile_id`? | Lower the confidence of a behavioural assertion | full, power | write |
| `get_learned_patterns` | `pattern_type`?, `limit`?, `profile_id`? | Learned behavioural patterns: interests, refinements, archival habits | whole only | read |
| `correct_pattern` | `pattern_id`, `correction`, `profile_id`? | Correct or annotate a learned pattern to improve retrieval | whole only | write |
| `get_behavioral_patterns` | `limit`?, `profile_id`? | Behavioural patterns detected for the active profile | power | read |

### Brain and receipts

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `get_brain_evidence_status` | `profile_id`? | Profile-scoped, observation-only Living Brain evidence totals | code, full, power | read |
| `record_agent_experience` | `payload`, `profile_id`? | Record a contract-validated terminal experience receipt. Supply only evidence you can substantiate; it does not alter recall or ranking by itself | code, full, power | write |
| `record_cognitive_turn` | `payload`, `profile_id`? | Open one contract-validated cognitive-turn provenance receipt | code, full, power | write |
| `finalize_cognitive_turn` | `receipt_id`, `outcome`, `profile_id`? | Close an open cognitive turn with outcome evidence | code, full, power | write |
| `observe_bounded_loop_evidence` | `workspace` | Import one explicit, read-only snapshot from an installed Bounded Loops producer; observation only | code, full, power | host only |
| `observe_bounded_loop_execution_learning` | `workspace` | Ingest negotiated bridge-v2 terminal evidence into the execution plane; writes no semantic facts or preferences | whole only | host only |

### Optimize

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `slm_compress` | `content`, `mode`?, `reversible`?, `ttl_seconds`? | Compress text or tool output. `mode` is `normalize` (lossless), `auto` (default) or `aggressive`; a lossy, reversible result returns a `ccr_id` | core, code, full, power | write |
| `slm_retrieve` | `ccr_id` | Recover the original text of a lossy compression from its `ccr_id` | core, code, full, power | read |
| `slm_cache_set` | `key`, `value`, `ttl_seconds`? | Cache a result under a key, namespaced per agent (default TTL 86,400 seconds) | core, code, full, power | write |
| `slm_cache_get` | `key` | Fetch a cached result; `hit` is false on a miss | core, code, full, power | read |
| `slm_optimize_stats` | none | Compression and cache statistics | core, code, full, power | read |

### Memory kinds

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `set_memory_kind` | `fact_id`, `kind`, `profile_id`? | Set (confirm) the kind of one memory | code, full, power | write |
| `memory_kinds_status` | `profile_id`? | Counts per kind (confirmed and suggested), the classifying backend and any run in progress | code, full, power | read |
| `review_memory_kinds` | `kind`?, `limit`?, `profile_id`? | List kind suggestions awaiting confirmation | code, full, power | read |
| `confirm_memory_kinds` | `items`, `profile_id`? | Confirm kinds for 1 to 200 facts at once | code, full, power | write |

### Views and summaries

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `run_view` | `name`?, `profile_id`? | Run a saved view by name, or list your views when `name` is empty | full, power | read |
| `manage_view` | `action`, `name`, `query`?, `filters`?, `limit`?, `new_name`?, `profile_id`? | Create, rename or delete a saved view (`action`); never changes a memory. `filters` may hold `kind`, `window` and `as_of` | full, power | write |
| `get_memory_summary` | `kind`?, `target`?, `profile_id`?, `limit`?, `offset`? | Summarise a day, project, session or community, or read a compiled page. `kind` is `day` (default), `project`, `session`, `community` or `page` (a page built only from word-for-word spans of the memories it cites and fixed templates, with the count of withheld sentences; empty `target` lists the pages); the result carries `coverage` and `source_fact_ids` | core, code, full, power | read |

### Skills

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `evolve_skill` | `skill_name`, `evolution_type`?, `reason`? | Run the evolution pipeline for one skill (`evolution_type`: fix, derived or captured). Needs evolution enabled | full, power | host only |
| `skill_health` | `skill_name`?, `include_history`?, `profile_id`? | Health metrics for one skill or all skills | full, power | read |
| `skill_lineage` | `skill_name`?, `profile_id`? | Evolution lineage of a skill | full, power | read |

### Mesh

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `mesh_summary` | `summary`? | Register this session and say what it is working on | full, power, mesh | host only |
| `mesh_peers` | none | List active peer sessions | full, power, mesh | mesh key |
| `mesh_send` | `to`, `message`, `refs`?, `reply_to`? | Send a message to a peer id, `broadcast`, or `project:/path`; optional `refs` and `reply_to` | full, power, mesh | mesh key |
| `mesh_inbox` | `ack`? | Read messages sent to this session; `ack` is used by connected web apps | full, power, mesh | mesh key |
| `mesh_wait` | `timeout_s`?, `ack`? | Wait up to `timeout_s` (default 20) for new messages, then return what arrived | full, power, mesh | mesh key |
| `mesh_state` | `key`?, `value`?, `action`? | Get or set shared, non-secret state (`action` get or set); credentials are rejected | full, power, mesh | mesh key |
| `mesh_lock` | `file_path`, `action`? | Advisory file locks (`action`: query, acquire or release) | full, power, mesh | host only |
| `mesh_events` | none | Recent mesh events: joins, leaves, messages, state changes | full, power, mesh | host only |
| `mesh_status` | none | Mesh broker health and statistics | full, power, mesh | host only |

### Pictures and documents

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `remember_media` | `path`?, `download_url`?, `base64`?, `content`?, `tags`?, `profile_id`?, `idempotency_key`?, `scope`?, `shared_with`?, `file`? | Save an image from a `path`, an https `download_url`, `base64` or a chat attachment (`file`) | full, power (images on) | media key |
| `get_media` | `media_id`, `variant`?, `profile_id`? | Return the thumbnail of a saved image as an image (`variant` thumb) | full, power (images on) | media key |
| `remember_document` | `path`?, `base64`?, `file_name`?, `content`?, `tags`?, `profile_id`?, `idempotency_key`?, `scope`?, `shared_with`?, `download_url`?, `file`? | Save a PDF the same ways as an image; work continues in the background and the answer carries a `job_id` | full, power (images on) | media key |
| `media_status` | `job_id`, `profile_id`? | Progress of a saved document's background work, by `job_id` | full, power (images on) | media key |
| `media_upload_link` | `kind`?, `note`?, `profile_id`? | Get a one-time link a person opens to add a picture or PDF from their own device (`kind`: image or document); for remote callers | full, power (images on) | media key |

### Bounded loops

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `slm_loop_run` | `name`, `gate_query`, `gate_contract`?, `gate_min_score`?, `max_iterations`?, `max_wallclock_s`?, `poll_interval_s`?, `max_tokens`?, `no_progress_window`?, `require_support`? | Run one bounded loop that ends only when an independent gate passes or a bound trips: a recall of `gate_query` finds a locator memory and, through `gate_contract={"job_id": ...}`, an operator-registered check has a passing receipt for the current files | code, full, power | host only |
| `slm_loop_history` | `name`, `limit`?, `profile_id`? | Recorded runs for a loop name, newest first | code, full, power | read |
| `slm_loop_show` | `run_id`, `limit`?, `profile_id`? | Every lap of one run, in order | code, full, power | read |

### Code graph

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `build_code_graph` | `repo_path`, `languages`?, `exclude_patterns`? | Build a code knowledge graph for a repository | code | host only |
| `update_code_graph` | `repo_path`?, `changed_files`? | Incrementally update the graph for changed files (default: detect via git diff) | whole only | host only |
| `get_blast_radius` | `changed_files`, `max_depth`?, `max_nodes`? | Impact radius of changed files, by breadth-first search over callers and callees | code | host only |
| `get_review_context` | `changed_files`, `include_source`? | Token-optimised review context for changed files | code | host only |
| `query_graph` | `pattern`, `target`?, `limit`? | Relationship query: `pattern` is callers_of, callees_of, imports_of, imported_by, tests_for, inherits_from, inherited_by or contains | code | host only |
| `semantic_search_code` | `query`, `kind`?, `limit`? | Hybrid full-text and vector search over code entities (`kind`: Function, Class, File or Test) | code | host only |
| `list_graph_stats` | none | Code graph size and health | whole only | host only |
| `find_large_functions` | `threshold`?, `limit`? | Functions above a line-count threshold (default 50) | whole only | host only |
| `list_flows` | `sort_by`?, `limit`? | Execution flows sorted by criticality or size | whole only | host only |
| `get_flow` | `flow_name` | Detail of one execution flow | whole only | host only |
| `get_affected_flows` | `changed_files` | Flows impacted by changed files | whole only | host only |
| `list_communities` | `sort_by`?, `limit`? | Detected code communities, sorted by cohesion or size | whole only | host only |
| `get_community` | `community_id` | Detail of one code community | whole only | host only |
| `get_architecture_overview` | none | High-level map of communities and their relationships | whole only | host only |
| `detect_changes` | `base`? | Detect code changes against a git ref (default HEAD~1) and score their risk | code | host only |
| `refactor_preview` | `action`, `target`, `new_name`? | Preview a refactor without running it: `action` is rename, find_dead_code or find_duplicates | whole only | host only |
| `apply_refactor` | `action`, `target`, `new_name`?, `dry_run`? | Run a refactor. Only `rename` is supported, and the tool is an MVP stub; `dry_run` is true by default | whole only | host only |
| `code_memory_search` | `code_entity`, `link_type`?, `limit`? | Bridge tool: search the memories linked to a code entity | whole only | host only |
| `code_entity_history` | `code_entity` | Bridge tool: the memory timeline of a function or class | whole only | host only |
| `enrich_blast_radius` | `changed_files`, `max_depth`? | Bridge tool: blast radius plus the memories about each impacted node | whole only | host only |
| `code_stale_check` | `scope`? | Bridge tool: memories that reference deleted or changed code | whole only | host only |
| `link_memory_to_code` | `fact_id`, `code_entity`, `link_type`? | Bridge tool: link a memory to a code entity by hand | whole only | host only |

### Status, health and diagnostics

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `get_status` | `profile_id`? | Fact count, entity count, mode, profile and database size | full, power | read |
| `health` | `profile_id`?, `memory`? | System health: scoring, consistency and lifecycle layers, database integrity, components; `memory=true`: the Memory Health report (what was delivered, how often each memory is used, whether memory measurably helps, judge certificates, second opinions, and a Trust section: where memories came from, those waiting for the user's OK, credentials held as placeholders by type) | power | read |
| `get_version` | none | SLM, Python and platform versions | power | read |
| `get_mode` | none | Current operating mode and what it means | power | read |
| `set_mode` | `mode` | Switch the operating mode (a, b or c) | full, power | host only |
| `consistency_check` | `limit`? | Run the sheaf consistency check; returns pairs of contradicting facts with severity | power | host only |
| `recall_trace` | `query`, `limit`?, `as_of`?, `known_as_of`?, `valid_at`?, `include_unknown`?, `project`?, `prefer_project`?, `profile_id`?, `tags`?, `tags_match`?, `project_strict`? | Recall with a per-channel score breakdown, same filters as `recall` | power | read |
| `audit_trail` | `limit`? | Compliance audit trail for the active profile | power | host only |
| `backup_status` | none | Backup state, last backup time and available backup files | whole only | host only |
| `memory_used` | `profile_id`? | Memory usage by fact type and lifecycle state | whole only | read |
| `get_attribution` | none | Author, version, license and provenance metadata | whole only | read |
| `build_graph` | none | Rebuild knowledge-graph edges for all facts in the active profile | whole only | host only |
| `prestage_context` | `query`, `limit`?, `profile_id`?, `as_of`? | Top-K memories for a query, with optional `as_of`. Registered but kept off every profile | whole only | read |

### Lifecycle, maintenance and operations

| Tool | Parameters | What it does | Profiles | Remote |
|---|---|---|---|---|
| `forget` | `dry_run`? | Run the Ebbinghaus forgetting decay cycle, a dry run by default (`dry_run` true). It does NOT delete memories matching a query, unlike `slm forget` | core, code, full, power | host only |
| `quantize` | `dry_run`? | Run the embedding quantization cycle; dry run by default | power | host only |
| `consolidate_cognitive` | none | Run cognitive consolidation: cluster cold memories into gist summaries | full, power | host only |
| `run_maintenance` | none | Run decay, Langevin stepping and pattern mining in one call; clients can call it at session end | full, power | host only |
| `compact_memories` | `dry_run`? | Archive cold or stale facts; dry run by default | power | host only |
| `set_retention_policy` | `cold_after_days`?, `archive_after_days`? | Set the cold and archive thresholds in days (defaults 30 and 90) | power | host only |
| `get_lifecycle_status` | `limit`?, `profile_id`? | Counts per lifecycle state (active, warm, cold, archived) and recent facts in each | power | read |
| `get_retention_stats` | `profile_id`? | Zone distribution and decay rates from the retention table | power | read |
| `get_soft_prompts` | `profile_id`? | Active soft prompts built from learned patterns | full, power | read |
| `reap_processes` | `dry_run`? | Find and stop orphaned SLM processes; dry run by default | whole only | host only |
| `list_failed_operations` | `profile_id`? | Failed, stuck or degraded memory operations (dead letters, degraded manifests, exhausted retries) | whole only | host only |
| `resolve_operation` | `operation_id`, `action` | Resolve one operation: `action` retry, force_reconcile or cancel (administrator action) | whole only | host only |
## Resources

Resources are read-only data a client can read passively. They are registered separately from the tools and are not counted in the 110.

| Resource | What it holds |
|---|---|
| `slm://context` | Active session context: the most relevant recent decisions, patterns and project context, read on connect |
| `slm://recent` | A plain-text summary of the 20 most recent facts in the active profile |
| `slm://stats` | Fact, entity and edge counts, and storage breakdown by type |
| `slm://clusters` | Memory scenes (topic clusters) with themes and fact counts |
| `slm://identity` | Active profile name, mode and top entities |
| `slm://learning` | Adaptive-learning health, pattern summary and outcome statistics |
| `slm://engagement` | Memory usage activity: recent stores, recalls and top-accessed facts |
| `slm://media/{media_id}` | A saved image: its id and where to read its thumbnail (apps on this computer only) |
| `slm://media/{media_id}/thumb` | The image thumbnail as WebP (apps on this computer only) |

## What a remote app may call

A remote caller is an AI tool on another computer using a key from `slm remote keys add`, or a web app connected through [Web Access and Connected Apps](Web-Access-and-Connected-Apps). The policy in `server/remote_tool_policy.py` is default-deny. 112 tools are registered. The policy classifies 109 of them: 31 read tools, 23 write-only tools, 5 mesh tools, 5 media tools and 45 host-only tools. `expand` is not in those lists. A remote caller cannot run it, because anything outside the read, write, mesh and media lists is refused, and the denial is the generic one rather than the host-only message. The policy's classification test expects every registered name to appear in exactly one of those sets; `expand` does not.

| Class | Tools |
|---|---|
| Read (any remote key) | fetch, get_assertions, get_attribution, get_behavioral_patterns, get_brain_evidence_status, get_learned_patterns, get_lifecycle_status, get_memory_summary, get_mode, get_retention_stats, get_soft_prompts, get_status, get_version, health, list_corrections, list_disputes, list_recent, memory_kinds_status, memory_used, prestage_context, recall, recall_trace, review_memory_kinds, run_view, search, skill_health, skill_lineage, slm_cache_get, slm_loop_history, slm_loop_show, slm_optimize_stats, slm_retrieve |
| Write (a key that is not read-only, in addition to the read tools) | close_session, confirm_memory_kinds, contradict_assertion, core_memory, correct_pattern, delete_memory, finalize_cognitive_turn, log_tool_event, manage_view, observe, record_agent_experience, record_cognitive_turn, reinforce_assertion, remember, report_feedback, report_outcome, review_correction, session_init, set_memory_kind, settle_dispute, settle_session_outcomes, slm_cache_set, slm_compress, update_memory |
| Mesh (opt-in) | mesh_peers, mesh_send, mesh_inbox, mesh_wait, mesh_state. Needs the grant's mesh scope and `slm remote keys allow KEY mesh` |
| Media (opt-in) | remember_media, get_media, remember_document, media_status, media_upload_link. Needs the grant's media scope and `slm remote keys allow KEY media`; saving also needs the write scope |
| Host only | The 45 tools classified as host-only: switching profile, maintenance, retention, decay, `forget`, the code graph, `slm_loop_run`, `mesh_lock`, `mesh_events`, `mesh_status`, `mesh_summary`, set_mode and the rest of that set. `expand` is not in it |

Web access is narrower than this table: an app connected through the dashboard reaches only the tools its approval named (recall, search, fetch and get_status always; save and session tools only if you ticked them), as set out in [Web Access and Connected Apps](Web-Access-and-Connected-Apps). On either path the daemon enforces the policy on every request. It accepts only the MCP methods `initialize`, `server/discover`, `ping`, `tools/list`, `tools/call` and the standard client notifications, matches tool names exactly, refuses a body over 1 MiB, a repeated JSON key or a batch, answers `tools/list` with only the tools the key may call, and answers a refused call as an MCP tool error with the code `remote_tool_not_allowed`. A response is capped at 4 MiB. Company mode refuses `slm remote` keys. That is an enterprise deployment, or a workspace that requires sign-in; if the mode cannot be read, the keys are refused as well ([Security, Company Mode and Teams](Security-Company-Mode-and-Teams)).

## Notes on specific tools

`slm_loop_run` blocks while it polls its gate, up to `max_wallclock_s` (default 15 seconds, at most 120), and writes every lap to memory under the tag `loop:NAME`. Its gate needs both a recall of `gate_query` that returns a confident match and an independent execution receipt named by `gate_contract`; without a valid contract it cannot reach DONE. `require_support=true` also needs the [answer check](Answer-Check) to judge the memories sufficient. Setting `SLM_LOOP_KILL` to any non-empty value stops loops. See [Bounded Loops](Bounded-Loops).

The Optimize tools cache and compress only what you route through them. They do not intercept the main conversation turn without the proxy ([Optimize](Optimize)). Do not cache secrets, and treat a `ccr_id` like a capability.

`get_memory_summary` is extractive unless the profile runs a model, and the response carries `coverage` and `source_fact_ids` because session ids are sparse; a session summary is usually partial. For `community`, `target` is the `community_id` from a recall's `thematic_context`.

`recall`, `search` and `recall_trace` follow the [Retrieval Score Contract](Retrieval-Score-Contract): `relevance_score` is query relevance, `ranking_score` is diagnostic, and `memory_confidence` belongs to the stored assertion.

The optional `observe_bounded_loop_evidence` tool needs the separate `bounded-loops-mcp` executable. Receipts are observation-only and cannot change recall or ranking.

Code: `mcp/server.py` (registration and precedence), `mcp/profiles.py`, `mcp/profile_names.py`, `mcp/agent_context.py`, `mcp/resources.py`, `mcp/tools_core.py`, `mcp/tools_active.py`, `mcp/tools_brain.py`, `mcp/tools_code_graph.py`, `mcp/tools_mesh.py`, `mcp/tools_media.py`, `mcp/tools_optimize.py`, `mcp/tools_loops.py`, `server/remote_tool_policy.py`.
