---
name: slm-cli
description: Use SuperLocalMemory from the shell when its MCP tools are not available to you. Every MCP tool runs as `slm tool <name>` with the same arguments, checks and JSON result, and the session protocol (start, recall before answering or searching, cue check before editing, remember decisions, pictures and PDFs, close) has a command for each step. Includes the full tool-to-command table.
version: "4.2.33"
agent: agent
tools:
  - Bash(slm:*)
---

# slm-cli — SuperLocalMemory from the shell

When the SuperLocalMemory MCP server is not available to the agent, the
command line is. `slm tool` hosts the same tools the MCP server hosts: the
names, input schemas, descriptions and permission checks come from the one
registration both use, and a call reaches the local SLM service the same way
an MCP call does. Decisions made this way are recorded on the `cli` channel.
Your agent name comes from `SLM_AGENT_ID`, never from an argument.

```bash
slm tool list                      # every tool, one line each (--json for data)
slm tool describe recall           # description and input schema (--json)
slm tool recall --query "which database does project X use" --limit 5
slm tool remember --json '{"content": "Project X uses PostgreSQL 16", "tags": "project-x"}'
echo '{"query": "auth flow"}' | slm tool recall -       # arguments on stdin
slm tool recall --args-file args.json                    # arguments in a file
```

A call prints the tool's JSON result on stdout. Exit code 0: the tool
succeeded. 1: the tool answered with an error or a refusal (the JSON says
why). 2: the call could not be made (unknown tool, an argument the tool does
not take, a wrong type). `--key value` sets one argument (`--profile-id` and
`--profile_id` both mean `profile_id`); numbers, booleans and lists are read
as JSON. `--pretty` indents the output. `--image-dir DIR` writes returned
images to files.

---

## The session protocol, by command

| Step | Command |
|---|---|
| Session start (once) | `slm session-context "<topic>" --project-dir "$PWD" --json` |
| Before a non-trivial answer | `slm recall "<the question>" --json` |
| Before a web search | `slm recall "<what you are about to look up>" --json` (the answer may already be stored) |
| Before editing a file (cue check) | `printf '%s' '{"hook_event_name":"PreToolUse","tool_name":"Edit","tool_input":{"file_path":"/abs/path/file.py"},"cwd":"'"$PWD"'"}' \| slm hook cue` |
| After a decision or rule | `slm remember "<the decision>" --kind decision --cue path:src/module/**` (`--kind rule`, `status`, `procedure`, `correction`) |
| A picture | `slm tool remember_media --path /abs/screenshot.png --content "<why it matters>"` |
| A PDF | `slm tool remember_document --path /abs/spec.pdf`, then `slm tool media_status --job-id <job_id>` |
| See a saved picture | `slm tool get_media --media-id <id> --image-dir /tmp/slm-images` |
| Messages from other agents | `slm tool mesh_inbox --session "$SLM_SESSION_ID"` |
| Session end | `slm session close` |

The cue check prints `{}` when no memory is tied to that file, else the
memories to read before the edit (as notes, not instructions). Rules and
decisions saved with `--kind` load at the start of later sessions.

**Mesh identity.** A mesh tool needs a session: `--session <id>` or
`SLM_SESSION_ID`, else one shared by every command run by the same agent in
the same folder. A command-line session is reachable while it keeps calling
mesh tools (the service forgets a session after 30 minutes without one);
for messages that must wait longer, send to `project:<path>` or `broadcast`.

**Tool sets.** `slm tool` reaches every tool. `--tool-set core|code|full|power|mesh`
narrows it to one MCP tool set, and `--tool-set mcp` to what an MCP
connection on this computer would see now. Pictures and documents answer
with the reason when they are turned off (`slm media status`).

---

## Every tool, by command

Required arguments are shown; `slm tool describe <name>` lists the rest.

### Memory

| Tool | Command | What it does |
|---|---|---|
| `backup_status` | `slm tool backup_status` | Get backup system status, last backup time, and available backup files. |
| `build_graph` | `slm tool build_graph` | Rebuild knowledge graph edges for all facts in the active profile. |
| `correct_pattern` | `slm tool correct_pattern --pattern-id <pattern_id> --correction <correction>` | Correct or annotate a learned behavioral pattern to improve retrieval. |
| `delete_memory` | `slm tool delete_memory --fact-id <fact_id>` | Delete a specific memory by exact fact ID. |
| `expand` | `slm tool expand --memory-id <memory_id>` | Drill into one recall result: the full fact plus, at depth 2, its |
| `fetch` | `slm tool fetch --fact-ids <fact_ids>` | Fetch full details for specific fact IDs (comma-separated or a list). |
| `get_attribution` | `slm tool get_attribution` | Get system attribution: author, version, license, and provenance metadata. |
| `get_learned_patterns` | `slm tool get_learned_patterns` | Get learned behavioral patterns (interests, refinements, archival habits). |
| `get_memory_summary` | `slm tool get_memory_summary` | Summarise your memories: a day, a project, a session, or a community. |
| `get_status` | `slm tool get_status` | Get memory system status: fact count, entity count, mode, profile, db size. |
| `list_corrections` | `slm tool list_corrections` | List correction cases for a human or host reviewer. |
| `list_disputes` | `slm tool list_disputes` | What needs an answer: open disputes and changed source files. |
| `list_recent` | `slm tool list_recent` | List most recently stored memories, newest first. |
| `manage_view` | `slm tool manage_view --action <action> --name <name>` | Create, rename or delete a saved view. No memory is ever changed. |
| `memory_used` | `slm tool memory_used` | Get memory usage breakdown by fact type and lifecycle state. |
| `prestage_context` | `slm tool prestage_context --query <query>` | Proactively return top-K memories for a query. |
| `recall` | `slm tool recall --query <query>` | Search memories through hybrid retrieval, RRF fusion, and reranking. |
| `remember` | `slm tool remember --content <content>` | Store content to memory with intelligent indexing. |
| `review_correction` | `slm tool review_correction --case-id <case_id> --action <action> --expected-version <expected_version>` | Apply, reject, or roll back a review-gated correction case. |
| `run_view` | `slm tool run_view` | Run one of your saved views, or list them. |
| `search` | `slm tool search --query <query>` | Full-text search across memories using FTS5 with BM25 ranking. |
| `settle_dispute` | `slm tool settle_dispute --dispute-id <dispute_id> --keep <keep>` | Settle one dispute from ``list_disputes`` or recall, or take a decision back. |
| `switch_profile` | `slm tool switch_profile --profile-id <profile_id>` | Switch the active memory profile. All operations scope to this profile. |
| `update_memory` | `slm tool update_memory --fact-id <fact_id> --content <content>` | Update the content of a specific memory by exact fact ID. |

### Memory kinds

| Tool | Command | What it does |
|---|---|---|
| `confirm_memory_kinds` | `slm tool confirm_memory_kinds --items <items>` | Confirm kinds for 1-200 facts at once. |
| `memory_kinds_status` | `slm tool memory_kinds_status` | Memory-kind status for the active profile (or ``profile_id``): counts |
| `review_memory_kinds` | `slm tool review_memory_kinds` | List memory-kind suggestions awaiting confirmation. |
| `set_memory_kind` | `slm tool set_memory_kind --fact-id <fact_id> --kind <kind>` | Set (confirm) the kind of one memory you already know the type of. |

### Sessions and learning

| Tool | Command | What it does |
|---|---|---|
| `close_session` | `slm tool close_session` | Close the current session and create temporal summary events. |
| `contradict_assertion` | `slm tool contradict_assertion --assertion-id <assertion_id>` | Contradict a behavioral assertion (decrease confidence). |
| `core_memory` | `slm tool core_memory --action <action>` | Manage the explicit Core Memory pin set (v3.4.65). |
| `evolve_skill` | `slm tool evolve_skill --skill-name <skill_name>` | Manually trigger evolution for a specific skill. |
| `finalize_cognitive_turn` | `slm tool finalize_cognitive_turn --receipt-id <receipt_id> --outcome <outcome>` | Finalize an active-profile cognitive turn with outcome evidence. |
| `get_assertions` | `slm tool get_assertions` | Get learned behavioral assertions. |
| `get_brain_evidence_status` | `slm tool get_brain_evidence_status` | Get profile-scoped, observation-only Living Brain evidence totals. |
| `log_tool_event` | `slm tool log_tool_event --tool-name <tool_name>` | Log a tool usage event for behavioral learning. |
| `observe` | `slm tool observe --content <content>` | Observe conversation content for automatic memory capture. |
| `observe_bounded_loop_evidence` | `slm tool observe_bounded_loop_evidence --workspace <workspace>` | Import one explicit, read-only snapshot from installed Bounded Loops. |
| `observe_bounded_loop_execution_learning` | `slm tool observe_bounded_loop_execution_learning --workspace <workspace>` | Ingest negotiated bridge-v2 terminal evidence into the execution plane. |
| `record_agent_experience` | `slm tool record_agent_experience --payload <payload>` | Record a contract-validated terminal experience receipt. |
| `record_cognitive_turn` | `slm tool record_cognitive_turn --payload <payload>` | Open one contract-validated cognitive-turn provenance receipt. |
| `reinforce_assertion` | `slm tool reinforce_assertion --assertion-id <assertion_id>` | Reinforce a behavioral assertion (increase confidence). |
| `report_feedback` | `slm tool report_feedback --fact-id <fact_id>` | Report whether a recalled memory was useful. |
| `session_init` | `slm tool session_init` | Load project context, confirmed rules, pins and session learning. |
| `settle_session_outcomes` | `slm tool settle_session_outcomes --session-id <session_id>` | Settle pending recall outcomes for one exact host session. |
| `skill_health` | `slm tool skill_health` | Get health metrics for a skill or all skills. |
| `skill_lineage` | `slm tool skill_lineage` | Get evolution lineage for a skill. |

### Pictures and documents

| Tool | Command | What it does |
|---|---|---|
| `get_media` | `slm tool get_media --media-id <media_id>` | Show the thumbnail of a saved image as an image. Apps on other computers need the owner's permission. |
| `media_status` | `slm tool media_status` | How far a saved document's background work has got (``job_id``), or with ``setup=true`` where pictures, |
| `media_upload_link` | `slm tool media_upload_link` | Get a one-time link the person opens to add a picture or a PDF from their own device. |
| `remember_document` | `slm tool remember_document` | Save a PDF as a memory, from a file on this computer, an https link, base64 or a chat attachment. |
| `remember_media` | `slm tool remember_media` | Save an image as a memory, from a file on this computer, an https link, base64 or a chat attachment. |

### Agent mesh

| Tool | Command | What it does |
|---|---|---|
| `mesh_events` | `slm tool mesh_events` | Get recent mesh events (peer joins, leaves, messages, state changes). |
| `mesh_inbox` | `slm tool mesh_inbox` | Read messages sent to this session. |
| `mesh_lock` | `slm tool mesh_lock --file-path <file_path>` | Manage file locks across sessions. |
| `mesh_peers` | `slm tool mesh_peers` | List all active peer sessions on this machine. |
| `mesh_send` | `slm tool mesh_send --to <to> --message <message>` | Send a message to another peer session, broadcast, or project. |
| `mesh_state` | `slm tool mesh_state` | Get or set shared state across all sessions. |
| `mesh_status` | `slm tool mesh_status` | Get mesh broker health and statistics. |
| `mesh_summary` | `slm tool mesh_summary` | Register this session and describe what you're working on. |
| `mesh_wait` | `slm tool mesh_wait` | Wait for new messages, up to 20 seconds, then return what arrived. |

### Code graph

| Tool | Command | What it does |
|---|---|---|
| `apply_refactor` | `slm tool apply_refactor --action <action> --target <target>` | Execute a refactoring operation (stub for MVP). |
| `build_code_graph` | `slm tool build_code_graph --repo-path <repo_path>` | Build a complete code knowledge graph from a repository. |
| `code_entity_history` | `slm tool code_entity_history --code-entity <code_entity>` | Get the complete memory timeline for a code entity. |
| `code_memory_search` | `slm tool code_memory_search --code-entity <code_entity>` | Search SLM memories linked to a code entity. |
| `code_stale_check` | `slm tool code_stale_check` | Find SLM memories that reference deleted or changed code. |
| `detect_changes` | `slm tool detect_changes` | Detect code changes and compute risk scores. |
| `enrich_blast_radius` | `slm tool enrich_blast_radius --changed-files <changed_files>` | Compute blast radius PLUS institutional memory for each impacted node. |
| `find_large_functions` | `slm tool find_large_functions` | Find functions exceeding a line count threshold. |
| `get_affected_flows` | `slm tool get_affected_flows --changed-files <changed_files>` | Find execution flows impacted by file changes. |
| `get_architecture_overview` | `slm tool get_architecture_overview` | Get high-level architecture map showing communities and their relationships. |
| `get_blast_radius` | `slm tool get_blast_radius --changed-files <changed_files>` | Compute impact radius for changed files. |
| `get_community` | `slm tool get_community --community-id <community_id>` | Get detailed information about a single code community. |
| `get_flow` | `slm tool get_flow --flow-name <flow_name>` | Get detailed information about a single execution flow. |
| `get_review_context` | `slm tool get_review_context --changed-files <changed_files>` | Get token-optimized review context for changed files. |
| `link_memory_to_code` | `slm tool link_memory_to_code --fact-id <fact_id> --code-entity <code_entity>` | Manually link an SLM memory to a code graph node. |
| `list_communities` | `slm tool list_communities` | List detected code communities (clusters of related code). |
| `list_flows` | `slm tool list_flows` | List execution flows sorted by criticality or size. |
| `list_graph_stats` | `slm tool list_graph_stats` | Get code graph size and health metrics. |
| `query_graph` | `slm tool query_graph --pattern <pattern>` | Query the code graph for relationships. |
| `refactor_preview` | `slm tool refactor_preview --action <action> --target <target>` | Preview a refactoring operation without executing it. |
| `semantic_search_code` | `slm tool semantic_search_code --query <query>` | Search code entities by semantic meaning using hybrid FTS5 + vector search. |
| `update_code_graph` | `slm tool update_code_graph` | Incrementally update the code graph for changed files. |

### Context optimization

| Tool | Command | What it does |
|---|---|---|
| `slm_cache_get` | `slm tool slm_cache_get --key <key>` | Retrieve a previously cached result. |
| `slm_cache_set` | `slm tool slm_cache_set --key <key> --value <value>` | Cache a result you want to reuse (tool output, file read, search result). |
| `slm_compress` | `slm tool slm_compress --content <content>` | Compress text with one tenant-scoped, TTL-bound recovery copy. |
| `slm_optimize_stats` | `slm tool slm_optimize_stats` | Return compression and cache statistics. |
| `slm_retrieve` | `slm tool slm_retrieve --ccr-id <ccr_id>` | Retrieve original text stored during a lossy slm_compress call. |

### Bounded loops

| Tool | Command | What it does |
|---|---|---|
| `slm_loop_history` | `slm tool slm_loop_history --name <name>` | List recorded bounded-loop runs for a loop name, newest laps summarised. |
| `slm_loop_run` | `slm tool slm_loop_run --name <name> --gate-query <gate_query>` | Wait under hard bounds for independently executed verification. |
| `slm_loop_show` | `slm tool slm_loop_show --run-id <run_id>` | Show every lap of one bounded-loop run, in order, from SLM memory. |

### Maintenance and governance

| Tool | Command | What it does |
|---|---|---|
| `audit_trail` | `slm tool audit_trail` | Get compliance audit trail for the active profile. |
| `compact_memories` | `slm tool compact_memories` | Compact memory store by archiving cold/stale facts. |
| `consistency_check` | `slm tool consistency_check` | Run sheaf consistency check on stored memories. |
| `consolidate_cognitive` | `slm tool consolidate_cognitive` | Run CCQ cognitive consolidation pipeline. |
| `forget` | `slm tool forget` | Run Ebbinghaus forgetting decay cycle. |
| `get_behavioral_patterns` | `slm tool get_behavioral_patterns` | Get detected behavioral patterns for the active profile (or ``profile_id``). |
| `get_lifecycle_status` | `slm tool get_lifecycle_status` | Get lifecycle state distribution for stored memories. |
| `get_mode` | `slm tool get_mode` | Get current operating mode and its capabilities. |
| `get_retention_stats` | `slm tool get_retention_stats` | Get memory retention statistics (zone distribution, decay rates). |
| `get_soft_prompts` | `slm tool get_soft_prompts` | Get active soft prompts (auto-learned user patterns). |
| `get_version` | `slm tool get_version` | Get SuperLocalMemory version, Python version, and platform info. |
| `health` | `slm tool health` | Get system health including math layer status. |
| `list_failed_operations` | `slm tool list_failed_operations` | List all failed, stuck, or degraded memory operations. |
| `quantize` | `slm tool quantize` | Run EAP quantization cycle. |
| `reap_processes` | `slm tool reap_processes` | Find and kill orphaned SLM processes. |
| `recall_trace` | `slm tool recall_trace --query <query>` | Recall with per-channel score breakdown. |
| `report_outcome` | `slm tool report_outcome --memory-ids <memory_ids> --outcome <outcome>` | Report outcome of using recalled memories. |
| `resolve_operation` | `slm tool resolve_operation --operation-id <operation_id> --action <action>` | Resolve a stuck, failed, or degraded operation. |
| `run_maintenance` | `slm tool run_maintenance` | Run all periodic maintenance tasks in a single call. |
| `set_mode` | `slm tool set_mode --mode <mode>` | Switch operating mode (a, b, or c). |
| `set_retention_policy` | `slm tool set_retention_policy` | Set memory retention policy thresholds. |

112 tools. Generated from the MCP tool registry by `python scripts/build_cli_skill.py`; do not edit by hand.
