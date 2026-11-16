# Profiles

The word "profile" means two unrelated things in SLM, and mixing them up is the most common source of confusion. A memory profile is a namespace for memories: work, personal, one per client. An MCP tool profile is a named list of which tools an AI client can see. Switching one never changes the other. This page covers both, and then the three memory scopes (personal, shared, global) that decide whether one memory profile can see another's memories.

## Memory profiles

A memory profile is a namespace inside one database. Every memory row carries a profile id and every query filters on it, so recall in one profile does not return another profile's personal memories. Each profile has its own memories, knowledge graph, learned patterns, saved views and audit trail. After installation there is one profile, `default`, and everything goes there until you create and switch to another.

Profiles are not a security boundary between people. They all live in the same SQLite database on the same computer. If you need separation between users, that is [Company mode](Security-Company-Mode-and-Teams), which adds sign-in and per-workspace roles on top of profiles.

```bash
slm profile list                 # add --json for structured output
slm profile create work
slm profile switch work
```

A name must start with a letter or digit, and after that may use only letters, digits, underscore and hyphen, up to 50 characters. Switching sets the active profile for the whole installation: the CLI, the dashboard and every connected agent use it until you switch again. The choice is persisted in both `config.json` (`active_profile`) and `profiles.json`. From an MCP client, the same switch is `switch_profile`.

```json
{ "tool": "switch_profile", "arguments": { "profile_id": "work" } }
```

### Using another profile for one call

Most memory tools accept a `profile_id`. A non-empty value routes that single call to the named profile and leaves the active profile alone. The profile must already exist: a name that matches no profile is refused with `unknown_profile`, never created and never treated as empty. If the background service is not running, a named profile cannot be served and the call fails with `routing_needs_daemon`; `slm restart` starts it. This is how an agent works in a second workspace without moving everyone else.

### Deleting a profile

Deletion is available in the dashboard's profile list, not on the CLI. It cannot delete `default` or the profile that is active. Memories, facts, the graph and the indexes that make them findable move into `default` in one database transaction. If that transaction fails, the profile row is still there. Saved views, Answer Check history and learned state are cleared before that transaction, so a later failure does not put them back. Pictures and documents move after the transaction commits. If that move fails, the profile is still deleted, the move is recorded, and it finishes at the next daemon start or before the next profile is created.

Not everything moves. Saved views, Answer Check history, the receipts of how agents used that workspace, and what the workspace learned (its ranker, bandit and signals) are removed. Folding that learned state into `default` would retrain it. Pending memories that were not stored yet move with the rest. Write receipts and erasure receipts stay, still labelled with the deleted profile, and so do the markers that refuse a late write for that profile. If you want the views or the Answer Check history, export or note them before you delete.

### Profiles and remote access

A key for `slm remote`, the listener for other computers on your network ([Web Access and Connected Apps](Web-Access-and-Connected-Apps)), is bound to exactly one profile. `slm remote keys add <name>` binds it to the profile active at that moment; add `--profile <profile>` to name a different one. A key that still has no profile is refused. Keys made before this binding are bound once, to the profile active when the daemon next starts or when any `slm remote keys` command runs. Every remote tool call is served for the key's profile, whatever profile the computer is using at that moment. A `profile_id` on the call that names a different profile is refused.

## Scopes: how one profile can see another's memories

Each memory has a scope.

| Scope | Who can recall it | How to set it |
|---|---|---|
| `personal` (default) | Only the owning profile. | Nothing, it is the default. |
| `shared` | The owner and the profiles listed in `shared_with`. | `--scope shared --shared-with a,b` |
| `global` | Every profile on the machine. | `--scope global` |

Visibility is a separate, second switch with both flags off by default: `include_global` adds other profiles' global memories to a recall, and `include_shared` adds memories other profiles shared with you. Your own personal memories are always returned, and another profile's personal memories are never returned whatever the flags say. The design is default-deny: a fresh install behaves as if sharing did not exist, and turning the flags on can only add rows, never remove or reorder your own results.

```bash
slm remember "Deploy freeze starts Friday" --scope global
slm remember "Q3 roadmap draft" --scope shared --shared-with alice,bob
slm recall "deploy freeze" --include-global --include-shared
slm recall "private note" --no-global --no-shared
```

Leaving a flag unset means "use the configured default"; an explicit true or false overrides it for that one call. Set the defaults persistently with `slm config set scope.default_scope personal`, `scope.recall_include_global true` and `scope.recall_include_shared true`, or in Settings under "Memory Scope". An invalid value is ignored with a warning and the safe default applies. Every read path (`recall`, `search`, `list_recent`, the recent-memories resource) honours this. [Answer Check](Answer-Check) runs after scope filtering, on the results the call already produced, so it never sees a memory the caller could not.

Two limits to know. Cognitive-consolidation summary blocks are always `personal` for now. A consolidated summary takes its cluster's scope only when the whole cluster agrees; a mixed cluster yields a `personal` summary, the most restrictive choice.

## MCP tool profiles

An MCP tool profile controls how many of SLM's 112 tools an AI client sees. Many editors cap how many tools they will load from all servers together, and registering every SLM tool can crowd out other servers. The profile is chosen when the MCP server starts, with the environment variable `SLM_MCP_PROFILE` or `slm connect <ide> --profile <name>`, which writes it into the IDE's MCP block. It cannot be changed mid-session, and the dashboard's "MCP & Tools" pane shows the active profile but only gives instructions for changing it.

| Profile | Tools | What it is for |
|---|---:|---|
| `core` | 21 | The smallest useful surface: remember, recall, search, fetch, expand, list_recent, update_memory, forget, the session tools, the five Optimize tools, correction review and listing, summaries and `switch_profile`. |
| `code` | 41 | Core plus the portable Brain evidence tools, memory-kind tools, six code-graph tools, the three bounded-loop tools, `report_outcome` and `report_feedback`. |
| `full` | 60 (63 with images on) | Everyday memory, Brain evidence, kinds, saved views, loops, skill evolution, Optimize, learning signals, and the nine mesh tools. |
| `power` | 72 (75 with images on) | Full plus `get_version`, `get_mode`, `health`, `consistency_check`, `recall_trace`, lifecycle and retention tools, `compact_memories`, `get_behavioral_patterns`, `audit_trail`, `quantize`, `get_retention_stats`. |
| `mesh` | 9 | Only the nine `mesh_*` coordination tools. |
| `whole` | 112 | The raw server with every tool. `SLM_MCP_ALL_TOOLS=1` does the same. |

The "with images on" counts add the five picture and document tools (`remember_media`, `get_media`, `remember_document`, `media_status`, `media_upload_link`) when [Pictures and Documents](Pictures-and-Documents) is enabled. The counts come from `mcp/profiles.py` and were recomputed by executing that file.

With no selection at all, SLM serves a default set that is identical to `full`. The Claude Code and Codex plugins set only an agent id, so they get that default set unless you add `SLM_MCP_PROFILE=code`. `SLM_MCP_TOOLS=a,b,c` gives a custom allowlist, and the precedence is `SLM_MCP_ALL_TOOLS`, then the allowlist, then the profile, then the default. An unknown profile name stops `slm mcp` with exit code 2 and lists the valid names; old count-suffixed names such as `core14` or `full50` still resolve, with a deprecation warning.

Two things surprise people. `switch_profile` is in `core`, `code`, `full` and `power` but not `mesh`. And the description text for `core` ("store, recall, search, sessions") undersells what the set actually contains. The full tool list per profile is on [MCP Tools](MCP-Tools).

Code: `core/profiles.py`, `server/routes/profiles.py`, `server/routes/helpers.py` (`delete_profile_from_db`), `storage/profile_fold.py`, `mcp/request_profile.py`, `mcp/tools_core.py` (`switch_profile`), `core/config.py` (`ScopeConfig`), `storage/database.py` (scope clauses), `server/remote_profile_binding.py`, `mcp/profiles.py`, `mcp/profile_names.py`, `mcp/server.py`.
