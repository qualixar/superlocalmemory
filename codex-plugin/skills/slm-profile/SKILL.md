---
name: slm-profile
description: Memory profiles (separate namespaces inside one SuperLocalMemory store) and MCP tool sets (which tools your host sees). Use switch_profile or slm profile switch to change the active memory profile, the profile_id argument to read or write another profile for one call, and SLM_MCP_PROFILE to choose a tool set. Required when working across multiple projects, clients, or tenants.
when_to_use: |
  - "Switch to my work profile"
  - "What profile am I currently in?"
  - "Create a profile for this client"
  - "Read the work profile once without leaving this one"
  - "I need mesh or code-graph tools" (a tool-set question, not a profile switch)
  - "Why can't I see tool X?"
  - Multi-project workflows needing memory isolation
  - Switching between personal and team workspaces
allowed-tools: switch_profile, get_status, Bash
---

# slm-profile — Memory Profiles and Tool Sets

Two different things are called "profile" in SuperLocalMemory. Keep them apart.

- A **memory profile** is a namespace inside one store. It has its own memories,
  knowledge graph, learned patterns, retention settings and audit trail. A fresh
  install has one, called `default`. This is what `switch_profile`,
  `slm profile` and the `profile_id` argument are about.
- An **MCP tool set** (`SLM_MCP_PROFILE`: `core`, `code`, `full`, `power`,
  `mesh`) decides which tools the server shows your host. `switch_profile` does
  not change it.

A memory profile organises memory. It is not a security boundary: any local
caller that can name a profile can read it, unless the user has turned on company
mode with roles. For memories that must be unreachable from another bot or user,
use a separate data directory (`SLM_DATA_DIR`) or company mode (`slm-governance`).

---

## Memory profiles

```bash
slm profile list [--json]
slm profile create <name> [--json]
slm profile switch <name> [--json]
slm status --json          # data.profile is the active profile
```

Creating a profile is a CLI action; there is no MCP tool for it. Over MCP:

```
switch_profile(
  profile_id: str,   # the name of an existing profile
)
```

`switch_profile` changes the **active profile**. The change is not private to
your session: it is written as the machine's active profile, so every later call
from every session on this machine, and the dashboard, now work in it until
someone switches again. An unknown name is refused (`Profile '<name>' does not
exist`); nothing is created. In company mode it also needs a role on the target
profile. Only switch when the user asks to move; do not switch to peek at
something.

### Reaching another profile for one call

Most tools take an optional `profile_id`: `remember`, `recall`, `search`,
`fetch`, `list_recent`, `update_memory`, `delete_memory`, `session_init`,
`close_session`, `report_outcome`, `report_feedback`, `get_status`,
`get_memory_summary`, the correction and memory-kind tools, and others. A
non-empty value serves that one call from that profile (which must already
exist) and **never moves the active profile**. An empty value means the active
profile. Prefer this to switching and switching back.

```
recall(query="rollout plan", profile_id="work", session_id="<sid>")
remember(content="...", tags="ops", profile_id="client-acme")
```

`forget` and the mesh tools take no `profile_id`; they act on the active profile.

### What a profile isolates

- `recall` returns only the profile's own memories, plus shared or global ones
  only when you opt in (see `slm-scope`).
- `remember` writes to the profile you name or the active one.
- The code graph is not partitioned by profile: it lives in `code_graph.db` in the data directory.
- The `slm_cache_*` key-value cache and `slm_compress` recovery store are keyed
  by the calling agent, not by profile, so switching profiles does not give you
  a clean cache (see `slm-cache`).

### Separate stores

To keep two bots or two clients completely apart, give each its own
`SLM_DATA_DIR` in its MCP config. You can run several SLM MCP servers at once,
named differently (for example `superlocalmemory-personal` and
`superlocalmemory-work`), each pointed at its own directory.

---

## MCP tool sets

| Profile | Tools | What it is |
|---------|-------|------------|
| `core` | 18 tools — remember, recall, search, fetch, list_recent, update_memory, forget, session, optimize, corrections, summaries, switch_profile | Smallest set; the Grok Bot / Cursor plugin uses it |
| `code` | 38 tools — core + portable Brain evidence, report_outcome/report_feedback, 6 code-graph tools, memory kinds, bounded loops | For coding agents that need the graph; no mesh, no `get_status` |
| `full` | 57 tools — everyday memory, delete_memory, get_status, observe, saved views, learning tools, skills, optimize, kinds, loops, mesh | Same set as the no-profile default |
| `power` | 69 tools — full + audit_trail, retention, compaction, consistency_check, behavioral and diagnostic tools | Governance and admin work |
| `mesh` | 9 tools — mesh coordination only | Lightweight cross-session signalling |

A host that sets no profile gets the 57-tool `full` set (this is what the Claude
Code and Codex plugins do; the Antigravity plugin sets `power`). While pictures
and documents are on (`slm media status`), `full` lists 62 tools and `power` 74:
the five picture tools `remember_media`, `get_media`, `remember_document`,
`media_status` and `media_upload_link` are added to both, and never to `core`,
`code` or `mesh` (see `slm-media`). The server reads this when it starts. Two
more environment variables widen or narrow it: `SLM_MCP_ALL_TOOLS=1` registers
every tool (109), and `SLM_MCP_TOOLS=remember,recall,...` registers exactly the names
listed. `SLM_MCP_PROFILE=whole` is also every tool. The code-graph tools beyond
the six in `code` (such as `update_code_graph`, `list_graph_stats`) and tools
such as `core_memory` and `settle_session_outcomes` exist only with
`SLM_MCP_ALL_TOOLS=1`, `whole` or an explicit `SLM_MCP_TOOLS` list. Tools whose
function is to manage the SLM computer are never available to a remote caller.

The tool set is read when the MCP server starts. To change it, set it in the
host's MCP config and restart the host; asking the server at runtime does not
work:

```json
"env": {
  "SLM_MCP_PROFILE": "code",
  "SLM_AGENT_ID": "codex"
}
```

`slm connect <ide> --profile <name>` writes that variable into a supported
host's config. Setting `SLM_MCP_PROFILE` to a name that does not exist stops
`slm mcp` with a message listing the valid names. Older count-suffixed names
such as `code20`, `full38` and `power50` still resolve, with a startup warning.
On an install that already has memories, do not set `SLM_DATA_DIR` in the MCP
config: it points the host at a different, empty store.

`switch_profile` is in `core`, `code`, `full` and `power`, and in the default
set. It is not in `mesh`.

---

## If a tool is missing

Check the tool set first (`slm status` does not show it; look at the host's MCP
config for `SLM_MCP_PROFILE`). Mesh tools need `full`, `power`, `mesh` or the
default. Picture and PDF tools need `full`, `power` or `whole` and the feature
turned on (`slm media enable`). Code-graph tools need `code` or `SLM_MCP_ALL_TOOLS=1`. Audit and
retention tools need `power`. `report_outcome` and `report_feedback` need
anything but `core` or `mesh`.

---

## Related skills

- `slm-scope` — opt-in fact sharing across profiles (personal/shared/global)
- `slm-graph` — code-graph tools and the tool set they need
- `slm-mesh` — mesh tools and the tool set they need
- `slm-status` — check the active profile name
- `slm-governance` — roles and company mode

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
