---
name: slm-profile
description: Workspace isolation and runtime profile switching for SuperLocalMemory. Each profile is a fully independent memory namespace — separate facts, code graphs, and tool sets. Use switch_profile (MCP, requires code/full/power profile) to change the active workspace without restarting. Check the active profile with slm status. Required when working across multiple projects, clients, or tenants.
when_to_use: |
  - "Switch to my work profile"
  - "Use the code profile for this session"
  - "What profile am I currently in?"
  - "I need mesh tools — switch to full profile"
  - "Load the devops workspace"
  - Multi-project workflows needing memory isolation
  - Switching between personal and team workspaces
  - Enabling additional MCP tools by activating a richer profile
allowed-tools: switch_profile, Bash
---

# slm-profile — Workspace Isolation and Profile Switching

A profile is a fully isolated memory workspace. Each profile has its own:
- Memory facts (nothing bleeds across profiles by default)
- Code graph index (separate per repo/project)
- Active MCP tool set (determined by profile tier)
- Retention policy and decay settings

Profiles are the right tool when you have genuinely separate contexts: a personal
project, a client engagement, a production vs staging environment.

---

## Available profiles and their tool sets

| Profile | Tools | When to use |
|---------|-------|-------------|
| `core` | 14 tools — remember, recall, search, session, optimize | Minimal footprint, no code tools |
| `code` | 24 tools — core + 6 code-graph tools + switch_profile + 3 bounded-loop tools | Default for IDE/coding agents |
| `full` | 42 tools — code + all memory ops + mesh + bounded loops | Multi-session, team workflows |
| `power` | 54 tools — full + governance + behavioral tools | Enterprise, admin, audit use cases |
| `mesh` | 8 tools — mesh coordination only | Lightweight cross-session signalling |

The profile is set at MCP server startup via `SLM_MCP_PROFILE` in the MCP config.
`switch_profile` lets you change it at runtime without a restart.

---

## Checking the active profile

```bash
slm status --json
```

The `profile` field in the output is the currently active profile name.

Or via MCP (works in any profile):

```bash
slm status
```

---

## Switching profiles at runtime (v3.8.0+)

`switch_profile` is available in `code`, `full`, and `power` profiles.

```
switch_profile(
  profile: str,   # one of: "core", "code", "full", "power", "mesh"
)
```

### Example: activate full profile to access mesh tools

```
# You started in code profile but need mesh coordination
switch_profile(profile="full")

# Now mesh_peers, mesh_send, mesh_inbox, etc. are available
mesh_peers()
```

### Example: switch workspaces

```
# Switch from personal to work workspace
switch_profile(profile="work-project")
```

Wait — profile names and workspace names are distinct concepts:
- **Profile tier** (`core`, `code`, `full`, `power`, `mesh`) controls which MCP
  tools are registered.
- **Workspace / data directory** (`SLM_DATA_DIR`) controls which memory database
  is used.

`switch_profile` changes the **tool tier** within the current workspace. To
switch to a completely different memory database (workspace), you need to change
`SLM_DATA_DIR` — this requires restarting the MCP server or using a separately
configured MCP server instance.

---

## Configuring the initial profile

In your `.mcp.json` (Claude Code) or `.codex/config.toml` (Codex):

```json
"env": {
  "SLM_MCP_PROFILE": "code",
  "SLM_DATA_DIR": "~/.superlocalmemory"
}
```

Profile aliases from older versions still resolve: `code20` → `code`,
`full38` → `full`, `power50` → `power`. Stale configs get a startup warning
but continue to work.

---

## Profile isolation guarantees

- `recall` returns only memories in the active profile (plus opt-in shared/global
  facts — see `slm-scope`).
- `remember` writes to the active profile only unless `scope="shared"/"global"`.
- Code graph tools (`build_code_graph`, `get_blast_radius`, etc.) index into the
  active profile's graph store.
- KV cache entries are namespaced per profile — switching profiles gives you a
  clean cache.

---

## Multiple concurrent profiles

You can run multiple SLM MCP server instances simultaneously, each pointed at a
different `SLM_DATA_DIR`, to serve different workspaces in the same IDE session.
Name them differently in your MCP config (e.g. `superlocalmemory-personal` and
`superlocalmemory-work`) and route tool calls to the appropriate server.

---

## Related skills

- `slm-scope` — opt-in fact sharing across profiles (personal/shared/global)
- `slm-graph` — code-graph tools available in code/full/power profiles
- `slm-mesh` — mesh tools available in full/power/mesh profiles
- `slm-status` — check active profile name and tool inventory
- `slm-governance` — enterprise role-based access to profiles

---

*SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later*
