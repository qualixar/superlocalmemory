# Profiles
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Profiles organize memory contexts inside one installation. Personal facts are
profile-scoped by default; shared and global recall are opt-in and subject to
the configured scope policy. Profiles are not an operating-system or tenant
security boundary.

---

## What Profiles Are

A profile is an isolated memory namespace. Each profile has its own:

- Memories and knowledge graph
- Learned patterns and behavioral data
- Trust scores and provenance records
- Retention policies
- Audit trail

The supported contract excludes another profile's personal facts. Release
verification must also prove authorized shared/global inclusion and
unauthorized, deleted, and archived exclusion across every retrieval surface.

## Default Profile

After installation, you have one profile called `default`. All memories go here unless you create and switch to another profile.

## Managing Profiles

### List profiles

```bash
slm profile list
```

Output:

```
Profiles:
  * default     (142 memories, active)
    work        (89 memories)
    personal    (34 memories)
    client-acme (67 memories)
```

The `*` marks the active profile.

### Create a profile

```bash
slm profile create work
slm profile create client-acme
slm profile create personal
```

### Switch profiles

```bash
slm profile switch work
```

All subsequent `remember`, `recall`, and auto-memory operations use this profile until you switch again.

From an MCP-connected agent (v3.8.0, `code`/`full`/`power` profiles):

```json
{ "tool": "switch_profile", "arguments": { "profile": "work" } }
```

The `switch_profile` MCP tool is available in the `code` (21), `full` (39), and `power` (51) profiles. It is not included in `core` (14) or `mesh` (8). See [MCP Profiles →](../README.md#mcp--profiles).

## Use Cases

### Work vs Personal

```bash
# Morning: switch to work
slm profile switch work
# Work memories are captured and recalled all day

# Evening: switch to personal
slm profile switch personal
# Personal project memories are now active
```

### Per-Client Isolation

For consultants and agencies working across multiple clients:

```bash
slm profile create client-alpha
slm profile create client-beta

# Working on Alpha's project
slm profile switch client-alpha
# Only Alpha's architecture, decisions, and context are available

# Switch to Beta
slm profile switch client-beta
# Alpha's data is completely invisible
```

### Per-Project Isolation

```bash
slm profile create mobile-app
slm profile create backend-api
slm profile create infrastructure
```

### Temporary Profiles

For experiments or short-term work:

```bash
slm profile create experiment-graphql
slm profile switch experiment-graphql
# ... do your experiment ...

# Done — switch back (the profile and its memories remain; profiles have no delete command)
slm profile switch default
```

## Profile-Specific Settings

Retention policies are set globally via `slm config set` or the `set_retention_policy`
MCP tool. To apply a policy after switching to a profile:

```bash
slm profile switch client-acme
slm config set retention.default_policy gdpr-30d   # GDPR compliance

slm profile switch internal
slm config set retention.default_policy indefinite  # Keep internal memories forever
```

All `remember` and `recall` operations run against the active profile. To work in a
different profile, switch first with `slm profile switch <name>`, then run your commands.

## How Profiles Work Internally

Each profile stores memories in the same SQLite database but with a profile identifier on every row. Queries are filtered by profile at the database level, ensuring complete isolation.

The entity graph, BM25 index, and all math layer state are also per-profile. Building the graph for one profile does not affect another.

## Limits

- No hard limit on the number of profiles
- Each profile adds minimal overhead (a few KB for metadata)
- Performance is determined by per-profile memory count, not total profiles
- Switching profiles is instant (no data loading required)

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
