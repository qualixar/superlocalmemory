# Profiles
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Profiles let you maintain completely isolated memory contexts. Work memories never mix with personal memories. Client A never sees Client B's data.

---

## What Profiles Are

A profile is an isolated memory namespace. Each profile has its own:

- Memories and knowledge graph
- Learned patterns and behavioral data
- Trust scores and provenance records
- Retention policies
- Audit trail

There is zero data leakage between profiles. Searching in one profile never returns results from another.

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

### Delete a profile

```bash
slm profile delete old-project
```

This permanently deletes all memories in that profile. You will be prompted for confirmation.

### Export a profile

```bash
slm profile export work > work-backup.json
```

### Import a profile

```bash
slm profile import < work-backup.json
```

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

# Done — delete it
slm profile switch default
slm profile delete experiment-graphql
```

## Profile-Specific Settings

Each profile can have its own retention policy:

```bash
slm profile switch client-acme
slm retention set gdpr-30d        # GDPR compliance for this client

slm profile switch internal
slm retention set indefinite       # Keep internal memories forever
```

## Using Profiles With CLI Commands

Most commands operate on the active profile. You can override this per-command:

```bash
# Recall from a specific profile without switching
slm recall "database config" --profile client-acme

# Store to a specific profile without switching
slm remember "Acme uses Aurora PostgreSQL" --profile client-acme
```

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
