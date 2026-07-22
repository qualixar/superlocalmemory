# Scale Engine
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

The Scale Engine manages optional CozoDB and LanceDB projections of the
canonical SQLite store. SQLite (and sqlite-vec) remain the authoritative
database; CozoDB and LanceDB are derived projections that the daemon can
use for graph queries and approximate vector search when they are available
and verified.

The staged lifecycle prevents an optional dependency from silently mutating
an active data root at daemon startup. Every projection is prepared outside
the active paths, verified against the canonical store, and only then
promoted atomically.

---

## When to Use It

You need the Scale Engine when:

- You have CozoDB or LanceDB installed and want SLM to route graph or
  vector queries through them.
- You upgraded from a pre-v3.7 install that created CozoDB/LanceDB files
  under a prior scheme and want SLM to adopt them safely.
- A failed promotion left the data root in an inconsistent state and you
  need to inspect or roll back.

If you are running a standard SLM install (SQLite + sqlite-vec only), you
do not need these commands. The daemon falls back to canonical SQLite
retrieval automatically.

---

## Lifecycle Overview

```
[status] → [adopt] → [prepare] → [verify] → [promote]
                                                 │
                                            [rollback]
```

| Action | What it does | Mutates active paths? |
|--------|-------------|:---------------------:|
| `status` | Show current lifecycle state, staged projections, and available backups | No |
| `adopt` | Confirm a detected legacy (pre-v3.7) projection and run the full prepare → verify → promote cycle automatically | Yes (via promote) |
| `prepare` | Build a new staged CozoDB/LanceDB projection from the canonical SQLite data. Output goes to a staging directory | **No** |
| `verify` | Check that the staged projection matches the canonical store (row counts, vector fingerprints, graph edges). Fails closed if parity is not met | **No** |
| `promote` | Atomically move the verified staging projection to the active paths. Creates a backup first | Yes |
| `rollback` | Restore the active paths from a named backup | Yes |

`prepare` and `verify` never mutate active paths — they are safe to run on
a live system. `promote` and `rollback` swap directories atomically; the
daemon must be restarted after a successful promotion or rollback to pick up
the new paths.

---

## Commands

### `slm db scale status`

Show the current lifecycle state, staged projection manifests, and available
backup directories.

```bash
slm db scale status
slm db scale status --json
```

Key fields in the output:

- `state` — `local_core` (SQLite only), `prepared`, `verified`, or `promoted`
- `legacy_projection_candidate` — `true` if pre-v3.7 CozoDB/LanceDB files
  are present and eligible for adoption
- `stages` — list of in-progress or rejected stage manifests
- `backups` — list of backup directory names available for rollback

### `slm db scale adopt`

Adopt a detected legacy (pre-v3.7) projection. Runs prepare → verify →
promote in one step after confirming the pre-v3.7 files are present. The
legacy directories become the rollback copy.

```bash
slm db scale adopt
```

Use this when upgrading from a pre-v3.7 install that already has CozoDB or
LanceDB files. If `legacy_projection_candidate` is `false`, the command
returns without making changes.

### `slm db scale prepare`

Build a new staged projection from the current canonical SQLite data. The
staging directory is isolated; active paths are untouched.

```bash
slm db scale prepare
```

On success, the output includes a `stage_id` — record it. You need it for
`verify` and `promote`.

### `slm db scale verify --stage-id STAGE_ID`

Verify that the staged projection at `STAGE_ID` matches the canonical store.
Checks row counts, vector fingerprints, and graph edge parity. Fails closed:
if parity is not met, the stage is marked rejected and must be re-prepared.

```bash
slm db scale verify --stage-id <stage_id>
```

`--stage-id` is required. Obtain it from the `prepare` output or
`slm db scale status`.

### `slm db scale promote --stage-id STAGE_ID`

Atomically promote a verified staging projection to the active paths. Creates
a backup before swapping. Restart the daemon after promotion so it picks up
the new paths.

```bash
slm db scale promote --stage-id <stage_id>
slm restart
```

`--stage-id` is required.

### `slm db scale rollback --backup-id BACKUP_ID`

Restore the active CozoDB/LanceDB paths from a named backup. Use
`slm db scale status` to list available backup IDs. Restart the daemon
after rollback.

```bash
slm db scale rollback --backup-id <backup_id>
slm restart
```

`--backup-id` is required. Obtain backup IDs from `slm db scale status`.

---

## Typical Upgrade Workflow (pre-v3.7 → v3.7+)

```bash
# 1. Check whether legacy files need adoption
slm db scale status

# If legacy_projection_candidate is true:
slm db scale adopt
slm restart

# 2. Verify retrieval still works
slm trace "a known memory"
slm health
```

If `adopt` fails, the scale state is reset to `local_core` and the daemon
continues using canonical SQLite — no data is lost. Inspect the rejected
stage manifest in `~/.superlocalmemory/scale-staging/` for the failure
reason.

---

## Manual Stage Workflow

Use this when you want to build and verify a projection before committing.

```bash
# Build
slm db scale prepare
# → outputs: {"stage_id": "abc123", "state": "prepared", ...}

# Verify
slm db scale verify --stage-id abc123

# Promote when verified
slm db scale promote --stage-id abc123
slm restart

# Rollback if needed
slm db scale status          # find backup_id
slm db scale rollback --backup-id <backup_id>
slm restart
```

---

## Error Recovery

If the daemon is in an inconsistent state after a failed promotion (the
`scale_engine_state` is `verified` but the promotion journal exists):

```bash
# Inspect state
slm db scale status --json

# Option A: roll back to the last good backup
slm db scale rollback --backup-id <backup_id>

# Option B: re-run from prepare
slm db scale prepare
slm db scale verify --stage-id <new_stage_id>
slm db scale promote --stage-id <new_stage_id>
slm restart
```

In all failure cases, canonical SQLite retrieval remains active — recall and
remember operations continue without interruption.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
