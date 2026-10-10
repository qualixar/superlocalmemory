---
name: slm-governance
description: Governed-workspace behavior for SuperLocalMemory. Covers roles (admin/member/viewer) and company mode, who administers a company-mode workspace (a signed-in admin, or `slm team` on the SLM computer), `slm token show` in strict mode, what remote web apps can never see or change, retention and lifecycle settings, the audit trail, GDPR export and erasure, and how agents must behave when operating under workspace governance. The audit and retention tools need the power MCP profile. Agents must never bypass governance controls.
version: "4.1.25"
agent: agent
tools:
  - audit_trail
  - set_retention_policy
  - get_retention_stats
  - get_lifecycle_status
  - compact_memories
  - consistency_check
  - recall
  - search
  - remember
  - Bash
---

# slm-governance — Governed Workspace Behavior

SuperLocalMemory can run with named users and roles per workspace (company
mode), keeps a compliance audit trail, applies a retention lifecycle, and ships
GDPR export and erasure commands. This skill documents how an agent must behave
in a governed workspace and what the governance tools really do. The MCP audit
and retention tools are in the `power` tool set (see `slm-profile`).

---

## Role model

By default SLM is single-user: whoever runs it is the owner and nothing asks for
a login. Company mode adds named users, each with one role per workspace
(profile). A role in one workspace grants nothing in another.

| Role | Read | Write | Share (`shared`/`global` writes) | Delete | Manage users and settings |
|------|------|-------|-------|--------|-----------------|
| `viewer` | Yes | No | No | No | No |
| `member` | Yes | Yes | Yes | No | No |
| `admin` | Yes | Yes | Yes | Yes | Yes |

**Agent behavior by role:**

- **Viewer**: Only call `recall`, `search`, `fetch`, `list_recent` and other
  reads. Never call `remember`, `update_memory`, `delete_memory` or any write
  tool. If a write is refused, say so: "This workspace is read-only for my role."
- **Member**: May write memories, including `scope="shared"` and `scope="global"`
  when the user explicitly asks for them. May not delete memories or change
  users, roles or settings.
- **Admin**: Everything above plus deletion and user administration.

No MCP tool or `slm` command reports your role (only a signed-in dashboard session can ask the daemon, at `GET /api/rbac/whoami`). Do not guess it: attempt
what the user asked and treat a permission refusal as final. Setting up users,
roles and "Require login" is done in the dashboard under **Settings → Access**;
there is no `slm user` or `slm role` command. The only company-mode commands
are `slm team status` and `slm team policy` (next section).

Roles apply to every route that touches data, not only to `remember` and
`recall`. In 4.1.25 that includes pictures, documents, connected folders and
turning features on: reading a file by path, connecting a folder
(`slm sources add`), turning on pictures and documents and cleaning up with
`slm media gc --apply` or `slm media repair` need the owner or an admin. SLM's
own data folder can never be named as a path or a source. With login
required, every page of data (export, compliance records, profiles, learning
and trust details) needs a signed-in user; a test walks every route so a new
one cannot slip through.

---

## Who administers a company-mode workspace

While users are enrolled and login is required, the install token or an API
key alone no longer administers the workspace: any local program can fetch the
install token, so it would let anyone switch company mode off. Administration
needs one of these:

- **A signed-in admin** (a user with the `admin` role on that workspace),
  through the dashboard.
- **`slm team` on the SLM computer**, which uses a private capability file that
  only the user running SLM can read, so the owner can never be locked out
  even if every admin is:

```bash
slm team status                          # "Require login: on. Users: N."  (read-only, safe)
slm team policy --require-login on       # every user signs in
slm team policy --require-login off      # the machine owner is the user again
```

Run them as the same user that runs SLM, with the daemon running. Otherwise the
daemon refuses them. Users, roles and sessions are kept when login is turned
off. Switching the requirement off lowers protection for everyone, so do it
only when the user asks for it in this conversation; `slm team status` first.
With no users enrolled yet, the machine owner can still create the first
administrator.

### Strict mode: `slm token show`

With `SLM_REQUIRE_CREDENTIALS=1`, the dashboard is not handed the SLM key; the
person pastes it into a one-time box. `slm token show` prints that key. It is a
secret that opens write access: do not run it on your own, never paste it into
chat, a memory or a log, and tell the user to run it in their own terminal.
`slm rotate-token` replaces it (then `slm restart`).

### What remote web apps can never do

A connected web app (see `slm-web-access`) cannot correct, delete, share or
switch profile, cannot read another profile, and cannot change, delete, pin or
replace a memory it is not allowed to see: it is answered as if that memory did
not exist. It never sees memories that came from a connected folder, and sees
a picture or page only when its text held no secret or personal data.

---

## require-login

When login is required, every operation on memories needs a signed-in user,
including the connection your assistant uses. SLM handles this at the daemon
level; agents do not pass credentials in tool calls. If a tool call returns an
authentication or permission error, you must:

1. Stop the current operation immediately.
2. Report the requirement to the user.
3. Never cache, retry, or work around the block.

In this mode, `include_global=True` and `include_shared=True` on `recall` are
quietly turned off while the recall policy forbids cross-profile reads (the
default), so an opt-in recall can come back with only personal facts. Do not try
to work around it. See `slm-scope`.

---

## Retention and lifecycle

Every memory moves through lifecycle states as it goes unused: active, warm, cold,
archived. Two tools set and inspect that, and a third runs the forgetting cycle.
They need the `power` tool set.

### Set the thresholds

```
set_retention_policy(
  cold_after_days: int = 30,      # days of inactivity before a memory goes cold
  archive_after_days: int = 90,   # days before it is archived
)
```

It sets two thresholds and returns them. There is no `profile_id` argument and
there are no named retention zones or per-tag policies; tagging a memory does not
route it to a different policy.

### Look at the state

```
get_retention_stats(profile_id="")
get_lifecycle_status(limit=50, profile_id="")
```

`get_retention_stats` reports, from the retention table, the count and average
retention score per Ebbinghaus zone (`active`, `warm`, `cold`, `archive`,
`forgotten`) and the totals. `get_lifecycle_status` counts the active, warm, cold
and archived state of up to `limit` memories and returns up to ten short
samples of each. Neither says when the next cycle runs.

### Apply it

```
forget(dry_run=True)          # preview the decay cycle for the active profile
compact_memories(dry_run=True) # preview lifecycle-state transitions
```

The MCP `forget` tool is not a delete: it recomputes retention scores and moves
memories between zones. `compact_memories` moves memories whose lifecycle state
has become due (for example cold to archived); it does not merge duplicates. Both
default to a dry run. Run the preview first, show it to the user, and only then
pass `dry_run=False`. Do not run either without the user's say-so.

---

## Audit trail

```
audit_trail(limit: int = 50)
```

Returns the newest `limit` rows of the compliance audit for the active profile,
as `{"success", "entries", "count"}`. Each entry has `audit_id`, `profile_id`,
`action`, `target_type`, `target_id`, `details` and `timestamp`. There are no
filter arguments. Entries are compliance actions (store, retrieve, delete, export
and the like).

Use it for compliance reviews, for investigating an unexpected change, and for
audit reports to a data controller.

---

## GDPR

```bash
slm gdpr status [--profile P] [--json]            # posture: receipts, audit counts, known gaps; read-only
slm gdpr export --profile P [--output FILE] [--json]   # Art. 15/20 access and portability
slm gdpr erase --profile P --dry-run [--json]     # preview an erasure
slm gdpr erase --profile P --yes [--json]         # Art. 17 erasure, IRREVERSIBLE
slm gdpr verify --receipt-id ID [--profile P]     # check an erasure receipt (exit 0 ok, 1 tampered, 2 not found)
```

`slm gdpr erase` erases a **profile**. It refuses to run without both `--profile`
and `--yes`, and without them it only previews. It is the only irreversible
operation here, so confirm the subject, the profile and the authority to erase
with the user before running it. `slm gdpr status` lists the known gaps (for
example backups and the code graph) rather than claiming completeness.

To remove individual memories rather than a whole profile, use the deletion
commands from `slm-remember`:

```bash
slm forget "<subject or project name>" --dry-run --json   # ALWAYS first
slm forget "<subject or project name>" --yes --json
slm delete <fact_id> --yes --json
```

Memories that belong to a reviewed correction cannot be deleted this way; the
refusal names the case. After an erasure, confirm with `slm recall "<content>"`
that nothing comes back, and never try to re-derive erased content from other
stored facts.

---

## Scope enforcement in governed workspaces

- **Viewers** cannot write anything, whatever the `scope` argument.
- **Members and admins** can write `shared` and `global` facts; writing either
  scope needs the share permission, which both roles hold.
- Agents must not split a `global` fact into several `shared` facts, or otherwise
  accumulate visibility the user did not ask for.

---

## Integrity checks

```
consistency_check(limit: int = 100)
```

Runs a sheaf-consistency check over up to `limit` memories and returns pairs of
facts that contradict each other, with a severity (`fact_a`, `fact_b`,
`severity`, `content_a`), plus `facts_checked`, `facts_errored` and
`total_contradictions`. If the checker is disabled it says so in `note`. For the
health of the store itself (orphan rows, erased words left behind, unfinished
deletes) use `slm db integrity` and `slm db repair`; see `slm-status`.

---

## Agent checklist for governed workspaces

Before each write operation:
- [ ] The user asked for it, and a refusal from the workspace is respected
- [ ] Scope is `personal` unless the user explicitly asked to share
- [ ] Pass `session_id` so the write is attributed

Before running any destructive or state-changing operation (`forget`,
`compact_memories`, `slm forget --yes`, `slm gdpr erase --yes`):
- [ ] The user authorized it in this conversation
- [ ] Ran the dry run or preview and showed it to the user
- [ ] GDPR: confirmed the subject or controller authorized the erasure

---

## Related skills

- `slm-scope` — scope model details (personal/shared/global)
- `slm-profile` — memory profiles and tool sets
- `slm-remember` — fact storage, corrections and deletion
- `slm-recall` — retrieval reference (includes scope read flags)
- `slm-mesh` — mesh tools
- `slm-media` — pictures, documents and folders, and the roles that gate them
- To use this memory from a web assistant or another computer, see the slm-web-access skill.

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
