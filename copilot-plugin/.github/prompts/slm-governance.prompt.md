---
name: slm-governance
description: Enterprise compliance and governed workspace behavior for SuperLocalMemory. Covers role-based access (admin/member/viewer), retention policies, audit trail, GDPR data export/erase, and how agents must behave when operating under workspace governance. Requires power MCP profile for audit/retention tools. Agents must never bypass governance controls.
version: "3.8.1"
agent: agent
tools:
  - audit_trail
  - set_retention_policy
  - get_retention_stats
  - get_lifecycle_status
  - recall
  - search
  - remember
  - Bash
---

# slm-governance — Enterprise Compliance and Governed Workspace Behavior

SuperLocalMemory supports enterprise deployments with role-based access control,
retention policies, audit logging, and GDPR compliance tooling. This skill
documents how agents must behave when operating in a governed workspace and how
to use the governance MCP tools (available in the `power` profile).

---

## Role model

Governed workspaces have three roles:

| Role | Read | Write personal | Write shared/global | Admin operations |
|------|------|---------------|---------------------|-----------------|
| `viewer` | Yes | No | No | No |
| `member` | Yes | Yes | Yes (within access list) | No |
| `admin` | Yes | Yes | Yes (unrestricted) | Yes |

**Agent behavior by role:**

- **Viewer**: Only call `recall`, `search`, `fetch`, `list_recent`. Never call
  `remember`, `update_memory`, `forget`, or any write tool. If a write is
  attempted, fail gracefully: "This workspace is read-only in viewer mode."
- **Member**: May write personal facts and shared facts with permitted profiles.
  May NOT write `scope="global"` facts without explicit admin authorization.
  May NOT call `set_retention_policy`, `audit_trail`, or `compact_memories`.
- **Admin**: Full access including governance tools in the `power` profile.

An agent operating in a governed workspace must check its role before any write
operation. Role information is visible in workspace configuration or via
`slm status --json` (the `role` field, if present).

---

## Retention policies

Retention policies control how long facts are stored before they become eligible
for decay. Available in the `power` MCP profile.

### Set a retention policy

```
set_retention_policy(
  profile_id: str = "",   # "" = active profile
  days: int = 90,         # facts older than this become decay-eligible
  zone: str = "default",  # retention zone name
)
```

Retention zones let you apply different policies to different fact categories:

```
# Standard facts: 90-day retention
set_retention_policy(profile_id="", days=90, zone="default")

# Security findings: 365-day retention (compliance requirement)
set_retention_policy(profile_id="", days=365, zone="security")
```

Tag your facts with the zone name to route them to the right policy:
```
remember(content="Critical auth bypass in v2.1", tags="security,cve,finding", ...)
```

### Check retention statistics

```
get_retention_stats()
```

Returns zone distribution, average fact age, and decay-eligible counts. Use this
to verify policies are working as expected.

### Check lifecycle status

```
get_lifecycle_status()
```

Reports the state of the retention and decay subsystem — whether decay cycles are
running, when the next cycle runs, and any backlog.

---

## Audit trail

`audit_trail` is available in the `power` profile. It returns a structured log of
recent memory operations (writes, reads, profile switches, policy changes).

```
audit_trail(
  limit: int = 50,          # number of entries to return
  operation: str = "",      # filter by operation type (e.g. "remember", "forget")
  profile_id: str = "",     # filter by profile; "" = active profile
)
```

Use this for:
- Compliance reviews ("what data was written in the last 30 days?")
- Investigating unexpected memory changes
- Generating audit reports for data controllers

The audit trail covers MCP and CLI operations. It does not record the content of
facts by default — only operation type, timestamp, agent ID, and fact ID.

---

## GDPR compliance

### Data export

SLM does not have a dedicated MCP export tool. For GDPR data subject access
requests, use the CLI:

```bash
# Export all memories in a profile to JSON
slm status --json     # confirm active profile
slm list --limit 9999 --json > export.json
```

For a complete export including entity graph data, run:
```bash
slm status --json
```

Contact your workspace admin to arrange a full database-level export if the CLI
output is insufficient for compliance purposes.

### Right to erasure

To erase all memories for a subject or project:

```bash
# Step 1: preview what will be deleted (ALWAYS do this first)
slm forget "<subject or project name>" --dry-run --json

# Step 2: review the preview, then execute
slm forget "<subject or project name>" --yes --json
```

For targeted deletion by fact ID:
```bash
slm delete <fact_id> --yes --json
```

For data reconstruction prevention: after erasure, confirm the fact is gone by
running `slm recall "<content>"`. A successful erasure returns no results. Never
attempt to re-derive erased content from other stored facts.

---

## require-login

When `require_login` is enabled in workspace configuration, agents must
authenticate before any memory operation. SLM handles authentication at the
daemon level — agents do not need to pass credentials in tool calls. If an
agent receives an authentication error from any MCP tool, it must:

1. Stop the current operation immediately.
2. Report the authentication requirement to the user.
3. Never cache, retry, or work around the authentication block.

---

## Scope enforcement in governed workspaces

In a governed workspace, scope restrictions are enforced server-side:
- **Viewers** cannot write any fact regardless of `scope` parameter.
- **Members** cannot write `scope="global"` unless their access list includes
  the global scope — attempts return a permission error.
- **Admins** can write any scope.

Agents must not attempt to work around scope restrictions by splitting a global
fact into multiple shared facts to accumulate equivalent visibility.

---

## Compact memories (admin-only)

`compact_memories` deduplicates and consolidates stored memories. This is an
admin operation — it can change fact IDs and remove content.

```
compact_memories(
  profile_id: str = "",   # "" = active profile
  dry_run: bool = True,   # ALWAYS true first — inspect before running
)
```

Always run with `dry_run=True` first and review the impact report. Never run
compaction without admin authorization.

---

## Consistency check (admin-only)

```
consistency_check(profile_id: str = "")
```

Verifies data integrity of the memory store — checks for orphaned entities,
broken references, and index-database mismatches. Use after migrations or
unexpected shutdowns. Returns a structured report.

---

## Agent checklist for governed workspaces

Before each write operation:
- [ ] Confirm my role allows writes (viewer → skip; member/admin → proceed)
- [ ] Confirm scope is appropriate for my role (member → no global)
- [ ] Set correct tags including zone name if retention policy applies
- [ ] Pass `session_id` for full audit attribution

Before running any destructive operation (`forget`, `compact_memories`):
- [ ] Admin authorization confirmed
- [ ] Ran with `dry_run=True` and reviewed output
- [ ] GDPR: confirmed the subject or controller authorized the erasure

---

## Related skills

- `slm-scope` — scope model details (personal/shared/global)
- `slm-profile` — workspace isolation and profile switching
- `slm-remember` — fact storage reference (includes scope parameters)
- `slm-recall` — retrieval reference (includes scope read flags)
- `slm-mesh` — mesh tools (full/power profiles)

---

*SuperLocalMemory v3.8.1 · Qualixar · AGPL-3.0-or-later*
