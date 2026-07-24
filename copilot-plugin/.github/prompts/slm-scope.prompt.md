---
name: slm-scope
description: Controls memory visibility across profiles — personal (private, default), shared (selected profiles), or global (all profiles on this machine). Default is always personal. Only change scope when the user explicitly asks to share a memory across workspaces. Works with both remember (write scope) and recall (read scope flags).
version: "3.8.2"
agent: agent
tools:
  - remember
  - recall
  - search
  - list_recent
  - Bash
---

# slm-scope — Memory Scope and Sharing Model

SuperLocalMemory is profile-isolated by default. Every profile is a fully
independent memory namespace. The scope model adds controlled, opt-in sharing
between profiles on the same machine — it is never active unless you explicitly
set it.

---

## The Three Scopes

| Scope | Visibility | When to use |
|-------|-----------|-------------|
| `personal` | Active profile only | Default — all facts. Never set explicitly. |
| `shared` | Active profile + `shared_with` profiles | Team handoffs, shared project context |
| `global` | Every profile on this machine | Machine-wide conventions, org-wide rules |

**The default is always `personal`.** A fact stored without a `scope` argument
is private to the profile that stored it. Recall without scope flags returns only
personal facts. This is identical to single-profile SLM.

---

## Opt-in is mandatory

Never set `scope="shared"` or `scope="global"` (or the recall equivalents
`include_shared`, `include_global`) unless the user explicitly asks you to share
or read across profiles. This is a hard rule — scope expansion is an action with
team-wide or machine-wide impact, and it must always be user-initiated.

---

## Writing a shared or global fact

### share with specific profiles

```
remember(
  content="API gateway rate limit is 1000 req/min per client",
  tags="api,limits,ops",
  project="platform",
  importance=8,
  session_id="<sid>",
  scope="shared",
  shared_with="devops-profile,backend-profile",
)
```

`shared_with` is a comma-separated list of profile IDs that should be able to
recall this fact. Use `slm status` to see the current profile name — that is the
`profile_id` to supply.

### share with all profiles on the machine

```
remember(
  content="All services must enforce TLS 1.3 minimum",
  tags="security,tls,global",
  project="infra",
  importance=9,
  session_id="<sid>",
  scope="global",
)
```

`scope="global"` makes the fact visible to every profile on this machine. Use
this only for machine-wide conventions (security rules, org-wide constraints).

---

## Reading shared or global facts

By default, `recall` returns only the active profile's personal facts. To surface
shared and global facts, pass the opt-in flags:

```
recall(
  query="rate limiting rules",
  limit=20,
  include_global=True,   # include global-scope facts
  include_shared=True,   # include facts shared with this profile
  session_id="<sid>",
)
```

You can opt in to either or both flags independently. Only the active profile's
permitted shared facts are returned — you cannot read another profile's personal
facts.

### CLI equivalents

```bash
# Recall with global facts included
slm recall "rate limiting rules" --include-global

# Recall with shared facts included
slm recall "rate limiting rules" --include-shared

# Both
slm recall "rate limiting rules" --include-global --include-shared

# Store a global fact
slm remember "TLS 1.3 minimum" --scope global

# Store a shared fact (shared-with is a comma-separated list of profile IDs)
slm remember "API rate limit 1000/min" --scope shared --shared-with devops,backend
```

CLI flags verified in source: `--include-global` / `--no-global`, `--include-shared` / `--no-shared`,
`--scope personal|shared|global`, `--shared-with <profile-ids>`.

---

## Configuring scope defaults

Per-profile defaults live in `mode_a/b/c.json` configuration. To make a profile
always include global facts without passing flags each time:

```json
{
  "recall": {
    "include_global": true,
    "include_shared": false
  }
}
```

Changing the default requires editing the profile config directly — this is an
admin-level operation.

---

## Enterprise governance

In a governed workspace (admin/member/viewer roles), scope expansion may be
restricted:
- **Viewers** cannot write `scope="global"` facts — their writes are always `personal`.
- **Members** can write `scope="shared"` with profiles in their access list.
- **Admins** can write `scope="global"` unrestricted.

See `slm-governance` for the full enterprise behavior model.

---

## Scope deletion and retention

A `scope="global"` fact deleted by one profile is deleted for all profiles. A
`scope="shared"` fact is deleted from all profiles that had access to it.
Use `slm forget "<query>" --dry-run` before any deletion involving shared facts
to review the impact. See `slm-remember` for the full deletion discipline.

---

## Related skills

- `slm-remember` — full fact storage reference including scope parameters
- `slm-recall` — full retrieval reference including include_global/include_shared
- `slm-profile` — what a profile is and how to switch between them
- `slm-governance` — enterprise role restrictions on scope

---

*SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later*
