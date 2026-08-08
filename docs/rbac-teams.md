# Teams, Users, and Access Control (RBAC)
> SuperLocalMemory V4 Documentation
> https://superlocalmemory.com | Part of Qualixar

V4 includes multi-user access control to SLM (from the 3.8 line). This page covers roles, workspace
membership, the login gate, and how to operate RBAC from the CLI, API, and
dashboard.

---

## Roles

SLM uses three roles, scoped per workspace (profile):

| Role | Read memory | Write memory | Manage users and config |
|------|:-----------:|:------------:|:-----------------------:|
| **admin** | yes | yes | yes |
| **member** | yes | yes | no |
| **viewer** | yes | no | no |

A user can have different roles in different workspaces. Role assignment is
per-workspace, not global.

---

## Workspaces

In SLM, a workspace is a profile. Each profile has its own:

- Isolated memory database (personal facts never cross profiles)
- User membership list and role assignments
- Retention policy
- Audit trail

A user's access to profile A does not grant any access to profile B.

---

## Login gate

The login gate controls whether the dashboard and API require authenticated sessions.

| Setting | Behavior | Recommended for |
|---------|----------|-----------------|
| `require_login = false` | Loopback caller is trusted as owner. No authentication enforced. | Personal single-user installs |
| `require_login = true` | Every dashboard and API request requires a valid session. | Team and enterprise deployments |

Personal installs ship with `require_login = false`. No default credentials are
included. Enterprise installs (set via the installer or `slm reconfigure`) ship
with `require_login = true`.

### First-run admin creation

When login is enabled, the first run of the daemon triggers an admin creation
flow. The admin sets a password of their choosing. No default password is
shipped. Do not configure a blank or trivially guessable password: SLM
does not enforce complexity rules, but the password protects all workspaces on
the machine.

### Enabling login

```bash
slm config set security.require_login true
slm restart
```

After restart, the dashboard prompts for credentials. Session cookies are
`HttpOnly`. Set `SLM_DASHBOARD_HTTPS=1` to add the `Secure` flag.

---

## Managing users

### Via CLI

The `slm profile` CLI manages workspace profiles only — not users:

```bash
slm profile list              # list workspaces
slm profile switch <name>     # switch active workspace
slm profile create <name>     # create a new workspace
```

Verified in `src/superlocalmemory/cli/main.py`: `profile` accepts only `list | switch | create` (`choices=["list", "switch", "create"]`). There is no `slm profile users …` subcommand. User and membership administration is via the RBAC HTTP API and the dashboard (see below).

### Via dashboard

The **Governance → Access & Users** tab shows the user list for the current workspace. Admins can invite, assign roles, and remove users.

### Via HTTP API (RBAC routes in `src/superlocalmemory/server/routes/rbac.py`)

All routes are under `/api/rbac/*` and require machine auth (`require_http_mutation_actor`); user/membership/policy admin further requires `MANAGE` (`src/superlocalmemory/server/rbac_enforce.py: require_manage`).

| Method & Path | Purpose |
|---|---|
| `POST /api/rbac/login` | Authenticate; sets `HttpOnly` `slm_session` cookie (add `Secure` with `SLM_DASHBOARD_HTTPS=1`) |
| `POST /api/rbac/logout` | Revoke session |
| `GET /api/rbac/whoami` | Current principal |
| `GET /api/rbac/status` | `rbac_active`, `require_login`, `user_count` |
| `GET /api/rbac/users` | List users (MANAGE) |
| `POST /api/rbac/users` | Create user; optional `role` + `profile_id` to set initial membership (requires MANAGE on that profile) |
| `PATCH /api/rbac/users/{user_id}` | Update `display_name`, `password`, `status` |
| `DELETE /api/rbac/users/{user_id}` | Delete user |
| `GET /api/rbac/members?profile_id=` | List workspace members (MANAGE on target profile) |
| `POST /api/rbac/members` `{user_id, role, profile_id}` | Set membership role (`admin`/`member`/`viewer`) |
| `DELETE /api/rbac/members` `{user_id, profile_id}` | Remove membership |
| `POST /api/rbac/policy` `{require_login}` | Toggle login gate |

Workspace membership commands require `MANAGE` on the **target** profile, and modifying a user other than yourself requires authority over that user (owner or an admin who shares a `MANAGE` workspace).

---

## Memory scopes and sharing

Memory scopes control which profiles can recall a given fact.

| Scope | Recallable by | Set with |
|-------|--------------|----------|
| `personal` | Owner profile only (default) | `slm remember "..." --scope personal` |
| `shared` | Named profiles the owner grants | `slm remember "..." --scope shared --shared-with profile-a,profile-b` |
| `global` | Any authorized user on this machine | `slm remember "..." --scope global` |

Recall is default-deny: shared and global facts are not returned unless the
caller explicitly opts in or the scope policy allows it.

```bash
# Store a global fact (team-wide)
slm remember "Production DB host is db.internal:5432" --scope global

# Recall including global and shared
slm recall "production database" --include-global --include-shared
```

Full docs: [docs/shared-memory.md](./shared-memory.md)

---

## Audit trail

Every access-control event is recorded in the hash-chained audit trail:
user creation, role change, login attempt, session creation, workspace membership
change. The audit trail cannot be retroactively modified without breaking the
chain.

View the audit trail in the **Governance → Audit** dashboard tab or via CLI:

```bash
slm diagnostics export /tmp/audit-report.json
```

---

## Security notes

- Role checks are enforced at the API layer. A member with read/write access
  cannot escalate to admin by manipulating request parameters.
- IDOR protection: an admin of workspace A cannot modify users in workspace B.
- Session timing: `resolve_session` is debounced to avoid writer contention
  on the `rbac_sessions` table under high request rates.
- Login rate-limiting: failed login attempts are tracked; repeated failures
  result in a lockout window.
- Install token: the install token (used by the dashboard and IDE configs)
  is stored in memory only and scoped to loopback addresses in company mode.
  Rotate it with `slm rotate-token`.

---

## Related

- [Deployment tiers: Personal vs Enterprise](./deployment-tiers.md)
- [GDPR and compliance controls](./compliance.md)
- [Profiles and workspace isolation](./profiles.md)
- [Shared memory scopes](./shared-memory.md)

---

*SuperLocalMemory V4 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
