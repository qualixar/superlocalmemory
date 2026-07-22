# RBAC and Teams

SuperLocalMemory V3.8.0 introduces multi-user access control. This page summarizes
the key concepts; full documentation is in [docs/rbac-teams.md](../docs/rbac-teams.md).

## Roles

Three roles, scoped per workspace (profile):

| Role | Read | Write | Manage users/config |
|------|:----:|:-----:|:-------------------:|
| admin | yes | yes | yes |
| member | yes | yes | no |
| viewer | yes | no | no |

A user may have different roles in different workspaces.

## Login gate

| Mode | Behavior |
|------|---------|
| `require_login = false` | Personal installs, loopback owner trusted |
| `require_login = true` | Team/enterprise, every request needs a session |

Enable with: `slm config set security.require_login true && slm restart`

No default credentials are shipped. First-run prompts the admin to set a password.

## Memory scopes

| Scope | Who can recall |
|-------|---------------|
| `personal` | Owner profile only (default) |
| `shared` | Named profiles the owner grants |
| `global` | Any authorized user on this machine |

Recall is default-deny: `--include-shared` and `--include-global` opt in explicitly.

## Dashboard

The **Governance → Access & Users** tab shows users, roles, and workspace membership.
Admins can invite users, change roles, and remove access from this tab.

## Related pages

- [[GDPR Compliance]]
- [[Compliance]] (full compliance.md mirror)
- [[Getting Started]]
