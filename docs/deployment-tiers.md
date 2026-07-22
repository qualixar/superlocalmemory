# Deployment Tiers: Personal and Enterprise
> SuperLocalMemory V3.8.0 Documentation
> https://superlocalmemory.com | Part of Qualixar

SLM ships one binary. The deployment tier is a configuration preset applied
at install time or via `slm reconfigure`. Each setting is independently
overridable at runtime with `slm config set`.

---

## Tier comparison

| Capability | Personal | Enterprise |
|-----------|:--------:|:----------:|
| Login gate (`require_login`) | off | on |
| PII redaction before memory storage | off | on |
| Retention rules (time-based expiry) | off | on |
| Hash-chained audit trail | on | on |
| RBAC (users, roles, workspaces) | on | on |
| GDPR export and erasure | on | on |
| Mesh coordination | on | on |
| Cloud backup | opt-in | opt-in |
| Scale Engine projections | opt-in | opt-in |

All capabilities are present in both tiers. The difference is default
configuration, not binary capability.

---

## Personal tier

The default for individual developers. No login required; the loopback owner is
trusted. PII redaction and retention rules are off.

```bash
# Default after install, or after:
slm config set security.require_login false
slm config set privacy.pii_redaction false
slm config set retention.enabled false
slm restart
```

Use the personal tier when:
- You are the sole user of the machine or the SLM install.
- You do not share memory data with other users.
- You want zero authentication friction.

---

## Enterprise tier

For teams, shared infrastructure, or any deployment where multiple users access
the same SLM instance.

```bash
# Set via installer, or manually:
slm config set security.require_login true
slm config set privacy.pii_redaction true
slm config set retention.enabled true
slm config set retention.default_policy gdpr-30d
slm restart
```

Or use the interactive configurator:

```bash
slm reconfigure
```

The enterprise preset applies:
1. `require_login = true` — every API and dashboard call needs a valid session.
2. `pii_redaction = true` — recognized PII patterns are redacted before memory content crosses trust boundaries.
3. `retention.enabled = true` — time-based retention rules enforce expiry. Default policy is `gdpr-30d` (configurable).
4. `audit = true` — already on in personal tier; remains on.

### First-run after enabling login

After setting `require_login = true` and restarting, the daemon runs an admin
creation flow on the first access. Set a strong admin password at this point —
SLM ships no default credentials.

---

## Retention policies

| Policy | Behavior |
|--------|----------|
| `indefinite` | No automatic expiry (default in personal tier) |
| `gdpr-30d` | Expire memories older than 30 days unless pinned |
| `hipaa-7y` | Retain for 7 years, then expire |
| `custom` | Operator-specified day count |

```bash
slm config set retention.default_policy gdpr-30d
# Or via MCP:
# set_retention_policy(policy="gdpr-30d")
```

Retention is enforced per-profile. Different workspaces can have different policies.

---

## PII redaction

When `pii_redaction = true`, SLM applies pattern-based redaction to memory
content before it is stored and before it is emitted in recall responses that
cross trust boundaries (e.g., MCP responses to non-loopback clients).

Redaction is heuristic and not guaranteed to catch all PII patterns. Operators
responsible for a compliance program should verify redaction coverage for their
specific content domain.

```bash
slm config set privacy.pii_redaction true
```

---

## Installer presets

The interactive installer (`npm install -g superlocalmemory` → `slm setup`) or
`slm reconfigure` lets you select a tier:

```
? Deployment type:
  > Personal (single user, no login required)
    Enterprise (multi-user, login required, PII redaction, retention)
```

This writes the appropriate `slm config set` values and restarts the daemon.
Any individual setting can be changed afterward with `slm config set`.

---

## Related

- [Teams, users, and RBAC](./rbac-teams.md)
- [GDPR and compliance controls](./compliance.md)
- [Configuration reference](./configuration.md)

---

*SuperLocalMemory V3.8.0 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
