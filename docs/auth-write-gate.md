# Auth Write Gate
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

The SLM daemon protects mutating operations (store, delete, update, config
writes) through a single authoritative write gate. This page explains what
credentials the gate accepts, how to enable opt-in API key auth, how to
rotate the install token, and the context behind the v3.7.6 double-gate fix.

---

## Credential Hierarchy

The write gate accepts one of four credentials in priority order:

| Credential | Who holds it | When it applies |
|-----------|-------------|-----------------|
| **Daemon capability** | Internal daemon process (process/filesystem state) | MCP `remember` / `recall` calls routed through the resident daemon itself |
| **Install token** | Same-origin dashboard browser | Dashboard writes and config tests at `http://127.0.0.1:8765` |
| **API key** (`X-SLM-API-Key` header) | Remote callers with a configured key | Non-loopback HTTP MCP and direct API writes when API key auth is enabled |
| **Uncredentialed loopback** | Any caller on `127.0.0.1` | Local CLI, local MCP clients, and local IDE connections (the default local-first posture) |

A caller on loopback with no credentials is trusted as the local OS-user
boundary. This is the default and covers all standard single-machine use.

Read endpoints are always open regardless of auth configuration.

---

## Enabling API Key Auth

API key auth is opt-in. To enable it, write a key to the key file:

```bash
# Generate a random key and write it
python3 -c "import secrets; print(secrets.token_urlsafe(32))" \
  > ~/.superlocalmemory/api_key
chmod 600 ~/.superlocalmemory/api_key
```

Once the file exists, non-loopback write callers must present the key in
the `X-SLM-API-Key` header:

```bash
curl -X POST http://<slm-host>:8765/api/memories \
  -H "X-SLM-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"content": "..."}'
```

Loopback callers (CLI, local IDE) are still trusted without a credential.
To require the key even on loopback (shared-host operators), set:

```bash
export SLM_REQUIRE_API_KEY_LOOPBACK=1
```

This opt-in flag restores the stricter pre-v3.7.6 posture for operators
running SLM on a multi-user machine. It is a no-op unless an `api_key` file
is configured.

---

## Rotating the Install Token

The install token is an auto-generated credential that the same-origin
dashboard browser uses to authenticate writes. Rotate it when the daemon
host is shared or after a security incident:

```bash
slm rotate-token
```

The daemon generates a new token and the dashboard picks it up on the next
page load. There are no further arguments.

---

## v3.7.6 Double-Gate Fix

Before v3.7.6, the write path ran two independent authorization checks:

1. The mutation-actor gate — which correctly accepted daemon capability,
   install token, API key, and uncredentialed loopback.
2. A redundant legacy check that only understood `X-SLM-API-Key`.

When API key auth was enabled, the redundant second gate rejected
capability-authenticated MCP `remember` calls and install-token dashboard
writes with `401 "Invalid or missing API key"`. This broke MCP writes from
Claude Code, Cursor, and other clients that authenticated via capability
rather than a key header.

**v3.7.6 fix:** The redundant second gate is removed. The mutation-actor gate
is now the single authoritative write boundary. The four accepted credentials
above remain unchanged.

**v3.7.8 note:** The `SLM_REQUIRE_API_KEY_LOOPBACK` opt-in was added to
allow shared-host operators to restore the strict posture selectively, without
reverting the local-first default.

---

## Remote HTTP MCP

Remote HTTP MCP clients (non-loopback) must present the configured API key.
Wire the host to the `SLM_MCP_ALLOWED_HOSTS` allowlist and configure the key:

```bash
# On the SLM host
export SLM_DAEMON_HOST=0.0.0.0
export SLM_MCP_ALLOWED_HOSTS=192.168.1.100:*
# api_key file must exist — see "Enabling API Key Auth" above
slm serve start
```

Remote callers then present `X-SLM-API-Key` in their MCP config or HTTP
headers. See [distributed-deployment.md](distributed-deployment.md) for the
full LAN setup guide.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
