# Auth Write Gate
> SuperLocalMemory V4 Documentation
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

## v3.8.4 — IPv4-Mapped Loopback Fix (issue #90)

**Symptom:** In a container or VM where `SLM_DAEMON_HOST=0.0.0.0`, curl
commands using `X-Install-Token` fail with `403 Write rejected` even though
the caller is on the same machine. This affects LXC, Docker, and any
dual-stack Linux host.

**Root cause:** When the daemon binds to `0.0.0.0`, the OS creates an IPv6
socket. IPv4 clients connecting to `localhost` are reported to uvicorn as
`::ffff:127.0.0.1` (IPv4-mapped IPv6 loopback, RFC 4291 §2.5.5.2). The
auth gate's literal check `("127.0.0.1", "::1", "localhost")` did not
include this form, so the connection was incorrectly treated as non-loopback
and the install token was rejected.

**Fix (3.8.4):** The centralized `is_loopback()` helper in
`server/loopback.py` uses `ipaddress.ip_address(host).is_loopback`, which
correctly returns `True` for all of:

| Address form | Loopback? |
|---|---|
| `127.0.0.1` | True |
| `127.0.0.2` … `127.255.255.255` | True (full 127.0.0.0/8) |
| `::1` | True |
| `::ffff:127.0.0.1` | True (fixes #90) |
| `localhost` | True |
| `::ffff:192.168.1.1` | **False** (private, not loopback) |
| `192.168.1.1` | **False** |

**Security invariants preserved:**
- The install token is still accepted **only** from loopback addresses.
  `::ffff:192.168.1.1` (IPv4-mapped LAN IP) is not loopback and is rejected.
- `SLM_REQUIRE_CREDENTIALS=1` still forces credentials on all callers,
  including loopback.
- Non-loopback callers must use `X-SLM-API-Key` (the API key is the designed
  credential for container/remote access).

---

## Networked Deployment Recipe (containers, VMs, LAN)

Use this recipe when running the SLM daemon in a container or when the HTTP
client is not on the same loopback interface:

```bash
# 1. Start the daemon accessible from the container/VM network:
SLM_DAEMON_HOST=0.0.0.0 SLM_REQUIRE_CREDENTIALS=1 slm serve start

# 2. Generate an API key (one-time setup):
python3 -c "import secrets; print(secrets.token_urlsafe(32))" \
  > ~/.superlocalmemory/api_key
chmod 600 ~/.superlocalmemory/api_key

# 3. Call from the container using the API key:
curl -X POST http://localhost:8765/api/memories \
     -H "X-SLM-API-Key: $(cat ~/.superlocalmemory/api_key)" \
     -H "Content-Type: application/json" \
     -d '{"content": "test"}'
```

**Why not use the install token from a container?**
The install token is embedded in the dashboard JavaScript served over HTTP,
so a LAN observer can read it. It is intentionally restricted to loopback
peers. Use the API key (`X-SLM-API-Key`) for all container and remote access.

**On SLM 3.8.4+:** If you upgrade to 3.8.4 without changing anything, the
install token will now work from within the same container over
`::ffff:127.0.0.1` (dual-stack loopback) when `SLM_REQUIRE_CREDENTIALS` is
not set. For production deployments with `SLM_DAEMON_HOST=0.0.0.0`, always
set `SLM_REQUIRE_CREDENTIALS=1` and use the API key.

---

*SuperLocalMemory V4 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
