# Auth Write Gate

The SLM daemon protects mutating operations (store, delete, update, config
writes) through a single authoritative write gate. This page explains what
credentials the gate accepts, how to enable opt-in API key auth, how to
rotate the install token, and what counts as a loopback caller.

---

## Credential Hierarchy

The write gate accepts one of these credentials:

| Credential | Who holds it | When it applies |
|-----------|-------------|-----------------|
| **Daemon capability** | Internal daemon process (process/filesystem state) | MCP `remember` / `recall` calls routed through the resident daemon itself |
| **Install token** | Same-origin dashboard browser | Dashboard writes and config tests at `http://127.0.0.1:8765` |
| **API key** (`X-SLM-API-Key` header) | Remote callers with a configured key | Non-loopback HTTP MCP and direct API writes when API key auth is enabled |
| **Remote key** (`Authorization: Bearer slmr_...`) | AI tools on other computers (`slm remote keys add`) | MCP only, over HTTPS, scoped `read` or `write`; host-management tools are refused (see [distributed-deployment.md](distributed-deployment.md#remote-access-over-tls)) |
| **Uncredentialed loopback** | Any caller on `127.0.0.1` | Local CLI, local MCP clients, and local IDE connections (the default local-first posture) |

A caller on loopback with no credentials is trusted as the local OS-user
boundary. This is the default and covers all standard single-machine use.

Read endpoints are open to this computer. Other computers need a key or the
`SLM_REMOTE` LAN allowlist.

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

`curl` without `-L` never follows a redirect; do not add `-L` (or any
"follow redirects" option) to a request that carries `X-SLM-API-Key`, because
clients forward custom headers to wherever a redirect points. SLM itself never
answers these routes with a redirect.

Loopback callers (CLI, local IDE) are still trusted without a credential.
To require the key even on loopback (shared-host operators), set:

```bash
export SLM_REQUIRE_API_KEY_LOOPBACK=1
```

This opt-in flag gives the stricter posture to operators
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

## Strict mode and the dashboard

By default the dashboard asks the daemon for the install token (`GET
/internal/token`) and the daemon hands it to any program on this computer,
because any such program could already read the token file.

With `SLM_REQUIRE_CREDENTIALS=1` you have said that even a local program must
hold a key. Handing the key to whoever asks would defeat that, so in this mode
`/internal/token` answers 403 with the reason "This computer requires the
SuperLocalMemory key". The dashboard then shows a one-time box asking you to
paste the key. Print it in a terminal with:

```bash
slm token show
```

The key is the install token. `slm token show` tightens the token file to
owner-only before printing, and refuses to print a key owned by another
account. The dashboard keeps the pasted key in memory for that browser tab only;
it is never written to browser storage, so reloading the page asks again. If you
choose "Not now", writes from the dashboard stay blocked and the box does not
reappear for 30 seconds.

---

## One gate for every write

The mutation-actor gate is the single authoritative write boundary. It accepts
the daemon capability, the install token, an API key, a remote key or an
uncredentialed loopback caller, as listed above. An MCP `remember` that arrives
through the daemon's own capability, or a dashboard write that carries the
install token, is therefore not rejected for lacking an API key header even when
API key auth is on.

[Web access](remote-access/README.md) does not use this gate for the app's own
sign-in. An app signs in with OAuth at the gateway, and the companion then
forwards the admitted request to the local MCP endpoint with the local install
credential. Cloud credentials never replace it.

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

For MCP clients on other computers, use the remote listener and a named
remote key (`slm remote keys add`), sent as `Authorization: Bearer <key>`. If
you keep the API key for MCP, send it the same way, as `Authorization: Bearer
<api key>`: MCP clients drop `Authorization` when a redirect points at another
site, but they forward `X-SLM-API-Key`. `X-SLM-API-Key` remains accepted for
HTTP API calls. See [distributed-deployment.md](distributed-deployment.md) for the
full LAN setup guide.

---

## What counts as loopback

When the daemon binds to `0.0.0.0` in a container or VM (LXC, Docker, any
dual-stack Linux host), IPv4 clients connecting to `localhost` can be reported
as `::ffff:127.0.0.1`, the IPv4-mapped IPv6 loopback form. SLM treats every
loopback form as loopback, so the install token works for a caller on the same
machine:

| Address form | Loopback? |
|---|---|
| `127.0.0.1` | True |
| `127.0.0.2` … `127.255.255.255` | True (full 127.0.0.0/8) |
| `::1` | True |
| `::ffff:127.0.0.1` | True (fixes #90) |
| `localhost` | True |
| `::ffff:192.168.1.1` | **False** (private, not loopback) |
| `192.168.1.1` | **False** |

**Invariants:**
- The install token is still accepted **only** from loopback addresses.
  `::ffff:192.168.1.1` (IPv4-mapped LAN IP) is not loopback and is rejected.
- `SLM_REQUIRE_CREDENTIALS=1` still forces credentials on all callers,
  including loopback, and stops the dashboard from being handed the token (see
  "Strict mode and the dashboard").
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

Inside the same container the install token works over `::ffff:127.0.0.1`
(dual-stack loopback) when `SLM_REQUIRE_CREDENTIALS` is not set. For production
deployments with `SLM_DAEMON_HOST=0.0.0.0`, always set
`SLM_REQUIRE_CREDENTIALS=1` and use the API key.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
