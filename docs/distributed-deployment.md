# Distributed Deployment

This guide covers running SLM in distributed environments: multiple containers, LXC hosts, VMs, or any topology where the daemon runs on a different host than the MCP clients.

---

## Quick start (single host)

```bash
npm install -g superlocalmemory
slm setup
slm serve start   # or: systemctl start slm-http
```

Python deployments can instead create and activate a dedicated virtual
environment, then run `python -m pip install superlocalmemory` before setup.

For multi-container setups, read on.

---

## Binding the daemon to a LAN address

By default the daemon binds `127.0.0.1` (loopback only). To serve other containers:

```bash
SLM_DAEMON_HOST=0.0.0.0 slm serve start
# or in a systemd unit:
Environment=SLM_DAEMON_HOST=0.0.0.0
```

> **Security:** Host allowlists are DNS-rebinding/origin controls, not
> authentication. For AI tools on other computers, use SLM's own encrypted
> remote listener ([Remote access over TLS](#remote-access-over-tls)) rather
> than binding the main daemon to the network. Remote HTTP MCP requires HTTPS
> and a key; mesh routes require their configured shared secret. Do not expose
> the daemon directly to the public internet.

### Other computers must sign in

A request from another computer — including a browser on the LAN opening the
dashboard by IP — gets `401 remote_auth_required` for anything that carries
memory, reads included, unless it presents one of:

- the SLM API key in `X-SLM-API-Key` (the key is the `api_key` file in the
  SLM data folder; see [auth-write-gate.md](auth-write-gate.md));
- a team-account session token in `X-SLM-User-Session` (see [rbac-teams.md](rbac-teams.md));
- the mesh shared secret, on `/mesh/*` routes only;
- or it comes from an address you allowlisted for LAN use:

```bash
export SLM_REMOTE=1
export SLM_MCP_ALLOWED_HOSTS=192.168.50.0/24   # the computers you trust
```

The dashboard page, its static files and `/health` load without credentials;
`/mcp` keeps its own check (HTTPS plus a remote key or the API key; see
[Remote access over TLS](#remote-access-over-tls)). A LAN dashboard opened by IP
needs one of the above, usually the `SLM_REMOTE=1` allowlist. A Docker port mapping makes your own
computer's requests arrive from the bridge address (for example `172.17.0.1`),
not loopback: allowlist that address or send the API key.

---

## Remote access over TLS

The recommended way for AI tools on other computers (Hermes, Claude Code
`type: http`, `mcp-remote`) to use this SLM. It is off by default. The main
daemon keeps listening on `127.0.0.1` exactly as before; remote access is a
second listener that:

- speaks **TLS only** (plain HTTP to its port fails);
- serves **only** `/mcp`, `/mcp/<agent>` and `GET /health` — the dashboard,
  the HTTP API and the internal hook endpoints return 404 there;
- treats **every** caller as remote, even one on `127.0.0.1`;
- never answers with a redirect.

```bash
slm remote tls init --name slm.lan --ip 192.168.50.144   # CA + server certificate
slm remote enable --listen 192.168.50.144:8443
slm restart
slm remote keys add hermes-laptop             # write key for the active profile, shown once
slm remote keys add dashboards --read-only --profile clientx   # recall-only, profile clientx
slm remote check                              # exit 1 on any problem
```

`tls init` writes `ca.pem`, `ca.key`, `server.pem` and `server.key` under
`remote/tls/` in the SLM data folder. The CA can only vouch for the names and
addresses you give it, so even a stolen CA key cannot impersonate another
site. Copy `ca.pem` (public) to each client. `--force` issues a new server
certificate with the same CA; the server certificate lasts at most 397 days
and `slm remote check` warns 30 days before it expires.

**Keys.** Each client gets its own named key, sent as `Authorization: Bearer
slmr_...`. SLM stores only a hash. `slm remote keys list` shows names, scopes,
profiles and dates, never secrets. `slm remote keys revoke <name>` takes effect on the
next request, without a restart. The key file must be readable only by the SLM
user; otherwise every remote key is refused. The SLM API key also
works as a write key; send it as `Authorization: Bearer <api key>`, not as
`X-SLM-API-Key`, because MCP clients drop `Authorization` on a redirect to
another site but forward custom headers. SLM never answers `/mcp` with a
redirect.

**One profile per key.** Every key is bound to exactly one profile: the one
you name with `--profile`, or the profile active when you create it. A key
never reaches another profile:

- a `profile_id` other than the key's profile, in any tool's arguments
  (including inside structured ones such as `payload`), is refused with
  `remote_profile_not_allowed` — never quietly redirected;
- every tool that reads or writes memory (`recall`, `remember`, `search`,
  `fetch`, `list_recent`, `update_memory`, `delete_memory`, `session_init`,
  `observe`, status, kinds, views, assertions and the rest) is served for the
  key's profile whatever profile this computer is using; the host's active
  profile is never moved. A write is checked exactly as a write on this
  computer to that profile is (a write key, the role on that profile, the
  writer's own profile check) and is logged with the key's id and profile.
  Tools that touch no profile's memory (`slm_cache_*`, `slm_compress`,
  `slm_retrieve`, `slm_optimize_stats`, `get_version`, `get_mode`,
  `get_attribution`) always work;
- a remote save always stays in the key's profile: `scope` must be `personal`
  (a save that names no scope is made `personal`, whatever this computer's
  default), and `shared_with` is refused. Share a memory from this computer.
  A recall may still pass `include_shared` / `include_global`: it then sees only
  what other profiles shared with the key's profile, or made global.

Keys made by an older release had no profile. Each is bound once to the profile
active when the upgrade first runs (the daemon at start, or any `slm remote keys`
command); `slm remote keys list` says so and marks it `(bound on upgrade)`.
Until then the key is refused (`remote_key_unbound`). An older SLM refuses the
upgraded key file (every remote key off) rather than ignore the binding. The SLM API key, used as a key, has no stored profile: it reaches only
the profile active at the time of each call.

**What a key holder can read.** A key's holder can read the full text of every
memory in its profile, exactly as written — including any file paths, names or
pasted error output inside those memories. Bind each key to a profile that
holds only what that tool should see. Everything else in a tool answer that
describes this computer (data folder, home folder, account name, process ids,
environment, paths in errors, tracebacks, notes and receipts) is withheld from
remote callers.

**Separate caches.** `slm_cache_*` and reversible `slm_compress` entries made
over remote access belong to the key that made them (and the `/mcp/<agent>`
segment). A remote key never reads or overwrites a local agent's cache, even
if it sends the same agent name, and two keys never share entries.

**What a remote key can do.** A `read` key can recall, search, fetch, list and
read status. A `write` key can also save, update and delete individual
memories and record session activity. Tools that manage the SLM computer —
switching its active profile, indexing local folders for the code graph,
maintenance, retention, mesh, loops, `forget` by pattern, `set_mode` — are
refused to every remote caller, and are not listed in `tools/list`. Only
`initialize`, `ping`, `tools/list` and `tools/call` are accepted, and only as
`POST`: any other HTTP method gets `405` at once (there is no event stream to
open). Batched requests and bodies over 1 MiB are refused. One audit line per
remote tool call records the key name, its profile, the tool and the decision
(never arguments or content).

**Limits.** Company mode (team accounts with
required sign-in) refuses remote keys in this release. Remote calls share the
rate limiter with other network callers (`SLM_RATE_LIMIT_WRITE`, default 30
per minute per computer); raise it for a busy agent.

**Client recipes.** Hermes: see [Hermes: remote](hermes.md#use-an-slm-that-runs-on-another-computer). No SLM
environment variable points a client at a server; the address lives in each
client's own MCP configuration: `https://<slm-host>:8443/mcp/<agent>` with `Authorization:
Bearer <remote key>`. Stdio-only clients can use the `mcp-remote` bridge:
`npx -y mcp-remote https://<slm-host>:8443/mcp/<agent> --header
"Authorization:${AUTH_HEADER}"` with `AUTH_HEADER="Bearer <key>"` and
`NODE_EXTRA_CA_CERTS=/path/to/slm-ca.pem` in its `env`. Never use
`--allow-http` against another computer.

**Hooks stay local.** SLM's own hook scripts only ever talk to the daemon on
this computer (`SLM_HOOK_DAEMON_URL` accepts loopback addresses only), and the
install token, hook token and daemon capability are never accepted on the
remote listener.

## Host names SLM answers to

SLM refuses any request — dashboard, HTTP API or `/mcp` — that is not
addressed to this computer, so a web page on another site cannot reach it
through your browser. Requests addressed to `localhost` or to an IP address
are always served, so reaching a LAN install by its address needs nothing
new. To reach it by a host **name**, list the name in `SLM_ALLOWED_HOSTS`
(comma-separated; `*.office.lan` covers a whole suffix):

```bash
SLM_ALLOWED_HOSTS=slm.lan,*.office.lan slm serve start
```

Names you already list in `SLM_MCP_ALLOWED_HOSTS` (for example `slm.lan:*`)
are accepted too, so existing LAN setups keep working unchanged.

## Behind a reverse proxy

SLM decides whether a caller is on this computer from the TCP connection
alone. It ignores `X-Forwarded-For`, `Forwarded`, `X-Real-IP` and similar
headers, and a request that arrives from `127.0.0.1` carrying any of them is
treated as coming from another computer: it gets no local trust and must
present credentials like any network caller.

If you run a TLS proxy on another host and want SLM to see each client's real
address (for `SLM_MCP_ALLOWED_HOSTS` or rate limiting), name the proxy:

```bash
SLM_TRUSTED_PROXIES=10.0.0.5 slm serve start      # addresses or networks, comma-separated
```

SLM then reads `X-Forwarded-For` / `X-Forwarded-Proto` from those addresses
only. Naming a proxy never makes its callers local.

> **Warning:** a proxy on the same computer that strips every forwarding
> header makes all of its callers look like `127.0.0.1`, and SLM cannot tell
> them from a local program. Do not run such a proxy in front of SLM: configure
> it to send `X-Forwarded-For` (most proxies do by default).

## Opening the HTTP MCP transport to LAN clients

The `/mcp` endpoint uses MCP's DNS-rebinding protection, which defaults to localhost-only even when the daemon is bound on `0.0.0.0`. Set `SLM_MCP_ALLOWED_HOSTS` to open it:

```bash
# Allow one specific host (recommended)
SLM_MCP_ALLOWED_HOSTS=192.168.50.144:* slm serve start

# Allow multiple hosts
SLM_MCP_ALLOWED_HOSTS=192.168.50.144:*,slm.lan:* slm serve start

# Broad wildcard (not recommended; still does not replace authentication)
SLM_MCP_ALLOWED_HOSTS=* slm serve start
```

The value is a comma-separated list of `host:port-wildcard` patterns, e.g.
`192.168.50.144:*` allows any port on that IP. Prefer exact hosts or bounded
CIDRs. `*` widens the rebinding/origin surface and is not recommended, even on
a private LAN.

Remote HTTP MCP requests must also arrive over HTTPS and present a key (a
named remote key, or the SLM API key). The allowlist decides which hosts may
reach the transport; it does not create an authenticated actor. MCP from another
computer over plain HTTP is refused (`403
remote_requires_tls`) unless you set `SLM_REMOTE_ALLOW_PLAINTEXT=1`; prefer the
remote listener below.

---

## One-switch LAN mode: `SLM_REMOTE=1`

SLM historically assumes every dashboard browser, MCP client, and API caller is on `127.0.0.1`. That breaks three things when you reach SLM across a LAN:

1. **The Brain page can't load** from a remote browser — `/internal/token` refuses non-loopback clients, so the dashboard never gets the install token.
2. **Older stateful MCP clients can return `-32600 Session not found`** when a gateway/hub fails to replay `Mcp-Session-Id`. The MCP transport is stateless by default, so this is not the normal V4 path.
3. **The dashboard CSRF origin guard** only accepts loopback origins.

`SLM_REMOTE=1` flips all three at once — **default OFF**, so the loopback-only posture is unchanged for local installs. LAN access is still gated by your existing `SLM_MCP_ALLOWED_HOSTS` allowlist:

```bash
export SLM_DAEMON_HOST=0.0.0.0
export SLM_MCP_ALLOWED_HOSTS=192.168.50.0/24   # exact IP, CIDR, prefix*, or *
export SLM_REMOTE=1
slm serve start
```

What `SLM_REMOTE=1` does:
- **`/internal/token`** can serve the install token to allowlisted clients.
  Treat every allowlisted host as trusted local-operator infrastructure; an IP
  allowlist is not user authentication.
- **MCP transport remains stateless by default** so gateways/hubs/forwarders work without replaying a session ID. Set `SLM_MCP_STATEFUL=1` only when a compatibility integration explicitly requires stateful Streamable HTTP; it is not enabled by `SLM_REMOTE`.
- **The dashboard CSRF origin guard** also accepts allowlisted LAN origins.
- **The dashboard rate limiter exempts** allowlisted LAN browsers (they poll like the local dashboard does), so normal use doesn't trip `429`.

> **Security:** Remote mode widens the attack surface. Stateless MCP relaxes
> per-session isolation, and serving an install token to a LAN host gives that
> host a local dashboard credential. Keep `SLM_MCP_ALLOWED_HOSTS` specific,
> use the remote listener (TLS built in) for remote MCP, configure
> `SLM_MESH_SHARED_SECRET` for mesh, and if you put a reverse proxy in front of
> SLM, make it send `X-Forwarded-For`: SLM never treats a forwarded request as
> local.

### Tuning the dashboard rate limiter

If you hit `429 Too Many Requests` while debugging over a LAN, raise the limits (defaults: 30 writes / 120 reads per 60s):

```bash
export SLM_RATE_LIMIT_WRITE=100
export SLM_RATE_LIMIT_READ=500
export SLM_RATE_LIMIT_WINDOW=60
```

In `SLM_REMOTE=1` mode, allowlisted LAN clients are exempt from rate limiting anyway, so this is mainly for non-allowlisted callers or extra headroom.

---

## LXC / multi-container example (the #36 reporter's setup)

```
SLM container:     192.168.50.144   (runs daemon + HTTP MCP)
Hub container:     192.168.50.143   (runs SLM Hub MCP client)
OpenClaw container:192.168.50.142
```

On the SLM container:
```bash
export SLM_DAEMON_HOST=0.0.0.0
export SLM_MCP_ALLOWED_HOSTS=192.168.50.143:*,192.168.50.142:*
export SLM_MESH_SHARED_SECRET=<random 32-char string>
slm serve start
```

On Hub / OpenClaw containers, point the MCP client config at the SLM host.
There is no environment variable for this: the address goes in the client's
MCP server entry.

If the client speaks HTTP MCP:
```json
{
  "type": "http",
  "url": "http://192.168.50.144:8765/mcp/"
}
```

If the client is stdio-only, run it through the `mcp-remote` bridge (see
[IDE setup](ide-setup.md) for the global-install and `npx` routes). The bridge
refuses plain HTTP to anything other than localhost unless you pass
`--allow-http`, so only do this on a private network you trust:
```json
{
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "mcp-remote", "http://192.168.50.144:8765/mcp/", "--allow-http"]
}
```

Only MCP traffic goes to the remote daemon. The `slm` command line on a client
container always works against its own local store.

---

## stdio + HTTP coexistence

`slm mcp` (stdio transport) now reuses the running daemon instead of starting a second one on the same port. If you run `slm mcp` on the same machine as a systemd `slm-http` service, they coexist cleanly — one daemon, many front-ends.

---

## Per-agent identity over HTTP — `/mcp/{agent_id}`

The HTTP MCP endpoint accepts an **agent-id path segment** so that every AI tool
sharing the one daemon gets its own audit attribution — without spawning a
separate `slm mcp` stdio process per tool (which wastes RAM).

| URL | Resolved `agent_id` |
|-----|---------------------|
| `http://127.0.0.1:8765/mcp/` | `mcp_client` (default, backward compatible) |
| `http://127.0.0.1:8765/mcp/claude` | `claude` |
| `http://127.0.0.1:8765/mcp/hermes` | `hermes` |
| `http://127.0.0.1:8765/mcp/gemini` | `gemini` |
| `http://127.0.0.1:8765/mcp/codex` | `codex` |
| `http://127.0.0.1:8765/mcp/kimi` | `kimi` |

The daemon extracts the segment from the URL path into a per-request
`ContextVar`, so `remember`, `recall`, `observe`, `delete_memory`,
`update_memory`, `session_init`, and event emission all tag the correct agent —
no MCP-protocol changes are required.

The path segment is metadata, not authentication. Mutation authority derives
from the verified local capability, same-origin install token, configured SLM
API key, or documented mesh credential. A caller cannot grant itself trust by
choosing a different `{agent_id}`.

**Precedence:** URL path segment → `SLM_AGENT_ID` env var (stdio) → `mcp_client`.
The bare `/mcp/` endpoint is unchanged, so existing configs keep working.

### Client config examples

Claude Code / Claude Desktop (`~/.claude.json` → `mcpServers`):

```json
"superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/claude" }
```

Gemini CLI / Codex / Kimi — point each tool's MCP HTTP URL at its own segment
(`/mcp/gemini`, `/mcp/codex`, `/mcp/kimi`). Any string is accepted as the
agent id; pick a stable, lowercase name per tool.

> **Rollout order matters:** point a client at `/mcp/{agent_id}` only after the
> daemon is running a current release (`slm --version`). An older daemon mounts a bare
> `/mcp` without the extractor and will not recognise the extra path segment.

---

## Complete `SLM_*` environment variable reference

> Read from the source; every variable below is read by the code.

### Daemon / bind / paths

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_DAEMON_HOST` | Bind address for the HTTP daemon | `127.0.0.1` |
| `SLM_HOST` | Alias for `SLM_DAEMON_HOST` | — |
| `SLM_DAEMON_PORT` | Port for the HTTP daemon  | `8765` |
| `SLM_DAEMON_IDLE_TIMEOUT` | Seconds of inactivity before auto-shutdown (0 = always-on) | `0` |
| `SLM_DATA_DIR` | Override the base data directory | `~/.superlocalmemory` |
| `SLM_HOME` | Alias for `SLM_DATA_DIR` | — |
| `SLM_MEMORY_DB` | Override memory.db path | `$SLM_DATA_DIR/memory.db` |
| `SLM_CACHE_DB` | Override cache.db path | `$SLM_DATA_DIR/cache.db` |
| `SLM_DISABLE_LEGACY_PORT` | Set `1` to disable the 8767 backward-compat redirect | — |
| `SLM_PROFILE_ID` | Default profile ID | `default` |
| `SLM_AGENT_ID` | Override agent identifier for multi-agent attribution | — |
| `SLM_SESSION_ID` | Override session ID (usually generated by `session_init`) | — |
| `SLM_VERSION` | Read-only: current package version | — |

### MCP transport

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_MCP_EMBEDDED` | Set `1` when running MCP inside the daemon (suppresses warmup threads) | — |
| `SLM_MCP_ALLOWED_HOSTS` | Comma-separated allowlist (`host:port*`, exact IP, CIDR, prefix`*`, or `*`) for HTTP MCP + LAN token/origin/rate-limit (see above) | localhost-only |
| `SLM_REMOTE_LISTEN` | `HOST:PORT` for the TLS remote listener; overrides `slm remote enable`. Unset and not enabled: no remote listener | — |
| `SLM_REMOTE_TLS_CERT` / `SLM_REMOTE_TLS_KEY` | Server certificate and private key (PEM) for the remote listener; the key must not be readable by other users | `remote/tls/server.pem`, `server.key` |
| `SLM_REMOTE_ALLOW_PLAINTEXT` | `1` accepts MCP from other computers over plain HTTP on the main listener (logged). Never applies to the remote listener | — |
| `SLM_TRUSTED_PROXIES` | Proxies whose `X-Forwarded-For` / `X-Forwarded-Proto` SLM reads (addresses or networks). Unset: forwarding headers are ignored. Never grants local trust | — |
| `SLM_REMOTE` | One-switch LAN mode: serves token to allowlisted LAN clients, relaxes origin guard, and exempts LAN from rate limit. It does not change the default stateless MCP transport. Default OFF | — |
| `SLM_MCP_STATELESS` | Explicitly select stateless MCP transport; stateless is already the default | — |
| `SLM_MCP_STATEFUL` | Set `1` only for a compatibility integration that requires stateful Streamable HTTP | — |
| `SLM_MCP_TOOLS` | Comma-separated list of MCP tools to expose (default: all) | — |
| `SLM_RATE_LIMIT_WRITE` | Max dashboard write requests per window | `30` |
| `SLM_RATE_LIMIT_READ` | Max dashboard read requests per window | `120` |
| `SLM_RATE_LIMIT_WINDOW` | Rate-limit window in seconds | `60` |
| `SLM_MCP_ALL_TOOLS` | Set `1` to force-enable all tools regardless of mode | — |
| `SLM_MCP_MESH_TOOLS` | Set `1` to always include mesh tools | — |

### Mesh

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_MESH_HOST` | Bind address for mesh WebSocket broker | `127.0.0.1` |
| `SLM_MESH_WS_PORT` | WebSocket port for mesh broker | `7900` |
| `SLM_MESH_SHARED_SECRET` | Auth secret for the mesh HTTP API. Required when `SLM_MESH_HOST` is not localhost. Send as `Authorization: Bearer <secret>` (canonical) or `X-Mesh-Secret: <secret>` (legacy). | — |
| `SLM_MESH_PEER_URL` | Explicit peer URL to register with at startup | — |
| `SLM_MESH_DISCOVERY` | Discovery mode: `local` / `manual` | `local` |

> **Mesh API auth:** When `SLM_MESH_SHARED_SECRET` is set, non-loopback callers must authenticate every `/mesh/*` request. The canonical header is `Authorization: Bearer <your-secret>` — this is what `RemoteSyncClient` sends automatically. The legacy `X-Mesh-Secret: <your-secret>` header is also accepted for backwards compatibility.
>
> Example: `curl http://192.168.50.144:8765/mesh/status -H "Authorization: Bearer <your-secret>"`

> **Note:** The variable is `SLM_MESH_WS_PORT` (not `SLM_MCP_WS_PORT`).

### Inspecting the mesh

Check broker health and connected peer sessions from the terminal:

```bash
slm mesh status   # broker up/down, peer count, uptime
slm mesh peers    # active peer sessions on this machine
```

These are read-only and query the running daemon's mesh broker; agents also
have the `mesh_*` MCP tools and the dashboard **Mesh Peers** tab.
> `SLM_DAEMON_HOST` is canonical; `SLM_HOST` is the alias (not the other way around).

### Memory / health / workers

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_RSS_BUDGET_MB` | Global RSS budget for the health monitor watchdog (0 = auto, 40% of RAM) | auto |
| `SLM_MAX_WORKER_MB` | Per-worker RSS limit before the per-worker watchdog triggers | `2048` |
| `SLM_MAX_EMBEDDING_WORKERS` | Max parallel embedding worker processes | `1` |
| `SLM_EMBED_WORKER_RSS_LIMIT_MB` | RSS limit per embedding worker process | `2500` |
| `SLM_EMBED_IDLE_TIMEOUT` | Seconds before an idle embedding worker exits | `1800` |
| `SLM_EMBED_RECYCLE_AFTER` | Recycle embedding worker after N requests | `5000` |
| `SLM_EMBED_RESPONSE_TIMEOUT` | Timeout (s) for a single embedding request | `180` |
| `SLM_RERANKER_ENABLED` | Process-only override for whether the cross-encoder reranker runs, regardless of the persisted `retrieval.use_cross_encoder` choice — never written back to config.json. `false`/`0`/`no`/`off` disables it; anything else (including unset) leaves the persisted choice in force | unset |
| `SLM_RERANKER_IDLE_TIMEOUT` | Seconds before an idle reranker worker exits | `1800` |
| `SLM_DAEMON_HEALTH_PROBE_S` | Seconds a client waits for a ready daemon's `/health` answer before calling it unavailable (clamped 0.25–10; a starting daemon always gets 0.5) | `2` |
| `SLM_MIN_AVAILABLE_MEMORY_GB` | Minimum free system RAM before SLM defers heavy operations | `1.0` |
| `SLM_TRIGRAM_BOOTSTRAP_RAM_MB` | Max RAM for trigram index bootstrap | `512` |

**Lite bot-host profile** (a shared, RAM-constrained box — e.g. the Cursor/Grok Bot
plugin's `mcp.cursor.json`, which sets these by default): `SLM_RERANKER_ENABLED=false`,
`SLM_RERANKER_IDLE_TIMEOUT=120`, `SLM_MAX_EMBEDDING_WORKERS=1`. Measured on macOS (daemon +
workers, warm, after one remember and one recall on a fresh store): default profile (reranker
on) ≈ 390 MB resident (daemon ~90 MB + reranker worker ~200 MB + embedding worker ~100 MB)
vs. lite profile (reranker off) ≈ 190 MB resident (daemon ~100 MB + embedding worker ~90 MB,
no reranker worker) — roughly halved, with the reranker subprocess being the single biggest
line item. Quality cost: the cross-encoder reranker measurably improves top-of-list ranking
precision in SLM's own benchmarks; with it off, recall still runs every other channel (BM25,
semantic, entity graph, temporal, spreading activation, Hopfield) and fuses them by RRF, just
without the cross-encoder re-ordering pass on top. These are macOS/arm64 numbers (PyTorch
backend for both workers on this platform); a Linux/x86_64 CPU-only host will differ (its
embedding worker tries ONNX first) and has not been measured here.

### Learning / bandit / signals

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_SIGNALS_ENABLED` | Enable implicit reward signal collection | `1` |
| `SLM_SIGNAL_QUEUE_MAX` | Max buffered signals before flush | `500` |
| `SLM_BANDIT_DISABLED` | Set `1` to disable the contextual bandit ranker | — |
| `SLM_BANDIT_ALPHA_CAP` | Maximum bandit learning rate | `0.3` |
| `SLM_BANDIT_REWARD_WINDOW_SEC` | Window (s) for reward aggregation | `3600` |
| `SLM_BANDIT_PLAYS_RETENTION_DAYS` | Keep bandit play history for N days | `30` |

### Evolution

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_EVOLUTION_ENABLED` | Enable skill evolution | `0` |
| `SLM_EVOLUTION_BACKEND` | Evolution backend: `local` / `remote` | `local` |
| `SLM_EVOLUTION_RETRY_CAP` | Max retries per evolution attempt | `3` |

### Rate limiting

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_RATE_LIMIT_READ` | Max read requests per window | `120` |
| `SLM_RATE_LIMIT_WRITE` | Max write requests per window | `30` |
| `SLM_RATE_LIMIT_WINDOW` | Rate-limit window in seconds | `60` |

### Adapters / sync

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_CURSOR_*` | Cursor IDE adapter settings | — |
| `SLM_COPILOT_*` | GitHub Copilot adapter settings | — |
| `SLM_ANTIGRAVITY_*` | Antigravity adapter settings | — |
| `SLM_ADAPTER_FORCE_*` | Force-enable a specific adapter | — |
| `SLM_CROSS_PLATFORM_SYNC_DISABLED` | Set `1` to disable cross-tool sync | — |
| `SLM_CROSS_PLATFORM_SYNC_INTERVAL` | Sync interval in seconds | `60` |

### Recall / ingest / misc

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_RECALL_NO_FLOOR` | Set `1` to disable the relevance floor (returns all results) | — |
| `SLM_RECALL_TIMING` | Set `1` to log per-channel recall timing | — |
| `SLM_RANKING` | Adaptive ranking mode: `off`, `v1`, `v2` or `v2-ensemble`. Unset means `off` | `off` |
| `SLM_INGEST_NO_GATE` | Set `1` to skip the ingest quality gate | — |
| `SLM_OBSERVE_DEBOUNCE_SEC` | Debounce window (s) for `observe` auto-capture | `5` |
| `SLM_TOPIC_SHIFT_LOG` | Set `1` to log topic-shift detection | — |
| `SLM_HOOK_DAEMON_URL` | Override daemon URL for hook integrations | — |
| `SLM_HOOK_DAEMON_TIMEOUT` | Timeout (s) for hook→daemon requests | `5` |
| `SLM_DISABLE_WARMUP_SIDE_EFFECTS` | Set `1` to suppress daemon auto-start in tests | — |
| `SLM_SKIP_DEP_CHECK` | Set `1` to skip dependency version checks | — |
| `SLM_NON_INTERACTIVE` | Set `1` to suppress interactive prompts | — |
| `SLM_DISABLE` | Set `1` to disable SLM entirely (no-op MCP tools) | — |
| `SLM_V2_PIPELINE_DISABLED` | Set `1` to force v1 store pipeline | — |
| `SLM_INJECTION_LEGACY` | Set `1` to use legacy context injection format | — |
| `SLM_SIGNER_KEY` | HMAC key for memory signing (anti-tamper) | — |

---

## Health config via config.json

You can now tune the health monitor via `~/.superlocalmemory/config.json`:

```json
{
  "health": {
    "global_rss_budget_mb": 0,
    "heartbeat_timeout_sec": 60,
    "health_check_interval_sec": 15,
    "enable_structured_logging": true
  }
}
```

`global_rss_budget_mb: 0` means auto (40% of physical RAM, floor 2500 MB). `SLM_RSS_BUDGET_MB` env takes priority over the config file value.
