# Distributed Deployment
> SuperLocalMemory V3.6.9+ — Multi-machine / LXC / container setup
> https://superlocalmemory.com | Part of Qualixar

This guide covers running SLM in distributed environments: multiple containers, LXC hosts, VMs, or any topology where the daemon runs on a different host than the MCP clients.

---

## Quick start (single host, already works in v3.6.8)

```bash
pipx install superlocalmemory   # or: uv tool install superlocalmemory
slm setup
slm serve start   # or: systemctl start slm-http
```

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
> authentication. Remote HTTP MCP requires a configured SLM API key; mesh
> routes require their configured shared secret. Use TLS and network policy in
> front of any non-loopback listener. Do not expose the daemon directly to the
> public internet.

---

## Opening the HTTP MCP transport to LAN clients (v3.6.9+)

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

Remote HTTP MCP requests must also present the configured SLM API key. The
allowlist decides which hosts may reach the transport; it does not create an
authenticated actor.

---

## One-switch LAN mode: `SLM_REMOTE=1` (v3.6.12)

SLM historically assumes every dashboard browser, MCP client, and API caller is on `127.0.0.1`. That breaks three things when you reach SLM across a LAN (issues #39 / #40):

1. **The Brain page can't load** from a remote browser — `/internal/token` refuses non-loopback clients, so the dashboard never gets the install token.
2. **Mesh / forwarded MCP tools return `-32600 Session not found`** — the Streamable-HTTP transport is stateful, so any gateway/hub that doesn't replay the `Mcp-Session-Id` is rejected.
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
- **MCP transport runs stateless** so gateways/hubs/forwarders work without replaying the session id (fixes the mesh `-32600`). Available standalone as `SLM_MCP_STATELESS=1` if you only need the gateway fix without opening the token endpoint.
- **The dashboard CSRF origin guard** also accepts allowlisted LAN origins.
- **The dashboard rate limiter exempts** allowlisted LAN browsers (they poll like the local dashboard does), so normal use doesn't trip `429`.

> **Security:** Remote mode widens the attack surface. Stateless MCP relaxes
> per-session isolation, and serving an install token to a LAN host gives that
> host a local dashboard credential. Keep `SLM_MCP_ALLOWED_HOSTS` specific,
> configure the SLM API key for remote MCP, configure
> `SLM_MESH_SHARED_SECRET` for mesh, and terminate TLS at a trusted gateway.

### Tuning the dashboard rate limiter (v3.6.12)

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

On Hub / OpenClaw containers, point to the SLM host:
```bash
# In mcp client config or env:
SLM_DAEMON_URL=http://192.168.50.144:8765
```

---

## stdio + HTTP coexistence (v3.6.9+)

`slm mcp` (stdio transport) now reuses the running daemon instead of starting a second one on the same port. If you run `slm mcp` on the same machine as a systemd `slm-http` service, they coexist cleanly — one daemon, many front-ends.

---

## Per-agent identity over HTTP — `/mcp/{agent_id}` (v3.6.10+)

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
> daemon is running **v3.6.10+** (`slm --version`). An older daemon mounts a bare
> `/mcp` without the extractor and will not recognise the extra path segment.

---

## Complete `SLM_*` environment variable reference

> Generated from source at v3.6.9. **NEW** marks variables added in this release.

### Daemon / bind / paths

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_DAEMON_HOST` | Bind address for the HTTP daemon | `127.0.0.1` |
| `SLM_HOST` | Alias for `SLM_DAEMON_HOST` | — |
| `SLM_DAEMON_PORT` | **NEW** Port for the HTTP daemon (**fully wired** as of v3.6.9) | `8765` |
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
| `SLM_MCP_ALLOWED_HOSTS` | **NEW** Comma-separated allowlist (`host:port*`, exact IP, CIDR, prefix`*`, or `*`) for HTTP MCP + LAN token/origin/rate-limit (see above) | localhost-only |
| `SLM_REMOTE` | **NEW (v3.6.12)** One-switch LAN mode: serves token to allowlisted LAN clients, runs MCP stateless, relaxes origin guard, exempts LAN from rate limit. Default OFF | — |
| `SLM_MCP_STATELESS` | **NEW (v3.6.12)** Run MCP transport stateless only (gateway/hub fix) without opening the token endpoint | — |
| `SLM_MCP_TOOLS` | Comma-separated list of MCP tools to expose (default: all) | — |
| `SLM_RATE_LIMIT_WRITE` | **NEW (v3.6.12)** Max dashboard write requests per window | `30` |
| `SLM_RATE_LIMIT_READ` | **NEW (v3.6.12)** Max dashboard read requests per window | `120` |
| `SLM_RATE_LIMIT_WINDOW` | **NEW (v3.6.12)** Rate-limit window in seconds | `60` |
| `SLM_MCP_ALL_TOOLS` | Set `1` to force-enable all tools regardless of mode | — |
| `SLM_MCP_MESH_TOOLS` | Set `1` to always include mesh tools | — |

### Mesh

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_MESH_HOST` | Bind address for mesh WebSocket broker | `127.0.0.1` |
| `SLM_MESH_WS_PORT` | WebSocket port for mesh broker | `8766` |
| `SLM_MESH_SHARED_SECRET` | Auth secret for the mesh HTTP API. Required when `SLM_MESH_HOST` is not localhost. Send as `Authorization: Bearer <secret>` (canonical) or `X-Mesh-Secret: <secret>` (legacy). | — |
| `SLM_MESH_PEER_URL` | Explicit peer URL to register with at startup | — |
| `SLM_MESH_DISCOVERY` | Discovery mode: `local` / `manual` | `local` |

> **Mesh API auth (v3.6.20):** When `SLM_MESH_SHARED_SECRET` is set, non-loopback callers must authenticate every `/mesh/*` request. The canonical header is `Authorization: Bearer <your-secret>` — this is what `RemoteSyncClient` sends automatically. The legacy `X-Mesh-Secret: <your-secret>` header is also accepted for backwards compatibility.
>
> Example: `curl http://192.168.50.144:8765/mesh/status -H "Authorization: Bearer <your-secret>"`

> **Note:** The variable is `SLM_MESH_WS_PORT` (not `SLM_MCP_WS_PORT`).
> `SLM_DAEMON_HOST` is canonical; `SLM_HOST` is the alias (not the other way around).

### Memory / health / workers

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_RSS_BUDGET_MB` | **NEW** Global RSS budget for the health monitor watchdog (0 = auto, 40% of RAM) | auto |
| `SLM_MAX_WORKER_MB` | Per-worker RSS limit before the per-worker watchdog triggers | `2048` |
| `SLM_MAX_EMBEDDING_WORKERS` | Max parallel embedding worker processes | `1` |
| `SLM_EMBED_WORKER_RSS_LIMIT_MB` | RSS limit per embedding worker process | `1500` |
| `SLM_EMBED_IDLE_TIMEOUT` | Seconds before an idle embedding worker exits | `120` |
| `SLM_EMBED_RECYCLE_AFTER` | Recycle embedding worker after N requests | `1000` |
| `SLM_EMBED_RESPONSE_TIMEOUT` | Timeout (s) for a single embedding request | `30` |
| `SLM_RERANKER_IDLE_TIMEOUT` | Seconds before an idle reranker worker exits | `120` |
| `SLM_MIN_AVAILABLE_MEMORY_GB` | Minimum free system RAM before SLM defers heavy operations | `1.0` |
| `SLM_TRIGRAM_BOOTSTRAP_RAM_MB` | Max RAM for trigram index bootstrap | `512` |

### Learning / bandit / signals

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLM_SIGNALS_ENABLED` | Enable implicit reward signal collection | `1` |
| `SLM_SIGNAL_QUEUE_MAX` | Max buffered signals before flush | `500` |
| `SLM_BANDIT_DISABLED` | Set `1` to disable the contextual bandit ranker | — |
| `SLM_BANDIT_ALPHA_CAP` | Maximum bandit learning rate | `0.3` |
| `SLM_BANDIT_REWARD_WINDOW_SEC` | Window (s) for reward aggregation | `3600` |
| `SLM_BANDIT_PLAYS_RETENTION_DAYS` | Keep bandit play history for N days | `30` |
| `SLM_DNA_SEED` | Random seed for reproducible bandit initialisation | — |

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
| `SLM_RATE_LIMIT_PER_AGENT` | Per-agent request cap per window | — |
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
| `SLM_RANKING` | Override ranking algorithm: `bandit` / `bm25` / `semantic` | auto |
| `SLM_INGEST_NO_GATE` | Set `1` to skip the ingest quality gate | — |
| `SLM_OBSERVE_DEBOUNCE_SEC` | Debounce window (s) for `observe` auto-capture | `5` |
| `SLM_TOPIC_SHIFT_LOG` | Set `1` to log topic-shift detection | — |
| `SLM_HOOK_DAEMON_URL` | Override daemon URL for hook integrations | — |
| `SLM_HOOK_DAEMON_TIMEOUT` | Timeout (s) for hook→daemon requests | `5` |
| `SLM_DISABLE_WARMUP_SIDE_EFFECTS` | Set `1` to suppress daemon auto-start in tests | — |
| `SLM_DISABLE_HF_DOWNLOAD` | Set `1` to block HuggingFace model downloads | — |
| `SLM_SKIP_DEP_CHECK` | Set `1` to skip dependency version checks | — |
| `SLM_NON_INTERACTIVE` | Set `1` to suppress interactive prompts | — |
| `SLM_DISABLE` | Set `1` to disable SLM entirely (no-op MCP tools) | — |
| `SLM_V2_PIPELINE_DISABLED` | Set `1` to force v1 store pipeline | — |
| `SLM_INJECTION_LEGACY` | Set `1` to use legacy context injection format | — |
| `SLM_SIGNER_KEY` | HMAC key for memory signing (anti-tamper) | — |

---

## Health config via config.json (v3.6.9+)

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
