# Multi-Machine Mesh (v3.4.48)

SuperLocalMemory v3.4.48 enables two machines on the same LAN to share agent peer lists and route messages cross-machine — zero manual configuration required when mDNS is available.

## How It Works

```
┌─ Mac M4 (192.168.1.100) ──────────────┐     ┌─ Mac M5 (192.168.1.101) ──────────────┐
│  SLM daemon (:8765)                    │     │  SLM daemon (:8765)                    │
│  RemoteSyncClient → polls M5 /peers    │◄───►│  RemoteSyncClient → polls M4 /peers    │
│                                        │HTTP │                                        │
│  ┌──────────┐  ┌──────────┐           │     │  ┌──────────┐  ┌──────────┐           │
│  │  Claude  │  │  Cursor  │           │     │  │  Claude  │  │  iTerm   │           │
│  └──────────┘  └──────────┘           │     │  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘     └────────────────────────────────────────┘
```

- Each machine runs its own SLM daemon.
- `RemoteSyncClient` runs a background thread that fetches `/mesh/peers` from the remote machine every 30 seconds.
- Remote peers appear in `list_all_peers()` and the `mesh_peers` MCP tool.
- When an agent sends to a remote peer, SLM proxies the request to the remote machine's `/mesh/send`.
- mDNS auto-discovery (`_slm-mesh._tcp`) eliminates manual IP configuration when available.

## Quick Start

### On M4 (the remote machine — set `SLM_MESH_SHARED_SECRET`)

```bash
export SLM_MESH_SHARED_SECRET=your-shared-secret-here
slm start   # or however you start SLM
```

### On M5 (add `SLM_MESH_PEER_URL` to your env/MCP config)

```bash
export SLM_MESH_PEER_URL=http://192.168.1.100:8765
export SLM_MESH_SHARED_SECRET=your-shared-secret-here
```

Or in Claude Code `settings.json`:
```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"],
      "env": {
        "SLM_MESH_PEER_URL": "http://192.168.1.100:8765",
        "SLM_MESH_SHARED_SECRET": "your-shared-secret-here"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SLM_MESH_PEER_URL` | unset | Full URL of remote SLM (e.g., `http://192.168.1.100:8765`) |
| `SLM_MESH_SHARED_SECRET` | unset | Shared auth token — same value on both machines |
| `SLM_MESH_HOST` | `127.0.0.1` | Host to bind this SLM's mesh listener |
| `SLM_MESH_WS_PORT` | `7900` | Port for mDNS service announcement |
| `SLM_MESH_DISCOVERY` | `on` | Set to `off` to disable mDNS auto-discovery |

## mDNS Auto-Discovery

When `SLM_MESH_DISCOVERY=on` (default) and `zeroconf` is installed, SLM automatically:
1. Announces itself as `_slm-mesh._tcp` on the local network.
2. Discovers other SLM instances and updates `SLM_MESH_PEER_URL` accordingly.
3. No manual IP configuration needed — works on Apple Bonjour networks out of the box.

Install zeroconf support:
```bash
python -m pip install "superlocalmemory[mesh]"
# or
python -m pip install zeroconf
```

Run these commands only while the SLM Python virtual environment is active.

## MCP Tools — Unchanged

All 8 MCP tools work identically in multi-machine mode. No agent code changes needed.

| Tool | Multi-Machine Behavior |
|---|---|
| `mesh_peers` | Returns local + remote peers merged |
| `mesh_send` | Auto-routes to remote SLM if target is remote |
| `mesh_inbox` | Shows messages from both local and remote peers |
| `mesh_state` | Local state only (no distributed consensus) |
| `mesh_lock` | Local locks only (per-machine) |
| `mesh_status` | Shows local broker status |
| `mesh_events` | Local events only |
| `mesh_summary` | Updates local peer summary |

## Security

- **LAN-only.** Not designed for internet exposure.
- **Shared secret** (`SLM_MESH_SHARED_SECRET`) is used as a bearer token. Required when `SLM_MESH_HOST` is not localhost.
- The `/mesh/peers` endpoint requires `Authorization: Bearer <shared_secret>` from remote callers.
- Never commit your shared secret to git. Use environment variables or secrets management.

## Sync Behavior

- Remote peer sync runs every **30 seconds** in a background thread.
- Stale peers (absent from remote for one sync cycle) are removed from local `_remote_peers`.
- Sync failures are logged but do NOT crash SLM. Single-machine mode continues unaffected.
- Message proxying is synchronous — `mesh_send` to a remote peer waits for the proxy response.

## Troubleshooting

**Remote peers not appearing:**
1. Confirm `SLM_MESH_PEER_URL` is set on M5.
2. Check M4's SLM is running: `curl http://192.168.1.100:8765/health`
3. Verify shared secret matches on both machines.
4. Wait up to 30 seconds for the first sync cycle.

**Authentication errors:**
- Ensure `SLM_MESH_SHARED_SECRET` is identical on both machines.
- The secret must not be empty.

**mDNS not working:**
- Install zeroconf: `pip install zeroconf`
- Check firewall allows mDNS (UDP port 5353).
- Disable with `SLM_MESH_DISCOVERY=off` and use manual `SLM_MESH_PEER_URL` instead.

**Send to remote peer fails:**
- Verify M4's SLM is reachable from M5: `curl http://192.168.1.100:8765/mesh/status -H 'Authorization: Bearer <secret>'`
- Check the target peer is listed by M4's `/mesh/peers` endpoint.

## Non-Goals

- No distributed consensus (each machine has its own SQLite, no synced state).
- No internet-wide mesh (LAN only).
- No real-time WS push from Python SLM (polling-based sync, 30s interval). For sub-second push, use `slm-mesh` (TypeScript) which has full WebSocket push routing.
- No multi-user support (single user, multiple machines).
