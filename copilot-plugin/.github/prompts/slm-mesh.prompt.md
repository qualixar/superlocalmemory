---
name: slm-mesh
description: Cross-session peer coordination via the SLM mesh network. Lets multiple AI agent sessions on the same machine discover each other, send messages, share lightweight state, and lock files to avoid conflicts. Requires full, power, or mesh MCP profile. All 8 tools are MCP-only — there is no CLI fallback.
version: "3.8.2"
agent: agent
tools:
  - mesh_summary
  - mesh_peers
  - mesh_send
  - mesh_inbox
  - mesh_state
  - mesh_lock
  - mesh_events
  - mesh_status
  - Bash
---

# slm-mesh — Cross-Session Peer Coordination

The mesh network lets multiple AI agent sessions on the same machine discover
each other and coordinate in real time — without writing to the persistent
memory store. Mesh messages are transient (48-hour TTL); they complement memory
(which is durable) rather than replacing it.

Mesh is local-only: it uses the SLM daemon as a local broker. No data leaves
the machine.

---

## Profile requirement

Mesh tools are available in the `full`, `power`, and `mesh` MCP profiles.
Confirm the active profile with `slm status` before calling mesh tools. If the
tools are not available, switch to `full` profile with `switch_profile("full")`
(requires `code` or higher active profile). See `slm-profile`.

---

## Tool reference

### 1. `mesh_summary` — announce what this session is doing

```
mesh_summary(summary: str = "") -> dict
```

Call at session start to register on the mesh and announce your purpose. Other
sessions can see your summary via `mesh_peers`. The session stays alive via
automatic heartbeat.

```
mesh_summary(summary="Refactoring auth module in api/src/auth/")
```

Response: `{peer_id, summary, project_path, registered, heartbeat_active, broker_response}`

Call this once at the start of any session that will participate in the mesh.
The peer registration happens automatically at MCP startup, but calling
`mesh_summary` sets the human-readable description that other agents see.

---

### 2. `mesh_peers` — list active sessions

```
mesh_peers() -> dict
```

Returns all active peer sessions on this machine.

```
mesh_peers()
```

Response: `{peers: [{peer_id, summary, project_path, last_seen}], count, my_peer_id}`

Use this to discover other sessions before sending a message or checking for
conflicts.

---

### 3. `mesh_send` — send a message to another session

```
mesh_send(
  to: str,       # peer_id | "broadcast" | "project:/path/to/dir"
  message: str,  # max 4 KB — use file paths for large data
) -> dict
```

Send a targeted, broadcast, or project-wide message.

```
# Direct message to a specific peer
peers = await mesh_peers()
target_id = peers["peers"][0]["peer_id"]
mesh_send(to=target_id, message="I'm starting work on auth/handler.py — please hold off")

# Broadcast to all sessions
mesh_send(to="broadcast", message="Deploying to staging in 5 minutes")

# Message all sessions working in the same project
mesh_send(to="project:/Users/me/myproject", message="Tests are green on main")
```

**4 KB message cap.** For large payloads (diffs, file contents), write to a file
and send the path instead. The circuit breaker opens automatically if the daemon
is repeatedly unreachable — `mesh_send` returns `ok: false` in that case.

---

### 4. `mesh_inbox` — read messages sent to this session

```
mesh_inbox() -> dict
```

Returns unread messages (direct, broadcast, and project-targeted). Messages are
automatically marked as read after retrieval.

```
inbox = await mesh_inbox()
for msg in inbox["messages"]:
    print(msg["from"], msg["content"])
```

Response: `{messages: [{id, from, content, sent_at, read}], count, unread}`

Messages auto-expire after 48 hours.

---

### 5. `mesh_state` — get or set shared coordination state

```
mesh_state(
  key: str = "",
  value: str = "",
  action: str = "get",   # "get" | "set"
) -> dict
```

Shared state is visible to all authenticated peers. Use it for non-secret
coordination metadata: feature flags, task assignments, progress markers.

```
# Set state
mesh_state(key="deploy_in_progress", value="true", action="set")
mesh_state(key="current_reviewer", value=my_peer_id, action="set")

# Read one key
mesh_state(key="deploy_in_progress", action="get")

# Read all state
mesh_state(action="get")
```

**Security constraint:** Credentials, tokens, passwords, and API keys are
rejected by the broker. Never store secrets in shared state.

---

### 6. `mesh_lock` — file lock coordination

```
mesh_lock(
  file_path: str,         # must be an absolute path
  action: str = "query",  # "query" | "acquire" | "release"
) -> dict
```

Check, acquire, or release a file lock before editing a shared file.

```
# Step 1: check if the file is already locked
lock = await mesh_lock(file_path="/abs/path/to/auth/handler.py", action="query")

if lock.get("locked"):
    print(f"File is locked by {lock['locked_by']} — wait")
else:
    # Step 2: acquire the lock
    mesh_lock(file_path="/abs/path/to/auth/handler.py", action="acquire")

    # ... edit the file ...

    # Step 3: release the lock when done
    mesh_lock(file_path="/abs/path/to/auth/handler.py", action="release")
```

`file_path` must be an absolute path (starts with `/` on Unix, drive letter on
Windows). Relative paths are rejected.

---

### 7. `mesh_events` — recent mesh activity log

```
mesh_events() -> dict
```

Returns the activity log for the mesh network: peer joins, leaves, messages sent,
and state changes. Use to understand what other sessions have been doing.

---

### 8. `mesh_status` — mesh broker health

```
mesh_status() -> dict
```

Returns broker uptime, peer count, and connection health. Use at session start
to confirm the mesh is available before relying on coordination.

Response includes: `broker_up`, `peer_count`, `uptime_seconds`, `my_peer_id`,
`heartbeat_active`.

---

## Common workflow: parallel agents coordinating on a shared repo

```
# Both agents call at session start:
await mesh_summary(summary="Working on feature/auth-refactor")

# Agent A: check who else is active
peers = await mesh_peers()
# → sees Agent B working on the same project

# Agent A: before editing a shared file
lock = await mesh_lock("/repo/src/auth/handler.py", action="query")
if not lock.get("locked"):
    await mesh_lock("/repo/src/auth/handler.py", action="acquire")
    # ... edit handler.py ...
    await mesh_lock("/repo/src/auth/handler.py", action="release")

# Agent A: after finishing a phase
await mesh_send(to="project:/repo", message="Auth refactor complete — handler.py ready for review")

# Agent B: check inbox
inbox = await mesh_inbox()
```

---

## Mesh vs memory: when to use which

| Need | Use |
|------|-----|
| Ephemeral coordination signal (< 48h) | `mesh_send` / `mesh_state` |
| Durable fact across sessions/days | `remember` |
| File conflict prevention | `mesh_lock` |
| Cross-profile fact sharing | `scope="shared"/"global"` on `remember` |
| Session announcement | `mesh_summary` |
| Finding parallel agents | `mesh_peers` |

---

## Error handling

All 8 mesh tools return structured errors — they never raise exceptions.

| Error | Cause | Action |
|-------|-------|--------|
| `broker_up: false` from `mesh_status` | Daemon not running or mesh not configured | Run `slm status` to check daemon health |
| `ok: false` from `mesh_send` with circuit-breaker message | Repeated daemon unreachability | Daemon unreachable; stop sending until broker is up |
| `ok: false` from `mesh_lock` | Lock operation failed | Check `file_path` is absolute; retry once |
| Empty `peers` from `mesh_peers` | No other sessions registered | You're the only active session |

Mesh failures are non-fatal for the primary task. If `mesh_status` shows
`broker_up: false`, proceed without mesh coordination — do not block work on
mesh availability.

---

## Related skills

- `slm-profile` — activate full/power/mesh profile to access mesh tools
- `slm-scope` — for durable cross-profile sharing (complement to transient mesh state)
- `slm-remember` — persist coordination decisions that should survive session end
- `slm-governance` — enterprise governance of mesh (who can send/receive)

---

*SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later*
