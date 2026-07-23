# Multi-Agent Memory

SuperLocalMemory provides three coordination mechanisms for multi-agent
deployments: shared memory scopes, SLM Mesh peer coordination, and bounded
loops. Each mechanism serves a distinct use case and operates independently.

## Shared memory scopes

By default, a memory is `personal` — visible only to the writing profile.
Two additional scopes allow cross-agent access:

| Scope | Visibility |
|---|---|
| `personal` | Owning profile only (default) |
| `shared` | Explicitly named profiles, authorized at write time |
| `global` | All profiles on the same instance |

Write a shared memory:

```bash
slm remember "Deploy approved for v2.3.1" --scope shared --shared-with agent-b,agent-c
```

Or via MCP:

```python
await remember(
    content="Deploy approved for v2.3.1",
    scope="shared",
    shared_with="agent-b,agent-c",
)
```

`shared` and `global` recall is default-deny until scope visibility is
explicitly enabled for the reading profile. This is a local authorization
contract on the same SLM instance, not a synonym for Mesh peer coordination
across machines.

## SLM Mesh

Mesh provides authenticated peer messaging, distributed locks, shared state,
and an inbox/outbox queue across SLM instances on different machines or
processes. It does not replicate the full memory database; it coordinates peers.

Mesh tools are available in the `full`, `power`, and `whole` MCP profiles.
The `mesh` profile exposes only the eight Mesh tools without the rest of the
full profile.

| Tool | Description |
|---|---|
| `mesh_send` | Send a message to a named peer |
| `mesh_inbox` | Read incoming peer messages |
| `mesh_peers` | List known peers and their status |
| `mesh_lock` | Acquire a distributed lock |
| `mesh_state` | Read or write shared mesh state |
| `mesh_events` | Read the mesh event stream |
| `mesh_status` | Connectivity and health |
| `mesh_summary` | Summary of recent mesh activity |

CLI equivalent: `slm mesh peers | send | inbox | status`

Mesh coordinates peers; it is not a distributed replicated memory database.
Deployment configuration determines the security and availability posture of
a Mesh network.

## Bounded loops

Bounded loops let one agent wait, under strict iteration and time limits, for
a condition that another agent will write into shared SLM memory.

```python
# Agent A writes the signal
await remember("review complete: PR #42 is approved", tags="review,pr-42")

# Agent B waits for that signal (via MCP slm_loop_run)
result = await slm_loop_run(
    name="wait-for-pr-review",
    gate_query="PR #42 approved",
    gate_min_score=0.5,
    max_iterations=30,
    max_wallclock_s=60.0,
)
```

The gate is a recall query: the loop converges the first lap a memory
matching the query becomes retrievable at the required confidence level.
Neither agent polls a shared queue or a file; coordination flows through
memory both agents already use.

See [Bounded Loops](Bounded-Loops) for the full tool reference.

## Dashboard: Multi-Agent panes

The local dashboard exposes two panes relevant to multi-agent deployments:

**Mesh Peers** — live view of connected peers, last seen, and recent mesh
events. Use alongside `slm mesh peers` and `slm mesh status`.

**MCP & Tools** — profile management pane, added in V3.8.0. Shows the active
profile, tool count, and lets an operator switch profiles without restarting
the daemon. Tools active under each profile:

| Profile | Tools |
|---|---|
| `core` | 14 |
| `code` | 24 |
| `full` | 42 |
| `power` | 54 |
| `whole` | 84 |

## Choosing a coordination mechanism

| Need | Mechanism |
|---|---|
| One agent reads what another agent wrote on the same instance | Shared memory scope (`--scope shared`) |
| Agent waits for a condition to appear in memory before proceeding | Bounded loop (`slm_loop_run`) |
| Agents on different machines exchange signals or share locks | SLM Mesh |
| Agent needs to know when a parallel agent completes a task | Bounded loop on the memory the other agent writes |

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
