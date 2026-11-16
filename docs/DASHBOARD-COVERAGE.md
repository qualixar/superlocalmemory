# Dashboard Coverage

The dashboard is a local operational surface over the SLM daemon (`slm dashboard`).
It helps an operator inspect and control a deployment; it does not replace
CLI/MCP traces or make a health signal a guarantee of recall quality.

## Navigation structure

The sidebar groups the panes as follows.

### Overview group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Answer Check | which check runs, a Try it panel, how often recall said it did not have the answer, timings, recent checks. See [Answer check](answer-check.md) | `slm recall` output and the response fields |
| Dashboard | runtime summary, daemon identity, storage and runtime health, diagnostics and recent activity, a **Needs your answer** card (disputed values with one-click answers, memories whose source file changed; hidden when empty), a **Help it get things right** card (blind audit questions, one at a time, never showing the judge's answer; hidden when none), and a dismissible **What's new** card that shows only what is not yet on | `slm status`, `slm doctor`, `slm disputes` |
| Brain | consolidation, behavioral patterns, feedback and outcomes, reward and soft-prompt state | mode and configuration, and the underlying data |

### Memory group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Knowledge Graph | graph neighborhoods, entities, scenes and relationships; click a node to see the memory behind it | `slm trace` for query-time graph participation |
| Memories | Recall Lab, **Summaries** and **Saved views** tabs, memory browsing, filtering, update and delete, provenance-oriented inspection. The **All memories** search shows the recall tier's sentence ("That isn't in SuperLocalMemory.", the closest memories, or the warming-up note) instead of an empty list, has a **Show closest matches** switch, and marks stubs. A memory's detail shows **When it was true**: the value it states, each value over time with its start date (shown now, replaced, in question), an open dispute with one-click answers, and a changed source file | write identity and scope policy for mutations |
| Entity Explorer | compiled entity summaries and timelines | source facts and entity recompilation |
| Multi-Agent Memory | which agents save and read memory | the agent ids on the memories themselves |
| Documents & Images | turn pictures and documents on (with progress and a restart button), a drop zone for pictures and PDFs, **Find a picture**, connected folders with the computer's own folder picker, and each PDF's state with **Try again** | `slm media status`, [Pictures and documents](pictures-and-documents.md) |

### Intelligence group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Skill Evolution | opt-in lineage, budgets, trigger and verification outcomes | the operator's evolution policy and model/provider boundary |

### System group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Health | the **Memory health** card (background improvement progress, repairs that wait for **Fix**, **Undo**; the post-upgrade store check and **Repair now** live in it), math-layer health, trust signals, connected agents and component health; a **Truth checks** card that runs `slm doctor --truth` read-only on request and says each result in plain words, which judges may act on their own (certificates) and how often a stronger judge, rechecking in the background, disagreed (second opinions) | `slm health`, `slm db integrity`, deployment resource limits |
| Governance | tabs for **Lifecycle** (memory ages, compaction preview, transitions), **Access & Users**, **Trust**, **Compliance** (retention policies, GDPR export and erasure, audit trail) and **Ingestion** | effective RBAC role on each API call; erasure completeness |
| Optimize | cache and compression controls, cache state and savings telemetry | actual proxy and MCP routing and invalidation behavior |

### Integrations group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| MCP & Tools | the active MCP tool profile and its tool count, the available profiles, per-IDE MCP configuration snippets, and a pointer to Connected apps. Changing the tool profile means editing the client's `.mcp.json` and restarting the client | `slm connect --list`, [MCP Tools Reference](mcp-tools.md) |
| Connected apps | optional [Web access](remote-access/README.md): link this computer with GitHub, choose what an app may do, copy the server URL and instructions, see each connected app and remove its access, turn Web access off | a real recall from the app |
| Mesh Peers | configured peers, inbox and outbox, pending messages and locks | authenticated transport on the real peer topology |
| Bot messages | every bot that sent or received through the mesh, by name, with mute, rename and retire | `mesh_peers`, [Host guides](remote-access/hosts.md#bot-messages-and-the-two-new-permissions) |
| Bounded Loops | the optional Bounded Loops evidence bridge and recorded loop runs | `slm loop history` |

### Config group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Settings | mode, provider, answer check (including **Calibrate on my store** and, under Online with Jev, **Background judges**: the owner agrees once to let the background judges ask Jev, or stops it; `slm consent`), auto-capture and auto-recall, scope defaults and other product configuration | effective configuration and secret management |
| Cloud Backup | backup destinations, schedule status, last backup time, manual trigger, encryption | actual backup file presence and restore path |

## Data and trust boundary

Panels can be empty when a feature is disabled, no data has been produced, or
the selected mode lacks a configured dependency. Recalled or displayed memory
is evidence, not an instruction authority. SLM keeps dynamic memory out of
high-trust IDE instruction files and renders it through the bounded context
path at runtime.

The **Access & Users** tab needs the login gate switched on to be meaningful in
a multi-user deployment. In personal mode (single user, loopback owner), the
RBAC controls are visible but authentication is not enforced.

## Release verification

For a release or deployment witness, validate each enabled surface end to end:

```bash
slm doctor
slm health
slm remember "dashboard witness" --sync --json
slm trace "dashboard witness"
```

For optional systems, also execute a real cache invalidation, peer
authentication exchange, provider call, adapter import, or Scale Engine
prepare/verify/promote/rollback cycle as applicable. Do not infer that these
subsystems are active solely because their dashboard workspace is visible.
