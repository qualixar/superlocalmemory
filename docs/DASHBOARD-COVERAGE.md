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
| Dashboard | runtime summary, daemon identity, storage and runtime health, diagnostics and recent activity, a **Needs your answer** card (disputed values with one-click answers, memories whose source file changed; hidden when empty), a **Help it get things right** card (blind audit questions, one at a time, never showing the judge's answer; hidden when none), a **Proposed changes** card (sets of changes SLM suggests, in plain words, with **Apply**, **Reject** and, on a set you applied, **Undo**; hidden when there is nothing to show), a **Living wiki** card (pages compiled from your memories, each sentence shown with the memory it cites and **Open memory**, the count of sentences withheld in words, **Rebuild pages** and, while the Markdown wiki is on, **Read my edits**; hidden when there are no pages and the Markdown wiki is off; `GET /api/wiki/pages`, `GET /api/wiki/pages/{id}`, `POST /api/wiki/rebuild`, `POST /api/wiki/sync`), a **Held back from instructions** card (memories that asked to be a pin, rule or core memory but did not come from you, where each came from, **Promote** and **Promote all**; shown once the instruction firewall is on or in shadow, `GET /api/trust/pending`, `POST /api/trust/promote/{fact}`), a **Recent decisions by your agents** card (who decided what, through which channel and when: promotions, settled disputes, applied or rolled-back proposal sets, blind audit labels, retired memories brought back, each with an **Undo**; hidden until something was decided; `GET /api/decisions`, `POST /api/decisions/{id}/undo`), a **Bring your memories from another app** card (WS-9: **Turn on importing** while the pack is off, then **choose a file** from ChatGPT, Claude, Mem0, Letta or a notes folder, a preview in words such as "We found 214 conversations from ChatGPT, March 2024 to October 2026. 12 empty ones will be skipped.", **Import**, a progress bar with **Stop**, and when it is done **Undo this import** and the earlier imports; plain error words, no ids; `GET /api/import/runs`, `POST /api/import/preview`, `POST /api/import/run`, `POST /api/import/runs/{id}/cancel`, `POST /api/import/runs/{id}/undo`, `POST /api/import/switch`), a dismissible banner about what SLM measures (**Learn more**, **Turn off**, **OK**; shown until the notice is acknowledged), a dismissible **What's new** card that shows only what is not yet on, and under it a dismissible **What memory can do now** card: one plain line for each 4.2.30 feature (memory on every prompt, reminders on cues, what SLM measures, where a memory came from, held back from instructions, secrets stored as placeholders, signed memories, why was this shown, proposed changes, living wiki, bring memories from other apps, better pictures and documents, your agents decide and you can undo), each with a link to its page or guide and shown only while that feature is not turned on (`GET /api/whatsnew/state`, read-only) | `slm status`, `slm doctor`, `slm disputes`, `slm proposals`, `slm trust`, `slm wiki`, `slm import`, `slm measurement status` |
| Brain | consolidation, behavioral patterns, feedback and outcomes, reward and soft-prompt state | mode and configuration, and the underlying data |

### Memory group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Knowledge Graph | graph neighborhoods, entities, scenes and relationships; click a node to see the memory behind it | `slm trace` for query-time graph participation |
| Memories | Recall Lab, **Summaries** and **Saved views** tabs, memory browsing, filtering, update and delete, provenance-oriented inspection. The **All memories** search shows the recall tier's sentence ("That isn't in SuperLocalMemory.", the closest memories, or the warming-up note) instead of an empty list, has a **Show closest matches** switch, and marks stubs. A memory's detail shows **Where this came from** (the origin class in plain words: you said this, an agent's own words, a tool's output, a web page, an imported file, written by a model, or unknown for memories saved before 4.2.30, with a note when it was built from something less trusted) and **When it was true**: the value it states, each value over time with its start date (shown now, replaced, in question), an open dispute with one-click answers, and a changed source file; and, on request, **Why was this shown?** (where it came from and why that counts, whether the instruction firewall holds it back and who approved it, whether its signature still matches, and the last times an agent was shown it with the reason for each; `GET /api/trust/why/{fact_id}`, read-only) | write identity and scope policy for mutations |
| Entity Explorer | compiled entity summaries and timelines | source facts and entity recompilation |
| Multi-Agent Memory | which agents save and read memory | the agent ids on the memories themselves |
| Documents & Images | turn pictures and documents on (with progress and a restart button), a drop zone for pictures and PDFs, **Find a picture**, connected folders with the computer's own folder picker, and each PDF's state with **Try again**, and a card **Better pictures and documents** (4.2.30, WS-8) with three plain switches: describe pictures (a 0.5 GB download), read scanned documents better (1.9 GB) and remember where text was found; the size is shown and confirmed before anything downloads, progress is shown, and Turn off keeps your memories (the memory detail's **Why was this shown?** then says "from page 3, top left") | `slm media status`, `slm media pack`, [Pictures and documents](pictures-and-documents.md), [Better pictures and documents](media-pack.md) |

### Intelligence group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Skill Evolution | opt-in lineage, budgets, trigger and verification outcomes | the operator's evolution policy and model/provider boundary |

### System group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Health | the **Memory health** card (background improvement progress, repairs that wait for **Fix**, **Undo**; the post-upgrade store check and **Repair now** live in it), math-layer health, trust signals, connected agents and component health; a **Truth checks** card that runs `slm doctor --truth` read-only on request and says each result in plain words, which judges may act on their own (certificates) and how often a stronger judge, rechecking in the background, disagreed (second opinions); a **What memory did for you** card (on request, `GET /api/outcomes/health`): deliveries in the last 30 days, how often memories are used with the most and rarely used (each opens the memory), and whether memory measurably helps, a claim only when the anytime-valid range excludes zero, and whether each part of SLM's learning is learning, stalled, off, waiting for agents or not connected; the same **Truth checks** card also says whether signed memories still match their signatures and whether the audit log is intact; a **Shown to your agents** card (on request, `GET /api/delivered/recent`, read-only): what your agents were recently given, grouped by conversation (start of a conversation, a prompt, a cue, or when an agent asked), each memory with how it was chosen, whether it is current, disputed or from a file that changed, how old it is and, once the origin check is on, where it came from, plus how many prompts the need check skipped and the bound it was measured at (the rules skip a needed prompt at most 10% of the time, P(skipped | needed) <= 10%, and the score skips nothing until the prompts that needed memory weigh at least 180, so a young store shows no score skips) and how many memories the harm check would withhold (it only watches, and a memory is tested only when it was both shown and left out in enough sessions: with none of those the card says "Not testable yet: no memory has been left out often enough to compare" instead of claiming none was measured as making an agent worse); a **Trust overview** card (on request, `GET /api/trust/overview`, read-only): how many memories came from each kind of source, how many wait for your OK (a link to that card), the state of signed memories (signed, not signed yet, changed since signing, waiting to be signed again), credentials held as placeholders by type, and the audit log's state | `slm health`, `slm health --memory`, `slm db integrity`, deployment resource limits |
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
| Settings | mode, provider, answer check (including **Calibrate on my store** and, under Online with Jev, **Background judges**: the owner agrees once to let the background judges ask Jev, or stops it; `slm consent`), the **Help SLM measure what works** card (the notice in plain words and one On/Off switch for withholding and swapping; `slm measurement`), the **Turn off everything new in 4.2.30** card (D-23: what the kill switch does and that it does not undo retirements, the state, one button that acts at once, and **Bring back memories that 4.2.30 retired**, which counts first and asks before it does it; `GET` and `POST /api/flags/kill`, `POST /api/flags/rollback-retirements`; `slm flags`), auto-capture and auto-recall, scope defaults and other product configuration | effective configuration and secret management |
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
