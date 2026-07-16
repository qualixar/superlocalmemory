# Dashboard Coverage — V3.7

The dashboard is a local operational surface over the SLM daemon. It helps an
operator inspect and control a deployment; it does not replace CLI/MCP traces
or make a health signal a guarantee of recall quality.

## Workspaces

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Dashboard | runtime summary, daemon activity and recent memory operations | `slm status`, `slm doctor` |
| Brain | consolidation, behavioral patterns, feedback/outcomes, reward and soft-prompt state | mode/configuration and the underlying data tables |
| Knowledge Graph | graph neighborhoods, entities, scenes and relationships | `slm trace` for query-time graph participation |
| Memories | memory browsing, filtering, update/delete and provenance-oriented inspection | write identity and scope policy for mutations |
| Health | dependency, math/lifecycle and process health | `slm health`, deployment resource limits |
| Operations | ingestion-operation state, traces, maintenance and lifecycle work | a representative `remember --sync` receipt |
| Entity Explorer | compiled entity summaries and timelines | source facts and entity recompilation behavior |
| Skill Evolution | opt-in lineage, budgets, trigger and verification outcomes | the operator’s evolution policy and model/provider boundary |
| Mesh Peers | configured peers, inbox/outbox, pending messages and locks | authenticated transport on the real peer topology |
| Settings | mode, provider and product configuration | effective configuration and secret management |
| Optimize | cache/compression controls, cache state and savings telemetry | actual proxy/MCP routing and invalidation behavior |

## Data and trust boundary

Panels can be empty when a feature is disabled, no data has been produced, or
the selected mode lacks a configured dependency. Recalled or displayed memory
is evidence, not an instruction authority. SLM keeps dynamic memory out of
high-trust IDE instruction files and renders it through the bounded context
path at runtime.

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
