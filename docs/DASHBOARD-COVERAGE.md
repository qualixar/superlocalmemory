# Dashboard Coverage — V3.8.0

The dashboard is a local operational surface over the SLM daemon. It helps an
operator inspect and control a deployment; it does not replace CLI/MCP traces
or make a health signal a guarantee of recall quality.

## Navigation structure

V3.8.0 reorganized and expanded the navigation. The former "Operations" section
is now **Governance**, with dedicated sub-tabs for access control, data privacy,
and audit. A new **Integrations** group was added for MCP and tool management.

### Memory group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Memories | memory browsing, filtering, update/delete and provenance-oriented inspection | write identity and scope policy for mutations |
| Knowledge Graph | graph neighborhoods, entities, scenes and relationships | `slm trace` for query-time graph participation |
| Entity Explorer | compiled entity summaries and timelines | source facts and entity recompilation behavior |

### Brain group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Brain | consolidation, behavioral patterns, feedback/outcomes, reward and soft-prompt state | mode/configuration and the underlying data tables |
| Skill Evolution | opt-in lineage, budgets, trigger and verification outcomes | the operator's evolution policy and model/provider boundary |

### Governance group (new in v3.8.0)

The former "Operations" pane is now "Governance". Sub-tabs cover all compliance
and access controls in one place.

| Sub-tab | What it covers | What to verify separately |
|---|---|---|
| **Access & Users** | user list, role assignments (admin/member/viewer), workspace membership, login gate toggle | effective RBAC role on each API call; test with `slm doctor` |
| **Data Privacy** | PII redaction toggle, retention rule list (gdpr-30d / hipaa-7y / custom), GDPR export button, erasure with confirmation | erasure completeness — run `slm diagnostics export` after erasure |
| **Audit** | hash-chained audit trail viewer (store/recall/mutate/erase events), tamper detection status | `slm evidence verify` for bundle integrity |
| **Lifecycle & Trust** | ingestion-operation state and traces, maintenance work, consolidation triggers | a representative `slm remember --sync` receipt |

### Integrations group (new in v3.8.0)

| Workspace | Product surface | What to verify separately |
|---|---|---|
| **MCP & Tools** | active MCP profile name and tool count, profile switcher (runtime, no restart), per-IDE MCP config snippets for 15 IDEs | daemon restart is not required for profile changes; verify with `slm status` |
| **Cloud Backup** | backup destinations, schedule status, last backup timestamp, manual trigger | actual backup file presence and restore path |

### Operations group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Dashboard | runtime summary, daemon identity, storage/runtime health, diagnostics and recent activity | `slm status`, `slm doctor` |
| Health | dependency, math/lifecycle and process health | `slm health`, deployment resource limits |
| Mesh Peers | configured peers, inbox/outbox, pending messages and locks | authenticated transport on the real peer topology |

### Optimize group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Settings | mode, provider and product configuration | effective configuration and secret management |
| Optimize | cache/compression controls, cache state and savings telemetry | actual proxy/MCP routing and invalidation behavior |

## Data and trust boundary

Panels can be empty when a feature is disabled, no data has been produced, or
the selected mode lacks a configured dependency. Recalled or displayed memory
is evidence, not an instruction authority. SLM keeps dynamic memory out of
high-trust IDE instruction files and renders it through the bounded context
path at runtime.

The **Access & Users** tab requires `require_login = true` to be meaningful in
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
