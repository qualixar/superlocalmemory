# Compliance Controls and Limits

> SuperLocalMemory V3 documentation · not legal advice or certification

SuperLocalMemory exposes local storage, scoped recall, erasure mutations,
provenance, policy hooks, and audit records that can support a compliance
program. Whether a deployment meets the EU AI Act, GDPR, HIPAA, SOC 2, or any
other requirement depends on the use case, operator role, configuration,
surrounding systems, and verified operating procedures.

## Data-flow boundary

| Mode | Core memory path | Network considerations |
|---|---|---|
| A | Can store and retrieve through the configured local data root without a cloud model provider | Model/dependency downloads, connectors, backup, proxy providers, remote embedding endpoints, and other enabled integrations remain separate network paths |
| B | Adds an operator-configured Ollama endpoint | Verify where Ollama runs and which data it receives |
| C | Adds configured provider calls | Provider terms, retention, region, logging, access, and data-processing agreements become part of the deployment |

“Local-first” describes the default core state location. It does not mean that
every optional feature is offline or that operating-system processes cannot
access the data root.

## Erasure and lifecycle

Use the supported mutation commands for the installed release:

```bash
slm forget "query" --dry-run
slm forget "query" --yes
slm delete <fact_id> --yes
```

A complete erasure claim requires proof that deletion propagates through:

- relational facts and raw evidence;
- FTS, graph, temporal, vector, cache, and optional backend projections;
- context injection and active sessions;
- exports, backups, snapshots, logs, queues, and recovery artifacts; and
- any configured provider, connector, mesh peer, or external system.

The V3.7 release gate includes this lifecycle propagation matrix. Do not delete
only `memory.db` and call the deployment erased: the data root can also contain
configuration, logs, queues, credentials, models, derived indexes, and optional
backend state.

## Scope and access control

Personal memory is profile-scoped by default. Shared and global scopes are
opt-in and remain subject to the configured scope policy.

Profiles are a product-level organization and recall boundary inside one
installation. They are not a substitute for operating-system accounts,
filesystem permissions, process isolation, host isolation, or a multi-tenant
authorization service. Sensitive deployments should isolate data roots and
run the frozen release's cross-scope negative matrix.

Mutation authority does not come from a caller-selected `agent_id`. It derives
from a verified daemon capability and target instance, same-origin install
token, configured SLM API key, local loopback principal, or documented mesh
credential. Agent and IDE names remain attribution metadata.

## Transparency and scores

`slm trace` and the MCP/HTTP trace surfaces expose recorded channel
contributions and score metadata. Under [Score Contract
v2](retrieval-score-contract.md):

- `relevance_score` is query-relative relevance;
- `ranking_score` is internal diagnostic utility;
- `memory_confidence` is stored assertion metadata; and
- `answer_confidence` is `null` while calibration is unproven.

Traceability supports technical review; it is not automatically a legally
sufficient explanation. The current release must not present an uncalibrated
retrieval score as answer certainty.

## Audit and provenance

SLM can record provenance, policy events, and hash-chained audit entries. A hash
chain is tamper-evident when verified; it is not tamper-proof storage. Operators
must validate which operations are covered, who can modify or delete the log,
how long it is retained, where it is backed up, and how verification failures
are handled.

Use the installed dashboard and MCP audit/provenance tools documented by the
active tool profile. The repository does not claim a standalone `slm audit` or
`slm retention` CLI command unless it appears in `slm --help` for that release.

## Deployment checklist

1. Inventory the canonical data root and every external data flow.
2. Choose the minimum necessary mode, connectors, providers, backup, and mesh
   configuration.
3. Isolate operating-system identity, filesystem permissions, network access,
   API keys, mesh secrets, and backups.
4. Verify store, recall, scope, update, forget, export, backup, restore, and
   recovery on the frozen artifact.
5. Verify rendered context remains untrusted evidence under stored prompt-
   injection attacks.
6. Verify audit coverage, integrity checks, retention, and operator response.
7. Obtain qualified legal, privacy, and security review for the actual use
   case. SuperLocalMemory does not provide a BAA, DPA, or certification by
   itself.
