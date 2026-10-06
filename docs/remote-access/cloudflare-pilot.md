# Cloudflare pilot: operator guide

This is the provider/operator sequence, not a task for end users. Finish public authorization, enrollment and automatic companion delivery before calling this an end-to-end service.

## What Cloudflare hosts

A Worker acts as the public HTTPS front door. An installation-bound Durable Object maintains relay connection state. A separate authorization boundary validates owners, clients, consent and remote entitlement. The laptop opens the connection outward; Cloudflare does not host the user's canonical memory database.

Cloudflare supports SQLite-backed Durable Objects on Workers Free. Free limits stop operations when exceeded; they do not provide unlimited pilot capacity. Hibernating sockets can avoid idle duration charges. Consult the [current pricing and limits](https://developers.cloudflare.com/durable-objects/platform/pricing/) before deployment. A Free website zone and a Workers plan are different products; do not alter the existing website/Vercel DNS routes to enable this service.

## Deployment sequence

| Phase | Operator work | Exit evidence |
| --- | --- | --- |
| Build | Complete protected local enrollment, public OAuth/grants/entitlement, discovery filtering and packaged companion | Reviewed source and composed local tests; no unauthenticated origin forwarding |
| Staging | Configure a separate Worker environment, SQLite-backed DO namespace, authorization storage and protected secrets | Deployment artifact/version, denied anonymous access and working authenticated synthetic fixture |
| Endpoint | Configure operator-owned staging hostname/HTTPS and OAuth callback metadata | Certificate, discovery metadata, redirect validation and audience tests |
| Laptop | Enroll the candidate through its dashboard against an isolated data root | Journal recovery, owner-only credential storage, verified connector, preserved local configuration |
| Native host | Inspect the target host's actual custom-MCP controls and authentication support | Real login, consent, tool discovery and roundtrip; account/host limitations recorded |
| Acceptance | Run isolated real-memory tests, failure tests and concurrent local-host regressions | [Acceptance evidence](acceptance.md), rollback instructions and unresolved gaps |
| Release | Reconcile the completed 4.1.22 base; validate npm/PyPI packaging and publish only after release gates pass | Release provenance, host certification, user-facing status/docs match shipped behavior |

Planned production hostnames are `mcp.superlocalmemory.com`, `auth.superlocalmemory.com` and `connect.superlocalmemory.com`. They are proposed architecture names, not claims of active endpoints. Use separate staging names and register exact supported URLs in the candidate configuration. Current connector validation accepts the curated connection hostname; any staging adjustment requires explicit code/config validation and regression tests, never arbitrary endpoint input from an AI request.

The runtime test Worker under `integrations/remote-gateway/tests/runtime/` is a private fixture, not a production public entry point. Do not deploy it as the service.

## Access and secrets

Authenticate the Cloudflare operator through the normal account flow; use narrowly scoped deployment credentials and protected secret storage. Never place tokens, local installation keys or user memory content in public docs, screenshots, Git, browser storage or diagnostic logs. Cloud login, AI-client OAuth and laptop connector authentication have separate identities.

End users receive a dashboard workflow and host recipe. They never need a Cloudflare account, API token, named Tunnel or DNS setup. Connection grants must bind owner, installation, profile and client; a valid device socket alone does not authorize a public tool call.

## Usage and rollback

Measure requests, CPU, active DO duration, storage operations, reconnect frequency and memory-call latency during the pilot. Do not infer a supported user count or hard monthly cap from a plan's entry price. Choose the Workers plan against measured rollout usage; a bounded Free pilot can be evaluated before paid activation.

Keep the prior Worker deployment and schema-compatible connector artifact. Disable new enrollment if admission regresses; revoke affected remote grants through the authoritative registry. Local SLM must remain operational. Local engine rollback and memory-schema rollback are separate procedures; do not downgrade an upgraded live database without a verified recovery plan.
