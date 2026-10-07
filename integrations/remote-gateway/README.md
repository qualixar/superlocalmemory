# SLM remote gateway — preparatory domain package

Implemented: permission intersection, enrollment reducer, bounded relay codec, private hibernating Durable Object delivery, dashboard initiation/cancellation, a durable local enrollment journal and authenticated local API, and outbound connector request/lifecycle components. This private package is not a deployed public gateway, a packaged companion, or a completed release. Merge and verify the completed 4.1.22 before final release; ongoing development does not require a separate M4 acceptance decision.

## Local enrollment integration checkpoint

The existing daemon registers `/api/v3/connections/status`, `/initiate`, and `/{connection_id}/cancel`. Remote access remains unavailable without an explicitly configured service; optional router import failures preserve local startup. The journal stores enrollment metadata separately from memory databases with owner/profile isolation, durable idempotency, dispatch leases and cancellation fences. Mutations require explicit installation credentials and same-origin loopback requests.

The dashboard accepts only a connection-bound HTTPS owner-login URL. It recovers an uncertain request through its non-secret intent key, permits authenticated versioned cancellation, and clears the cancelled browser intent so a fresh request works after reload. Pending receipts never establish connectivity. `cleanup_pending` remains true until authoritative gateway revocation; no cloud cleanup is fabricated.

On 2026-10-07, verification passed 105 Python tests (56 enrollment/API/ACL probes plus 49 existing authentication/loopback regressions), 218 Node tests and 18 local workerd tests; TypeScript checks passed. Enrollment/API branch-aware coverage is 89%. Windows ACL probes are mocked. Five additional profile/MCP integration checks remain blocked by the isolated environment's broken OpenTelemetry entry-point metadata and missing MCPServer module. See [batch evidence](evidence/enrollment-dashboard-green.json). Historical counts below describe earlier checkpoints.

## Local core and optional remote service

Local SLM stays free through npm/PyPI with its complete local tools, mesh, configured Jev/Laya, and Claude Code/Codex integrations. No hosted account or subscription may gate local use. Remote access is off by default and initiated from the existing dashboard. Future subscription enforcement applies only to hosted enrollment and remote requests. Gateway loss, expiry, revocation, and companion failure must leave local SLM operational. Existing local profiles/providers/credentials/hooks are preserved. Remote consent limits the remote catalog only. These are release requirements; fresh installer/concurrent-client/provider regression evidence remains pending.

## Approved user journey

SLM works locally with remote access off. The user optionally selects MCP & Integrations → Add AI connection in the SLM UI. SLM will enroll the owned installation, provision its shared relay binding, manage its local connector and guide host consent. No end-user Cloudflare account, DNS, terminal command or infrastructure secret-paste step. Private-server mode B remains deferred.

## Module boundaries

- `src/contracts.ts`: actor, connection and immutable per-client authorization types from the reviewed design.
- `src/request-policy.ts`: ownership/audience checks, scope/tool intersection, era method allowlist and explicit profile/correction/sharing restrictions. It returns an internal effective grant snapshot; do not serialize that whole object into client responses. Discovery response filtering must use its allowedTools.
- `src/enrollment.ts`: pure enrollment transitions with opt-in, owner/device binding, expiry, version and connector generation checks. It selects known persistence fields and has no provider/credential side effects.

Inputs to policy must already have validated schemas, verified provider identity and current authoritative registry records. Inputs to the reducer must come from authenticated UI operations or validated internal provider/connector events. Never trust a client-provided `verified` event. Capture async operation version/generation when dispatching it; never replace them with the latest record values before accepting its callback.

Still required: public MCP schema/token admission, authoritative grant registry/CAS, OAuth refresh/revocation adapter, protected enrollment journal/credential loader, dashboard backend handlers, remote entitlement, capacity/abuse controls, admission-to-origin composition/discovery filtering, automatic companion packaging, and native-host proof. Domain state/version checks alone cannot establish these integration guarantees.

## Reproduce checks

Use Node >=22.18.0; TypeScript7.0.2 is pinned in package-lock. From this directory install dependencies with `npm ci --ignore-scripts --no-audit --no-fund`. Run `npm run typecheck`, `npm test`, and `npm run test:coverage`. For release evidence use a minimal process environment and the direct pinned compiler/test runner, as done for the GREEN checkpoint. Tests use only synthetic identifiers and reserved example.com hostnames; they make no network/database/provider calls.

The original domain evidence records57 RED cases and98 GREEN cases. Current checkpoints have214 Node and18 local workerd cases passing. Tests include real loopback WebSocket/HTTP transport and forced DO eviction, using synthetic credentials/data. They do not establish real SLM engine, cloud deployment, OAuth or Musebot compatibility. Coverage is not a live-security or end-to-end certification.

## Local connector checkpoint

`src/local-connector.ts` validates socket generations and forwards bounded requests to one fixed loopback MCP origin using local `X-Install-Token`/`X-SLM-API-Key` headers. It disables redirects, cancels stalled body reads, enforces deadlines and fences late responses. Locally curated schema parameter headers are copied/frozen; credentials stay out of payloads/URLs.

`src/connector-supervisor.ts` supplies opt-in-only startup, epoch-fenced reconnects, credential expiry, ready handshake, heartbeat, bounded backpressure and cleanup. `src/node-connector.mjs` supplies the real ws adapter; ws8.21.0 is pinned. This is a companion implementation checkpoint, not an installer instruction: users must never be required to install Node/npm themselves for this add-on. Protected enrollment loading, packaged runtime delivery and daemon integration are pending. Request executor22cases and lifecycle/wire26cases pass; both reviewers approved these bounded components.

## Approved relay transport and codec checkpoint

The user adopted Cloudflare shared outbound WSS relay with hibernating installation objects. Public clients retain HTTPS Streamable HTTP/OAuth; no per-installation Cloudflare named Tunnel/DNS/cloudflared. Free supports local development and bounded synthetic evaluation; Paid activation is not a local build prerequisite.

`src/relay-protocol.ts` implements a private v1 whole-message codec, not WebSocket delivery or a public MCP parser. Exact compact JSON.stringify-compatible envelopes carry original MCP payload bytes in canonical padded base64. Request/response body caps are1MiB/4MiB, outerframe8MiB. Caller supplies exact schema-validated annotated request header names via RelayCodecOptions; unapproved and response-direction parameter headers are denied. Outer UTF8 BOM, duplicate/noncanonical frames, unknown fields and forbidden credential headers reject safely. Python connector serialization must match this private contract; public MCP JSON is not constrained to this canonical formatting.

Final packet has148passing tests total (98prior+50codec), strict typecheck and100%lines/functions,97.14%totalbranches. Both required reviews approved after fixes. Runtime stream framing/closure, generation/request correlation, authentication, deadline/cancel delivery, actual hibernation and native host/UI proof remain pending. There is no deployable production relay in this checkpoint.
