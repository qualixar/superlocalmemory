# SLM remote gateway — preparatory domain package

Implemented: PRE-01 permission intersection and the ENR-01 enrollment state reducer, with synthetic tests. This private package is not a deployed HTTP server, an installer, a finished UI feature or a release artifact. Engine/UI integration may develop on the current baseline now; merge and verify M4's completed 4.1.22 before final release. See docs/release-plans/4.1.23/UPSTREAM-RECONCILIATION.md in the repository.

## Approved user journey

SLM works locally with remote access off. The user optionally selects Connections → Add AI connection in the SLM UI. SLM will enroll the owned installation, provision its tunnel, manage its local connector and guide host consent. No end-user Cloudflare account, DNS, terminal command or infrastructure secret-paste step. Private-server mode B remains deferred.

## Module boundaries

- `src/contracts.ts`: actor, connection and immutable per-client authorization types from the reviewed design.
- `src/request-policy.ts`: ownership/audience checks, scope/tool intersection, era method allowlist and explicit profile/correction/sharing restrictions. It returns an internal effective grant snapshot; do not serialize that whole object into client responses. Discovery response filtering must use its allowedTools.
- `src/enrollment.ts`: pure enrollment transitions with opt-in, owner/device binding, expiry, version and connector generation checks. It selects known persistence fields and has no provider/credential side effects.

Inputs to policy must already have validated schemas, verified provider identity and current authoritative registry records. Inputs to the reducer must come from authenticated UI operations or validated internal provider/connector events. Never trust a client-provided `verified` event. Capture async operation version/generation when dispatching it; never replace them with the latest record values before accepting its callback.

Not yet implemented: schema/token/wire validation, atomic Durable Object admission/CAS, OAuth refresh/revocation adapter, provisioning idempotency/resource journal, capacity/abuse controls, encrypted credential delivery, authenticated connector/replica verification, origin forwarding/binding and response filtering, installer lifecycle, SLM UI and native host proof. Domain state/version checks alone cannot establish these integration guarantees.

## Reproduce checks

Use Node >=22.18.0; TypeScript7.0.2 is pinned in package-lock. From this directory install dependencies with `npm ci --ignore-scripts --no-audit --no-fund`. Run `npm run typecheck`, `npm test`, and `npm run test:coverage`. For release evidence use a minimal process environment and the direct pinned compiler/test runner, as done for the GREEN checkpoint. Tests use only synthetic identifiers and reserved example.com hostnames; they make no network/database/provider calls.

The committed RED evidence records57 expected assertion failures before implementation. GREEN evidence records98 passing tests, with100% source lines/functions and98.20% branches in these two executable domain modules. Coverage is not a live-security or end-to-end certification.

## Approved relay transport and codec checkpoint

The user adopted Cloudflare shared outbound WSS relay with hibernating installation objects. Public clients retain HTTPS Streamable HTTP/OAuth; no per-installation Cloudflare named Tunnel/DNS/cloudflared. Free supports local development and bounded synthetic evaluation; Paid activation is not a local build prerequisite.

`src/relay-protocol.ts` implements a private v1 whole-message codec, not WebSocket delivery or a public MCP parser. Exact compact JSON.stringify-compatible envelopes carry original MCP payload bytes in canonical padded base64. Request/response body caps are1MiB/4MiB, outerframe8MiB. Caller supplies exact schema-validated annotated request header names via RelayCodecOptions; unapproved and response-direction parameter headers are denied. Outer UTF8 BOM, duplicate/noncanonical frames, unknown fields and forbidden credential headers reject safely. Python connector serialization must match this private contract; public MCP JSON is not constrained to this canonical formatting.

Final packet has148passing tests total (98prior+50codec), strict typecheck and100%lines/functions,97.14%totalbranches. Both required reviews approved after fixes. Runtime stream framing/closure, generation/request correlation, authentication, deadline/cancel delivery, actual hibernation and native host/UI proof remain pending. There is no deployable production relay in this checkpoint.
