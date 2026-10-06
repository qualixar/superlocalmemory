# SLM remote gateway — preparatory domain package

Implemented: PRE-01 permission intersection and the ENR-01 enrollment state reducer, with synthetic tests. This private package is not a deployed HTTP server, an installer, a finished UI feature or a release artifact. Engine/UI integration awaits accepted M4 4.1.22 and its verdict.

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
