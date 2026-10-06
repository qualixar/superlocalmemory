# Acceptance and host certification

Component tests are useful evidence, but do not substitute for installed-package, real-engine, deployed or native-host checks.

## Build first, then acceptance

Development checks run while components are built. Full installation and Musebot acceptance start after the candidate is complete. The first real-memory test uses a consistent isolated database copy; production use follows successful acceptance and documented rollback.

| Gate | Required proof |
| --- | --- |
| Local core | Fresh npm/PyPI installation and upgrade work without an account/subscription; local tools, hooks, mesh and configured Jev/Laya retain their behavior |
| Enrollment | Installation authentication/Origin/CSRF, explicit opt-in, owner/profile binding, durable idempotency, changed-intent conflicts, cancellation and restart recovery |
| Public authorization | Login/PKCE, token audience/client/owner/profile binding, immutable consent ceiling, discovery filtering, remote entitlement, access/refresh expiry and revocation |
| Relay | Authenticated outbound connector, generation fencing, hibernation, bounded buffers, deadlines/cancellation, honest offline/timeout outcomes and no unsafe write replay |
| Real memory | Remember/recall, kind/tag persistence, correction, profile/scope boundaries and concurrent local/remote writes through the existing writer |
| Packaging | Automatic companion delivery; no end-user Node/npm infrastructure setup; missing companion cannot break local startup |
| Native host | Actual authenticated custom-MCP enrollment, permitted tool discovery and roundtrip inside each claimed host |
| Failure isolation | Laptop sleep, network loss, gateway outage, remote revoke/expiry/subscription failure leave local SLM functional |
| Release | Reconciled 4.1.22 source, installer regressions, privacy-safe logs, usage/latency evidence, recovery procedure and documentation matched to shipped status |

## Musebot test

1. Inspect the actual host's custom-MCP configuration, transport and authentication controls. Record the tested host/account surface; do not infer compatibility from a generic MCP claim.
2. Add the staged SLM HTTPS endpoint and complete owner login/consent. Verify that only permitted tools are listed.
3. Recall a known synthetic marker; compare the tool response with the underlying SLM result, not only the assistant's paraphrase.
4. With write permission, remember a uniquely named synthetic decision using durable idempotency; verify canonical persistence and recall it in a fresh host session.
5. Check read-only denial, cross-profile denial, correction consent and revoked-grant rejection.
6. Sleep/disconnect the laptop; confirm a useful unavailable state. Reconnect without duplicate writes.
7. Continue local Claude Code/Codex calls during remote activity; validate mesh and configured answer checks separately.

Use the same certification approach for ChatGPT Web and other hosts. Do not advertise a host until its native evidence passes. Local answer-provider success is distinct from hooks, MCP and native-host proof; SLM's Jev features are distinct from the Jev Decision Layer plugin.

## Timing and evidence

For the established local recall target, a correct result taking over three seconds is slow, not proof recall is broken. Record local engine latency separately from network/OAuth/host overhead and judge success against correctness as well as timing. Report timeout, abstention, unavailable and genuinely empty results separately.

Capture source SHA, package/runtime version, data-root isolation, fixture/live classification, host surface, elapsed times, observed tool results and unresolved gaps. Keep secrets and raw user memories out of public evidence. Existing component evidence: [connector checkpoint](../../integrations/remote-gateway/evidence/local-connector-green.json), [runtime/dashboard checkpoint](../../integrations/remote-gateway/evidence/runtime-dashboard-green.json).
