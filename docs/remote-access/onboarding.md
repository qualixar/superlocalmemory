# Dashboard onboarding

Existing local installation remains the normal free path. Remote access starts only when the user chooses it.

## User journey

1. Install SLM using its normal npm or PyPI route. Set up local tools as usual; no remote account is required.
2. Open the existing dashboard's **MCP & Integrations** pane and select **Add AI connection** for an enabled service.
3. Choose a certified host and local profile. Explicitly opt in to remote access. Start with read-only permissions; saving, corrections and broader visibility require separate consent.
4. Sign in through the hosted authorization flow. The local service journals enrollment, protects its credentials and starts the companion automatically.
5. Follow that host's tested custom-MCP recipe using the supplied HTTPS endpoint. Complete the host's consent flow. Do not ask the user to configure Cloudflare, DNS, a tunnel or a terminal process.
6. Show **Connected** only after current ownership/grant/connector proof. An enrollment acknowledgment alone shows **Pending**.
7. Run the guided recall check. Offer a synthetic save-and-recall check only after explicit write consent. Link to connection management and disconnect controls.

Host account features and authentication requirements must be verified before a recipe is advertised. Each host recipe must have native compatibility evidence; listing a host in a mock UI does not certify it.

## Honest connection states

| State | User meaning | Required behavior |
| --- | --- | --- |
| Off | Local-only use | No gateway enrollment or hosted entitlement checks |
| Pending | Enrollment/login not complete | Resume the journaled operation; do not create duplicates |
| Cancelled | Enrollment was stopped | Permit a fresh request; show remote cleanup pending until revocation is confirmed |
| Connecting | Companion is establishing the route | No false connected indication |
| Connected | Current remote access is verified | Display host/profile and granted permissions |
| Reconnecting / offline | Laptop route is unavailable | Explain availability; leave local SLM usable |
| Authorization required | Credential/grant/entitlement needs attention | Stop remote access until reauthorized |
| Disconnected / revoked | Owner removed remote access | Block new remote requests; preserve local memory/configuration |

## Configuration preservation

Do not rewrite existing modes, profiles, providers, hooks, local MCP client wiring, mesh configuration or database paths. The companion and connection journal use separate remote state. Optional runtime failures must not become a local startup requirement.

GitHub sign-in covers optional web connections only. The local dashboard binds approval to the installation, current profile and selected permissions. Local SLM continues to work without a hosted account. Google sign-in is not required.
