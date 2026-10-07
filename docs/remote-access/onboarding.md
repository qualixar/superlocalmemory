# Dashboard onboarding

Existing local installation remains the normal free path. Remote access starts only when the user chooses it.

## User journey

1. Install SLM using its normal npm or PyPI route. Set up local tools as usual; no remote account is required.
2. Open the existing dashboard's **MCP & Tools** pane. The **Connect your AI** card shows the setup steps and service choices directly.
3. Select Composio, ChatGPT Web, Musebot, Claude Web, Claude Code Web or **Other MCP client**. The choice describes the connection method; it does not certify compatibility with every account. The active local profile remains the scope.
4. Check **Enable remote access for this connection**, then select **Link this computer with GitHub**. Reading is the default. **Allow this AI to save new memories** requires separate consent. The local service journals enrollment, protects credentials and starts the companion automatically.
5. Complete GitHub sign-in in the opened page. If the browser blocks the new page, use **Continue sign-in**. Wait for **Ready for AI client — GitHub connected**: this requires a live companion and a successful canonical MCP health check. A pending receipt is not readiness.
6. Follow the service instructions shown after readiness. Composio uses **Custom MCP**, **OAuth**, the supplied MCP URL and the OAuth metadata URL. Musebot requires its private adapter and secure OAuth connector. Other clients require support for custom remote MCP with OAuth. Approve the same GitHub account and profile in the client. End users do not configure Cloudflare, DNS or a tunnel.
7. Verify a real recall in the AI client. Only a completed client round trip establishes client acceptance. Test a disposable save-and-recall separately after write consent. Use **Cancel connection** to revoke the route; confirmed cancellation preserves local memory and configuration.

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

## Stable controls and profile isolation

Status polling preserves the selected service, consent controls and unchanged connection buttons. Switching installation or profile clears the old sign-in link and resets consent. Retrying an uncertain request retains its idempotency key; changing an unconfirmed request is rejected. Cancelling restores the controls without requiring a manual refresh.
