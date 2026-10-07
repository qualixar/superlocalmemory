# Cloudflare operator guide

SLM users start web connections from the existing local dashboard. They do not configure Cloudflare, DNS, tunnels or servers. The database and canonical memory engine remain on the user's computer. GitHub sign-in applies to the optional web connection; local CLI, stdio, Codex, Claude Code and mesh do not require a hosted account.

## Services

| Host | Purpose | Deployment configuration |
| --- | --- | --- |
| `auth.superlocalmemory.com` | GitHub owner identity, OAuth discovery and client authorization, native enrollment | `wrangler.auth.jsonc` |
| `mcp.superlocalmemory.com/mcp` | OAuth-protected public MCP resource | `wrangler.mcp.jsonc` |
| `connect.superlocalmemory.com/connector` | Outbound authenticated laptop WebSocket | `wrangler.connect.jsonc` |
| No public route | Durable connection, device, grant and revocation authority | `wrangler.authority.jsonc` |

The native owner token has resource `https://auth.superlocalmemory.com/owner` and scope `slm:connect`. It cannot execute memory tools. AI-client tokens have resource `https://mcp.superlocalmemory.com/mcp`; read, write and session permissions are separately consented. Device credentials never go into Composio or Muse.

## Operator setup

1. Create the GitHub OAuth application named **SuperLocalMemory**, with homepage `https://superlocalmemory.com` and callback `https://auth.superlocalmemory.com/github/callback`. Keep wildcard callback matching and GitHub Device Flow disabled. SLM's native flow uses its own authorization-code PKCE enrollment.
2. Sign into the Cloudflare operator account with Wrangler. Grant the deployment scopes required by Workers, routes, KV and zone lookup; do not broaden scopes merely to remove unrelated Wrangler warnings.
3. Create one `OAUTH_KV` namespace and put its namespace ID in the auth configuration. Set the public GitHub client ID. Neither value is a client secret.
4. Run `npm run typecheck`, `npm test`, `npm run test:runtime` and a Wrangler dry run for each configuration in `integrations/remote-gateway`.
5. Deploy the private authority Worker first, then the auth Worker, the resource Worker and the connector Worker. Cross-Worker Durable Object bindings refer to the authority Worker; the resource validates tokens through the auth service binding.
6. Put a newly generated 32-byte random hex value into the encrypted **DEVICE_WRAP_KEY** Worker secret using protected input. Do not rotate it during retries: existing encrypted device deliveries depend on it.
7. In the Cloudflare auth Worker's **Settings → Runtime variables and secrets**, add **GITHUB_CLIENT_SECRET** with **Secret** selected. The operator generates/copies this from GitHub and submits it directly. Never paste it into a chat, repository, `.env` file, screenshot or log.
8. Verify public HTTPS discovery, the OAuth challenge, DCR and native enrollment before configuring an AI client. A deployed hostname alone does not prove a successful owner or client connection.

The GitHub client secret belongs to the SLM operator application, not to individual SLM users. Users authorize that application through GitHub.

## API security compatibility

MCP clients and the outbound laptop connector are machine clients. Browser challenges must not intercept these API routes. Cloudflare Error 1010 indicates a browser-signature rejection; consult [Cloudflare's explanation](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/).

When Browser Integrity Check blocks legitimate SDKs, use a configuration rule restricted to the dedicated SLM service hosts. The expression is:

```text
http.host in {"auth.superlocalmemory.com" "mcp.superlocalmemory.com" "connect.superlocalmemory.com"}
```

Disable **Browser Integrity Check** only for the matching hosts. Review and approve this security change before deploying the rule. Do not disable zone-wide protection or bypass all WAF rules. OAuth audience, PKCE, state/callback validation, per-connection permissions, device proof, revocation and setup admission remain mandatory.

The auth Worker limits anonymous setup before allocating state. Per-address admission is evaluated before the shared limiter. Missing limiter bindings fail closed. [Cloudflare rate-limit counters](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/) are approximate per-location controls, not a globally strict quota or spending cap. Monitor legitimate users sharing a network before changing thresholds.

## Client setup

First complete the local dashboard connection and wait for **Ready for AI client — GitHub connected**. Then:

- **Composio:** Add Custom MCP, name **SuperLocalMemory**, server URL `https://mcp.superlocalmemory.com/mcp`, authentication **OAuth**. Complete GitHub sign-in, select the correct laptop/profile connection and approve the requested scopes. Verify its discovered tools and a recall round trip before calling it connected.
- **Muse:** use the reviewed private adapter and its hosted OAuth connector, rather than assuming a generic custom-MCP screen. Give the exact resource URL and issuer metadata. Verify the real helper supplies PKCE S256, `resource`, state validation and its exact registered callback. Secrets and tokens stay in the hosted connector's secure store, not chat or adapter files.

Discovery: `https://auth.superlocalmemory.com/.well-known/oauth-authorization-server`. Expected authorization, token and registration endpoints are `/authorize`, `/oauth/token` and `/oauth/register` on that issuer. The discovered token-auth methods are authoritative for the client. No static edge bearer is supplied to AI clients.

## Capacity and recovery

[SQLite Durable Objects are available on Workers Free](https://developers.cloudflare.com/durable-objects/platform/pricing/). Test within the account's actual Free limits before purchasing anything. Free quota exhaustion can stop operations; a website zone plan and a Workers compute plan are distinct. Count Worker requests, Durable Object RPCs/storage operations and KV operations. A user count is not a capacity unit.

Use hibernating outbound WebSockets, monitor reconnect frequency and measure end-to-end memory latency. Do not cache user memory bodies at the edge. Logs remain disabled by default to avoid recording OAuth query parameters or memory payloads; diagnostic events use bounded, non-secret codes.

Failed provider-grant cleanup persists only grant references in the private `slm-cleanup:` KV retry ledger. Token requests schedule a bounded retry pass, at most once per minute per active isolate. No request traffic means no automatic pass; operator monitoring must detect accumulated cleanup entries. Local and registry revocation remain authoritative even if provider cleanup is temporarily unavailable.

For rollback, use the prior Worker deployment versions and disable web access through the local dashboard. Preserve the named local keys, enrollment cleanup state and existing database until revocation is confirmed. A local database/runtime upgrade has its own backed-up rollback procedure; do not conflate it with gateway rollback.
