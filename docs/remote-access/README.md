# Local memory, optional remote access

SuperLocalMemory gives local agents a shared memory and knowledge system. The remote-access architecture adds an optional internet connection for compatible web MCP clients, using the same local engine and database.


![SLM integrated local memory and optional web connection architecture](assets/slm-integrated-architecture.svg)

## Choose your path

| Path | What it does | Account and cost boundary |
| --- | --- | --- |
| Local SLM | Existing npm/PyPI installation, local Claude Code/Codex/MCP clients, tools, mesh and configured answer checks | The free local core requires no hosted SLM account or subscription. Optional model providers retain their own requirements. |
| Optional remote service | Lets a compatible web MCP client reach the installed laptop through an authenticated gateway | Off by default. Hosted-service accounts and entitlements apply only to remote access. |

Both paths are intended to work concurrently. Enabling the remote path must preserve local profiles, modes, provider settings, hooks and IDE configurations. A remote outage or revoked subscription must not stop local SLM.

## Documentation map

- [Architecture and trust boundaries](architecture.md): components, data flow, authorization, local-core isolation and current implementation evidence.
- [Dashboard onboarding](onboarding.md): the intended nontechnical user journey and connection states.
- [Cloudflare pilot](cloudflare-operations.md): operator setup, deployment sequence, rollback and usage checks.
- [Acceptance and host certification](acceptance.md): synthetic, real-engine, native-host and regression gates.
- [Offline visual preview](architecture-preview.html): desktop/mobile layouts with editable SVG source.
- [Local engine architecture](../ARCHITECTURE.md): the existing memory pipeline, canonical store and projections.

## What local means when remote access is enabled

The canonical memory database stays on the user's machine. Remote tool arguments and results travel through the gateway and the selected AI host. This is a network-access feature, not a promise that memory content never leaves the laptop. The cloud service stores connection and authorization state; it is not a replacement hosted memory database. The laptop and connector must be online for remote calls.

Hosted database mode is outside this release's approved scope. SLM-Mesh remains trusted-peer coordination; this gateway is not database replication. SLM's Jev answer-check integration is separate from the Jev Decision Layer plugin.
