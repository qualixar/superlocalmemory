---
name: slm-web-access
description: Set up and troubleshoot SuperLocalMemory Web access, the optional link that lets web apps (ChatGPT, Claude on the web, Muse, Composio, other remote MCP clients) use the memory on this computer. Covers turning it on from the dashboard, adding and removing apps, read/save/session permissions, renewal, the connection states, the errors a web app can see, and the copy-paste instructions for web agents.
version: "4.1.25"
agent: agent
tools:
  - get_status
  - Read
  - Bash
---

# slm-web-access — Web apps and bots

## What it is

Web access is optional. Without it, SuperLocalMemory works fully on this
computer. With it, web apps reach the same memory through SuperLocalMemory's
connection gateway: the app signs in with OAuth, the gateway relays each tool
call to this computer over an outbound link, and the memory database never
leaves the computer. Nobody configures Cloudflare, DNS or a tunnel.

Web access is not `slm remote`. `slm remote` is direct access over TLS with
named keys, for tools on another computer in your own network.

## Turning it on (the owner does this, in the dashboard)

1. Open the dashboard (`slm dashboard`, default `http://localhost:8765`) and go
   to **Connected apps**.
2. Under **Add an app**, choose the app: ChatGPT, Claude (web), Claude Code
   (web), Composio, Muse, or **Other app (MCP)** for any client that supports
   remote MCP with OAuth.
3. Under **What this app can do**, tick **Turn on internet access for this
   app** (reading is always included). **Allow saving memories** adds the
   `remember` tool and **Allow session tools** adds `session_init`,
   `close_session`, `report_feedback` and `report_outcome`; both are separate
   opt-ins.
4. Select **Link this computer with GitHub** and finish sign-in in the page
   that opens. Wait until the row reads "On. Apps you approve can reach your
   memory while this computer is online."
5. In the app, add a remote MCP connector with the **MCP server URL**
   `https://mcp.superlocalmemory.com/mcp` and OAuth. Some apps (Composio, Muse)
   also want the **OAuth metadata URL**
   `https://auth.superlocalmemory.com/.well-known/oauth-authorization-server`.
   Both are under **Technical details** with copy buttons.
6. Approve the same GitHub account and profile in the app, then test one
   recall from the app.

An agent cannot do steps 3–6 for the owner: they need the owner's consent and
sign-in. Point the owner to the dashboard; never ask for tokens or codes.

## What a web app can call

`recall`, `search`, `fetch`, `get_status`; `remember` only with Save; the four
session tools only with session tools. It reaches only the profile chosen at
setup. `scope` must be personal, `shared_with` is refused, and `replaces` is
refused (`CORRECTION_DENIED`): corrections, deletes and sharing stay on this
computer.

## Instructions for the web agent

The app also needs to know when to use the tools. Give the owner
`docs/web-agents/instructions.md` (full and short blocks) or the Agent Skill
folder `docs/web-agents/superlocalmemory-web/`. The dashboard's **How to add an
app** panel has a **Copy instructions** button with the short block.
Exact per-app steps (ChatGPT, ChatGPT dots, Grok Bot, Composio, Muse) are in
`docs/remote-access/hosts.md`.

## Renewal

The link renews its own credential while the computer is online; the row says
**Renews automatically**. If the computer stays offline for weeks, the row
warns before access ends ("Web access ends on <date> unless this computer
reconnects"). After it ends, or if sign-in is needed again, the row says so and
the owner turns Web access on again or signs in again. Local memory is never
affected.

## When a web app reports an error

| What the app sees | Meaning | What to do |
|---|---|---|
| `connector_asleep` | This computer was silent for 45 seconds (usually asleep) | Wake the computer; the link reconnects by itself |
| `connector_offline`, `connector_unavailable`, `relay_timeout` | The link is down or the call took too long | Check the computer is online and SLM is running (`slm status`) |
| `relay_busy` | Too many calls in flight at once | Make calls one at a time |
| `DAILY_LIMIT_REACHED` (HTTP 429) | The free daily allowance is used up | Resets at midnight UTC (`Retry-After` says when) |
| `TOOL_DENIED`, `INSUFFICIENT_SCOPE` | The app was not given that permission | Remove its access and add it again with the permission ticked |
| `CORRECTION_DENIED`, `SHARING_DENIED`, `PROFILE_DENIED` | The app tried something only this computer may do | Do it locally |
| `REVOKED` | The owner removed the app | Add it again if wanted |
| `ENTITLEMENT_REQUIRED` | Web access has ended | Turn it on again in **Connected apps** |

To check from this computer, call `get_status` and run `slm status`; the
**Connected apps** page shows each connection's state and access date.

## Removing an app

**Connected apps** → the app's row → **Remove access**. The gateway refuses its
next call.
Memory and local configuration are kept.
