---
name: slm-web-access
description: Set up and troubleshoot SuperLocalMemory Web access, the optional link that lets web apps (ChatGPT, Claude on the web, Muse, Composio, other remote MCP clients) use the memory on this computer. Covers turning it on from the dashboard, adding and removing apps, read/save/session permissions, the second yes for bot messages and pictures (the Web access switches in Connected apps), one-time upload links for pictures and PDFs, what leaves the computer, renewal, the connection states, the errors a web app can see, and the copy-paste instructions for web agents.
when_to_use: "web access, connected apps, connect chatgpt, connect claude web, connect muse, connect composio, connect grok bot, remote mcp, mcp.superlocalmemory.com, connector_asleep, DAILY_LIMIT_REACHED, web app cannot reach memory, instructions for web agent, web app bot messages, web app upload link, Let these apps message your other bots, Let these apps save and read pictures and documents, not_for_remote"
allowed-tools: get_status, Read, Bash
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
   opt-ins. Bot messages and pictures are two more opt-ins that the app asks
   for on its own approval page; see "The second yes" below.
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

Two more groups need the second yes (next section):

- **Bot messages:** `mesh_peers`, `mesh_send`, `mesh_inbox`, `mesh_wait`,
  `mesh_state` (read only). See `slm-mesh`.
- **Pictures and documents:** `get_media`, `media_status`; with Save also
  `remember_media`, `remember_document` and `media_upload_link`. See
  `slm-media`.

Everything else (`mesh_lock`, `mesh_summary`, `mesh_events`, `mesh_status`,
`switch_profile`, maintenance, code graph, loops, `forget`) stays on this
computer.

## The second yes: Web access switches

Bot messages and pictures need two yeses. Without both, the call is refused.

1. **The app's approval page.** Tick **Allow talking to your other bots** and
   **Allow images and documents**. Both start unticked and appear only when the
   app asks for them. Saving files also needs **Allow saving memories**.
2. **This computer.** In **Connected apps**, the **Web access** row has two
   switches: **Let these apps message your other bots (SLM Mesh)** and **Let
   these apps save and read pictures and documents**. They apply to every app on
   that connection on its next request. From a terminal the same switches are
   on the connection's remote key, named `web-<connection id>`:

```bash
slm remote keys list
slm remote keys allow web-<connection id> mesh
slm remote keys allow web-<connection id> media
slm remote keys disallow web-<connection id> mesh
```

Turning pictures off also ends any upload link the connection has not used.
The owner flips these; an agent must not. Pictures also need the feature
itself to be on (`slm media status`; see `slm-media`).

ChatGPT keeps the permissions it saw when the app was created: after a new
permission, uninstall the app and add it again. Composio needs a manual
toolkit re-sync to see new tools.

## Adding a picture or PDF from a web app

A chat cannot type a file into a tool call. The app asks for an upload link
(`media_upload_link`, kind `image` or `document`) and shows it to the person.
The person opens it in any browser, picks the file and presses Save; the page
says "Saved to your memory" when this computer has it. The file streams to this
computer through the gateway, which keeps no copy. A link belongs to the app
that asked for it, works once, expires in 10 minutes, and must not be shared.
Each connection can have 3 open links and 20 uploads a day. Pictures can be up
to 25 MB (PNG, JPEG, GIF, WebP), PDFs up to 100 MB. This computer must be awake
and running SLM. If SLM was updated, links made before the update stop working:
ask the app for a new one.

ChatGPT can instead hand over a file attached to the chat. This computer
downloads it only from ChatGPT's own file hosts plus hosts in
`SLM_MEDIA_URL_HOSTS`, which is empty by default; a refusal such as "That host
is not on the allowed list" means the owner must add the host. Per-app steps
are in `docs/remote-access/hosts.md`.

## What leaves the computer

The memory database never leaves the computer. Each tool call and its result
pass through SuperLocalMemory's connection gateway and the app that made the
call. Memory text the app recalls is visible to that app. Bot messages and peer
summaries are secret-redacted before a web app sees them; a picture or page
whose text held a secret or personal data is not shown to web apps at all, and
connected folders are never shown to them.
Uploaded files travel through the gateway to this computer and are not stored
there. An app cannot change, delete, pin or replace a memory it is not allowed
to see; it is answered as if that memory did not exist.

## Instructions for the web agent

The app also needs to know when to use the tools. Give the owner
`docs/web-agents/setup-prompt.md` (paste it into a chat with the app once
connected), `docs/web-agents/instructions.md` (full and short blocks) or the
Agent Skill folder `docs/web-agents/superlocalmemory-web/`. The dashboard's
**How to add an app** panel has a **Copy instructions** button with the short
block. All of them cover recall, saving, upload links for pictures, bot
messages with `mesh_wait` and `ack`, and what to do on errors.
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
| `not_for_remote`, `not_allowed` (pictures) or a refused mesh call | The app has the box but the Web access switch (or the key's `mesh`/`media` opt-in) is off, or the key is not a write key | Turn the switch on in **Connected apps**, or `slm remote keys allow web-<connection id> mesh\|media` |
| `MESH_SEND_LIMIT` | The app sent its 200 bot messages for today | Stop sending until the daily count resets |
| `CORRECTION_DENIED`, `SHARING_DENIED`, `PROFILE_DENIED` | The app tried something only this computer may do | Do it locally |
| `REVOKED` | The owner removed the app | Add it again if wanted |
| `ENTITLEMENT_REQUIRED` | Web access has ended | Turn it on again in **Connected apps** |

To check from this computer, call `get_status` and run `slm status`; the
**Connected apps** page shows each connection's state and access date.

## Removing an app

**Connected apps** → the app's row → **Remove access**. The gateway refuses its
next call.
Memory and local configuration are kept.

---

## Related skills

- `slm-media` — pictures, PDFs and upload links in detail
- `slm-mesh` — bot messages, `mesh_wait` and `ack`
- `slm-bot-memory` and `slm-getting-started-bot` — bots that share one computer
- `slm-governance` — roles and company mode
