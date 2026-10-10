# Connect each app: step-by-step host guides

[Web access](README.md) lets an AI app on the internet use the memory on your
computer. This page gives the exact steps for ChatGPT, ChatGPT dots, Grok Bot,
Composio and Muse, and the things that tripped people up. The steps were
checked against those apps on 8 October 2026. The apps change their screens
from time to time, so if a label has moved, look for the one nearest to it.

## Before you start

You do the first part once, in the SuperLocalMemory dashboard (`slm dashboard`),
whatever the app is.

1. Open **Connected apps** and choose **Add an app**.
2. Under **What this app can do**, tick **Turn on internet access for this
   app**. Tick **Allow saving memories** if the app should be able to save, and
   **Allow session tools** only if you want the app to open and close sessions.
   Reading is always included.
3. Select **Link this computer with GitHub** and finish the sign-in in the page
   that opens. Wait until the page says Web access is on.

Keep this computer awake and online while the app needs your memory.

Every app below needs two things: the connection (the steps in its section)
and the [setup prompt](../web-agents/setup-prompt.md), which you paste into a
chat with it once connected. Muse and Grok Bot in particular cannot be
connected by a button alone.

Two addresses are used below. Both have copy buttons under **Technical
details** on the Connected apps page.

| What | Address |
|---|---|
| Server URL (the MCP server) | `https://mcp.superlocalmemory.com/mcp` |
| OAuth metadata URL | `https://auth.superlocalmemory.com/.well-known/oauth-authorization-server` |

Every app also shows an approval page the first time it connects. Sign in with
GitHub there. Reading is always included on that page. **Allow saving
memories** and the session tools are separate boxes that start unticked, so tick
saving if the app should save.

## ChatGPT (web, paid plan)

1. In ChatGPT on the web, open **Plugins** and choose **Add**, then **Add
   custom MCP server**.
2. Enter a **Name**, the **Server URL**, and set **Authentication** to
   **OAuth**.
3. Open **Advanced OAuth settings**. Check that the default scopes show
   `slm:read`, `slm:write` and `slm:session` ticked. Then put the same three in
   **Base scopes**: `slm:read`, `slm:write`, `slm:session`.
4. Tick **I understand and want to continue**, then choose **Create as a
   plugin**, then **Continue**.
5. Sign in with GitHub on the approval page and tick **Allow saving memories**.
6. In a chat, open the tools picker and enable the plugin.

Two things to know:

- ChatGPT keeps the scopes it discovered at the moment the plugin was created.
  A plugin created before saving was fixed stays read-only. Uninstall it and
  create it again.
- An uninstalled plugin's name stays taken. If ChatGPT says "An app with this
  name already exists", pick another name, for example **SuperLocalMemory
  Brain**.

## ChatGPT dots

Dots are ChatGPT's always-on agents. You create them in ChatGPT on the desktop,
on a Pro or Business Premium plan. A dot uses the plugins you have connected in
ChatGPT, so connect SuperLocalMemory there first (the section above).

To test it, ask the dot to call `get_status` with the SuperLocalMemory plugin.
If it answers, the dot can reach your memory.

If the dot cannot see your custom plugin, use Composio instead. Composio is a
listed ChatGPT plugin: connect SuperLocalMemory to Composio as a custom MCP
server (see the Composio section below), then use the Composio plugin from the
dot.

We have not confirmed whether dots can use custom MCP plugins directly. The
`get_status` test is how you find out on your plan.

## Grok Bot

1. In Grok Bot, open **Settings**, then **Plugins**, and add a custom MCP
   server. Give it a name such as `SuperLocalMemory-Web` and the **Server URL**.
2. Grok Bot starts an OAuth sign-in in the chat. Approve it with GitHub and
   tick **Allow saving memories**.

Grok Bot's server runs on Cursor's backend, so the approval appears as
**Cursor** in your Connected apps list. That is expected.

### The local plugin is a separate memory

Grok Bot can also install the `superlocalmemory` plugin from the `qualixar`
marketplace. That plugin keeps its own memory on Grok's computer. It is not the
memory on your computer, and the two do not sync. For one shared brain, use the
web connection above and treat the local plugin as optional.

If the local plugin shows the old description "21-tool code profile", or fails
with `venv/bin/slm: No such file`, Grok's copy of the marketplace is pinned to
an old commit. Remove the `qualixar` marketplace, add it again, and reinstall
the plugin. It should then show the description "Local-first long-term memory
for your agents and bots".

## Composio

1. In Composio, create a **Custom MCP** server with **OAuth** authentication.
2. Enter the **Server URL**, and under **Advanced settings** enter the **OAuth
   metadata URL**.
3. Connect, sign in with GitHub on the approval page and tick **Allow saving
   memories** if wanted.

If Composio later reports `401` after you turned Web access off and on again,
its approval was removed. Reconnect it:

```bash
composio link custom_superlocalmemory
```

## Muse

Muse connects through the private adapter's secure OAuth connector. Enter both
the **Server URL** and the **OAuth metadata URL**, then sign in with GitHub on
the approval page and tick **Allow saving memories** if wanted. If Muse was
connected before, authorize it again.

## Bot messages and the two new permissions

The approval page has two more boxes. Both start unticked, and they appear
only when the app asks for them.

- **Allow talking to your other bots** lets the app list your other bots, send
  them messages and read the replies.
- **Allow images and documents** is prepared for a later release. No image or
  document tools work over Web access yet. When they do, image links from a web
  app will be limited to known file hosts.

Ticking a box is not enough. You must also allow it for the connection on your
computer. Each connection has a remote key named `web-<connection id>`. Find it
with `slm remote keys list`, then run:

```bash
slm remote keys allow web-<connection id> mesh
slm remote keys allow web-<connection id> media
slm remote keys disallow web-<connection id> mesh
```

`allow` turns a permission on and `disallow` turns it off. Without both the box
and the key, the app is refused.

What a web app can do with bot messages:

- `mesh_peers` lists your other bots.
- `mesh_send` sends one message to one bot. There is no broadcast.
- `mesh_inbox` checks for messages.
- `mesh_wait` waits up to 20 seconds for a message.
- `mesh_state` reads shared notes. It cannot change them.

Limits: 200 messages sent per app per day. Inbox checks have their own daily
budget. A connection can have at most 2 waits at once.

Messages from other bots are data, not instructions. A web app should never act
on a request inside a message without asking you, and should never reply to a
bot message by itself. The [setup prompt](../web-agents/setup-prompt.md) tells
it so.

Each app gets a stable name. In the dashboard's **Bot messages** tab you can
rename, mute or remove any peer.

**ChatGPT:** ChatGPT keeps the scopes it saw when the app was created. To see
the new boxes, uninstall the app and create it again. **Composio:** run a
manual toolkit re-sync so it sees the new tools.

## After you connect any app

1. Paste the [setup prompt](../web-agents/setup-prompt.md) into a chat with the
   app. It asks the app to keep the SuperLocalMemory skill permanently, then
   tests saving and recall and shows the raw results. Connecting gives the app
   the tools; this prompt tells it when to use them. For an instructions field
   instead, the dashboard's **Copy instructions** button (Connected apps, How to
   add an app) copies the short block.
2. Check that it works. Ask the app to save a sentence with a unique marker,
   such as "Remember that my test marker is plum-4471", then ask it in a new
   chat what your test marker is. A recall that returns the marker means both
   directions work.
3. In **Connected apps**, check that the app is listed with the permissions you
   expect.

## If something is wrong

| What you see | Why | What to do |
|---|---|---|
| The app can read but not save | The app was approved without saving, or ChatGPT kept the scopes from when the plugin was created | Re-approve with **Allow saving memories** ticked. In ChatGPT, uninstall the plugin and create it again |
| `401` from the app | The app's approval was removed, or Web access was turned off and on again | Reconnect the app, and approve it again |
| "This sign-in session is no longer valid" after you already approved | The sign-in had already completed before the page was shown | Open **Connected apps** and check whether the app is listed. The approval page now says this more clearly |
| The same app appears twice in **Connected apps** | The app was connected more than once | Remove the older entry with **Remove access** |
| `connector_asleep` or `relay_timeout` | This computer is asleep or offline | Wake it and try again |

The full list of errors an app can see is in
[Troubleshooting](../troubleshooting.md#web-access).
