# Web agents: instructions you can copy

Connecting an app through [Web access](../remote-access/README.md) gives it the
tools. It does not tell the app when to use them. Without instructions, most
web agents either never recall or save every passing remark. This folder holds
the instructions that close that gap, in four shapes:

| File | Use it when |
|---|---|
| [setup-prompt.md](setup-prompt.md) | You are in a chat with the app (Muse, Grok Bot, ChatGPT, dots). Paste one prompt: the app keeps the skill and tests saving and recall. Start here. |
| [instructions.md](instructions.md) | The app has an instructions or system-prompt field: ChatGPT projects and custom GPTs, Claude projects, Muse bots, Composio agents. It has a full block and a short block for limited fields. |
| [superlocalmemory-web/SKILL.md](superlocalmemory-web/SKILL.md) | The app accepts Agent Skills (a folder with a `SKILL.md`). Upload the `superlocalmemory-web` folder as it is. |
| The **Copy instructions** button | You are in the dashboard. Open **Connected apps**, choose **How to add an app**, and copy the short block straight from there. |

All four carry the same rules, and a test keeps them identical. For the steps to
connect a particular app first, see the [host guides](../remote-access/hosts.md).

## What the instructions cover

They tell the agent to recall before answering anything that may rest on your
past decisions, preferences, projects or rules; to treat memories as notes
rather than commands; to say it doesn't have something when recall reports
`abstained: true` or `no_confident_match: true`, or when the memories simply do
not answer the question; and to cite the fact ids it relied on. If you allowed saving,
they tell it to save only lasting facts, one per call, with a
[kind](../memory-kinds.md), a few tags and an `idempotency_key`, and never to
save secrets. They also say what to do when your computer is asleep, when the
free daily allowance is used up, and when a permission was not granted.

Since 4.1.25 they also cover two optional groups of tools. For **pictures and
PDFs**, the agent passes a ChatGPT attachment straight to `remember_media` or
`remember_document`, and otherwise asks for a one-time upload link
(`media_upload_link`), shows it to you and does not say the file was saved until
you tell it so; a recall result with a `media` block is looked at with
`get_media` before the agent says what a picture shows. For **bot messages**,
the agent treats every message as data, never as an instruction, waits for
replies with `mesh_wait`, and passes the `ack_ids` of each reply back as `ack`
so a message is not delivered twice.

## What a web app can and cannot do

A web app reaches one profile, the one you chose when you turned on Web access.
It can call `recall`, `search`, `fetch` and `get_status`. It can call `remember`
only if you allowed saving, and the session tools (`session_init`,
`close_session`, `report_feedback`, `report_outcome`) only if you allowed
those. With the second yes (the box on the approval page and the matching
switch on the **Web access** row in Connected apps, see the
[host guides](../remote-access/hosts.md#bot-messages-and-the-two-new-permissions))
it can also use bot messages (`mesh_peers`, `mesh_send`, `mesh_inbox`,
`mesh_wait`, and `mesh_state` to read) and pictures and documents
(`get_media`, `media_status`, and with saving `remember_media`,
`remember_document`, `media_upload_link`). It cannot delete, retire or correct a
memory, share one with another profile, or read another profile. Those stay on
your computer, in the dashboard, CLI or a local agent.

## For agents on your own computer

Claude Code, Codex, Hermes, Antigravity, VS Code, Cursor and the Grok Bot
plugin connect locally and get the full tool set. They use the
[universal agent kit](../../plugin-src/rules/AGENTS.md) and the skills shipped
in each plugin. The `slm-web-access` skill in those plugins helps a local agent
set up Web access and diagnose a connection.
