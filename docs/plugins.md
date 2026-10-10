# Editor and agent plugins

SuperLocalMemory works with any MCP client through `slm connect <ide>`, which
writes the MCP server into that tool's config. A plugin adds more on top: skills
that teach the agent when to recall and what to save, sub-agents, session hooks
and slash commands. Every plugin is built from one source (`plugin-src/`), so
they all carry the same 16 skills and 4 sub-agents, adapted to each host.

Install SuperLocalMemory first, from npm or PyPI ([getting started](getting-started.md)).
The npm package also carries every plugin folder, at
`$(npm root -g)/superlocalmemory/`; with a PyPI install, the same folders are in
this repository.

| Host | Install | What you get |
|---|---|---|
| Claude Code | `claude plugin marketplace add qualixar/superlocalmemory` then `claude plugin install superlocalmemory@qualixar` | MCP server, 16 skills, 4 sub-agents, hooks, `/slm-loop` |
| Codex | `slm connect codex` then `slm codex install` | MCP server, skills, sub-agents, lifecycle hooks |
| VS Code / GitHub Copilot | `slm connect vscode-copilot --here` (from the project root), then copy `copilot-plugin/.github/` add-ons | MCP server, agent rules, 15 prompt files, 4 agents, hooks |
| Cursor-format hosts (Grok Bot, Cursor) | Add the `qualixar` marketplace in the host's Plugins screen, then `superlocalmemory` | MCP server, a bot-sized skill set, no hooks or dashboard needed |
| Hermes | `hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'` | Skills, child-agent roles, `/slm` commands |
| Antigravity (`agy`) | `agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"` | MCP server, skills, agents, hooks |
| Any other agent | Paste the [universal agent rules](../plugin-src/rules/AGENTS.md) into `AGENTS.md`, `CLAUDE.md`, `.cursorrules` or the system prompt | Memory discipline for any MCP client |
| Web apps (ChatGPT, Claude on the web, Muse, Composio) | Web access in the dashboard, then the [host guides](remote-access/hosts.md) and [web agent instructions](web-agents/README.md) | Recall and, if allowed, save through Web access |

## The skills

| Skill | What it covers |
|---|---|
| `slm-recall` | Recall and search: filters, tags, kinds, time windows, the answer check |
| `slm-remember` | What to save, memory kinds, `replaces`, corrections |
| `slm-session` | Session start, pinned items and standing rules, closing a session |
| `slm-status` | Health, `slm db integrity` and `slm db repair`, embedder and model checks |
| `slm-governance` | Scopes, roles, retention, audit trail, GDPR export and erasure |
| `slm-scope` | Personal, shared and global memory across profiles |
| `slm-profile` | Memory profiles and switching between them |
| `slm-graph` | The knowledge graph and entity tools |
| `slm-mesh` | SLM-Mesh: messages, `mesh_wait`, locks and shared state between agent sessions and web apps (with `ack` for at-least-once delivery) |
| `slm-loop` | Bounded loops and their history |
| `slm-cache`, `slm-compress` | Context optimization: KV cache and reversible compression |
| `slm-web-access` | Setting up and diagnosing Web access for web apps: the Connected apps switches, upload links, what a web app can call |
| `slm-media` | Pictures, PDFs and folders: `remember_media`, `remember_document`, `get_media`, `media_status`, `media_upload_link`, `slm media`, `slm sources` |
| `slm-bot-memory`, `slm-getting-started-bot` | Several bots sharing one computer and one store |

The four sub-agents are the memory advisor, the optimize advisor, the governance
advisor and the loop runner.

## Claude Code

```bash
claude plugin marketplace add qualixar/superlocalmemory
claude plugin install superlocalmemory@qualixar
```

`slm connect claude-code` prints the same two commands. Updates arrive through
`claude plugin marketplace update qualixar`. Manual MCP setup without the plugin
is in [IDE setup](ide-setup.md#claude-code).

## Codex

```bash
slm connect codex --dry-run   # show what would be written
slm connect codex             # add the MCP server to your Codex config
slm codex install             # skills, sub-agents and lifecycle hooks (--dry-run to preview)
slm codex status              # what is installed
slm codex remove              # remove only what SLM installed
```

Codex also installs it as a plugin from the same marketplace:

```bash
codex plugin marketplace add qualixar/superlocalmemory --ref main
codex plugin add superlocalmemory-codex@qualixar
```

Review and trust the new hooks in Codex with `/hooks`. The agent rules in
`codex-plugin/AGENTS.md` are not written for you: append them to your
`AGENTS.md` ([details](../codex-plugin/README.md)).

## VS Code and GitHub Copilot

From the project root:

```bash
slm connect vscode-copilot --here
```

This merges the MCP server into `.vscode/mcp.json` and the agent rules into
`.github/copilot-instructions.md`, without overwriting what is there. The prompt
files, agents and hooks are in `copilot-plugin/.github/`: copy them into your
project's `.github/` and keep any files of your own
([details](../copilot-plugin/README.md)).

## Grok Bot and other Cursor-format hosts

Add the `qualixar` marketplace (`.cursor-plugin/marketplace.json` in this
repository) in the host's Plugins screen, then add `superlocalmemory`. The MCP
server starts with `uvx` pinned to the release, with no API key or sign-in. See
[the README section](../README.md#grok-bot-and-other-cursor-format-plugin-hosts)
for what bots on one computer share. This plugin keeps its own memory on the Grok Bot
computer. To reach the memory on your own computer, add Web access as a custom MCP
server in Grok Bot's Plugins screen ([steps](remote-access/hosts.md#grok-bot)).

## Hermes

```bash
hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'
```

To pin an exact release, pass its 40-character commit with `--ref`. Grant the
`superlocalmemory` MCP server to the plugin when Hermes asks. See
[Hermes](hermes.md) for settings and what the plugin records.

## Antigravity

`agy plugin install` takes a local folder only. With an npm install:

```bash
agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"
```

With a PyPI install, run `scripts/install-antigravity-plugin.sh` from this
repository: it fetches only `antigravity-plugin/` into a local cache and installs
it from there. Safe to re-run.

## Keeping plugins current

```bash
slm upgrade-hosts                 # preview which installed hosts are behind
slm upgrade-hosts --host codex --apply
slm doctor                        # warns when a plugin is older than SLM
```
