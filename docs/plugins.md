# Editor and agent plugins

SuperLocalMemory works with any harness that speaks MCP or can run a command
(`slm remember`, `slm recall`). `slm connect <ide>` writes the MCP server into a
tool's config, and the packaged plugins below (Claude Code, Codex, GitHub Copilot,
Antigravity, Hermes, the Cursor-format plugin for Grok Bot and Cursor, and the VS Code
extension) are examples of what a host-specific package adds, not a list of what is
supported. A plugin adds more on top: skills
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
| VS Code / GitHub Copilot | `slm connect vscode-copilot --here` (from the project root), then copy `copilot-plugin/.github/` add-ons | MCP server, agent rules, 16 prompt files, 4 agents, hooks |
| VS Code (extension) | Install `superlocalmemory-vscode-<version>.vsix` from the [GitHub release](https://github.com/qualixar/superlocalmemory/releases/latest): Extensions view → `...` → Install from VSIX | The MCP server in the agent's tool list, with nothing to edit |
| Cursor-format hosts (Grok Bot, Cursor) | Add the `qualixar` marketplace in the host's Plugins screen, then `superlocalmemory`. In Cursor you can also copy `cursor-plugin/` to `~/.cursor/plugins/local/superlocalmemory` | MCP server, a bot-sized skill set, no hooks or dashboard needed |
| Hermes | `hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'`, then `slm upgrade-hosts --host hermes --apply` | Skills, child-agent roles, `/slm` commands, recall and session hooks |
| Antigravity (`agy`) | `agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"` | MCP server, skills, agents, hooks |
| Grok CLI | `slm connect grok-cli`, then `grok plugin install "$(npm root -g)/superlocalmemory/plugin" --trust` | MCP server, 16 skills, 4 sub-agents, hooks |
| VS Code Agents window, GitHub Copilot CLI | `slm connect copilot-cli` (adds SLM to `~/.copilot/mcp-config.json`, which both read); for skills and agents, add the plugin from this repository in VS Code's Plugins screen | MCP server, 16 skills, 4 agents |
| OpenClaw | `slm connect openclaw` | MCP server in `openclaw.json`, 14 skills in `~/.agents/skills` |
| Muse CLI | `slm connect muse`, then `muse plugins install "$(npm root -g)/superlocalmemory/muse-plugin"` and `muse plugins approve superlocalmemory` | MCP server, 16 skills, 4 sub-agents, hooks, `/slm-loop` |
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
advisor and the loop runner. The Claude Code, Codex and Antigravity plugins also ship the
`/slm-loop` command; Hermes has `/slm <command>` and a `/slm-<command>` alias for each, and
Copilot has the 16 skills as prompt files.

## What the hooks do

Hooks run at session events without the agent doing anything, and every one fails open:
a hook problem never blocks your agent. In the Claude Code plugin, at session start one
hook prints the `session_init` guidance, another injects project context, and the launcher
makes sure the environment exists. `checkpoint` records files you edit. `post_tool_outcome`
and `stop_outcome` notice which recalled memories were actually used and settle each
outcome. `user_prompt_rehash` treats the same question asked again within a minute as a sign
the last answer missed, and `topic_shift` reminds the agent to recall when you change subject.
`before_web` runs a recall before a web search or fetch. `stop` saves a session summary.
Codex, Copilot and Antigravity carry the subset their host can fire (Copilot's hooks are a
preview in VS Code); the Cursor-format plugin ships none, and Hermes does its lifecycle observation inside the plugin. Per-prompt recall
and the session summary can be switched off in the rules in `config.json`
([Auto-memory](auto-memory.md)).

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

## VS Code extension

Download `superlocalmemory-vscode-<version>.vsix` from the
[latest GitHub release](https://github.com/qualixar/superlocalmemory/releases/latest),
then in VS Code open the Extensions view, choose `...` → Install from VSIX, and
pick the file. It needs VS Code 1.101 or later and SuperLocalMemory installed
(`slm` on your PATH). The extension adds SuperLocalMemory to the MCP servers the
agent can use and runs `slm mcp`. If your workspace or user `mcp.json` already
has SuperLocalMemory (for example from `slm connect vscode-copilot --here`), the
extension adds nothing, so you never get two copies. Antigravity IDE installs
the same `.vsix` and skips SuperLocalMemory when `slm connect antigravity`
already added it. Cursor does not support the VS Code API the extension uses,
so in Cursor use the Cursor plugin below or `slm connect cursor`. The extension
is MIT licensed; the engine it starts is AGPL.

## VS Code and Cursor: editor and Agents window

Both editors now have two places to work, and they do not read the same
settings.

**VS Code editor window** (Copilot Chat): the extension above, or
`slm connect vscode-copilot --here`. **VS Code Agents window**: it runs agents
in a separate process that does not see extension-provided servers and reads
`~/.copilot/mcp-config.json` instead, so run

```bash
slm connect copilot-cli
```

This also gives SLM to GitHub Copilot CLI, and the editor's chat discovers the
same file, so the extension then adds nothing to avoid a second copy. For the
16 skills and 4 agents in the Agents window, open its Plugins screen and add
`qualixar/superlocalmemory`: VS Code reads `.github/plugin/marketplace.json`,
which lists `vscode-agent-plugin/` (the Claude Code plugin with a plain
`slm mcp` server; its hooks are Claude Code's and are left out). To try it from
a local copy, add the folder to the `chat.pluginLocations` setting.

**Cursor** shares `~/.cursor/mcp.json` and installed plugins between the editor
and its Agents window, so `slm connect cursor` (or the Cursor plugin) covers
both. Cursor's "Include Third-Party Plugins" setting also imports the Claude
Code plugin; its server shows as Disabled there, and it should stay that way so
SuperLocalMemory runs once. Each of these shows as its own entry in Cursor's
MCP list (Cursor merges only URL-based servers), so keep one: the `slm connect`
entry or the Cursor plugin.

## OpenClaw

```bash
slm connect openclaw
```

This adds SuperLocalMemory under `mcp.servers` in `~/.openclaw/openclaw.json`
(or the file `OPENCLAW_CONFIG_PATH` names) and copies 14 skills into
`~/.agents/skills`, one of the folders OpenClaw loads skills from. OpenClaw's
config is JSON5; when the file has comments or unquoted keys, SLM leaves it
untouched and prints the `openclaw mcp add superlocalmemory --command slm --arg
mcp` command to run instead. Check it with `openclaw mcp doctor superlocalmemory
--probe`.

## Grok CLI

```bash
slm connect grok-cli
grok plugin install "$(npm root -g)/superlocalmemory/plugin" --trust
```

The first command adds the MCP server to `~/.grok/config.toml`. The second
installs the Claude Code plugin, which Grok CLI runs as is: the 16 skills, the
4 sub-agents and the session hooks. Grok asks for `--trust` because a plugin can
run hooks on your machine. With a PyPI install, use the `plugin/` folder of this
repository.

## Muse CLI

```bash
slm connect muse
muse plugins install "$(npm root -g)/superlocalmemory/muse-plugin"
muse plugins approve superlocalmemory
```

The first command adds the MCP server to Muse's `settings.json`
(`$XDG_CONFIG_HOME/muse/settings.json`, otherwise `~/.config/muse/settings.json`)
and keeps your other settings. If the file still uses the old `mcp_servers` key,
`slm connect` renames it to `mcpServers`, because Muse ignores every server when
both keys are present. The plugin adds the 16 skills, the 4 sub-agents,
`/slm-loop` and the session hooks; Muse runs a third-party plugin's hooks and
agents only after `muse plugins approve`. It is the Claude Code plugin with the
parts Muse does not accept left out, and it starts no MCP server of its own,
so run `slm connect muse` first. To use the hosted Muse bot instead, add it through Web
access ([host guide](remote-access/hosts.md)).

## Grok Bot and other Cursor-format hosts

Add the `qualixar` marketplace (`.cursor-plugin/marketplace.json` in this
repository) in the host's Plugins screen, then add `superlocalmemory`. The MCP
server starts with `uvx` pinned to the release, with no API key or sign-in. See
[the section below](#grok-bot-and-other-cursor-format-hosts)
for what bots on one computer share. This plugin keeps its own memory on the Grok Bot
computer. To reach the memory on your own computer, add Web access as a custom MCP
server in Grok Bot's Plugins screen ([steps](remote-access/hosts.md#grok-bot)).

In Cursor you can also copy `cursor-plugin/` (from this repository or
`$(npm root -g)/superlocalmemory/cursor-plugin`) to
`~/.cursor/plugins/local/superlocalmemory` and reload the window. Copy it rather
than symlinking a folder elsewhere: Cursor loads a symlink there only when it
points inside that folder.

If you added the `qualixar` marketplace to Cursor before, Cursor may still
serve the version from the day you added it; reinstalling does not move it
forward. Remove the marketplace in Cursor's plugin settings (or with
`cursor-agent plugin marketplace remove qualixar`), add it again, and reinstall
`superlocalmemory`. An older copy that fails with `/venv/bin/slm: No such file
or directory` is the Claude Code plugin from that old snapshot, not this one.

## Hermes

```bash
hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'
```

To pin an exact release, pass its 40-character commit with `--ref`. Hermes does
not ask for MCP access, so grant the plugin the `superlocalmemory` MCP server:

```bash
slm upgrade-hosts --host hermes --apply
```

This adds `mcp_allowlist: [superlocalmemory]` under
`plugins.entries.superlocalmemory` in `~/.hermes/config.yaml` and keeps a backup.
Without it the plugin's recall, session tracking and turn capture do nothing;
`/slm status` and `slm doctor` both say so. See [Hermes](hermes.md) for settings
and what the plugin records.

## Antigravity

`agy plugin install` takes a local folder only. With an npm install:

```bash
agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"
```

With a PyPI install, run `scripts/install-antigravity-plugin.sh` from this
repository: it fetches only `antigravity-plugin/` into a local cache and installs
it from there. Safe to re-run.

## Choosing how many tools

Every plugin and extension here starts SuperLocalMemory with the default tool
set, `full`: 60 tools covering memory, SLM-Mesh, optimize and bounded loops (5 more while images are on)
(63 while pictures and documents are on). None of them sets a tool set of its
own, so one saved choice applies to all of them:

```bash
slm tools                # the current set and how many tools each set has
slm tools set power      # core, code, full, power, mesh or whole
slm tools set default    # back to full
```

Then restart the apps, or reconnect them in Dashboard → Connected apps. The
choice is saved in SuperLocalMemory's data folder, so it reaches Claude Code,
Codex, Antigravity, Hermes, Grok CLI, Muse CLI, the VS Code extension and every
`slm connect` app alike.

| Set | Tools | When to pick it |
|---|---:|---|
| `core` | 21 | A host with a small tool limit, or a bot that only needs remember and recall |
| `full` | 60 | The default: everything most people use, including SLM-Mesh |
| `power` | 72 | Adds governance: audit trail, retention, lifecycle, diagnostics |
| `code` | 41 | Adds the code graph but leaves out the mesh |
| `whole` | 112 | Every tool; only for hosts with no tool limit |

Some hosts limit the total number of MCP tools across all your servers, often
to about 100, so a bigger set can push another server's tools out. To give one
app a different set, put `SLM_MCP_PROFILE` in that app's own MCP config; it wins
over the saved choice for that app only:

| App | Where its own setting goes |
|---|---|
| Apps set up with `slm connect` (Cursor, Windsurf, Zed, JetBrains, Gemini CLI, Claude Desktop, OpenCode, Continue, Codex, Grok CLI, Muse CLI, VS Code with Copilot) | `slm connect <app> --profile <name>` |
| Hermes | `env: {SLM_MCP_PROFILE: <name>}` under `mcp_servers.superlocalmemory` in `~/.hermes/config.yaml` |
| Claude Code, Antigravity, the Cursor plugin, the VS Code extension | Use `slm tools set`; their launch settings live inside the plugin |

## Keeping plugins current

```bash
slm upgrade-hosts                 # preview which installed hosts are behind
slm upgrade-hosts --host codex --apply
slm doctor                        # warns when a plugin is older than SLM
```
