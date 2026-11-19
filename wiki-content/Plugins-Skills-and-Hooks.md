# Plugins, Skills and Hooks

`slm connect <tool>` only tells an AI tool where SLM's MCP server is ([IDE Setup](IDE-Setup)). A plugin adds the parts that make an agent use that server well: skills that say when to recall and what to save, small advisor agents, slash commands and hooks that run at session start, around tool calls and at session end. SLM ships a plugin for Claude Code, Cursor-format hosts such as Grok Bot, Codex, GitHub Copilot, Hermes and Antigravity. They are all generated from one source folder, `plugin-src/`, so they carry the same 16 skills and 4 agents wherever the host can use them. The Cursor-format plugin is the exception: it carries 6 skills and no agents, commands or hooks.

Install SLM first ([Installation](Installation)). The npm package carries every plugin folder at `$(npm root -g)/superlocalmemory/`; with a PyPI install the same folders are in the repository.

## The plugins at a glance

| Host | Folder | How you install it |
|---|---|---|
| Claude Code | `plugin/`, listed by `.claude-plugin/marketplace.json` | `claude plugin marketplace add qualixar/superlocalmemory`, then `claude plugin install superlocalmemory@qualixar` |
| Cursor-format hosts (Grok Bot, Cursor) | `cursor-plugin/`, listed by `.cursor-plugin/marketplace.json` | Add the `qualixar` marketplace in the host's Plugins screen, then add `superlocalmemory`; in Cursor you can also copy `cursor-plugin/` to `~/.cursor/plugins/local/superlocalmemory` |
| Codex | `codex-plugin/` | `slm connect codex`, then `slm codex install`; or `codex plugin marketplace add qualixar/superlocalmemory --ref main` and `codex plugin add superlocalmemory-codex@qualixar` |
| GitHub Copilot (VS Code) | `copilot-plugin/` | `slm connect vscode-copilot --here` from the project root, then copy `copilot-plugin/.github/` into your project's `.github/` |
| Hermes | `hermes-plugin/` | `hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'`, then `slm upgrade-hosts --host hermes --apply` |
| Antigravity (`agy`) | `antigravity-plugin/` | `agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"` |
| VS Code (extension) | `vscode-extension/` | Install the `.vsix` from the GitHub release: Extensions view, `...`, Install from VSIX |
| Grok CLI | `plugin/` (the Claude Code plugin) | `slm connect grok-cli`, then `grok plugin install "$(npm root -g)/superlocalmemory/plugin" --trust` |
| VS Code Agents window, Copilot CLI | `vscode-agent-plugin/`, listed by `.github/plugin/marketplace.json` | `slm connect copilot-cli`, then add `qualixar/superlocalmemory` in the Agents window's Plugins screen |
| OpenClaw | none (MCP and shared skills) | `slm connect openclaw` |
| Muse CLI | `muse-plugin/` | `slm connect muse`, then `muse plugins install "$(npm root -g)/superlocalmemory/muse-plugin"` and `muse plugins approve superlocalmemory` |

What each one contains, counted from the folders:

| Host | Skills | Agents | Slash commands | Hooks | MCP tool set it asks for |
|---|---|---|---|---|---|
| Claude Code | 16 | 4 | `/slm-loop` | 10 hook commands on 5 events | none set, so the default |
| Cursor-format | 16 | none | none | none | none set, so the default |
| Codex | 16 (14 through `slm codex install`) | 4 | `/slm-loop` | 5 commands on 4 events in the plugin; 4 through `slm codex install` | none set, so the default |
| GitHub Copilot | 16 prompt files | 4 | none separate | 6 commands on 4 events | none set, so the default |
| Hermes | 16 (7 over a remote connection) | 4 child roles | `/slm`, 74 aliases and 4 agent commands | 10 plugin events, no shell hooks | whatever `superlocalmemory` server Hermes is configured with |
| Antigravity | 16 | 4 | `/slm-loop` | 4 events through a small adapter script | none set, so the default |

How many tools each named profile exposes is on [MCP Tools](MCP-Tools). The Claude Code, Codex and Copilot plugins deliberately set no profile and no data folder, because a plugin that narrows your tool set or points at a private store would override choices you made; only the agent id is set, so memories written from each host can be told apart. The Cursor-format plugin is the one documented exception: it sets `core` because Grok Bot lists every tool to the model on a shared computer with little memory.

## Per-host notes

**Claude Code.** `slm connect claude-code` prints the two install commands, and `slm setup` can run them. Updates come through the plugin lifecycle ([Host Upgrades](Host-Upgrades)). The plugin's launcher and its hooks use an `slm` that is already installed; only when none is found does `ensure-venv.sh` build a private Python environment for the plugin, and that needs Python 3.12 or newer. A hook that cannot run never stops your session: the wrapper script `slm-run` always exits 0 and writes the reason to `plugin-hooks.log` in the SLM `logs` folder. The folder also contains `README.md`, which lists every hook, what it captures and that the data stays on your computer, and the `LICENSE`.

**Cursor-format hosts.** Grok Bot reads plugins in Cursor's format (`.cursor-plugin/plugin.json`) rather than Claude Code's. Its own folder, `cursor-plugin/`, is MIT licensed (the engine it starts stays AGPL), and `.cursor-plugin/marketplace.json` at the repository root lists it. The Cursor manifest has an empty hook table and names no agents or commands. Its MCP definition starts the server with `uvx`, pinned to the SLM release, and sets the agent id `cursor_plugin`, `SLM_NON_INTERACTIVE=1`, one embedding worker, the cross-encoder reranker off (`SLM_RERANKER_ENABLED=false`) and CPU-only PyTorch. No `${...}` variable appears in it, because Grok Bot does not expand the Claude Code plugin-root variable. It sets no tool set, so it serves the default `full` set like every other host (`slm tools set` changes it), and it carries all 16 skills, the two written for bot hosts among them. The memory savings a shared bot computer needs come from the one embedding worker and the reranker being off, not from fewer tools. This plugin keeps its own memory on the bot computer; to reach the memory on your own computer, connect Web access instead ([Web Access and Connected Apps](Web-Access-and-Connected-Apps)). `slm connect cursor` is a different thing: it writes an MCP entry into the Cursor editor ([IDE Setup](IDE-Setup)).

**Codex.** There are two ways to get the same ideas, and they are not identical. `slm codex install` copies 14 skills into the shared agent skills folder (`~/.agents/skills`), writes the four agents as `.toml` files into Codex's agents folder and adds 4 lifecycle hooks to Codex's `hooks.json`; it leaves MCP wiring to `slm connect codex`, only ever touches files it owns, and `slm codex remove` undoes it. The two bot-host skills are not part of that install. The marketplace plugin in `codex-plugin/` carries all 16 skills, the agents as Markdown, `/slm-loop`, its own `hooks/hooks.json` and an `AGENTS.md` of rules that is not written into your project for you. Review and trust new hooks in Codex with `/hooks`.

**GitHub Copilot.** `slm connect vscode-copilot --here` merges the MCP server into `.vscode/mcp.json` and the rules into `.github/copilot-instructions.md` without overwriting what is there. The folder's `.github/` then adds 16 prompt files (one per skill), 4 agent files marked `target: vscode`, and `hooks/slm-hooks.json`. Copilot has no separate command concept, so `/slm-loop` exists only as the prompt generated from the skill of that name. The plugin README says hooks are a preview feature in VS Code and that the rules file works as a fallback when hooks are not available.

**Hermes.** Hermes asks before it enables the plugin. It does not ask for MCP access: grant the plugin the `superlocalmemory` MCP server with `slm upgrade-hosts --host hermes --apply` (or `mcp_allowlist: ["superlocalmemory"]` under `plugins.entries.superlocalmemory` in `config.yaml`); without it the plugin's recall, session tracking and turn capture do nothing, and `slm doctor` warns. It calls SLM through that MCP server for its hooks and, for `/slm` commands, through the installed `slm` command, which must be exactly the SLM release the plugin was built with. Set the plugin's `connection` setting to `remote` and `/slm` instead runs a fixed set of commands through an SLM on another computer, with no local install; only the 7 skills that work over a remote connection are registered then, and the governance and loop agents are refused because they manage the SLM computer. Details and settings are in `docs/hermes.md` in the repository.

**Antigravity.** `agy plugin install` accepts only a local folder. With a PyPI install, `scripts/install-antigravity-plugin.sh` fetches just `antigravity-plugin/` into a local cache and installs from there; re-running it is safe. The plugin sets no tool set, so the server gives it the default `full` set, under the vendor-reported ceiling of 100 tools the build script records for the host; `slm tools set` changes it. It never sets `SLM_MCP_ALL_TOOLS`, which would register every tool and pass that ceiling.

**VS Code extension.** The extension adds SuperLocalMemory to the MCP servers VS Code's agent can use and runs `slm mcp`, which must be on your PATH (it also looks in `~/.local/bin`, `/opt/homebrew/bin` and `/usr/local/bin`, because an editor opened from the Dock often lacks your shell PATH). If the workspace's `.vscode/mcp.json` or your user `mcp.json` already has SuperLocalMemory, for example from `slm connect vscode-copilot --here`, it adds nothing, so there is never a second copy. It needs VS Code 1.101 or later and is MIT licensed.

**Grok CLI.** Grok CLI runs the Claude Code plugin unchanged: the 16 skills, 4 agents and the hooks. `slm connect grok-cli` writes the MCP server into `~/.grok/config.toml`. Grok asks for `--trust` on install because a plugin can run commands on your computer.

**Muse CLI.** `slm connect muse` writes the MCP server into Muse's `settings.json` (`$XDG_CONFIG_HOME/muse/settings.json`, else `~/.config/muse/settings.json`) and keeps every other setting. A file that still uses the old `mcp_servers` key has it renamed to `mcpServers` in the same write, because Muse ignores every server when both keys are present. `muse-plugin/` is the Claude Code plugin rebuilt for Muse: no `allowed-tools` lines (Muse rejects short tool names there; in Claude Code the line only pre-approves tools), no MCP launcher (Muse gets the server from `slm connect muse`), and a session-start hook that answers in JSON, because Muse drops hook output that starts with `[` as malformed JSON. Muse runs its hooks and agents after `muse plugins approve superlocalmemory`. The hosted Muse bot connects through Web access instead.

Any other agent can use the same discipline without a plugin: paste `plugin-src/rules/AGENTS.md` into `AGENTS.md`, `CLAUDE.md`, `.cursorrules` or the system prompt.

## Skills

A skill is a short document the agent loads when a request matches its description. These 16 are in `plugin-src/skills/`.

| Skill | What it teaches the agent | In the Cursor-format plugin |
|---|---|---|
| `slm-recall` | Search and retrieve past facts and decisions; recall before saving anything new | yes |
| `slm-remember` | Save durable decisions, constraints and gotchas; recall first to avoid duplicates | yes |
| `slm-session` | Call `session_init` once at the start of a session and `close_session` when work is done | yes |
| `slm-scope` | Keep memory personal by default; share across profiles only when asked | yes |
| `slm-bot-memory` | Several bots on one computer: attribution is not access control, and what must never be saved | yes |
| `slm-getting-started-bot` | First session on a bot host with no hooks or dashboard: which tools of the `core` profile exist (the skill's text still counts 18, from before `expand`) and the first calls | yes |
| `slm-status` | Health and optimization counters with `slm status`, `slm doctor` and `slm_optimize_stats` | yes |
| `slm-profile` | Memory profiles versus MCP tool sets, and `SLM_MCP_PROFILE` ([Profiles](Profiles)) | yes |
| `slm-governance` | Roles, company mode, retention, audit trail, GDPR export and erasure; never bypass governance ([Governance](Governance)) | yes |
| `slm-graph` | Code graph: what a change breaks, callers, semantic code search, review context | yes |
| `slm-mesh` | Messages, locks and shared state between agent sessions and web apps ([Multi-Agent Memory](Multi-Agent-Memory)) | yes |
| `slm-loop` | Gate-verified bounded loops with SLM as the ledger ([Bounded Loops](Bounded-Loops)) | yes |
| `slm-cache` | Exact cache for repeated reads and searches ([Optimize](Optimize)) | yes |
| `slm-compress` | Reversible compression of large output ([Optimize](Optimize)) | yes |
| `slm-media` | Save and find pictures and PDFs; on by default, set up by itself from 16 GB ([Pictures and Documents](Pictures-and-Documents)) | yes |
| `slm-web-access` | Set up and diagnose Web access for web apps ([Web Access and Connected Apps](Web-Access-and-Connected-Apps)) | yes |

Every host in the table above that is not Cursor-format gets all 16. The exceptions are `slm codex install` (14, without the two bot-host skills) and Hermes over a remote connection (7: `slm-cache`, `slm-compress`, `slm-recall`, `slm-remember`, `slm-scope`, `slm-session`, `slm-status`).

## Agents

The four agents are advisers. Each is a Markdown file in `plugin-src/agents/` with a fixed tool list, and none of them does your main task.

| Agent | What it does | Tools it may use |
|---|---|---|
| `slm-memory-advisor` | Decides whether something should be saved or recalled, phrases queries, keeps memory clean | `session_init`, `recall`, `search`, `remember`, `update_memory`, `forget`, `list_recent`, Read |
| `slm-optimize-advisor` | Applies reversible compression and the cache when context is filling up or the same reads repeat; advisory and fail-open | `slm_compress`, `slm_retrieve`, `slm_cache_set`, `slm_cache_get`, `slm_optimize_stats`, Read, Bash |
| `slm-governance-advisor` | Enforces the least-permissive safe action in governed workspaces, retention and erasure questions | `recall`, `search`, `remember`, `update_memory`, `list_recent`, Read, Bash |
| `slm-loop-runner` | Runs a task as a bounded loop until an independent gate passes and reports the exact terminal status | Bash, `recall`, `remember`, Read, `slm_loop_run`, `slm_loop_history`, `slm_loop_show` |

Claude Code, Codex (plugin), Antigravity and Hermes carry all four as Markdown; Copilot carries them as `.agent.md` files; `slm codex install` writes them as `.toml` files. The Cursor-format plugin has none. In Hermes an agent is never started by a hook: you start one with `/slm-agent <memory|governance|optimize|loop> <goal>`, or the model starts one with the `slm_agent` tool, and it runs as a bounded child of the current chat turn.

## Slash commands

| Command | Hosts | What it does |
|---|---|---|
| `/slm-loop <task>` | Claude Code, Codex plugin, Antigravity | Hands the task to the `slm-loop-runner` agent: name the independent gate, state the bounds, iterate, and report `DONE`, `HALT`, `PAUSE`, `KILLED` or `ERROR` exactly |
| `/slm <command> [args]` | Hermes | Runs an SLM command; `/slm help` shows the tree |
| `/slm-<command>` | Hermes | One generated alias per top-level command (73 of them, listed in `hermes-plugin/command-inventory.json`); `search` is an alias of `recall` |
| `/slm-agent <role> <goal>` | Hermes | Starts one bounded advisor child agent |
| `/slm-agent-status`, `/slm-agent-cancel`, `/slm-agent-result` | Hermes | Take the JSON handle `/slm-agent` returned and inspect, cancel or fetch that child |

In Hermes, 26 commands are marked high-impact in the inventory (for example `forget`, `delete`, `migrate`, `db`, `gdpr`, `backup`, `token`), and `remember` becomes high-impact with `--replaces`. They show a preview and run only when you repeat the line with `CONFIRM`. Hermes's `/slm-loop` is the alias for the `slm loop` command, not the Claude-style loop command. Over a remote connection Hermes runs only `status`, `health`, `recall`, `remember`, `list`, `delete`, `update`, `summary`, `trace`, `kinds` and `help`. Copilot has the loop prompt but no command; Cursor-format hosts have no commands at all.

## Hooks

A hook is a command the host runs at a fixed moment. What a hook saves (a "file changed" note, a session summary, a captured observation) passes the same save step as a typed memory, and the hook event spool screens what a tool printed: with the secret screen on (`trust.secret_screen`), a credential is stored as a `[REDACTED:TYPE:LAST4]` placeholder and never written in clear. With the instruction firewall on (`trust.taint`) such memories are a tool's output or an agent's words, so they cannot become pins, standing rules or core memory until you approve them. Every hook the plugins install is built to fail open: its output is advisory text or nothing, errors are swallowed, and none of them blocks your work. They all run as `slm hook <action>`, handled in `src/superlocalmemory/hooks/hook_handlers.py` and the modules beside it.

| Action | What it does in plain words |
|---|---|
| `mandate` | Prints a short advisory note that `session_init` is available and how to load its schema if the host defers tools. It is a hint, not an instruction. |
| `start` | Clears the previous session's temporary markers, prints recent context for the project from a fast database read, says so if memory is unavailable because a restore is waiting for recovery, and reminds the agent to call `session_init`. |
| `checkpoint` | After a file write or edit, saves a one-line "file changed" memory tagged `hook-file-change`, at most once per file every 5 minutes, and asks the agent to follow up with an `observe` note saying what changed and why. |
| `post_tool_outcome` | When a tool's output contains the signed marker SLM attaches to a recalled memory, records that the agent used that memory (as an edit signal for edit and write tools, otherwise as a short dwell). This feeds the learning loop. |
| `user_prompt_rehash` | If you ask the same thing again within 60 seconds of a recall, records a re-query signal: the earlier answer did not satisfy you. |
| `topic_shift` | After at least two earlier prompts, if a prompt of at least 5 content words shares none with any of your last 5 prompts, prints a one-line reminder to call `recall` on the new topic. Silent otherwise. |
| `before_web` | Before a web search or fetch, runs `slm recall` on the query or URL and, if there is a useful hit, shows the top results as untrusted evidence with a note to read them first and skip the web call if they answer it. Silent when nothing matches; never blocks the call. |
| `stop` | Saves one session-end memory with the project, branch, uncommitted changes, commits from the last 3 hours and files touched, records a session-end event, and starts a full consolidation if the last one was more than a day ago. |
| `stop_outcome` | Settles every recall outcome still pending for the session so the learning loop can score it. |

Which host wires which action, with the event name and the timeout in the host's file (seconds for Claude Code and Codex; Copilot uses `timeoutSec`; the Antigravity file gives no unit):

| Action | Claude Code | Codex plugin | GitHub Copilot | Antigravity |
|---|---|---|---|---|
| `mandate` | SessionStart, 5 | SessionStart, 5 | sessionStart, 5 | PreInvocation, 15 with `start` |
| `start` | SessionStart, 15 | SessionStart, 15 | sessionStart, 15 | PreInvocation, 15 with `mandate` |
| `checkpoint` | PostToolUse on Write or Edit, 5 | not wired | postToolUse on create, edit, `str_replace_editor`, `apply_patch`; 5 | not wired |
| `post_tool_outcome` | PostToolUse on ten tools, 5 | PostToolUse on the same ten, 5 | not wired | PostToolUse on nine Antigravity tools, 5 |
| `user_prompt_rehash` | UserPromptSubmit, 5 | not wired | not wired | not wired |
| `topic_shift` | UserPromptSubmit, 3 | not wired | userPromptSubmitted, 3 | not wired |
| `before_web` | PreToolUse on WebSearch or WebFetch, 5 | PreToolUse on WebSearch or WebFetch, 5 | not wired | PreToolUse on `web_search` or `web_fetch`, 5 |
| `stop` | Stop, 10 | not wired | sessionEnd, 10 | Stop, 10 with `stop_outcome` |
| `stop_outcome` | Stop, 10 | Stop, 10 | sessionEnd, 10 | Stop, 10 with `stop` |

Claude Code's file also runs `ensure-venv.sh` at SessionStart, with no timeout set; it does nothing when an installed `slm` will be used. Antigravity cannot run `slm hook` directly, so its hooks call `antigravity-plugin/scripts/antigravity_hook_adapter.py`, which turns the host's JSON into the same `slm hook` calls and answers in the JSON shapes Antigravity documents. It runs `mandate` and `start` once per conversation, remembered with a marker file that `Stop` removes, and each call it makes to `slm` is cut off after 8 seconds. The plugin also keeps an older `hooks/hooks.json` in the Codex shape, which its `plugin.json` still names; the root `hooks.json` is the one current Antigravity reads, according to the build script.

`slm codex install` does not copy the plugin's hook file. It writes four Codex-native hooks: `codex-start` at SessionStart (15 s; opens the session and loads context in parallel), `checkpoint` after Edit or Write (5 s), `codex-prompt` at UserPromptSubmit (5 s; runs `user_prompt_rehash` and `topic_shift` side by side) and `codex-stop` at Stop (12 s; the same checkpoint as `stop`). Each has its own shorter internal deadline so it always answers before Codex's limit.

Hermes has no shell hooks. Its plugin registers ten Python lifecycle events that call the configured SLM MCP server with a 3-second ceiling per call (5 seconds for recall over a remote connection), and every failure is swallowed:

| Hermes event | What the plugin does |
|---|---|
| `on_session_start` | Calls `session_init` for the project |
| `pre_llm_call` | Recalls up to 5 memories for the user's message and offers them to the model labelled as untrusted evidence |
| `post_tool_call` | Logs a scrubbed record of the tool call |
| `post_llm_call` | Saves every completed turn as a memory, in any language with secrets redacted, only if you set `capture_turns: true` (off by default) |
| `on_session_end` | Settles recall outcomes that already carry real engagement evidence (Hermes fires this after every turn) |
| `on_session_finalize` | Settles all outcomes and closes the session |
| `on_session_reset` | Finalizes the old session and starts the new one |
| `on_skill_lifecycle`, `subagent_start`, `subagent_stop` | Log skill use and the start and result of an advisor child |

Text sent to SLM from these events is cut to a length limit and passed through a filter that masks API-key-like strings, bearer tokens, passwords and email addresses.

Two more things to know. First, installing the Claude Code plugin and running `slm hooks install` are two separate ways to get hooks: the second writes similar entries into Claude Code's own settings (plus `slm session close` at Stop and an experimental `--gate` that blocks other tools until `session_init` has run), so use one or the other ([Auto-Memory](Auto-Memory)). Second, the scripts in `ide/hooks/` ship in the npm package, but no installer in this repository registers them; the old standalone skill installer is now a stub that points at the plugin.

## How the plugin folders are built

Do not edit the generated folders. `plugin-src/` is the source and the `scripts/build-*.mjs` programs write the rest. Each generated folder has a `_GENERATED.md` saying so.

| You want to change | Edit | Then run |
|---|---|---|
| A skill, agent or command | `plugin-src/skills/<name>/SKILL.md`, `plugin-src/agents/*.md`, `plugin-src/commands/*.md` | `npm run build:plugins` |
| Claude Code hooks, launcher scripts, MCP definition | `plugin-src/hooks/hooks.json`, `plugin-src/scripts/`, `plugin-src/.mcp.json` (why it sets no profile: `plugin-src/MCP_JSON_NOTES.md`) | the same |
| Claude Code rules block | `plugin-src/rules/CLAUDE.md.fragment`, which becomes `plugin/CLAUDE.md` | the same |
| Which skills Cursor-format hosts get, and their server settings | `CURSOR_SKILLS` and `CURSOR_SERVER_ENV` in `scripts/build-cursor-plugin.mjs` | the same |
| Copilot or Antigravity hook files | the strings inside `scripts/build-copilot-plugin.mjs` and `scripts/build-antigravity-plugin.mjs` | the same |
| Codex rules, README, launcher scripts, `.codex/config.toml`, `hooks/hooks.json` | the files in `codex-plugin/` themselves; these are the authored part | the same, to restamp the version |
| Hermes | `plugin-src/hermes/` | the same |
| What a hook does | `src/superlocalmemory/hooks/` | the normal tests |

`scripts/build-all-plugins.mjs` runs the builders in the one order that works: `build-plugin.mjs` first (it writes `plugin/`, including the Cursor-format files, and the rules file the others read), then Copilot, Codex, Antigravity and Hermes. `node scripts/build-all-plugins.mjs --check` verifies that every generated folder is in sync without writing, and the package's `prepack` step runs the build. Versions are stamped from `plugin-src/manifest.json`, except the Hermes plugin, which is stamped from `pyproject.toml` because it refuses a mismatched `slm`.

Related pages: [IDE Setup](IDE-Setup), [Host Upgrades](Host-Upgrades), [MCP Tools](MCP-Tools), [Framework Adapters](Framework-Adapters) for code-level integrations, and [Auto-Memory](Auto-Memory) for what the capture hooks decide to save.

Code: `plugin-src/`, `plugin/`, `.claude-plugin/marketplace.json`, `.cursor-plugin/marketplace.json`, `codex-plugin/`, `copilot-plugin/`, `hermes-plugin/`, `antigravity-plugin/`, `scripts/build-*.mjs`, `src/superlocalmemory/hooks/`, `docs/plugins.md`.
