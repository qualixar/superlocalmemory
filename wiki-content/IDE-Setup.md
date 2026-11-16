# IDE Setup

`slm connect` wires SLM into an AI coding tool so the tool can call SLM's MCP server and, where the tool has a rules file, read a short instruction block that tells it when to recall and remember. The command knows 13 named targets: eleven that get an MCP entry written for them, and two, Claude Code and Hermes, that are installed as plugins instead. After any change, restart the tool completely, not just the window.

## What `slm connect` does

Naming a target, as in `slm connect cursor`, does the full job. It merges a `superlocalmemory` entry into that tool's MCP config, leaving every other server and key in the file untouched, writes atomically, and aborts without touching the file if it cannot parse it. It also appends SLM's short protocol block to the tool's rules file where one exists, between `<!-- SLM-START -->` and `<!-- SLM-END -->` markers, and never overwrites your own rules. That block holds only product-authored instructions; recalled memory text is never copied into these high-trust files. Live context arrives over MCP at run time as bounded, untrusted evidence. The command then says "Configuration written; host verification pending.", because SLM writes the entry but cannot see whether the tool has picked it up.

Running `slm connect --list` prints every target with how it is installed (`mcp-config` or `native-plugin`) and where its config file would be, and `--json` returns the same as data. Running `slm connect` with no target connects every target whose config file already exists on this computer, writing the full entry and rules block for each one; targets installed as plugins are skipped. To connect one tool only, name it.

**The setup wizard and the dashboard use an older, narrower connector.** The first-run wizard's "Connect them now?" step and the dashboard's connect action use a separate connector that knows seven editors (Claude Code, Cursor, VS Code / Copilot, Windsurf, Gemini CLI, Continue.dev and Zed). For each one it finds, it only appends the instruction block to the rules file (or merges JSON for Continue and Zed); it does not write the MCP entry. To get the MCP entry written, run `slm connect <target>`, which is also what the wizard tells you when you skip.

**What the stdio entry runs.** The entry that is written does not just say `slm`. It pins the Python that ran the command, so the tool starts the same installation you connected from: the command is that interpreter with `-m superlocalmemory.cli mcp`, and the entry's environment carries `SLM_MCP_HOST` (the target's id), `SLM_DATA_DIR` (your data folder) and the credential-enforcement flags that were set when you connected (`SLM_REQUIRE_CREDENTIALS`, `SLM_REQUIRE_API_KEY_LOOPBACK` and `SLM_DISABLE_LEGACY_PORT`, only those that are set). No secret is written. If you move to a different Python, run `slm connect` again.

## The 13 targets

| Target | MCP config written | Notes |
|---|---|---|
| `cursor` | `~/.cursor/mcp.json` (key `mcpServers`) | Rules file `.cursorrules`; `--here` writes `.cursor/mcp.json` in the project |
| `antigravity` | `~/.gemini/config/mcp_config.json` (`mcpServers`); `--here` writes `.agents/mcp_config.json` | No rules file |
| `windsurf` | `~/.codeium/windsurf/mcp_config.json` (`mcpServers`) | Rules file `.windsurfrules`; global only |
| `gemini-cli` | `~/.gemini/settings.json` (`mcpServers`) | Rules file `GEMINI.md`; global only |
| `vscode-copilot` | `.vscode/mcp.json` (key `servers`, not `mcpServers`) | Project-scoped: requires `--here` from the project root; rules file `.github/copilot-instructions.md` |
| `zed` | `~/.config/zed/settings.json` (key `context_servers`) | No rules file |
| `jetbrains` | `~/.config/JetBrains/mcp.json` (`mcpServers`); `--here` writes `.mcp.json` | Rules file `.junie/AGENTS.md` |
| `opencode` | `~/.config/opencode/opencode.json` (top-level key `mcp`); `--here` writes `opencode.json` | |
| `claude-desktop` | The Claude Desktop config file for your OS (`~/Library/Application Support/Claude/` on macOS, `%APPDATA%\Claude` on Windows, `~/.config/Claude` on Linux) | The desktop app, not Claude Code |
| `codex` | `~/.codex/config.toml` (TOML table `mcp_servers`) | Rules file `AGENTS.md` |
| `continue` | `~/.continue/config.yaml` (list `mcpServers`, matched by name) | Rules: none |
| `hermes` | Nothing | Prints the command that installs the native Hermes plugin |
| `claude-code` | Nothing | Prints the plugin install commands |

Names are exact: `slm connect vscode` is refused with "Unknown IDE" and a list of the valid ids, and the VS Code target is `vscode-copilot`. Several of these paths are marked best-effort in the source because the editors document them loosely, so if a tool does not pick the entry up, check where that version of the tool actually reads MCP config. A target with no project-scoped location (Windsurf, Gemini CLI, Zed, Claude Desktop, Codex) refuses `--here`.

Useful flags: `--dry-run` shows what would be written and changes nothing, `--here` uses project scope, `--json` returns the result as an object (`mcp_config` is one of `wrote`, `merged`, `unchanged`, `would_write`, `skipped`, `error`), `--verify` (with an HTTP transport) probes the daemon afterwards, and `--profile <name>` selects an MCP tool profile (see below). `slm connect --cross-platform` runs the adapters that keep instruction files fresh for Cursor (`.cursor/rules/*.mdc`), Antigravity (`.agent/skills/`) and Copilot (`.github/copilot-instructions.md`), and `--disable <adapter>` turns one off.

## Transports

By default the entry launches a stdio server: your Python interpreter running `-m superlocalmemory.cli mcp`, as described above. Each connection then starts its own MCP process, which costs memory per open tool. Two alternatives share the single daemon instead.

| `--transport` | What is written | Needs |
|---|---|---|
| `stdio` (default) | The pinned Python running the MCP server as a subprocess | Nothing; works with no daemon |
| `http` | A URL entry, `http://127.0.0.1:8765/mcp/`, in each host's own shape (`type: http` and `url` for most; `serverUrl` for Antigravity, `httpUrl` for Gemini CLI, `type: remote` for OpenCode, `type: streamable-http` for Continue) | The daemon running; `--port` changes 8765 |
| `http-mcp-remote` | A stdio entry running `mcp-remote` against that URL, for tools that speak only stdio | The daemon, plus `mcp-remote` installed globally with npm |

`slm connect <target> --transport http --verify` probes the daemon's health endpoint after writing and warns, with `slm serve start` as the fix, if it does not answer. A write is never blocked by a failed probe. If a hand-written config uses `npx` for the bridge instead of a global install, the command in the file must be `npx` with `-y mcp-remote`; naming `mcp-remote` directly fails with "command not found" unless it is installed globally.

With stdio, `--profile code` adds `SLM_MCP_PROFILE=code` to the entry's environment, which narrows the tool list ([MCP Tools](MCP-Tools)). With the HTTP transports `slm connect` appends `?profile=<name>` to the URL, but I found no code in the MCP endpoint that reads that parameter, so treat it as not narrowing the tool list; only `SLM_MCP_PROFILE` does that. To work on a different memory profile for one call, pass `profile_id` on the tool call: the named profile serves that call, the active profile never moves, and a name that is not a profile is refused rather than created.

To label what each tool writes, set `SLM_AGENT_ID` in the entry's `env` block, or connect over a URL such as `http://127.0.0.1:8765/mcp/cursor`; the Claude Code and Codex plugins already do this ([Multi-Agent Memory](Multi-Agent-Memory)). The Codex template in `ide/configs/codex-mcp.toml` sets `SLM_AGENT_ID = "codex"`.

## Claude Code and Codex

Claude Code is configured through the SLM plugin, not an MCP file. `slm connect claude-code` prints the two commands, and `slm setup` can run them for you:

```bash
claude plugin marketplace add qualixar/superlocalmemory
claude plugin install superlocalmemory@qualixar
```

The plugin brings the MCP server, 16 skills, four advisor agents, the `/slm-loop` command and lifecycle hooks. If you do not want the plugin, `slm hooks install` writes additive hooks directly, and you can add the MCP server by hand with `claude mcp add --transport http superlocalmemory http://127.0.0.1:8765/mcp/`. On native Windows install SLM first (`pipx install superlocalmemory` or `pip install superlocalmemory`) so `slm` is on PATH; the plugin's server runs your installed SLM and says so in its log if it is missing.

Codex gets the MCP entry from `slm connect codex`, lifecycle hooks from `slm hooks install --agent codex` (written to `~/.codex/hooks.json`), and its own skills and subagents from `slm codex install|remove|status`. Both plugins serve the default 60-tool profile unless you add `SLM_MCP_PROFILE=code`. Details and the upgrade path are in [Host Upgrades](Host-Upgrades).

## Tools without a connect target

Grok CLI, Kimi, CommandCode and pi.dev have no `slm connect` target; the repository documents hand-edited entries for them. All use the same server command or URL. Hermes has a `slm connect hermes` target that prints the command to install the native Hermes plugin (`hermes-plugin/`), which adds namespaced skills, `/slm` commands and lifecycle hooks, and `slm upgrade-hosts` recognises it by an `superlocalmemory` entry under `mcp_servers` in `~/.hermes/config.yaml`. Grok CLI supports stdio only, so use `command = "slm"` with `args = ["mcp"]` and `SLM_AGENT_ID = "grok"`, or the `mcp-remote` bridge. For pi.dev, add `{"mcpServers": {"superlocalmemory": {"command": "slm", "args": ["mcp"], "directTools": true}}}` to `~/.pi/agent/mcp.json` and enable it from pi.dev's `/mcp` menu; there is no plugin or connect target for it. For the plugins every host gets, see [Plugins, Skills and Hooks](Plugins-Skills-and-Hooks).

## Verifying

```bash
slm status                 # daemon, mode, profile, database size
slm doctor                 # dependencies, embedding worker, daemon connectivity
slm mcp                    # should sit waiting for stdio input; Ctrl+C to stop
```

Then ask the tool something that needs memory, such as "what do you know about my recent work?". If nothing is stored yet, `slm remember "test"` first. All connected tools share one memory database, so a memory saved from Claude Code is recallable from Cursor.

## When a tool does not see SLM

Restart the tool fully, not just the window. Confirm `which slm` returns a path, because stdio entries run `slm` from the tool's own environment, which may have a different PATH than your shell. Rerun `slm connect <target>`; it is idempotent. For HTTP entries, a 404 or connection refused means the daemon is down (`slm serve start` or `slm restart`). Editors do not conflict with each other, since each has its own config file. More in [Troubleshooting](Troubleshooting); the first-time wizard is `slm setup`, which detects tools and asks before connecting them ([Installation](Installation), [Getting Started](Getting-Started)).

Code: `src/superlocalmemory/hooks/portable_kit.py` (the target matrix and merge logic), `hooks/ide_connector.py` (the older seven-editor connector the wizard and dashboard use), `cli/connect_cmd.py`, `hooks/portable_install.py`, `cli/commands.py` (`cmd_connect`), `cli/main.py`, `ide/configs/`.
