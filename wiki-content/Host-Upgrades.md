# Host Integration Upgrades

Updating SLM through npm, pip or a repository checkout updates the `slm` program and nothing else. It does not rewrite Claude Code, Codex, Cursor, VS Code, Antigravity, Hermes or any other tool's configuration or instructions, and the only hooks it changes are SLM's own entries (see below). Those files are yours and may hold unrelated servers, policies or credentials. When you want the integrations refreshed, run `slm upgrade-hosts`: it previews by default, changes only SLM-owned pieces, and applies only to hosts you name. Installing and consenting are kept separate on purpose so an upgrade stays reversible and easy to inspect.

## Preview first

```bash
slm upgrade-hosts
```

With no flags this is read-only. It looks only at hosts that already contain an SLM entry, so it never discovers and touches an editor you have not connected. It does not create a data directory, start the daemon, edit any file or turn on the experimental tool gate. For each host it prints one line in the form `[status] host: detail`. The statuses are `preview` (a change that `--apply` would make), `updated` (a change made), `verified` (an existing SLM entry was checked and left alone), `blocked` (it could not proceed, with the reason).

How "already integrated" is decided: for each target in the connect matrix it reads the tool's global MCP config and looks for a `superlocalmemory` entry (a Continue entry is matched by its server name); for Claude Code it checks whether SLM's hooks are installed; for Hermes it looks for `superlocalmemory` under `mcp_servers` in `~/.hermes/config.yaml`.

## Apply

```bash
slm upgrade-hosts --host codex --apply
slm upgrade-hosts --host cursor --host zed --apply     # --host is repeatable
slm upgrade-hosts --all-detected --apply               # every host that already has SLM
```

Two rules are enforced before any file is read. `--apply` with neither `--host` nor `--all-detected` is rejected with "--apply requires at least one --host or --all-detected", so a package update cannot sweep every supported application. And `--host` together with `--all-detected` is rejected, so you choose one meaning. A host name that is not in the managed list (the editors in [IDE Setup](IDE-Setup) plus `claude-code` and `hermes`) is refused as unsupported.

## What actually changes

The rule is merge, never clobber. For a portable editor that already has an SLM entry, the entry is verified and preserved, with the status `verified`, so a custom command path, an agent identity set in its environment, or unrelated MCP servers are never downgraded to a generic block.

Codex is the one host with refreshable assets. If its SLM entry exists, apply refreshes only SLM-owned skills, subagent files and SLM's lifecycle hook entries (replaced in place, so a changed timeout lands; your own hooks stay and nothing is added to a hooks file that has no SLM entry), and the MCP block stays intact. The result line names what was written: how many skill files and subagent files, where they went, and which subagent files were preserved because SLM did not write them. If some skill paths resolve outside the location SLM writes, the line says those were not refreshed rather than reporting a no-op as a success.

For Claude Code, if SLM's hooks are in `settings.json` (`slm hooks install`), apply replaces and adds only SLM's entries there; a file with no SLM hooks is left alone. If you use the marketplace plugin, Claude Code owns it, so refresh it with:

```bash
claude plugin marketplace update qualixar
claude plugin update superlocalmemory@qualixar
```

A Claude Code without the plugin also gets its skills, subagents and `/slm-loop` refreshed. Only the `slm-*` skills in `~/.claude/skills`, the `slm-*` advisors in `~/.claude/agents` and `~/.claude/commands/slm-loop.md` are looked at, and the preview names each file it would add or refresh. The status line says "already current" only when the hooks and all of these match the release. A file SLM wrote and you did not change is refreshed; a new skill is added; a file you changed after SLM wrote it is kept and named; a skill or agent you deleted is not put back. A file from before SLM kept a record of what it wrote is replaced only by `slm upgrade-hosts --apply`, after the old copy is saved beside it as `<name>.bak-slm-<version>`. A path that resolves outside `~/.claude` is reported and left alone, and with the plugin installed `~/.claude` is not touched. SLM's record is `~/.claude/.slm-managed.json`.

Every host that takes hooks from a plugin (Claude Code, Codex, Hermes, Antigravity, Muse) gets its exact update command printed under its result line, and `slm doctor` prints the same.

Hermes is reported `verified` if its existing mapping is present, and the mapping is never overwritten with an invented format; if Hermes is not connected the result is `blocked` and points you to the setup steps.

Naming a host that is not yet connected is a first-time connection, not an upgrade: `slm upgrade-hosts --host windsurf --apply` writes the MCP entry the same way `slm connect windsurf` does. That path is never reached by `--all-detected`.

The command ends with a line, "Apps to restart or reconnect", that names the hosts you just applied plus any host whose running MCP session reports an older version than the installed SLM, and points to the dashboard's Connected apps page for reconnecting live sessions. Restart or reconnect those applications, then run `slm doctor` to check SLM itself. If nothing is detected, the command says so and suggests `slm setup` for first-time setup.

## SLM's own hooks upgrade by themselves

Hooks already live in people's harnesses, so an upgrade only replaces SLM's own hook entries and adds missing ones. It never replaces a whole file and never changes another hook, a setting or the order of keys. It runs once per installed version, inside the SLM background service a few minutes after start-up settles, for two files: Claude Code's `~/.claude/settings.json` and Codex's `~/.codex/hooks.json`. A file with no SLM hook entry is left exactly as it is, and SLM never creates one. The first change to a file keeps a copy beside it (`settings.json.bak-slm-<version>`), a file that is already current is not written again, and a file that cannot be read is reported and left alone. The same step adds the skills, subagents and command that a Claude Code without the plugin is missing, and refreshes the ones SLM wrote and you did not change; it never replaces a file from before SLM kept a record (the explicit `--apply` does that, with a backup). `SLM_NO_HOOK_UPGRADE=1` turns it off; `hooks-upgraded.json` in the SLM data folder records the version that ran. `slm upgrade-hosts --host claude-code --apply` does the same on demand, and `slm doctor` reports per file whether SLM's hooks are current.

## Your memory store after an upgrade

A separate safeguard covers data. An upgrade that changes the memory store first takes a verified copy; the restore points can be listed and used from the CLI (`slm db restore-points`, `slm db restore`), and there is no dashboard screen for them. A few minutes after SLM first starts on a new version it also checks the store once, read-only, and reports in the dashboard under Health > Memory store, with a Repair now button if something needs fixing ([Health and Repair](Health-and-Repair)).

## First-time setup instead

On a new machine use `slm setup`, the interactive wizard that detects installed tools and asks before connecting each one, or `slm connect <target>` for a single tool. The npm package creates a private Python runtime and the pip package installs into the active environment; neither performs host activation on its own.

Code: `src/superlocalmemory/cli/host_upgrades.py`, `hooks/portable_kit.py`, `hooks/codex_assets.py`, `hooks/codex_hooks.py`, `hooks/host_hook_upgrade.py`, `docs/host-upgrades.md`.
