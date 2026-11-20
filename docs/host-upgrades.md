# Host Integration Upgrades

Updating SuperLocalMemory updates the SLM executable. It does **not** silently
rewrite an AI application's configuration, plugin state, or workspace
instructions. Those files belong to the operator and may contain unrelated
tools, policies, or credentials. The one exception is SLM's own hook entries,
which are replaced and added in place, never touching anything else (see below).

Use the consented host-upgrade flow after upgrading through npm, pip, or a
repository checkout:

```bash
slm upgrade-hosts
```

This is a read-only preview. It detects existing SLM integrations and reports
what is safe to refresh. It does not create an SLM data directory, start a
daemon, change a configuration file, or enable the experimental PreToolUse
gate.

## Apply an upgrade

Review the preview, then target the hosts you approve:

```bash
slm upgrade-hosts --host codex --apply
slm upgrade-hosts --host cursor --apply
```

To refresh every *already integrated* host in one deliberate action:

```bash
slm upgrade-hosts --all-detected --apply
```

`--apply` without `--host` or `--all-detected` is rejected. This prevents a
package update from discovering and changing every supported application.

## What is changed

The command uses a merge-not-clobber rule. Existing portable MCP blocks are
verified and left intact so host-specific command paths, environment variables,
and unrelated MCP servers are not downgraded. For Codex, it may refresh only
SLM-owned skills, agents, and lifecycle hooks; the MCP block remains intact.

If you installed SLM's hooks into Claude Code's `settings.json`
(`slm hooks install`), `slm upgrade-hosts --host claude-code --apply` refreshes
only those entries. If you use the Claude Code marketplace plugin instead, refresh
it through Claude Code's own plugin manager:

```bash
claude plugin marketplace update qualixar
claude plugin update superlocalmemory@qualixar
```

The same command also refreshes the skills, subagents and `/slm-loop` command of a
Claude Code that has no plugin. Only files SLM ships are looked at: the
`slm-*` skills in `~/.claude/skills`, the `slm-*` advisors in `~/.claude/agents` and
`~/.claude/commands/slm-loop.md`. The advisors name SLM's tools in full
(`mcp__superlocalmemory__recall`), the only spelling a Claude Code without the plugin
resolves, so an advisor that already uses it stays byte-identical. The preview names each file it would add or
refresh, and the status line says "already current" only when the hooks and all of
these match this release.

- A file SLM wrote and you never changed is refreshed in place. A skill that is new
  in this release is added.
- A file you changed after SLM wrote it is kept and named in the output. Move it
  aside and run the command again to take the new version.
- A file from before SLM kept a record of what it wrote, and a file that differs
  from this release, is replaced only by `slm upgrade-hosts --apply`, and only
  after the old copy is saved next to it as `<name>.bak-slm-<version>`.
- A skill or agent you deleted is not put back. Anything SLM does not ship is never
  read or written.
- A path that resolves outside `~/.claude` (a skill folder linked to a checkout, or
  a skills folder kept in a dotfiles repository) is reported and left alone.
- With the plugin installed, `~/.claude` is not touched at all.

SLM's record of what it wrote is `~/.claude/.slm-managed.json`.

Then restart the affected host application. Finally verify the local runtime:

```bash
slm doctor
```

## SLM's own hooks upgrade by themselves (4.2.30)

Hooks already live in people's harnesses, so an upgrade only ever **replaces SLM's own
hook entries and adds missing ones**. It never replaces a whole file, and never changes
another hook, setting, or the order of keys. It happens once per installed version,
in the SLM background service, a few minutes after start-up settles, for two files:
Claude Code's `~/.claude/settings.json` and Codex's `~/.codex/hooks.json`.

- Only a file that **already holds SLM hook entries** is touched. A file with none is left
  exactly as it is, and SLM never creates one.
- The first change to a file keeps a copy next to it: `settings.json.bak-slm-<version>`.
- A file that already holds the current entries is not written again.
- A file that cannot be read is reported and left alone; the upgrade tries again next start.
- The same step adds the skills, subagents and command that are missing from a Claude
  Code without the plugin, and refreshes the ones SLM wrote and you did not change. It
  never replaces a file from before SLM kept a record; `slm upgrade-hosts --host
  claude-code --apply` does that, with a backup.
- `SLM_NO_HOOK_UPGRADE=1` turns the automatic upgrade off. The marker
  `hooks-upgraded.json` in the SLM data folder records the version that ran.
- On demand: `slm upgrade-hosts --host claude-code --apply` (or `--host codex`) does the
  same thing, and `slm doctor` says per file whether SLM's hooks are current.

Hosts that take hooks from a plugin (the Claude Code plugin, the Codex plugin, Hermes,
Antigravity, Muse) belong to the host, so SLM does not write them. `slm upgrade-hosts` and
`slm doctor` print the exact command each one needs to pick up the new version.

## Your memory store after an upgrade

An upgrade that changes the memory store first takes a verified copy; see
[Restore points](restore-points.md). A few minutes after SLM first starts on the
new version it also checks the store once, read-only, and reports the result in
the dashboard under **Health → Memory store**, with **Repair now** if anything
needs fixing. See [Troubleshooting](troubleshooting.md#memory-store-check).

## First-time setup

For a new machine, use the interactive setup flow instead of the upgrade flow:

```bash
slm setup
```

The wizard detects integrations and asks before connecting them. For a single
new host, use `slm connect <host>`; see [IDE Setup](ide-setup.md) for supported
hosts and host-specific restart instructions.

## Package-manager behavior

The npm package creates a private Python runtime. The pip package installs into
the activated Python environment. Neither package manager is used to perform
host activation automatically: installation and host consent are separate
operations. This separation is intentional and keeps an upgrade reversible,
inspectable, and compatible with developer-managed configuration.
