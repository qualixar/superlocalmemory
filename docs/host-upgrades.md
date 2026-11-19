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
