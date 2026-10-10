# Why `.mcp.json` sets no profile and no data directory

It used to set both:

```json
"SLM_MCP_PROFILE": "code",
"SLM_DATA_DIR": "${CLAUDE_PLUGIN_DATA}"
```

Both are wrong for anyone who already uses SLM, and neither is needed by anyone
who does not.

**`SLM_DATA_DIR` pointed the plugin at its own private directory.** On a machine
with an existing store that means the editor talks to an empty one: measured on
the author's machine, `${CLAUDE_PLUGIN_DATA}` was 28 KB while the real store was
611 MB with 5,370 memories in it. Installing the plugin would have looked like
losing every memory. Omitted, SLM resolves its canonical data root, which is the
same store every other surface uses — and on a fresh machine that is a new store
anyway, so nothing is lost either way.

**`SLM_MCP_PROFILE: code` narrowed the tool set.** `code` (38 tools) drops
the 8 mesh tools among others. Forcing it overrode a wider profile the user had
deliberately configured. Omitted, the server falls back to the same no-profile
default as every other install — the 57-tool `full` surface — which is the
user's decision to narrow or not, not the plugin's.

`SLM_AGENT_ID` stays: it is attribution, not configuration, and it is what lets
memories written from Claude Code be told apart from every other agent.

A plugin should add capability. It should not quietly re-point the data it reads
or take tools away.

# Why the Windows launcher is not what Claude Code runs

Until 4.1.20 `.mcp.json` named only `${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch`,
with no extension. On macOS and Linux that is the bash launcher. On Windows it is not resolved to
`slm-launch.bat`: a host that starts an MCP server without a shell goes through
`CreateProcess`, which cannot start a batch file and appends only `.exe` to a
name that has no extension. Node's `child_process.spawn` without `shell` and
libuv behave the same way (libuv tries the literal name, then `.com`, then
`.exe`). The Windows CI runner checks this directly:
`tests/test_plugin/test_windows_mcp_spawn_premise.py`.

So `slm-launch.bat` is never what a plugin install runs on Windows; it stays only
for hand-written `cmd /c` entries.

# How the one `.mcp.json` starts the server on each platform (#139)

A plugin `.mcp.json` has no per-platform command, so the entry chooses the
program with an environment default:

```json
"command": "${ComSpec:-${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch}",
"args": ["/d", "/s", "/c", "where.exe /q slm || (echo ... 1>&2 & exit 1) & slm serve start 1>&2 & slm mcp"]
```

- **Windows** always defines `ComSpec` (the path of `cmd.exe`). The host runs
  `cmd.exe /d /s /c "<line>"`, and the line runs **the SuperLocalMemory you
  installed**: pip and pipx put `slm.exe` on PATH. If `where.exe` finds no
  `slm`, it prints how to install it on stderr and exits 1, and nothing is
  written to the MCP stdout. The daemon is started first with its output
  sent to stderr, as the POSIX launcher does. The plugin venv is not used on
  Windows, and no plugin path is handed to `cmd.exe` (an unquoted
  forward-slash path is read as switches there).
- **macOS and Linux** do not define `ComSpec`, so the default applies and the
  host runs the bash launcher exactly as before. The launcher never reads its
  arguments, so the four `cmd.exe` arguments change nothing; the POSIX tests
  run the declared entry, arguments included.

Why this expands the way it does: Claude Code substitutes
`${CLAUDE_PLUGIN_ROOT}` literally first and then expands `${VAR}` /
`${VAR:-default}` (https://code.claude.com/docs/en/mcp — "Environment variable
expansion"; the two-pass order was read from the 2.1.229 binary). So by the
second pass the command is a single, un-nested `${ComSpec:-/path/to/slm-launch}`.
A plugin root containing `}` would break that, which no installer produces.

Tests: `tests/test_plugin/test_mcp_command_per_platform.py` (both expansions,
plus the POSIX entry run for real) and, on the Windows CI runner only,
`tests/test_plugin/test_windows_mcp_starts_installed_slm.py` (the expanded
entry spawned without a shell, through CreateProcess and through Node, against
a pip-style `slm.exe` and against a PATH with no `slm`).

The other hosts already name the installed `slm` directly — Codex
`.codex/config.toml`, VS Code/Copilot `.vscode/mcp.json` and Antigravity
`mcp_config.json` all use `slm mcp`; Hermes finds `slm` with `shutil.which`,
which honours `PATHEXT`. A shell-less spawn of a bare `slm` on Windows finds
`slm.exe` because CreateProcess appends `.exe` to a name with no extension
(Win32 `CreateProcessW`, lpCommandLine). It does not find an npm `slm.cmd`,
which is why the Windows install note says pipx or pip.

Hooks are different: Claude Code runs command hooks with bash (Git Bash on
Windows) or, without Git Bash, PowerShell. Under Git Bash the POSIX scripts —
`slm-run`, `ensure-venv.sh`, and the shared `slm-resolve.sh` — are what run.
