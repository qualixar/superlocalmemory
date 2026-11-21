## When SLM's MCP tools are not in your tool list

Some hosts block local MCP servers by policy, and many agents, bots and agent frameworks never had them. SuperLocalMemory works the same from any shell, so this skill still applies:

- Run `slm status --json` first. Never report "SLM unavailable" only because an MCP tool is missing.
- Every MCP tool this skill names runs as `slm tool <name> --json '{...}'`, with the same arguments, checks and results: a successful call prints exactly what the MCP tool returns, and a failed one prints `"success": false` with `error.code` and `error.message`. `slm tool describe <name>` shows one tool's arguments; `slm tool list` lists the tools available right now (pictures and documents appear once they are turned on, `slm media status`).
- Every other `slm` command answers in one JSON document on stdout: add `--json` after it, put `slm --json` before it, or set `SLM_JSON=1` in the environment. `slm mcp`, `slm session` and `slm session-context` ignore `SLM_JSON` so servers and hooks keep working; give them `--json` directly. Read `success` first; a failure carries `error.code` and `error.message`. Exit code 0 means success, 1 that the command ran and failed (`error.exit_code` keeps a more specific code), 2 that it could not run, 130 that it was interrupted. Under JSON a command never asks a question: one that needs a confirmation answers `confirmation_required` and names the flag. `slm gdpr export` streams its document over several lines.
- Shortcuts for everyday work: `slm recall "<question>" --json` and `slm remember "<text>" --json`. `slm help --json` lists every command with its options.

The slm-cli skill has the full command reference.
