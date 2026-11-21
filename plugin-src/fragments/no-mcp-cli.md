## When SLM's MCP tools are not in your tool list

Some hosts block local MCP servers by policy, and many agents, bots and agent frameworks never had them. SuperLocalMemory works the same from any shell, so this skill still applies:

- Run `slm status --json` first. Never report "SLM unavailable" only because an MCP tool is missing.
- Every MCP tool this skill names runs as `slm tool <name> --json '{...}'`, with the same arguments, checks and results. `slm tool list` lists all of them, and `slm tool describe <name>` shows one tool's arguments.
- Every `slm` command answers in one line of JSON: add `--json`, put `slm --json` before the command, or set `SLM_JSON=1` in the environment. The reply is `{"success": true, "data": {...}}` or `{"success": false, "error": {"code": ..., "message": ...}}`; exit code 0 means success, 1 that the command ran and failed, 2 that it could not run. Under JSON a command never asks a question: one that needs a confirmation answers `confirmation_required` and names the flag.
- Shortcuts for everyday work: `slm recall "<question>" --json` and `slm remember "<text>" --json`. `slm help --json` lists every command with its options.

The slm-cli skill has the full command reference.
