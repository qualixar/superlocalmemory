# SuperLocalMemory — GitHub Copilot plugin (v4.1.25)

Empowers GitHub Copilot (VS Code, Visual Studio, JetBrains, Eclipse, CLI) with SuperLocalMemory as its long-term brain — at parity with the Claude and Codex plugins.

## Install (non-destructive)

From your project root:

```
slm connect vscode-copilot --here
```

This merges, without overwriting what is there:
- `.vscode/mcp.json` — the SLM MCP server (GA on every Copilot IDE; the reliable baseline).
- `.github/copilot-instructions.md` — SLM agent rules (merged inside `<!-- SLM-START -->`/`<!-- SLM-END -->`).

Then copy the add-ons from this folder's `.github/` into your project's `.github/` (the npm package carries this folder at `$(npm root -g)/superlocalmemory/copilot-plugin/`); keep any files of your own:
- `.github/prompts/*.prompt.md` — 15 slash-command skills: slm-bot-memory · slm-cache · slm-compress · slm-getting-started-bot · slm-governance · slm-graph · slm-loop · slm-mesh · slm-profile · slm-recall · slm-remember · slm-scope · slm-session · slm-status · slm-web-access.
- `.github/agents/*.agent.md` — memory, optimize and governance advisors and the loop runner.
- `.github/hooks/slm-hooks.json` — session lifecycle (stable on Copilot CLI + cloud agent; Preview in VS Code).

## Surface support

MCP works on every Copilot IDE at GA. Prompts, agents, and hooks are additive and degrade gracefully where an IDE does not yet support them (hooks are VS Code Preview as of 2026). SLM lifecycle also has an instruction-level fallback in `copilot-instructions.md`, so memory works even without hooks.

SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later
