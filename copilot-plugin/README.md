# SuperLocalMemory — GitHub Copilot plugin (v3.8.2)

Empowers GitHub Copilot (VS Code, Visual Studio, JetBrains, Eclipse, CLI) with SuperLocalMemory as its long-term brain — at parity with the Claude and Codex plugins.

## Install (non-destructive)

```
slm connect copilot
```

Merges into your existing project without overwriting:
- `.vscode/mcp.json` — the SLM MCP server (GA on every Copilot IDE; the reliable baseline).
- `.github/copilot-instructions.md` — SLM agent rules (merged inside `<!-- SLM-START -->`/`<!-- SLM-END -->`).
- `.github/prompts/*.prompt.md` — 12 slash-command skills: slm-cache · slm-compress · slm-governance · slm-graph · slm-loop · slm-mesh · slm-profile · slm-recall · slm-remember · slm-scope · slm-session · slm-status.
- `.github/agents/*.agent.md` — memory / optimize / governance advisors.
- `.github/hooks/slm-hooks.json` — session lifecycle (stable on Copilot CLI + cloud agent; Preview in VS Code).

## Surface support

MCP works on every Copilot IDE at GA. Prompts, agents, and hooks are additive and degrade gracefully where an IDE does not yet support them (hooks are VS Code Preview as of 2026). SLM lifecycle also has an instruction-level fallback in `copilot-instructions.md`, so memory works even without hooks.

SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later
