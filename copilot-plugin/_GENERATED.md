# copilot-plugin/ — GENERATED, do not edit by hand

Built by `scripts/build-copilot-plugin.mjs` from the single source in `plugin-src/` (+ `ide/configs/vscode-copilot-mcp.json`, `plugin/CLAUDE.md`). Version stamped from `plugin-src/manifest.json`.

Version: **3.8.0**

| Output | Source |
|---|---|
| `.github/copilot-instructions.md` | `plugin/CLAUDE.md` (SLM-marker-wrapped) |
| `.github/prompts/slm-*.prompt.md` | `plugin-src/skills/*/SKILL.md` |
| `.github/agents/*.agent.md` | `plugin-src/agents/*.md` (+ `target: vscode`) |
| `.github/hooks/slm-hooks.json` | Copilot schema (this script) |
| `.vscode/mcp.json` | `ide/configs/vscode-copilot-mcp.json` |
| `scripts/*` | `plugin-src/scripts/*` |

Regenerate: `npm run build:copilot-plugin` (or `node scripts/build-copilot-plugin.mjs`).

## Files

- `.github/agents/slm-governance-advisor.agent.md`
- `.github/agents/slm-loop-runner.agent.md`
- `.github/agents/slm-memory-advisor.agent.md`
- `.github/agents/slm-optimize-advisor.agent.md`
- `.github/copilot-instructions.md`
- `.github/hooks/slm-hooks.json`
- `.github/prompts/slm-cache.prompt.md`
- `.github/prompts/slm-compress.prompt.md`
- `.github/prompts/slm-governance.prompt.md`
- `.github/prompts/slm-graph.prompt.md`
- `.github/prompts/slm-loop.prompt.md`
- `.github/prompts/slm-mesh.prompt.md`
- `.github/prompts/slm-profile.prompt.md`
- `.github/prompts/slm-recall.prompt.md`
- `.github/prompts/slm-remember.prompt.md`
- `.github/prompts/slm-scope.prompt.md`
- `.github/prompts/slm-session.prompt.md`
- `.github/prompts/slm-status.prompt.md`
- `.vscode/mcp.json`
- `README.md`
- `scripts/ensure-venv.bat`
- `scripts/ensure-venv.sh`
- `scripts/slm-launch`
- `scripts/slm-launch.bat`
