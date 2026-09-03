# codex-plugin/ — partly GENERATED

Built by `scripts/build-codex-plugin.mjs` from the single source in `plugin-src/`. Version stamped from `plugin-src/manifest.json`.

Version: **4.1.12**

## Derived — do not edit by hand

| Output | Source |
|---|---|
| `skills/*/SKILL.md` | `plugin-src/skills/*/SKILL.md`, with `SLM_AGENT_ID` retargeted to `codex` |
| version footers | `plugin-src/manifest.json` |

- `.claude-plugin/plugin.json`
- `.codex-plugin/plugin.json`
- `.mcp.json`
- `AGENTS.md`
- `README.md`
- `agents/slm-governance-advisor.md`
- `agents/slm-loop-runner.md`
- `agents/slm-memory-advisor.md`
- `agents/slm-optimize-advisor.md`
- `commands/slm-loop.md`
- `skills/slm-cache/SKILL.md`
- `skills/slm-compress/SKILL.md`
- `skills/slm-governance/SKILL.md`
- `skills/slm-graph/SKILL.md`
- `skills/slm-loop/SKILL.md`
- `skills/slm-mesh/SKILL.md`
- `skills/slm-profile/SKILL.md`
- `skills/slm-recall/SKILL.md`
- `skills/slm-remember/SKILL.md`
- `skills/slm-scope/SKILL.md`
- `skills/slm-session/SKILL.md`
- `skills/slm-status/SKILL.md`

## Authored here — no source to derive from

These are Codex-shaped and are maintained in this directory:

- `AGENTS.md` body — Codex has its own rules document; it is not a copy of the Claude one.
- `README.md` body
- `scripts/*` — the launcher and venv bootstrap resolve paths from
  `SLM_DATA_DIR`; the Claude versions require `CLAUDE_PLUGIN_ROOT`
  and `CLAUDE_PLUGIN_DATA`, which Codex does not set.
- `.codex/config.toml` — Codex MCP configuration
- `hooks/hooks.json` — Codex hook schema

Regenerate: `npm run build:codex-plugin` (or `node scripts/build-codex-plugin.mjs`).
