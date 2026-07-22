// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// build-copilot-plugin.mjs — generate copilot-plugin/ from the SINGLE source
// in plugin-src/ (skills, agents, scripts) + ide/configs/vscode-copilot-mcp.json
// + plugin/CLAUDE.md (agent rules), at parity with plugin/ (Claude) and
// codex-plugin/ (Codex). Self-contained (no import side effects).
//
//   node scripts/build-copilot-plugin.mjs           # write copilot-plugin/
//   node scripts/build-copilot-plugin.mjs --check    # verify in sync (exit 2 on drift)
//
// Copilot surfaces (2026): .github/copilot-instructions.md (always-on, merged
// via SLM markers), .github/prompts/*.prompt.md (skills), .github/agents/*.agent.md
// (advisors), .github/hooks/slm-hooks.json (CLI+cloud stable; VS Code preview),
// .vscode/mcp.json (GA everywhere). MCP is the reliable baseline; the rest are
// additive and degrade gracefully outside VS Code.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHECK = process.argv.includes('--check');

const SRC_SKILLS = join(ROOT, 'plugin-src', 'skills');
const SRC_AGENTS = join(ROOT, 'plugin-src', 'agents');
const SRC_SCRIPTS = join(ROOT, 'plugin-src', 'scripts');
const MCP_SRC = join(ROOT, 'ide', 'configs', 'vscode-copilot-mcp.json');
const RULES_SRC = join(ROOT, 'plugin', 'CLAUDE.md');
const MANIFEST = join(ROOT, 'plugin-src', 'manifest.json');
const OUT = join(ROOT, 'copilot-plugin');

const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
const VERSION = manifest.version;
if (!VERSION) throw new Error('manifest.version missing');

const nl = (s) => s.replace(/\r\n/g, '\n');

function splitFrontmatter(src) {
  const m = nl(src).match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!m) throw new Error('no YAML frontmatter');
  return { fm: m[1], body: m[2] };
}
function fmField(fm, key) {
  const m = fm.match(new RegExp('^' + key + ':\\s*(.+)$', 'm'));
  return m ? m[1].trim() : '';
}

// --- planned output files: path (relative to OUT) -> content string ----------
const files = new Map();

// 1. .github/copilot-instructions.md — agent rules, marker-bounded for merge.
{
  const rules = nl(readFileSync(RULES_SRC, 'utf8')).trim();
  const content =
    '<!-- SLM-START -->\n' +
    '<!-- SuperLocalMemory v' + VERSION + ' — managed block. Edit outside these markers; this section is regenerated. -->\n\n' +
    rules + '\n\n' +
    '<!-- SLM-END -->\n';
  files.set('.github/copilot-instructions.md', content);
}

// 2. .github/prompts/slm-<skill>.prompt.md — one per skill (skills equivalent).
for (const { name } of manifest.skills) {
  const skillPath = join(SRC_SKILLS, name, 'SKILL.md');
  const { fm, body } = splitFrontmatter(readFileSync(skillPath, 'utf8'));
  const description = fmField(fm, 'description');
  const allowed = fmField(fm, 'allowed-tools');
  const tools = allowed
    ? allowed.split(',').map((t) => t.trim()).filter(Boolean)
    : [];
  const toolsBlock = tools.length
    ? 'tools:\n' + tools.map((t) => '  - ' + t).join('\n') + '\n'
    : '';
  const pf =
    '---\n' +
    'name: ' + name + '\n' +
    'description: ' + description + '\n' +
    'version: "' + VERSION + '"\n' +
    'agent: agent\n' +
    toolsBlock +
    '---\n\n' +
    nl(body).trim() + '\n';
  files.set('.github/prompts/' + name + '.prompt.md', pf);
}

// 3. .github/agents/slm-<advisor>.agent.md — inject target + version, keep body.
for (const fname of readdirSync(SRC_AGENTS).filter((f) => f.endsWith('.md')).sort()) {
  const { fm, body } = splitFrontmatter(readFileSync(join(SRC_AGENTS, fname), 'utf8'));
  const name = fmField(fm, 'name') || fname.replace(/\.md$/, '');
  const newFm = fm.trimEnd() + '\ntarget: vscode\nversion: "' + VERSION + '"';
  const af = '---\n' + newFm + '\n---\n' + (body.startsWith('\n') ? body : '\n' + body);
  files.set('.github/agents/' + name + '.agent.md', nl(af).replace(/\n*$/, '\n'));
}

// 4. .github/hooks/slm-hooks.json — Copilot schema (bash/timeoutSec). Stable on
//    CLI + cloud agent; Preview in VS Code. Fail-open (|| true). Mirrors codex hooks.
{
  const hooks = {
    version: 1,
    hooks: {
      sessionStart: [
        { type: 'command', bash: 'slm hook mandate 2>/dev/null || true', timeoutSec: 5 },
        { type: 'command', bash: 'slm hook start 2>/dev/null || true', timeoutSec: 15 },
      ],
      sessionEnd: [
        { type: 'command', bash: 'slm hook stop 2>/dev/null || true', timeoutSec: 10 },
        { type: 'command', bash: 'slm hook stop_outcome 2>/dev/null || true', timeoutSec: 10 },
      ],
      userPromptSubmitted: [
        { type: 'command', bash: 'slm hook topic_shift 2>/dev/null || true', timeoutSec: 3 },
      ],
      postToolUse: [
        // matcher scopes the checkpoint to file-write tools on Copilot CLI (GA,
        // honours native-name matchers). VS Code (Preview) ignores matchers and
        // fires on every tool; harmless — the hook is cheap and fail-open.
        { matcher: 'create|edit|str_replace_editor|apply_patch', type: 'command', bash: 'slm hook checkpoint 2>/dev/null || true', timeoutSec: 5 },
      ],
    },
  };
  files.set('.github/hooks/slm-hooks.json', JSON.stringify(hooks, null, 2) + '\n');
}

// 5. .vscode/mcp.json — GA-everywhere baseline, verbatim from the shared config.
{
  const mcp = JSON.parse(readFileSync(MCP_SRC, 'utf8'));
  files.set('.vscode/mcp.json', JSON.stringify(mcp, null, 2) + '\n');
}

// 6. scripts/ — venv bootstrap + launcher (PATH-robust), from the single source.
for (const fname of readdirSync(SRC_SCRIPTS).sort()) {
  files.set('scripts/' + fname, readFileSync(join(SRC_SCRIPTS, fname), 'utf8'));
}

// 7. README.md + _GENERATED.md
{
  const skillNames = manifest.skills.map((s) => s.name).join(' · ');
  files.set('README.md',
    '# SuperLocalMemory — GitHub Copilot plugin (v' + VERSION + ')\n\n' +
    'Empowers GitHub Copilot (VS Code, Visual Studio, JetBrains, Eclipse, CLI) with SuperLocalMemory ' +
    'as its long-term brain — at parity with the Claude and Codex plugins.\n\n' +
    '## Install (non-destructive)\n\n' +
    '```\nslm connect copilot\n```\n\n' +
    'Merges into your existing project without overwriting:\n' +
    '- `.vscode/mcp.json` — the SLM MCP server (GA on every Copilot IDE; the reliable baseline).\n' +
    '- `.github/copilot-instructions.md` — SLM agent rules (merged inside `<!-- SLM-START -->`/`<!-- SLM-END -->`).\n' +
    '- `.github/prompts/*.prompt.md` — ' + manifest.skills.length + ' slash-command skills: ' + skillNames + '.\n' +
    '- `.github/agents/*.agent.md` — memory / optimize / governance advisors.\n' +
    '- `.github/hooks/slm-hooks.json` — session lifecycle (stable on Copilot CLI + cloud agent; Preview in VS Code).\n\n' +
    '## Surface support\n\n' +
    'MCP works on every Copilot IDE at GA. Prompts, agents, and hooks are additive and degrade gracefully ' +
    'where an IDE does not yet support them (hooks are VS Code Preview as of 2026). SLM lifecycle also has an ' +
    'instruction-level fallback in `copilot-instructions.md`, so memory works even without hooks.\n\n' +
    'SuperLocalMemory v' + VERSION + ' · Qualixar · AGPL-3.0-or-later\n');

  const manifestList = [...files.keys()].sort().map((p) => '- `' + p + '`').join('\n');
  files.set('_GENERATED.md',
    '# copilot-plugin/ — GENERATED, do not edit by hand\n\n' +
    'Built by `scripts/build-copilot-plugin.mjs` from the single source in `plugin-src/` ' +
    '(+ `ide/configs/vscode-copilot-mcp.json`, `plugin/CLAUDE.md`). Version stamped from `plugin-src/manifest.json`.\n\n' +
    'Version: **' + VERSION + '**\n\n' +
    '| Output | Source |\n|---|---|\n' +
    '| `.github/copilot-instructions.md` | `plugin/CLAUDE.md` (SLM-marker-wrapped) |\n' +
    '| `.github/prompts/slm-*.prompt.md` | `plugin-src/skills/*/SKILL.md` |\n' +
    '| `.github/agents/*.agent.md` | `plugin-src/agents/*.md` (+ `target: vscode`) |\n' +
    '| `.github/hooks/slm-hooks.json` | Copilot schema (this script) |\n' +
    '| `.vscode/mcp.json` | `ide/configs/vscode-copilot-mcp.json` |\n' +
    '| `scripts/*` | `plugin-src/scripts/*` |\n\n' +
    'Regenerate: `npm run build:copilot-plugin` (or `node scripts/build-copilot-plugin.mjs`).\n\n' +
    '## Files\n\n' + manifestList + '\n');
}

// --- write or check -----------------------------------------------------------
let drift = 0;
for (const [rel, content] of files) {
  const abs = join(OUT, rel);
  if (CHECK) {
    if (!existsSync(abs) || readFileSync(abs, 'utf8') !== content) {
      console.error('DRIFT: ' + rel);
      drift++;
    }
  } else {
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, content);
  }
}

if (CHECK) {
  if (drift) {
    console.error('copilot-plugin out of sync (' + drift + ' file(s)). Run: node scripts/build-copilot-plugin.mjs');
    process.exit(2);
  }
  console.log('copilot-plugin in sync (' + files.size + ' files, v' + VERSION + ').');
} else {
  console.log('copilot-plugin built: ' + files.size + ' files, v' + VERSION + '.');
}
