#!/usr/bin/env node
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// build-claude-code-assets.mjs — generate claude-code-assets/agents/ from plugin-src/agents/.
//
// USAGE
//   node scripts/build-claude-code-assets.mjs           # write
//   node scripts/build-claude-code-assets.mjs --check   # verify in sync (exit 2)
//
// WHY
//   `slm upgrade-hosts` and the daemon's once-per-version step install SLM's sub-agents into
//   ~/.claude/agents for a Claude Code that has no plugin. That Claude Code reaches SLM as a plain
//   MCP server, so a sub-agent must name each tool `mcp__superlocalmemory__<tool>`; the bare
//   names in plugin-src/agents (read directly by Codex, Antigravity, Hermes and Copilot) match
//   nothing there and the agent would receive no memory tools. This is the same rewrite the Muse
//   build applies (directAgentTools in agent-tool-names.mjs), done once at build time so the
//   Python installer and the wheel ship the finished files and no second transform exists.
//   Edit plugin-src/agents and rebuild; never hand-edit claude-code-assets/.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { directAgentTools } from './agent-tool-names.mjs';

const ROOT = join(dirname(new URL(import.meta.url).pathname), '..');
const SRC = join(ROOT, 'plugin-src', 'agents');
const OUT = join(ROOT, 'claude-code-assets');
const check = process.argv.includes('--check');

const mdFiles = (dir) => (existsSync(dir)
  ? readdirSync(dir).filter((n) => n.endsWith('.md')).sort() : []);

const files = new Map(); // path relative to OUT -> content
for (const name of mdFiles(SRC)) {
  files.set(`agents/${name}`, directAgentTools(readFileSync(join(SRC, name), 'utf8')));
}
if (files.size === 0) throw new Error('plugin-src/agents has no agents to build from');
files.set('_GENERATED.md', '# _GENERATED — DO NOT HAND-EDIT\n\nBuilt by '
  + '`scripts/build-claude-code-assets.mjs` from `plugin-src/agents/`: each SLM tool in an agent\'s '
  + '`tools:` line is spelled `mcp__superlocalmemory__<tool>`, the name a Claude Code without the '
  + 'plugin resolves. Edit `plugin-src/` and rebuild.\n');

const have = new Map();
for (const name of mdFiles(join(OUT, 'agents'))) have.set(`agents/${name}`, readFileSync(join(OUT, 'agents', name), 'utf8'));
if (existsSync(join(OUT, '_GENERATED.md'))) have.set('_GENERATED.md', readFileSync(join(OUT, '_GENERATED.md'), 'utf8'));

const stale = [...files].filter(([p, c]) => have.get(p) !== c).map(([p]) => p);
const extra = [...have.keys()].filter((p) => !files.has(p));

if (check) {
  if (stale.length || extra.length) {
    console.error(`claude-code-assets out of sync: ${[...stale, ...extra].join(', ')}`);
    process.exit(2);
  }
  console.log('claude-code-assets in sync.');
} else {
  for (const p of extra) rmSync(join(OUT, p));
  for (const [p, c] of files) {
    mkdirSync(dirname(join(OUT, p)), { recursive: true });
    writeFileSync(join(OUT, p), c);
  }
  console.log(`claude-code-assets built: ${files.size} files.`);
}
