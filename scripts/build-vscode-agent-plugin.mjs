#!/usr/bin/env node
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// build-vscode-agent-plugin.mjs — generate vscode-agent-plugin/ and the
// repository's .github/plugin/marketplace.json from the built plugin/.
//
// USAGE
//   node scripts/build-vscode-agent-plugin.mjs           # write
//   node scripts/build-vscode-agent-plugin.mjs --check   # verify in sync (exit 2)
//
// WHY
//   VS Code's agent plugins (the Plugins screen of the Agents window, and
//   "Chat: Install Plugin From Source") accept Claude Code plugins and look for
//   marketplace.json, .plugin/, .github/plugin/ and only then .claude-plugin/
//   (VS Code 1.141, observed 14 Oct 2026). Our Claude Code plugin starts its MCP
//   server with "${ComSpec:-${CLAUDE_PLUGIN_ROOT}/...}", a shell default that
//   VS Code (which only replaces ${CLAUDE_PLUGIN_ROOT} and ${PLUGIN_ROOT})
//   cannot run. This build is that plugin with a plain `slm mcp` server and no
//   Claude-only hooks or launcher; .github/plugin/marketplace.json lists it, so
//   VS Code finds it first while Claude Code keeps reading .claude-plugin/.

import {
  readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, rmSync,
} from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { bareAgentTools } from './agent-tool-names.mjs';

const ROOT = join(dirname(new URL(import.meta.url).pathname), '..');
const SRC = join(ROOT, 'plugin');
const OUT = join(ROOT, 'vscode-agent-plugin');
const MARKETPLACE = join(ROOT, '.github', 'plugin', 'marketplace.json');
const KEEP = [/^skills\//, /^agents\//, /^commands\//, /^README\.md$/, /^LICENSE$/];

const check = process.argv.includes('--check');
const files = new Map(); // absolute path -> content

function walk(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const abs = join(dir, e.name);
    if (e.isDirectory()) walk(abs);
    else if (e.isFile()) {
      const rel = relative(SRC, abs).split('\\').join('/');
      if (KEEP.some((re) => re.test(rel))) {
        const text = readFileSync(abs, 'utf8');
        // plugin/ spells tools the way Claude Code needs; VS Code keeps the bare source names.
        files.set(join(OUT, rel), /^agents\//.test(rel) ? bareAgentTools(text) : text);
      }
    }
  }
}

function build() {
  const manifestPath = join(SRC, '.claude-plugin', 'plugin.json');
  if (!existsSync(manifestPath)) throw new Error('plugin/ is not built: run scripts/build-plugin.mjs first');
  walk(SRC);
  const src = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const manifest = {
    ...src,
    description: 'SuperLocalMemory for VS Code agents: local-first agent memory, '
      + 'SLM-Mesh, skills and sub-agents, served by the `slm mcp` you installed.',
    mcpServers: './.mcp.json',
  };
  files.set(join(OUT, '.claude-plugin', 'plugin.json'), JSON.stringify(manifest, null, 2) + '\n');
  files.set(join(OUT, '.mcp.json'), JSON.stringify({
    mcpServers: {
      superlocalmemory: { command: 'slm', args: ['mcp'], env: { SLM_AGENT_ID: 'vscode_agent' } },
    },
  }, null, 2) + '\n');
  files.set(join(OUT, '_GENERATED.md'), '# _GENERATED — DO NOT HAND-EDIT\n\nBuilt by '
    + '`scripts/build-vscode-agent-plugin.mjs` from `plugin/`. Edit `plugin-src/` and rebuild.\n');
  files.set(MARKETPLACE, JSON.stringify({
    name: 'qualixar',
    owner: { name: 'Qualixar', email: 'varun.pratap.bhardwaj@gmail.com' },
    plugins: [{
      name: 'superlocalmemory',
      source: './vscode-agent-plugin',
      description: manifest.description,
      version: src.version,
      license: src.license,
      homepage: 'https://superlocalmemory.com',
      keywords: src.keywords,
      author: { name: 'Qualixar' },
    }],
  }, null, 2) + '\n');
}

function existing() {
  const have = new Map();
  const rec = (dir) => {
    if (!existsSync(dir)) return;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const abs = join(dir, e.name);
      if (e.isDirectory()) rec(abs);
      else have.set(abs, readFileSync(abs, 'utf8'));
    }
  };
  rec(OUT);
  if (existsSync(MARKETPLACE)) have.set(MARKETPLACE, readFileSync(MARKETPLACE, 'utf8'));
  return have;
}

build();
const have = existing();
const stale = [...files].filter(([p, c]) => have.get(p) !== c).map(([p]) => relative(ROOT, p));
const extra = [...have.keys()].filter((p) => !files.has(p));

if (check) {
  if (stale.length || extra.length) {
    console.error(`vscode-agent-plugin out of sync: ${[...stale, ...extra.map((p) => relative(ROOT, p))].join(', ')}`);
    process.exit(2);
  }
  console.log('vscode-agent-plugin in sync.');
} else {
  for (const p of extra) rmSync(p);
  for (const [p, c] of files) {
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, c);
  }
  console.log(`vscode-agent-plugin built: ${files.size} files.`);
}
