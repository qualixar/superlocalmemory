#!/usr/bin/env node
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// build-muse-plugin.mjs — generate muse-plugin/ from the built plugin/.
//
// USAGE
//   node scripts/build-muse-plugin.mjs           # write
//   node scripts/build-muse-plugin.mjs --check   # verify in sync (exit 2)
//
// WHY
//   Muse CLI (1.4.7) imports Claude Code plugins, but refused the Claude plugin
//   as built (14 Oct 2026):
//   - 8 of 16 skills: `allowed-tools` lists short MCP names (recall, remember,
//     search, fetch) that Muse rejects as malformed selectors. In Claude Code
//     that line only pre-approves tools, so dropping it costs nothing but a
//     prompt; renaming the tools in the shared skills would instead pre-approve
//     delete_memory and others in Claude Code.
//   - .mcp.json: Muse rejects the `${ComSpec:-...}` launcher placeholder. Muse
//     gets the MCP server from `slm connect muse` (settings.json) instead.
//   - settings.json: Claude Code permission rules, which Muse does not read.
//   - SessionStart: `slm hook start` prints plain text that opens with
//     "[BEGIN UNTRUSTED SLM EVIDENCE v1]". Muse reads stdout starting with
//     "[" as JSON, fails to parse it and drops the context. `slm hook
//     codex-start` emits the same context as hookSpecificOutput JSON, which
//     Muse (like Codex and Claude Code) accepts.
//   - agents: bare SLM tool names in `tools:` become mcp__superlocalmemory__<tool>.
//   Built from plugin/ (run build-plugin.mjs first) so the hooks, scripts,
//   agents and skill text cannot drift from the Claude Code plugin.

import {
  readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, rmSync, statSync, chmodSync,
} from 'node:fs';
import { join, dirname, relative } from 'node:path';

const ROOT = join(dirname(new URL(import.meta.url).pathname), '..');
const SRC = join(ROOT, 'plugin');
const OUT = join(ROOT, 'muse-plugin');
const AGENT_ID = 'muse';
// Other hosts' manifests stay out: Muse reads the Claude layout, and a root
// mcp.json here would register SLM a second time.
const DROP = new Set([
  '.mcp.json', 'settings.json', '_GENERATED.md', '.claude-plugin/plugin.json',
  'plugin.json', 'mcp.json', 'qwen-extension.json', 'kimi.plugin.json',
]);

const check = process.argv.includes('--check');
const files = new Map(); // relative path -> { content, mode }

function walk(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const abs = join(dir, e.name);
    if (e.isDirectory()) walk(abs);
    else if (e.isFile()) {
      const rel = relative(SRC, abs).split('\\').join('/');
      if (DROP.has(rel)) continue;
      const binary = /\.(png|jpe?g|gif|webp|ico)$/i.test(rel);
      files.set(rel, { content: readFileSync(abs, binary ? undefined : 'utf8'), mode: statSync(abs).mode });
    }
  }
}

/** Remove the `allowed-tools:` line from a skill's front matter only. */
export function dropAllowedTools(md) {
  const fm = md.match(/^---\n[\s\S]*?\n---\n/);
  if (!fm) return md;
  return fm[0].replace(/^allowed-tools:.*\n/m, '') + md.slice(fm[0].length);
}

/**
 * Muse names MCP tools mcp__<server>__<tool> (dev.meta.ai/docs/muse-code); an
 * agent's bare `recall` would match nothing. Capitalised names are host
 * built-ins (Read, Bash) and stay as they are.
 */
export function qualifyAgentTools(md) {
  const fm = md.match(/^---\n[\s\S]*?\n---\n/);
  if (!fm) return md;
  const head = fm[0].replace(/^tools:(.*)$/m, (_, list) => 'tools: ' + list.split(',')
    .map((t) => t.trim()).filter(Boolean)
    .map((t) => (/^[a-z][a-z0-9_]*$/.test(t) ? `mcp__superlocalmemory__${t}` : t))
    .join(', '));
  return head + md.slice(fm[0].length);
}

/** Use the SessionStart action that answers in hookSpecificOutput JSON. */
export function jsonSessionStart(hooksJson) {
  const out = hooksJson.replace(/(\/scripts\/slm-run\\" hook) start /g, '$1 codex-start ');
  if (out === hooksJson) throw new Error('hooks.json: no `slm-run hook start` to replace');
  return out;
}

const retarget = (s) => s.replace(/SLM_AGENT_ID(["']?\s*[:=]\s*["'])claude_code(["'])/g, `SLM_AGENT_ID$1${AGENT_ID}$2`);

function build() {
  if (!existsSync(join(SRC, '.claude-plugin', 'plugin.json'))) {
    throw new Error('plugin/ is not built: run scripts/build-plugin.mjs first');
  }
  walk(SRC);
  for (const [rel, f] of files) {
    if (Buffer.isBuffer(f.content)) continue;
    let content = retarget(f.content);
    if (/^skills\/[^/]+\/SKILL\.md$/.test(rel)) content = dropAllowedTools(content);
    if (rel === 'hooks/hooks.json') content = jsonSessionStart(content);
    if (/^agents\/[^/]+\.md$/.test(rel)) content = qualifyAgentTools(content);
    files.set(rel, { ...f, content });
  }
  const src = JSON.parse(readFileSync(join(SRC, '.claude-plugin', 'plugin.json'), 'utf8'));
  const { mcpServers: _omit, ...rest } = src;
  files.set('.claude-plugin/plugin.json', {
    content: JSON.stringify({
      ...rest,
      description: 'SuperLocalMemory for Muse CLI: local-first agent memory skills, agents and '
        + 'session hooks. Run `slm connect muse` first; it adds the MCP server to Muse.',
    }, null, 2) + '\n',
    mode: 0o644,
  });
  files.set('_GENERATED.md', {
    content: '# _GENERATED — DO NOT HAND-EDIT\n\nBuilt by `scripts/build-muse-plugin.mjs` from '
      + '`plugin/`. Edit `plugin-src/` and rebuild.\n',
    mode: 0o644,
  });
}

function existing() {
  const have = new Map();
  if (!existsSync(OUT)) return have;
  const rec = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const abs = join(dir, e.name);
      if (e.isDirectory()) rec(abs);
      else {
        const rel = relative(OUT, abs).split('\\').join('/');
        have.set(rel, readFileSync(abs, /\.(png|jpe?g|gif|webp|ico)$/i.test(rel) ? undefined : 'utf8'));
      }
    }
  };
  rec(OUT);
  return have;
}

build();
const have = existing();
const same = (a, b) => (Buffer.isBuffer(b) ? Buffer.isBuffer(a) && a.equals(b) : a === b);
const stale = [...files].filter(([p, f]) => !same(have.get(p), f.content)).map(([p]) => p);
const extra = [...have.keys()].filter((p) => !files.has(p));

if (check) {
  if (stale.length || extra.length) {
    console.error(`muse-plugin out of sync: ${[...stale, ...extra].join(', ')}`);
    process.exit(2);
  }
  console.log('muse-plugin in sync.');
} else {
  for (const p of extra) rmSync(join(OUT, p));
  for (const [p, f] of files) {
    const abs = join(OUT, p);
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, f.content);
    if (f.mode & 0o111) chmodSync(abs, 0o755);
  }
  console.log(`muse-plugin built: ${files.size} files.`);
}
