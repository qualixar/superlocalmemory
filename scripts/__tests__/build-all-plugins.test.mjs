/**
 * build-all-plugins.test.mjs — one script builds every plugin, in order.
 *
 * build-copilot-plugin.mjs reads plugin/CLAUDE.md, which build-plugin.mjs
 * writes, so build-plugin must run first. A build in the wrong order ships
 * the previous release's text.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const SCRIPT = path.join(ROOT, 'scripts', 'build-all-plugins.mjs');
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));

const OUTPUT_DIRS = [
  'plugin', 'copilot-plugin', 'codex-plugin', 'antigravity-plugin',
  'hermes-plugin', '.cursor-plugin',
];

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.name === '__pycache__' || e.name === 'node_modules') continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

// A note saying when something began ("Since 4.1.20 ...") is history, not a
// claim that this build is that version.
const HISTORY_PHRASE = /\b(since|introduced in|added in|fixed in|changed in)\s+$/i;

const cmp = (a, b) => {
  const x = a.split('.').map(Number);
  const y = b.split('.').map(Number);
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
};

describe('build-all-plugins', () => {
  test('dry run lists build-plugin before copilot, and every step in order', () => {
    const r = spawnSync(process.execPath, [SCRIPT, '--dry-run'], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    const steps = r.stdout.split('\n').filter((l) => /^step \d+:/.test(l))
      .map((l) => l.replace(/^step \d+: /, '').trim());
    assert.deepEqual(steps, [
      'build-plugin.mjs', 'build-copilot-plugin.mjs', 'build-codex-plugin.mjs',
      'build-antigravity-plugin.mjs', 'build-hermes-plugin.mjs',
    ]);
  });

  test('package.json prepack and build:plugins call the one script', () => {
    assert.match(pkg.scripts['build:plugins'], /build-all-plugins\.mjs/);
    assert.match(pkg.scripts.prepack, /^node scripts\/build-all-plugins\.mjs/);
    assert.doesNotMatch(pkg.scripts.prepack, /build-copilot-plugin/);
  });

  test('after a build no generated file names an older release', () => {
    const r = spawnSync(process.execPath, [SCRIPT], { encoding: 'utf8', cwd: ROOT });
    assert.equal(r.status, 0, r.stdout + r.stderr);
    const stale = [];
    for (const dir of OUTPUT_DIRS) {
      const abs = path.join(ROOT, dir);
      if (!fs.existsSync(abs)) continue;
      for (const file of walk(abs)) {
        if (/CHANGELOG|HISTORY/i.test(path.basename(file))) continue;
        let text;
        try { text = fs.readFileSync(file, 'utf8'); } catch { continue; }
        if (text.includes('\u0000')) continue;
        for (const m of text.matchAll(/(?<![\w.\-/])v?(4\.\d+\.\d+)(?![\w.\-])/g)) {
          const before = text.slice(Math.max(0, m.index - 24), m.index);
          if (HISTORY_PHRASE.test(before)) continue;
          if (cmp(m[1], pkg.version) < 0) stale.push(`${path.relative(ROOT, file)}: ${m[0]}`);
        }
      }
    }
    assert.deepEqual(stale, []);
  });
});
