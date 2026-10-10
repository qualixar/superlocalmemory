#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * Every repair command the installer prints for a global install is
 * `npm rebuild -g superlocalmemory`. Without -g npm rebuilds the project in the
 * current folder, so the printed command would do nothing for this package.
 *
 * Run with: node --test tests/postinstall/test_rebuild_command_is_global.js
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const POSTINSTALL = path.join(ROOT, 'scripts', 'postinstall.js');
const WRAPPER = path.join(ROOT, 'bin', 'slm-npm');

function rebuildLines(file) {
  return fs.readFileSync(file, 'utf8').split('\n')
    .map((text, i) => ({ text, line: i + 1 }))
    .filter(({ text }) => /npm rebuild/.test(text));
}

test('postinstall.js never prints npm rebuild without -g', () => {
  const found = rebuildLines(POSTINSTALL);
  assert.ok(found.length >= 5, 'expected the installer to print the repair command in several places');
  for (const { text, line } of found) {
    assert.match(text, /npm rebuild -g superlocalmemory/, `scripts/postinstall.js:${line}: ${text.trim()}`);
  }
});

test('bin/slm-npm never prints npm rebuild without -g', () => {
  const found = rebuildLines(WRAPPER);
  assert.ok(found.length >= 1);
  for (const { text, line } of found) {
    assert.match(text, /npm rebuild -g superlocalmemory/, `bin/slm-npm:${line}: ${text.trim()}`);
  }
});

test('a failed pip install prints the repair command once', () => {
  const lines = fs.readFileSync(POSTINSTALL, 'utf8').split('\n');
  const at = lines.findIndex((l) => l.includes('private-runtime installation failed'));
  assert.ok(at > 0);
  const block = lines.slice(at, at + 12);
  const printed = block.filter((l) => /npm rebuild/.test(l));
  assert.equal(printed.length, 1, block.join('\n'));
});

test('the wrapper, run with no runtime, tells a global install how to repair it', () => {
  // A copy of the wrapper in an empty package folder: there is no .slm-venv beside it.
  const dir = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'slm-wrap-'));
  fs.mkdirSync(path.join(dir, 'bin'));
  fs.copyFileSync(WRAPPER, path.join(dir, 'bin', 'slm-npm'));
  const result = spawnSync('node', [path.join(dir, 'bin', 'slm-npm')], { encoding: 'utf8', timeout: 15000 });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /npm rebuild -g superlocalmemory/);
  assert.doesNotMatch(result.stderr, /npm rebuild superlocalmemory/);
});
