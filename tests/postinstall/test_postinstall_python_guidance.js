#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * When no supported Python is found, the installer says how to get one on
 * this computer and the exact command that finishes a global install
 * (`npm rebuild -g superlocalmemory`; without -g npm looks in the current folder).
 *
 * Run with: node --test tests/postinstall/test_postinstall_python_guidance.js
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { pythonGuidance } = require('../../scripts/postinstall.js');

test('macOS points at Homebrew and the python.org installer', () => {
  const text = pythonGuidance('darwin').join('\n');
  assert.match(text, /brew install python@3\.13/);
  assert.match(text, /python\.org/);
  assert.match(text, /npm rebuild -g superlocalmemory/);
});

test('Linux names the distro packages, including deadsnakes for older Ubuntu', () => {
  const text = pythonGuidance('linux').join('\n');
  assert.match(text, /deadsnakes/);
  assert.match(text, /python3\.12-venv/);
  assert.match(text, /npm rebuild -g superlocalmemory/);
});

test('Windows points at winget and the py launcher', () => {
  const text = pythonGuidance('win32').join('\n');
  assert.match(text, /winget install Python\.Python\.3\.13/);
  assert.match(text, /npm rebuild -g superlocalmemory/);
});

test('every platform says the system Python is never changed', () => {
  for (const platform of ['darwin', 'linux', 'win32', 'freebsd']) {
    assert.match(pythonGuidance(platform).join('\n'), /private virtual environment/);
  }
});
