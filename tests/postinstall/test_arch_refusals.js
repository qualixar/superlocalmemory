#!/usr/bin/env node
/**
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later - see LICENSE file
 *
 * Computers SuperLocalMemory cannot run on are refused in plain words before pip
 * fails deep inside dependency resolution, and the "turn on pictures" request is
 * not recorded where the daemon would refuse it.
 *
 * Run with: node --test tests/postinstall/test_arch_refusals.js
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { unsupportedMachine, nodeRunsTranslated } = require('../../scripts/postinstall.js');
const media = require('../../scripts/postinstall/media-request.js');

test('Windows on ARM is refused', () => {
  const why = unsupportedMachine('win32', 'ARM64', true);
  assert.match(why, /Windows on ARM/);
  assert.doesNotMatch(why, /32-bit/);
});

test('32-bit Linux is refused, whether the machine or the Python is 32-bit', () => {
  for (const [machine, is64] of [['i686', false], ['i386', null], ['armv7l', false], ['x86_64', false]]) {
    const why = unsupportedMachine('linux', machine, is64);
    assert.match(why || '', /32-bit/, `${machine} ${is64}`);
  }
});

test('64-bit Linux and Windows on x64 still pass', () => {
  assert.equal(unsupportedMachine('linux', 'x86_64', true), null);
  assert.equal(unsupportedMachine('linux', 'aarch64', true), null);
  assert.equal(unsupportedMachine('win32', 'AMD64', true), null);
});

test('an Intel Python on a Mac still gets the Intel message when Node runs natively', () => {
  const why = unsupportedMachine('darwin', 'x86_64', true, false);
  assert.match(why, /Intel \(x86_64\)/);
});

test('a Python that reports Intel only because Node runs translated is not called Intel', () => {
  const why = unsupportedMachine('darwin', 'x86_64', true, true);
  assert.doesNotMatch(why, /This Python is an Intel/);
  assert.match(why, /Node/);
  assert.match(why, /natively/);
  assert.match(why, /npm rebuild -g superlocalmemory/);
});

test('nodeRunsTranslated reads sysctl.proc_translated on a Mac and nowhere else', () => {
  const asked = [];
  const run = (cmd, args) => { asked.push([cmd, ...args]); return { status: 0, stdout: '1\n' }; };
  assert.equal(nodeRunsTranslated('darwin', run), true);
  assert.deepEqual(asked, [['sysctl', '-n', 'sysctl.proc_translated']]);
  assert.equal(nodeRunsTranslated('darwin', () => ({ status: 0, stdout: '0\n' })), false);
  assert.equal(nodeRunsTranslated('darwin', () => ({ status: 1, stdout: '', error: new Error('no such key') })), false);
  assert.equal(nodeRunsTranslated('linux', run), false);
  assert.equal(asked.length, 1, 'no question asked off a Mac');
});

// -- the pictures request ----------------------------------------------------

const ROOMY = { SLM_MEDIA_ALLOW_LOW_RAM: '1', SLM_ENABLE_MEDIA: '1' };

async function choose(machine) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-arch-'));
  const lines = [];
  const recorded = await media.handleMediaChoice({
    args: { media: true }, env: ROOMY, slmDir: dir, interactive: false,
    log: (line) => lines.push(line), totalMem: 32 * 1024 ** 3, machine,
  });
  return { recorded, lines: lines.join('\n'), written: fs.existsSync(path.join(dir, 'features.json')) };
}

test('Linux ARM64 records nothing and says pictures are not supported yet', async () => {
  const out = await choose({ platform: 'linux', arch: 'arm64', translated: false });
  assert.equal(out.recorded, false);
  assert.equal(out.written, false);
  assert.match(out.lines, /not supported on this computer yet/);
  assert.doesNotMatch(out.lines, /will start setting up/);
});

test('an Intel Mac records nothing and says pictures are not supported yet', async () => {
  const out = await choose({ platform: 'darwin', arch: 'x64', translated: false });
  assert.equal(out.recorded, false);
  assert.equal(out.written, false);
  assert.match(out.lines, /not supported on this computer yet/);
});

test('Windows on ARM records nothing', async () => {
  const out = await choose({ platform: 'win32', arch: 'arm64', translated: false });
  assert.equal(out.recorded, false);
  assert.match(out.lines, /not supported on this computer yet/);
});

test('Node running translated on an Apple Silicon Mac is told to run Node natively, not that the Mac is Intel', async () => {
  const out = await choose({ platform: 'darwin', arch: 'x64', translated: true });
  assert.equal(out.recorded, false);
  assert.equal(out.written, false);
  assert.match(out.lines, /natively/);
  assert.doesNotMatch(out.lines, /Intel Mac/);
});

test('supported computers still record the request', async () => {
  for (const machine of [
    { platform: 'darwin', arch: 'arm64', translated: false },
    { platform: 'linux', arch: 'x64', translated: false },
    { platform: 'win32', arch: 'x64', translated: false },
  ]) {
    const out = await choose(machine);
    assert.equal(out.recorded, true, JSON.stringify(machine));
    assert.equal(out.written, true);
  }
});

test('mediaPlatformRefusal is empty for supported computers', () => {
  assert.equal(media.mediaPlatformRefusal({ platform: 'darwin', arch: 'arm64', translated: false }), '');
  assert.notEqual(media.mediaPlatformRefusal({ platform: 'linux', arch: 'arm64', translated: false }), '');
});
