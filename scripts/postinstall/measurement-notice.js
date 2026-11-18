/**
 * The measurement notice at install and upgrade (4.2.30 D-28).
 *
 * Asks the package-owned runtime for the notice text (one source of truth:
 * superlocalmemory.outcomes.disclosure) and prints it to stderr when this
 * version of it has not been acknowledged. It does not prompt, does not
 * acknowledge (npm often hides install output, so only a person at the CLI or
 * the dashboard counts as having seen it) and never fails the install: on any
 * problem it prints one plain line instead.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const { spawnSync } = require('child_process');

const SNIPPET = [
  'from superlocalmemory.infra.data_root import canonical_data_root',
  'from superlocalmemory.outcomes import disclosure',
  'home = canonical_data_root()',
  'print(disclosure.text() if disclosure.pending(home) else "", end="")',
].join('\n');

const FALLBACK = 'SuperLocalMemory measures whether memory helps your agents (ids and numbers, '
  + 'this computer only). To turn the withholding and swapping off: slm measurement off';

function readNotice(python, spawn = spawnSync) {
  const result = spawn(python, ['-c', SNIPPET], {
    stdio: 'pipe', timeout: 20000, env: process.env, encoding: 'utf8',
  });
  if (result.error || result.status !== 0) return null;
  return String(result.stdout || '').trim();
}

/** Print the notice if pending. Returns true when something was printed. Never throws. */
function printMeasurementNotice(python, { log = (line) => process.stderr.write(line + '\n'), spawn = spawnSync } = {}) {
  try {
    const text = readNotice(python, spawn);
    if (text === '') return false;  // already acknowledged
    log('');
    log(text === null ? FALLBACK : text);
    return true;
  } catch (error) {
    return false;
  }
}

module.exports = { FALLBACK, SNIPPET, printMeasurementNotice, readNotice };
