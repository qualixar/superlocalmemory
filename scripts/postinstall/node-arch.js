/**
 * Whether this Node.js is an Intel program run by Rosetta on an Apple Silicon Mac.
 *
 * Both the Python check and the pictures request need to tell that apart from a
 * real Intel Mac: a universal2 Python started by a translated Node reports x86_64.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const os = require('os');
const { spawnSync } = require('child_process');

// Only macOS has the question; any failure to ask answers false.
function nodeRunsTranslated(platform = os.platform(), run = spawnSync) {
  if (platform !== 'darwin') return false;
  try {
    const result = run('sysctl', ['-n', 'sysctl.proc_translated'], { stdio: 'pipe', timeout: 3000, encoding: 'utf8' });
    return Boolean(result) && !result.error && String(result.stdout || '').trim() === '1';
  } catch (_) {
    return false;
  }
}

module.exports = { nodeRunsTranslated };
