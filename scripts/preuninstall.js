#!/usr/bin/env node
/**
 * SuperLocalMemory NPM pre-uninstall notice.
 *
 * npm owns and removes the package directory, including its private Python
 * runtime. Durable SLM data has separate ownership and is never modified or
 * deleted by an application-code uninstall. The one thing worth saying: the
 * images-and-documents models (about 1.5 GB) live in the data folder, so this
 * names where they are, how big they are, and the command that removes them.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

function dataRoot(env = process.env) {
  return env.SLM_DATA_DIR
    || env.SL_MEMORY_PATH
    || env.SLM_HOME
    || path.join(os.homedir(), '.superlocalmemory');
}

// Total bytes under dir; read-only, bounded, never follows links, never throws.
function folderBytes(dir, limit = 200000) {
  let total = 0;
  let seen = 0;
  const stack = [dir];
  while (stack.length && seen < limit) {
    const current = stack.pop();
    let entries = [];
    try { entries = fs.readdirSync(current, { withFileTypes: true }); } catch (_e) { continue; }
    for (const entry of entries) {
      seen += 1;
      const full = path.join(current, entry.name);
      try {
        if (entry.isSymbolicLink()) continue;
        if (entry.isDirectory()) stack.push(full);
        else total += fs.lstatSync(full).size;
      } catch (_e) { /* skip unreadable entries */ }
    }
  }
  return total;
}

function humanSize(bytes) {
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1) + ' GB';
  return Math.max(1, Math.round(bytes / 1024 ** 2)) + ' MB';
}

function mediaEnvNotice(env = process.env, log = console.log) {
  try {
    const mediaDir = path.join(dataRoot(env), 'runtimes', 'media');
    if (!fs.existsSync(mediaDir)) return;
    log('Images & documents models are kept at ' + mediaDir + ' (' + humanSize(folderBytes(mediaDir)) + ').');
    log('To remove them, run: slm media disable --remove-files');
  } catch (_e) { /* a notice must never fail an uninstall */ }
}

console.log('SuperLocalMemory application code is being removed.');
console.log('Memory data is preserved. No database, profile, backup, or configuration was changed.');
mediaEnvNotice();

module.exports = { mediaEnvNotice, folderBytes, dataRoot };
