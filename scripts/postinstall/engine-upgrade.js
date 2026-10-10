/**
 * The "Upgrade memory engine" offer at install time.
 *
 * For someone who already has memories, the installer can offer to move them to the
 * newer memory engine. Like the images-and-documents request, it only records a request
 * in <data root>/features.json; the SLM daemon acts on it at its next start, once the
 * images-and-documents environment is ready. Nothing is downloaded here.
 *
 * Off until the memory check on the Mac passes (UPGRADE_PROMPT_ENABLED): the explicit
 * command (slm embedder upgrade) and the dashboard button work before that, an
 * unsolicited prompt does not. Never asks off a terminal or in CI; the default is No.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const { FEATURES_FILE, resolveDataRoot, recordMediaRequest } = require('./media-request.js');

const UPGRADE_PROMPT_ENABLED = false;
const MANAGED_PROVIDER = 'slm-media';

function printUpgradeOffer(log = console.log) {
  log('');
  log('SLM: you can move your memories to the newer memory engine (it also reads pictures and documents).');
  log('  Your memories are re-read in the background. Recall keeps working meanwhile, nothing is deleted,');
  log('  and you can roll back. It needs images and documents turned on (about 1.5 GB of models).');
  log('  Any time: slm embedder upgrade');
}

/** True for an existing store (memory.db present) that is not already on the managed model. */
function shouldOffer(slmDir) {
  if (!fs.existsSync(path.join(slmDir, 'memory.db'))) return false;
  try {
    const config = JSON.parse(fs.readFileSync(path.join(slmDir, 'config.json'), 'utf8'));
    return !(config && config.embedding && config.embedding.provider === MANAGED_PROVIDER);
  } catch (_e) {
    return true;
  }
}

/** Merge {"engine_upgrade": {"requested": true, ...}} into features.json (mode 0600). Never throws. */
function recordUpgradeRequest(slmDir, now = new Date()) {
  const file = path.join(slmDir, FEATURES_FILE);
  try {
    let data = { schema: 1 };
    if (fs.existsSync(file)) {
      data = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (!data || typeof data !== 'object' || Array.isArray(data)) return false;
    }
    data.engine_upgrade = { requested: true, requested_at: now.toISOString(), choice_source: 'npm' };
    const tmp = file + '.tmp';
    const fd = fs.openSync(tmp, 'w', 0o600);
    try {
      fs.writeSync(fd, JSON.stringify(data), 0, 'utf8');
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(tmp, file);
    if (process.platform !== 'win32') {
      try { fs.chmodSync(file, 0o600); } catch (_e) { /* best-effort */ }
    }
    return true;
  } catch (_e) {
    return false;
  }
}

function askYesNo(question) {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(question, (answer) => {
      rl.close();
      resolve(/^\s*y(es)?\s*$/i.test(answer || ''));
    });
  });
}

/**
 * Offer (when enabled, on a terminal, outside CI, for an existing store) and record on a yes.
 * Returns true when a request was recorded. Never downloads, never throws.
 */
async function runUpgradeStep({ enabled = UPGRADE_PROMPT_ENABLED, env = process.env, tty, ask,
  log = console.log, slmDir } = {}) {
  try {
    if (!enabled) return false;
    const isTty = tty !== undefined ? tty : Boolean(process.stdout.isTTY && process.stdin.isTTY);
    const ci = env.CI === 'true' || env.CI === '1';
    const dir = slmDir || resolveDataRoot(env);
    if (!isTty || ci || !shouldOffer(dir)) return false;
    printUpgradeOffer(log);
    const yes = await (ask || askYesNo)('Upgrade the memory engine after install? [y/N] ');
    if (!yes) return false;
    const media = recordMediaRequest(dir);
    if (!media.ok || !recordUpgradeRequest(dir)) {
      log('SLM: could not record the request. Run later: slm embedder upgrade');
      return false;
    }
    log('SLM: the upgrade will start the next time SLM starts, once images and documents are ready.');
    return true;
  } catch (e) {
    log('SLM: could not record the upgrade choice (' + e.message + '). Run later: slm embedder upgrade');
    return false;
  }
}

module.exports = { UPGRADE_PROMPT_ENABLED, printUpgradeOffer, recordUpgradeRequest, runUpgradeStep, shouldOffer };
