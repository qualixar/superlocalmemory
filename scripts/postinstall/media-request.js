/**
 * What's-new banner and the "turn on images and documents" request.
 *
 * The installer never downloads anything. It only records a request in
 * <data root>/features.json; the SLM daemon acts on it the next time it
 * starts. Nothing is written unless the person asked (a flag, the
 * SLM_ENABLE_MEDIA=1 environment variable, or a "yes" at a terminal).
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const readline = require('readline');

const FEATURES_FILE = 'features.json';

const GIB = 1024 ** 3;
// Same threshold and override as the Python side (MEDIA_MIN_RAM_BYTES and
// LOW_RAM_OVERRIDE_ENV in runtimes/media_env.py): 15 GiB, because machines sold as
// 16 GB report a little less.
const MIN_RAM_BYTES = 15 * GIB;
const LOW_RAM_OVERRIDE_ENV = 'SLM_MEDIA_ALLOW_LOW_RAM';

/**
 * Why images and documents cannot be turned on here, or '' when they can.
 * Unknown memory (0) is allowed; SLM_MEDIA_ALLOW_LOW_RAM=1 is the developer override.
 */
function ramRefusal(totalBytes, env = process.env) {
  if (!(totalBytes > 0 && totalBytes < MIN_RAM_BYTES)) return '';
  if (env && env[LOW_RAM_OVERRIDE_ENV] === '1') return '';
  return 'Images and documents need a computer with at least 16 GB of memory; this one has '
    + (totalBytes / GIB).toFixed(1) + ' GB. Your text memories keep working.';
}

function printWhatsNew(log = console.log) {
  log('');
  log("What's new in 4.1.25:");
  log('  + Images & documents: off until you turn them on (about 1.5 GB of models). Run: slm media enable');
  log('  + Bots on the web can message each other through the mesh (remote access is still off).');
  log('  + See what is on and what you can turn on: slm features');
}

// Same alias order as every other entry point.
function resolveDataRoot(env = process.env) {
  return env.SLM_DATA_DIR
    || env.SL_MEMORY_PATH
    || env.SLM_HOME
    || path.join(env.HOME || os.homedir(), '.superlocalmemory');
}

function mediaRequested(args, env) {
  const viaNpm = env && ['true', '1'].includes(String(env.npm_config_media || '').toLowerCase());
  return Boolean((args && args.media) || (env && env.SLM_ENABLE_MEDIA === '1') || viaNpm);
}

/**
 * Merge {"media": {"requested": true, ...}} into features.json (mode 0600).
 * Never throws; returns { ok, note }.
 */
function recordMediaRequest(slmDir, now = new Date()) {
  const file = path.join(slmDir, FEATURES_FILE);
  try {
    let data = { schema: 1 };
    if (fs.existsSync(file)) {
      try {
        data = JSON.parse(fs.readFileSync(file, 'utf8'));
      } catch (_e) {
        return { ok: false, note: 'features.json could not be read; left untouched. Run: slm media enable' };
      }
      if (!data || typeof data !== 'object' || Array.isArray(data)) {
        return { ok: false, note: 'features.json has an unexpected shape; left untouched. Run: slm media enable' };
      }
    }
    const media = data.media && typeof data.media === 'object' ? data.media : {};
    if (media.enabled) return { ok: true, note: 'images and documents are already on' };
    data.media = Object.assign({}, media, {
      requested: true,
      requested_at: now.toISOString(),
      choice_source: 'npm',
    });
    fs.mkdirSync(slmDir, { recursive: true });
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
    return { ok: true, note: 'recorded' };
  } catch (e) {
    return { ok: false, note: 'could not record the request (' + e.message + '). Run: slm media enable' };
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
 * Decide and (unless dry-run) record. `interactive` allows the question,
 * which defaults to No. Returns true when a request was recorded. On a computer
 * with less than 16 GB of memory it says so once and neither asks nor records.
 */
async function handleMediaChoice({ args, env, slmDir, interactive, ask = askYesNo, log = console.log,
  totalMem = os.totalmem() }) {
  let wanted = mediaRequested(args, env);
  const refusal = ramRefusal(totalMem, env);
  if ((wanted || interactive) && refusal) {
    log('');
    log('SLM: ' + refusal);
    return false;  // not asked, nothing recorded: the daemon would refuse it anyway
  }
  if ((wanted || interactive) && totalMem > 0 && totalMem < MIN_RAM_BYTES) {
    log('SLM: ' + LOW_RAM_OVERRIDE_ENV + '=1 is set; allowing images and documents on '
      + (totalMem / GIB).toFixed(1) + ' GB of memory (16 GB needed).');
  }
  if (!wanted && interactive) {
    log('');
    wanted = await ask('Turn on images and documents later? It downloads about 1.5 GB when SLM next starts. [y/N] ');
  }
  if (!wanted) return false;
  if (args && args.dryRun) {
    log('SLM dry-run: would record a request to turn on images and documents.');
    return false;
  }
  const result = recordMediaRequest(slmDir);
  if (result.ok) {
    log('SLM: images and documents will start setting up the next time SLM starts (' + result.note + ').');
  } else {
    log('SLM: ' + result.note);
  }
  return result.ok;
}

/**
 * The step npm's own postinstall runs: --media / npm_config_media /
 * SLM_ENABLE_MEDIA=1 record a request; on a real terminal (never in CI) a
 * default-No question is asked. Never downloads, never throws.
 */
async function runMediaStep({ argv = [], env = process.env, tty, ask, log = console.log } = {}) {
  try {
    const isTty = tty !== undefined ? tty
      : Boolean(process.stdout.isTTY && process.stdin.isTTY);
    const ci = env.CI === 'true' || env.CI === '1';
    return await handleMediaChoice({
      args: { media: argv.includes('--media'), dryRun: argv.includes('--dry-run') },
      env,
      slmDir: resolveDataRoot(env),
      interactive: isTty && !ci,
      ask: ask || askYesNo,
      log,
    });
  } catch (e) {
    log('SLM: could not record the images and documents choice (' + e.message + '). Run: slm media enable');
    return false;
  }
}

module.exports = { ramRefusal, MIN_RAM_BYTES, runMediaStep, resolveDataRoot, printWhatsNew, mediaRequested, recordMediaRequest, handleMediaChoice, FEATURES_FILE };
