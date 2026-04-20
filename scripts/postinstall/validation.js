/**
 * SuperLocalMemory postinstall — validation helpers.
 *
 * Extracted from scripts/postinstall-interactive.js in Stage 9 W4
 * (H-ARC-03) to keep the main installer under the 800-LOC cap while
 * preserving every check byte-for-byte.
 *
 * Exports:
 *   validateReplyFileSchema(obj)            — H-09 schema gate
 *   validateHomePath(homeArg, home, opt)    — H-10 + H-SEC-03 gate
 *   DENY_PREFIXES_POSIX                     — shared list (tests may read)
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');


// ------------------------------------------------------------------------
// H-09 — Reply-file schema validation
// ------------------------------------------------------------------------

function validateReplyFileSchema(obj) {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    return { ok: false, error: 'reply-file must decode to a JSON object' };
  }
  const schema = {
    profile: { type: 'string', enum: ['minimal', 'light', 'balanced', 'power', 'custom'] },
    home: { type: 'string' },
    accept_default: { type: 'boolean' },
    no_benchmark: { type: 'boolean' },
    ram_ceiling_mb: { type: 'number', integer: true, min: 1 },
    hot_path_hooks: { type: 'string' },
    reranker: { type: 'string' },
    context_injection_tokens: { type: 'number', integer: true, min: 0 },
    skill_evolution_enabled: { type: 'boolean' },
    evolution_llm: { type: 'string', enum: ['haiku', 'sonnet', 'ollama', 'skip'] },
    online_retrain_cadence: { type: 'string' },
    consolidation_cadence: { type: 'string' },
    inline_entity_detection: { type: 'boolean' },
    telemetry: { type: 'string' },
  };
  for (const key of Object.keys(obj)) {
    if (!Object.prototype.hasOwnProperty.call(schema, key)) {
      return { ok: false, error: 'unexpected key in reply-file: "' + key + '"' };
    }
    const rule = schema[key];
    const val = obj[key];
    if (rule.type === 'string') {
      if (typeof val !== 'string') {
        return { ok: false, error: 'reply-file key "' + key + '" must be a string' };
      }
      if (rule.enum && !rule.enum.includes(val)) {
        return {
          ok: false,
          error: 'reply-file key "' + key + '" must be one of: ' + rule.enum.join('|'),
        };
      }
    } else if (rule.type === 'boolean') {
      if (typeof val !== 'boolean') {
        return { ok: false, error: 'reply-file key "' + key + '" must be a boolean' };
      }
    } else if (rule.type === 'number') {
      if (typeof val !== 'number' || Number.isNaN(val) || !Number.isFinite(val)) {
        return { ok: false, error: 'reply-file key "' + key + '" must be a number' };
      }
      if (rule.integer && !Number.isInteger(val)) {
        return { ok: false, error: 'reply-file key "' + key + '" must be an integer' };
      }
      if (rule.min !== undefined && val < rule.min) {
        return { ok: false, error: 'reply-file key "' + key + '" must be >= ' + rule.min };
      }
    }
  }
  return { ok: true };
}


// ------------------------------------------------------------------------
// H-10 / H-SEC-03 — --home path validation
// ------------------------------------------------------------------------

const DENY_PREFIXES_POSIX = [
  '/etc', '/usr', '/bin', '/sbin', '/boot', '/proc', '/sys', '/dev',
  '/var/log', '/var/lib', '/var/spool', '/root',
];

function _violatesDenyList(resolvedPath) {
  if (process.platform === 'win32') return null;
  for (const prefix of DENY_PREFIXES_POSIX) {
    if (resolvedPath === prefix ||
        resolvedPath.startsWith(prefix + path.sep)) {
      return prefix;
    }
  }
  return null;
}

function validateHomePath(homeArg, userHomeDir, outsideOptIn) {
  if (typeof homeArg !== 'string' || homeArg === '') {
    return { ok: false, error: '--home must be a non-empty string' };
  }
  if (!path.isAbsolute(homeArg)) {
    return { ok: false, error: '--home must be an absolute path (rule: not-absolute)' };
  }
  const segments = homeArg.split(path.sep);
  if (segments.includes('..')) {
    return { ok: false, error: '--home must not contain ".." segments (rule: dotdot-segment)' };
  }

  // S9-W2 H-SEC-03: resolve symlinks BEFORE the insideHome check.
  let resolved = path.resolve(homeArg);
  try {
    if (fs.existsSync(resolved)) {
      resolved = fs.realpathSync.native
        ? fs.realpathSync.native(resolved)
        : fs.realpathSync(resolved);
    } else {
      let anc = path.dirname(resolved);
      const tail = [path.basename(resolved)];
      while (anc !== path.dirname(anc) && !fs.existsSync(anc)) {
        tail.unshift(path.basename(anc));
        anc = path.dirname(anc);
      }
      if (fs.existsSync(anc)) {
        const realAnc = fs.realpathSync.native
          ? fs.realpathSync.native(anc)
          : fs.realpathSync(anc);
        resolved = path.join(realAnc, ...tail);
      }
    }
  } catch (e) {
    // realpathSync failure → keep lexical resolve; downstream checks apply.
  }

  const resolvedHome = path.resolve(userHomeDir || os.homedir());
  const insideHome =
    resolved === resolvedHome || resolved.startsWith(resolvedHome + path.sep);
  if (!insideHome && !outsideOptIn) {
    return {
      ok: false,
      error:
        '--home resolves outside $HOME (' +
        resolvedHome +
        '); pass --home-outside-home to override (rule: outside-home)',
    };
  }

  const forbidden = _violatesDenyList(resolved);
  if (forbidden) {
    return {
      ok: false,
      error:
        '--home resolves to a system directory (' +
        forbidden +
        '); refusing (rule: deny-prefix)',
    };
  }

  try {
    const st = fs.lstatSync(resolved);
    if (st.isSymbolicLink()) {
      return {
        ok: false,
        error: '--home resolves to a symlink; refusing (rule: symlink)',
      };
    }
    if (!st.isDirectory()) {
      return { ok: false, error: '--home exists but is not a directory (rule: not-a-directory)' };
    }
  } catch (e) {
    // Path does not yet exist — OK, caller will mkdirSync.
  }
  return { ok: true, resolved };
}


module.exports = {
  validateReplyFileSchema,
  validateHomePath,
  DENY_PREFIXES_POSIX,
};
