/**
 * SuperLocalMemory postinstall — deployment-mode helpers.
 *
 * Extracted from scripts/postinstall-interactive.js (v3.8.0) to keep
 * the main installer under the 800-LOC cap per the 800-line hard cap rule.
 *
 * Exports:
 *   DEPLOYMENT_PRESETS          — canonical preset objects
 *   DEFAULT_DEPLOYMENT_MODE     — "personal" (safe default)
 *   promptDeploymentMode(rl)    — TTY question; returns deployment preset obj
 *   renderDeploymentBlock(dep)  — TOML lines for [deployment] section
 *                                 Returns [] when dep is personal (no churn).
 *
 * Design decision — no [deployment] block for personal installs:
 *   Personal is the default.  Writing `[deployment]\nmode = "personal"` to
 *   every config.toml would churn existing files on `slm reconfigure` even
 *   when nothing changed.  We only serialise the block when the user
 *   explicitly opts into enterprise mode.  The Python parser defaults to
 *   personal when the section is absent, so round-trip semantics are
 *   preserved.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

// ------------------------------------------------------------------------
// Presets
// ------------------------------------------------------------------------

const DEPLOYMENT_PRESETS = Object.freeze({
  personal: Object.freeze({
    mode: 'personal',
    require_login: false,
    pii_redaction: false,
    retention_enabled: false,
    audit: true,
  }),
  enterprise: Object.freeze({
    mode: 'enterprise',
    require_login: true,
    pii_redaction: true,
    retention_enabled: true,
    audit: true,
  }),
});

const DEFAULT_DEPLOYMENT_MODE = 'personal';

// ------------------------------------------------------------------------
// TTY prompt
// ------------------------------------------------------------------------

/**
 * Prompt the user to choose Personal or Enterprise deployment.
 *
 * @param {object} rl  readline.Interface already opened by the caller.
 * @returns {Promise<object>}  Resolved deployment preset object.
 */
async function promptDeploymentMode(rl) {
  return new Promise((resolve) => {
    console.log('');
    console.log('Deployment mode:');
    console.log('  [1] Personal (default) — single user, no sign-in required');
    console.log('  [2] Enterprise — team / company: sign-in required, PII');
    console.log('      redaction, retention scheduling, audit logging');
    rl.question('Choice [1]: ', (answer) => {
      const trimmed = (answer || '').trim();
      if (trimmed === '2' || trimmed.toLowerCase() === 'enterprise') {
        console.log('  → Enterprise deployment selected.');
        resolve(Object.assign({}, DEPLOYMENT_PRESETS.enterprise));
      } else {
        console.log('  → Personal deployment selected (default).');
        resolve(Object.assign({}, DEPLOYMENT_PRESETS.personal));
      }
    });
  });
}

// ------------------------------------------------------------------------
// TOML serialisation — only emitted for non-personal installs
// ------------------------------------------------------------------------

/**
 * Return TOML lines for the [deployment] section.
 *
 * Returns an empty array for personal mode — preserving the "no churn"
 * contract (personal installs produce identical config.toml output to
 * pre-3.8.0 installers).
 *
 * @param {object} dep  Deployment preset object from DEPLOYMENT_PRESETS.
 * @returns {string[]}  Array of TOML lines (empty = no section to write).
 */
function renderDeploymentBlock(dep) {
  if (!dep || dep.mode === DEFAULT_DEPLOYMENT_MODE) {
    // Personal is the default — do NOT write [deployment] block.
    // load_deployment_config() returns DEPLOYMENT_PERSONAL when the section
    // is absent, so omitting the block is semantically identical to writing
    // `mode = "personal"`.
    return [];
  }
  return [
    '[deployment]',
    `mode = "${dep.mode}"`,
    `require_login = ${dep.require_login}`,
    `pii_redaction = ${dep.pii_redaction}`,
    `retention_enabled = ${dep.retention_enabled}`,
    `audit = ${dep.audit}`,
    '',
  ];
}

// ------------------------------------------------------------------------
// Exports
// ------------------------------------------------------------------------

module.exports = {
  DEPLOYMENT_PRESETS,
  DEFAULT_DEPLOYMENT_MODE,
  promptDeploymentMode,
  renderDeploymentBlock,
};
