/**
 * SuperLocalMemory postinstall — IDE multi-select & non-destructive connect.
 *
 * Extracted from scripts/postinstall-interactive.js (v3.8.0) to keep the
 * main installer under the 800-LOC cap.
 *
 * Execution strategy (documented choice):
 *   At npm postinstall time `slm` may not be on PATH — the Python package
 *   is installed via pip, not npm.  We therefore:
 *     1. Try to execute immediately via `python3 -m superlocalmemory.hooks.portable_kit`
 *        (uses the __main__ entry we added to portable_kit.py).
 *     2. If Python is unavailable or the module can't be imported, record the
 *        selection to ~/.superlocalmemory/pending_ide_connections.json and
 *        print exact `slm connect <ide>` commands so the user can run them
 *        manually (or `slm setup` can pick them up).
 *   The actual file writes ALWAYS go through connect_ide (non-destructive
 *   merge-not-clobber).  We never shell out to a raw file write here.
 *
 * Exports:
 *   IDE_IDS            — ordered list of connectable IDE ids (from IDE_MATRIX)
 *   IDE_DISPLAY        — map from id to display name
 *   promptIdeMultiselect(rl)           — TTY multi-select; returns [ids]
 *   executeIdeConnections(ids, slmDir) — attempt connect; defers on failure
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ------------------------------------------------------------------------
// IDE id list — kept in sync with portable_kit.py IDE_MATRIX.
// claude-code is intentionally excluded: it uses the WP-06 plugin, not a
// direct MCP config write (connect_ide short-circuits it).
// ------------------------------------------------------------------------

const IDE_IDS = Object.freeze([
  'cursor',
  'windsurf',
  'vscode-copilot',
  'gemini-cli',
  'zed',
  'jetbrains',
  'opencode',
  'claude-desktop',
  'codex',
  'continue',
  'antigravity',
]);

const IDE_DISPLAY = Object.freeze({
  cursor: 'Cursor',
  windsurf: 'Windsurf',
  'vscode-copilot': 'VS Code / Copilot',
  'gemini-cli': 'Gemini CLI',
  zed: 'Zed Editor',
  jetbrains: 'JetBrains IDEs',
  opencode: 'OpenCode',
  'claude-desktop': 'Claude Desktop',
  codex: 'Codex CLI',
  continue: 'Continue.dev',
  antigravity: 'Antigravity (agy)',
});

// ------------------------------------------------------------------------
// TTY prompt — comma-separated numbers, "all", or "none"
// ------------------------------------------------------------------------

/**
 * Prompt the user to select zero or more IDEs/agents to wire SLM into.
 *
 * @param {object} rl  readline.Interface already opened by the caller.
 * @returns {Promise<string[]>}  Array of selected IDE ids.
 */
async function promptIdeMultiselect(rl) {
  return new Promise((resolve) => {
    console.log('');
    console.log('Which IDEs / agents would you like SLM to connect to?');
    console.log('(Your existing settings are preserved — merge, not replace.)');
    console.log('');
    IDE_IDS.forEach((id, i) => {
      const display = IDE_DISPLAY[id] || id;
      console.log(`  [${String(i + 1).padStart(2)}] ${display}`);
    });
    console.log('');
    console.log('Enter comma-separated numbers, "all", or press Enter to skip.');
    rl.question('Selection [none]: ', (answer) => {
      const trimmed = (answer || '').trim().toLowerCase();
      if (!trimmed || trimmed === 'none') {
        resolve([]);
        return;
      }
      if (trimmed === 'all') {
        resolve([...IDE_IDS]);
        return;
      }
      // Parse comma-separated indices (1-based)
      const parts = trimmed.split(',').map((s) => s.trim()).filter(Boolean);
      const selected = [];
      for (const part of parts) {
        const n = Number.parseInt(part, 10);
        if (Number.isFinite(n) && n >= 1 && n <= IDE_IDS.length) {
          const id = IDE_IDS[n - 1];
          if (!selected.includes(id)) selected.push(id);
        }
        // Unknown entries silently ignored (tolerant parsing)
      }
      resolve(selected);
    });
  });
}

// ------------------------------------------------------------------------
// Connection execution
// ------------------------------------------------------------------------

/**
 * Attempt to wire SLM into the selected IDEs via non-destructive merge.
 *
 * Execution path (in order):
 *   1. Try `python3 -m superlocalmemory.hooks.portable_kit <ids...>`
 *      — the module's __main__ calls connect_many (non-destructive).
 *   2. On failure (Python not found / module not importable): record to
 *      pending_ide_connections.json and print follow-up commands.
 *
 * @param {string[]} ids      IDE ids to connect.
 * @param {string}   slmDir   ~/.superlocalmemory path.
 * @returns {boolean}  True if connections were executed immediately.
 */
function executeIdeConnections(ids, slmDir) {
  if (!ids || ids.length === 0) return true;

  console.log('');
  console.log('Connecting IDEs (merge-not-replace — your existing settings are safe):');

  try {
    const { spawnSync } = require('child_process');
    const result = spawnSync(
      'python3',
      ['-m', 'superlocalmemory.hooks.portable_kit', ...ids],
      { timeout: 30000, encoding: 'utf8' }
    );
    if (!result.error && result.status === 0) {
      if (result.stdout) process.stdout.write(result.stdout);
      return true;
    }
    // python3 failed — fall through to deferred path
    const errMsg = result.stderr || (result.error && result.error.message) || 'unknown error';
    console.log(
      `  [note] python3 connect unavailable at install time: ${errMsg.split('\n')[0]}`
    );
  } catch (e) {
    console.log(`  [note] python3 not available at install time: ${e.message}`);
  }

  // Deferred path: record selection + print commands
  return _deferConnections(ids, slmDir);
}

/**
 * Record pending connections and print follow-up commands.
 * @private
 */
function _deferConnections(ids, slmDir) {
  // Save to pending_ide_connections.json so `slm setup` can execute them.
  try {
    const pendingPath = path.join(slmDir, 'pending_ide_connections.json');
    const existing = fs.existsSync(pendingPath)
      ? JSON.parse(fs.readFileSync(pendingPath, 'utf8'))
      : { pending: [] };
    const merged = Array.from(new Set([...existing.pending, ...ids]));
    const fd = fs.openSync(pendingPath, 'w', 0o600);
    try {
      fs.writeSync(fd, JSON.stringify({ pending: merged }, null, 2), 0, 'utf8');
    } finally {
      fs.closeSync(fd);
    }
  } catch (e) {
    // Non-fatal — the commands below are sufficient for the user.
  }

  console.log('  Run these commands after `pip install superlocalmemory` to');
  console.log('  complete IDE wiring (non-destructive merge, no overwrites):');
  console.log('');
  for (const id of ids) {
    console.log(`    slm connect ${id}`);
  }
  console.log('');
  console.log('  OR in one shot:');
  console.log(`    slm connect ${ids.join(' ')}`);
  return false;
}

// ------------------------------------------------------------------------
// Exports
// ------------------------------------------------------------------------

module.exports = {
  IDE_IDS,
  IDE_DISPLAY,
  promptIdeMultiselect,
  executeIdeConnections,
};
