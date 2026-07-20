#!/usr/bin/env node
/**
 * SuperLocalMemory V3 - Session Start Context Hook
 * Copyright (c) 2026 Varun Pratap Bhardwaj
 * Licensed under GNU Affero General Public License v3.0 or later
 *
 * Loads recent memories and learned patterns on Claude Code session start.
 * Outputs context to stderr (Claude Code reads hook stderr as context).
 * Fails gracefully -- never blocks session start if DB is missing or errors occur.
 */

const { execFile } = require('child_process');
const { promisify } = require('util');
const path = require('path');
const fs = require('fs');
const os = require('os');

const execFileAsync = promisify(execFile);

const MEMORY_DIR = process.env.SLM_DATA_DIR
  || process.env.SL_MEMORY_PATH
  || process.env.SLM_HOME
  || path.join(os.homedir(), '.superlocalmemory');
const DB_PATH = path.join(MEMORY_DIR, 'memory.db');

async function loadSessionContext() {
  // Fail gracefully if not installed
  if (!fs.existsSync(DB_PATH)) {
    return;
  }

  try {
    // Get stats via V3 CLI
    const { stdout: statsOutput } = await execFileAsync('python3', [
      '-m', 'superlocalmemory.cli.main', 'status', '--format', 'json'
    ], { timeout: 5000 });

    // Build context output
    let context = '';

    if (statsOutput && statsOutput.trim()) {
      try {
        const stats = JSON.parse(statsOutput.trim());
        const total = stats.total_memories || 0;
        if (total > 0) {
          context += 'SuperLocalMemory V3: ' + total + ' memories loaded.\n';
        }
      } catch (e) {
        // Stats output wasn't JSON -- use first line as-is
        context += 'SuperLocalMemory V3: ' + statsOutput.trim().split('\n')[0] + '\n';
      }
    }

    if (context) {
      process.stderr.write(context);
    }
  } catch (error) {
    // Never fail -- session start must not be blocked
    // Silently ignore errors (timeout, missing python, etc.)
  }
}

loadSessionContext();
