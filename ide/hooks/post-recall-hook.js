#!/usr/bin/env node
/**
 * SuperLocalMemory V3 - Post-Recall Hook
 * Copyright (c) 2026 Varun Pratap Bhardwaj
 * Licensed under GNU Affero General Public License v3.0 or later
 *
 * Claude Code hook that tracks recall events for implicit signal collection.
 * This hook fires after the slm-recall skill completes, recording timing
 * data that the signal inference engine uses to detect satisfaction/dissatisfaction.
 *
 * Installation: Automatically registered via install-skills.sh
 * The core hook reads local SLM state; the host client and optional integrations
 * retain their own network behavior.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const MEMORY_DIR = process.env.SLM_DATA_DIR
    || process.env.SL_MEMORY_PATH
    || process.env.SLM_HOME
    || path.join(os.homedir(), '.superlocalmemory');
const HOOK_LOG = path.join(MEMORY_DIR, 'recall-events.jsonl');

// Parse input from Claude Code hook system
function main() {
    try {
        const timestamp = Date.now();
        const args = process.argv.slice(2);

        // Extract query from args if available
        let query = '';
        for (let i = 0; i < args.length; i++) {
            if (args[i] && !args[i].startsWith('--')) {
                query = args[i];
                break;
            }
        }

        if (!query) return;

        // Append recall event to JSONL log (lightweight, append-only)
        const event = JSON.stringify({
            type: 'recall',
            query: query.substring(0, 100), // Truncate for privacy
            timestamp: timestamp,
            source: 'claude-code-hook',
        });

        fs.appendFileSync(HOOK_LOG, event + '\n', { flag: 'a' });
    } catch (e) {
        // Hook failures must be silent
    }
}

main();
