#!/usr/bin/env node
/**
 * SuperLocalMemory - Cross-platform prepack cleanup
 *
 * Removes __pycache__ directories and .pyc files before npm pack.
 * Works on Windows, macOS, and Linux.
 */

const fs = require('fs');
const path = require('path');

function removePycache(dir) {
    if (!fs.existsSync(dir)) return;

    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
            // Never walk environment/build VCS roots. Besides making prepack
            // needlessly slow, descending into .venv races active Python
            // processes and may delete their bytecode during an npm dry-run.
            if (entry.name === '__pycache__' || entry.name === 'node_modules') {
                if (entry.name === '__pycache__') {
                    fs.rmSync(fullPath, { recursive: true, force: true });
                }
                continue;
            }
            if (['.venv', '.git', 'dist', 'build'].includes(entry.name)) {
                continue;
            }
            removePycache(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.pyc')) {
            fs.unlinkSync(fullPath);
        }
    }
}

removePycache(process.cwd());
