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
            if (entry.name === '__pycache__' || entry.name === 'node_modules') {
                if (entry.name === '__pycache__') {
                    fs.rmSync(fullPath, { recursive: true, force: true });
                }
                continue;
            }
            removePycache(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.pyc')) {
            fs.unlinkSync(fullPath);
        }
    }
}

removePycache(process.cwd());
