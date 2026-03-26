#!/usr/bin/env node
/**
 * SuperLocalMemory V3 - NPM Preuninstall Script
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj
 * Solution Architect & Original Creator
 *
 * Licensed under MIT License (see LICENSE file)
 * Repository: https://github.com/qualixar/superlocalmemory
 *
 * ATTRIBUTION REQUIRED: This notice must be preserved in all copies.
 */

const path = require('path');
const os = require('os');
const fs = require('fs');

console.log('\n════════════════════════════════════════════════════════════');
console.log('  SuperLocalMemory V3 - Uninstalling');
console.log('════════════════════════════════════════════════════════════\n');

const SLM_DIR = path.join(os.homedir(), '.superlocalmemory');

if (fs.existsSync(SLM_DIR)) {
    console.log('⚠️  Your data is preserved at: ' + SLM_DIR);
    console.log('');
    console.log('   This includes:');
    console.log('   • Your memory database (memory.db)');
    console.log('   • All learned patterns');
    console.log('   • Knowledge graph');
    console.log('   • Profile data');
    console.log('');
    console.log('   To completely remove SuperLocalMemory:');
    if (os.platform() === 'win32') {
        console.log('   rmdir /s "' + SLM_DIR + '"');
    } else {
        console.log('   rm -rf "' + SLM_DIR + '"');
    }
    console.log('');
    console.log('   To backup your data first:');
    if (os.platform() === 'win32') {
        console.log('   xcopy "' + SLM_DIR + '" "%USERPROFILE%\\slm-backup\\" /E /I');
    } else {
        console.log('   cp -r "' + SLM_DIR + '" ~/slm-backup/');
    }
    console.log('');
} else {
    console.log('ℹ️  No data directory found at ' + SLM_DIR);
    console.log('');
}

console.log('📦 Removing NPM package...');
console.log('   (slm and superlocalmemory commands will be unavailable)');
console.log('');

// No actual deletion here - we preserve user data
// NPM will remove the package files automatically
