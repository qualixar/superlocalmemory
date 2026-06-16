#!/usr/bin/env node
/**
 * SuperLocalMemory V3 - NPM Postinstall Script
 *
 * ONE COMMAND INSTALL. Everything the user needs.
 * Python deps auto-installed. Embeddings auto-downloaded.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under Elastic License 2.0
 */

const { spawnSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

console.log('\n════════════════════════════════════════════════════════════');
console.log('  SuperLocalMemory V3 — The Unified Brain');
console.log('  by Varun Pratap Bhardwaj / Qualixar');
console.log('  https://github.com/qualixar/superlocalmemory');
console.log('════════════════════════════════════════════════════════════\n');

// --- Step 1: Create data directory ---
const SLM_HOME = path.join(os.homedir(), '.superlocalmemory');
if (!fs.existsSync(SLM_HOME)) {
    fs.mkdirSync(SLM_HOME, { recursive: true });
    console.log('✓ Created data directory: ' + SLM_HOME);
} else {
    console.log('✓ Data directory exists: ' + SLM_HOME);
}

// --- Step 2: Find Python 3 ---
function findPython() {
    const candidates = [
        'python3', 'python',
        '/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3',
    ];
    if (os.platform() === 'win32') candidates.push('py -3');
    for (const cmd of candidates) {
        try {
            const parts = cmd.split(' ');
            const r = spawnSync(parts[0], [...parts.slice(1), '--version'], {
                stdio: 'pipe', timeout: 5000,
                env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') },
            });
            if (r.status === 0 && (r.stdout || '').toString().includes('3.')) return parts;
        } catch (e) { /* next */ }
    }
    return null;
}

const pythonParts = findPython();
if (!pythonParts) {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║  ⚠  Python 3.11+ Required                              ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log('');
    console.log('  SuperLocalMemory V3 requires Python 3.11+');
    console.log('  Install from: https://python.org/downloads/');
    console.log('  After installing Python, run: slm setup');
    console.log('');
    process.exit(0); // Don't fail npm install
}
console.log('✓ Found Python: ' + pythonParts.join(' '));

// --- Step 3: Install ALL Python dependencies ---
console.log('\nInstalling Python dependencies (this may take 1-2 minutes)...\n');

// Detect if --user or --break-system-packages is needed
function pipInstall(packages, label) {
    // Try normal install first
    let result = spawnSync(pythonParts[0], [
        ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
        ...packages,
    ], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });

    if (result.status === 0) return true;

    // If PEP 668 blocks it, try --user
    const stderr = (result.stderr || '').toString();
    if (stderr.includes('externally-managed') || stderr.includes('PEP 668')) {
        result = spawnSync(pythonParts[0], [
            ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
            '--user', ...packages,
        ], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });
        if (result.status === 0) return true;

        // Last resort: --break-system-packages
        result = spawnSync(pythonParts[0], [
            ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
            '--break-system-packages', ...packages,
        ], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });
        return result.status === 0;
    }

    return false;
}

// Install the superlocalmemory package and all its pinned dependencies
// in one shot. pyproject.toml is the single source of truth for versions,
// so users via npm get exactly the same dep set as users via pip.
console.log('\nInstalling SuperLocalMemory and all dependencies...');
console.log('  (Single pip install — versions pinned in pyproject.toml)');
console.log('  This may take 1-3 minutes (downloads ~500MB of models on first use).');
console.log('');
console.log('  Includes: numpy, scipy, fastapi, sentence-transformers, onnxruntime,');
console.log('           torch, transformers, sqlite-vec, lightgbm, mcp, and more.');

// --- Step 3b: Install the superlocalmemory package itself ---
// This ensures `python -m superlocalmemory.cli.main` always resolves the
// correct version, even when invoked outside the Node.js wrapper (e.g.,
// via slm.bat on Windows or direct Python invocation).
console.log('\nInstalling superlocalmemory Python package...');
const pkgRoot = path.join(__dirname, '..');
const pipInstallPkg = spawnSync(pythonParts[0], [
    ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
    pkgRoot,
], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });

if (pipInstallPkg.status === 0) {
    console.log('✓ SuperLocalMemory Python package installed');
} else {
    // Try with --user if PEP 668
    const stderr = (pipInstallPkg.stderr || '').toString();
    if (stderr.includes('externally-managed') || stderr.includes('PEP 668')) {
        const retryResult = spawnSync(pythonParts[0], [
            ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
            '--user', pkgRoot,
        ], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });
        if (retryResult.status === 0) {
            console.log('✓ SuperLocalMemory Python package installed (--user)');
        } else {
            // WP-07: --user also failed — try --break-system-packages last-resort,
            // then surface the pipx hint (PEP 668 advisory).
            const retryBsp = spawnSync(pythonParts[0], [
                ...pythonParts.slice(1), '-m', 'pip', 'install', '--quiet',
                '--disable-pip-version-check', '--break-system-packages', pkgRoot,
            ], { stdio: 'pipe', timeout: 300000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });
            if (retryBsp.status === 0) {
                console.log('✓ SuperLocalMemory Python package installed (--break-system-packages)');
            } else {
                console.log('⚠ pip install failed (PEP 668 — externally managed Python).');
                console.log('  The Node.js wrapper (slm-npm) sets PYTHONPATH automatically,');
                console.log('  so the CLI will still work via npm.');
                console.log('');
                console.log('  To install into an isolated environment (recommended):');
                console.log('    pipx install superlocalmemory');
                console.log('  Last resort (may break OS packages):');
                console.log('    pip install --break-system-packages superlocalmemory');
            }
        }
    } else {
        console.log('⚠ Could not pip install the package. The Node.js wrapper (slm-npm)');
        console.log('  sets PYTHONPATH automatically, so CLI will still work.');
        console.log('');
        console.log('  If you see a PEP 668 error in future, install via:');
        console.log('    pipx install superlocalmemory');
    }
}

// --- Step 3c: Verify critical dependency versions ---
// sentence-transformers and onnxruntime MUST be exact versions to avoid
// memory blow-up on Apple Silicon. pip resolver can override pins via
// transitive deps — this step catches and fixes that.
console.log('\nVerifying critical dependency versions...');
const criticalDeps = { 'sentence_transformers': '5.3.0', 'onnxruntime': '1.24.4' };
for (const [mod, expected] of Object.entries(criticalDeps)) {
    const check = spawnSync(pythonParts[0], [
        ...pythonParts.slice(1), '-c',
        `import ${mod}; v=getattr(${mod},'__version__',''); print(v)`,
    ], { stdio: 'pipe', timeout: 10000, env: { ...process.env, PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || '') } });
    const actual = (check.stdout || '').toString().trim();
    if (actual && actual !== expected) {
        const pipName = mod.replace('_', '-');
        console.log(`⚠ ${pipName} is ${actual}, expected ${expected}. Fixing...`);
        pipInstall([`${pipName}==${expected}`], pipName);
        console.log(`✓ ${pipName}==${expected} installed`);
    } else if (actual === expected) {
        console.log(`✓ ${mod}==${expected}`);
    }
}

// --- Step 4: Detect V2 installation ---
const V2_HOME = path.join(os.homedir(), '.claude-memory');
if (fs.existsSync(V2_HOME) && fs.existsSync(path.join(V2_HOME, 'memory.db'))) {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║  V2 Installation Detected                                ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log('');
    console.log('  Found V2 data at: ' + V2_HOME);
    console.log('  Your memories are safe and will NOT be deleted.');
    console.log('');
    console.log('  To migrate V2 data to V3, run:');
    console.log('    slm migrate');
    console.log('');
}

// --- Step 5: Auto-install Claude Code hooks ---
// "Install once, forget forever" — hooks enable automatic memory lifecycle
const hooksDisabledFile = path.join(SLM_HOME, 'hooks', '.hooks-disabled');
if (fs.existsSync(hooksDisabledFile)) {
    console.log('⊘ Claude Code hooks: skipped (user opted out via slm hooks remove)');
} else {
    console.log('\nInstalling Claude Code hooks (auto-memory lifecycle)...');
    const hookResult = spawnSync(pythonParts[0], [
        ...pythonParts.slice(1), '-m', 'superlocalmemory.cli.main', 'hooks', 'install',
    ], {
        stdio: 'pipe', timeout: 15000,
        env: {
            ...process.env,
            PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || ''),
            PYTHONPATH: path.join(__dirname, '..', 'src') + ':' + (process.env.PYTHONPATH || ''),
        },
    });

    if (hookResult.status === 0) {
        console.log('✓ Claude Code hooks installed (auto-recall, auto-observe, auto-save)');
        console.log('  SLM: Hooks installed into Claude Code (slm hooks remove to undo)');
    } else {
        console.log('⚠ Claude Code hooks not installed (run: slm hooks install)');
        // Non-fatal — don't block npm install
    }
}

// --- Step 6: Run interactive setup wizard ---
// Downloads embedding + reranker models, configures mode, verifies installation.
// If TTY is available (interactive terminal), runs the full wizard.
// If not (CI, piped), uses defaults (Mode A, skip model download).
console.log('\n════════════════════════════════════════════════════════════');
console.log('  Running setup wizard (model download + verification)...');
console.log('════════════════════════════════════════════════════════════\n');

const isTTY = process.stdin.isTTY && process.stdout.isTTY;
const setupArgs = isTTY ? ['setup'] : ['setup'];
const setupEnv = {
    ...process.env,
    PATH: '/opt/homebrew/bin:/usr/local/bin:/usr/bin:' + (process.env.PATH || ''),
    PYTHONPATH: path.join(__dirname, '..', 'src') + ':' + (process.env.PYTHONPATH || ''),
    CUDA_VISIBLE_DEVICES: '',
    TOKENIZERS_PARALLELISM: 'false',
    TORCH_DEVICE: 'cpu',
};

// Non-interactive: set env flag so wizard uses defaults
if (!isTTY) {
    setupEnv.SLM_NON_INTERACTIVE = '1';
}

const setupResult = spawnSync(pythonParts[0], [
    ...pythonParts.slice(1), '-m', 'superlocalmemory.cli.main', ...setupArgs,
], {
    stdio: 'inherit',  // Show all output including download progress
    timeout: 900000,    // 15 min (model downloads can be slow)
    env: setupEnv,
});

if (setupResult.status === 0) {
    console.log('✓ Setup wizard completed successfully');
} else {
    console.log('⚠ Setup wizard had issues (run: slm setup)');
    console.log('  SuperLocalMemory will still work — models download on first use.');
}

// --- Done ---
console.log('\n════════════════════════════════════════════════════════════');
console.log('  ✓ SuperLocalMemory V3 — The Unified Brain installed!');
console.log('');
console.log('  Quick start:');
console.log('    slm remember "..."        # Store a memory');
console.log('    slm recall "..."          # Search memories');
console.log('    slm dashboard             # Open web dashboard');
console.log('    slm serve                 # Start 24/7 daemon');
console.log('    slm adapters enable gmail # Enable Gmail ingestion');
console.log('    slm setup                 # Re-run 9-step wizard');
console.log('');
console.log('  New in v3.4.3:');
console.log('    • Unified daemon (one process, 24/7, < 700MB)');
console.log('    • SLM Mesh (agent-to-agent P2P built in)');
console.log('    • Entity compilation (auto knowledge summaries)');
console.log('    • Ingestion adapters (Gmail, Calendar, Transcripts)');
console.log('');
console.log('  Docs: https://github.com/qualixar/superlocalmemory');
console.log('════════════════════════════════════════════════════════════\n');

console.log('────────────────────────────────────────────────────────────');
console.log('  ⭐ Help us grow!');
console.log('  If this saves you time, please star the repo:');
console.log('    https://github.com/qualixar/superlocalmemory');
console.log('  Part of the Qualixar AI Agent Reliability Platform:');
console.log('    https://qualixar.com  (7 OSS products, 19K+ monthly downloads)');
console.log('────────────────────────────────────────────────────────────\n');
