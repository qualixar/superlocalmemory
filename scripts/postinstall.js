#!/usr/bin/env node
/**
 * SuperLocalMemory V3 - safe NPM runtime installer.
 *
 * NPM owns a private Python virtual environment inside its package directory.
 * This script never installs into system Python and never creates or modifies
 * SLM data, hooks, IDE configuration, daemons, or model caches.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
 * Licensed under AGPL-3.0-or-later.
 */

'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const MIN_PYTHON = Object.freeze([3, 11]);
const MAX_PYTHON_EXCLUSIVE = Object.freeze([3, 15]);
const INSTALL_TIMEOUT_MS = 15 * 60 * 1000;

function parsePythonVersion(output) {
  const match = String(output || '').match(/Python\s+(\d+)\.(\d+)(?:\.(\d+))?/i);
  if (!match) return null;
  return [Number(match[1]), Number(match[2]), Number(match[3] || 0)];
}

function isSupportedPython(version) {
  if (!version) return false;
  const [major, minor] = version;
  const atLeastMinimum = major > MIN_PYTHON[0]
    || (major === MIN_PYTHON[0] && minor >= MIN_PYTHON[1]);
  const belowMaximum = major < MAX_PYTHON_EXCLUSIVE[0]
    || (major === MAX_PYTHON_EXCLUSIVE[0] && minor < MAX_PYTHON_EXCLUSIVE[1]);
  return atLeastMinimum && belowMaximum;
}

function pythonCandidates(platform = os.platform()) {
  if (platform === 'win32') {
    return [
      ['py', '-3.14'],
      ['py', '-3.13'],
      ['py', '-3.12'],
      ['py', '-3.11'],
      ['python3'],
      ['python'],
    ];
  }
  return [
    ['python3'],
    ['python'],
    ['/opt/homebrew/bin/python3'],
    ['/usr/local/bin/python3'],
    ['/usr/bin/python3'],
  ];
}

function findSupportedPython() {
  for (const candidate of pythonCandidates()) {
    try {
      const result = spawnSync(candidate[0], [...candidate.slice(1), '--version'], {
        stdio: 'pipe',
        timeout: 5000,
        env: process.env,
      });
      const output = `${(result.stdout || '').toString()} ${(result.stderr || '').toString()}`;
      const version = parsePythonVersion(output);
      if (result.status === 0 && isSupportedPython(version)) {
        return { command: candidate[0], prefixArgs: candidate.slice(1), version };
      }
    } catch (_error) {
      // Try the next interpreter without mutating PATH or the machine.
    }
  }
  return null;
}

function runtimePythonPath(packageRoot, platform = os.platform()) {
  return platform === 'win32'
    ? path.join(packageRoot, '.slm-venv', 'Scripts', 'python.exe')
    : path.join(packageRoot, '.slm-venv', 'bin', 'python');
}

function validateRuntimeLocation(venvRoot) {
  if (!fs.existsSync(venvRoot)) return { ok: true };
  try {
    const stat = fs.lstatSync(venvRoot);
    if (stat.isSymbolicLink()) {
      return { ok: false, error: `${venvRoot} is a symbolic link` };
    }
    if (!stat.isDirectory()) {
      return { ok: false, error: `${venvRoot} is not a directory` };
    }
    return { ok: true };
  } catch (error) {
    return { ok: false, error: `cannot inspect ${venvRoot}: ${error.message}` };
  }
}

function printPythonGuidance() {
  console.error('');
  console.error('SuperLocalMemory requires Python 3.11, 3.12, 3.13, or 3.14.');
  console.error('Install Python from https://www.python.org/downloads/ and rerun:');
  console.error('  npm rebuild superlocalmemory');
  console.error('The npm installer will create a private virtual environment;');
  console.error('it will not install packages into your system Python.');
}

function failureDetail(result) {
  if (result && result.error) return result.error.message || String(result.error);
  if (result && Number.isInteger(result.status)) return `exit code ${result.status}`;
  return 'process did not complete';
}

function main(argv = process.argv.slice(2)) {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log('Usage: node scripts/postinstall.js');
    console.log('Creates or repairs the package-owned .slm-venv runtime.');
    console.log('This command never configures SLM or writes durable memory.');
    return 0;
  }

  const packageRoot = path.resolve(__dirname, '..');
  const venvRoot = path.join(packageRoot, '.slm-venv');
  const packageVersion = require(path.join(packageRoot, 'package.json')).version;
  const python = findSupportedPython();

  console.log('');
  console.log('SuperLocalMemory: creating an isolated npm-owned Python runtime.');

  if (!python) {
    printPythonGuidance();
    return 1;
  }

  const locationCheck = validateRuntimeLocation(venvRoot);
  if (!locationCheck.ok) {
    console.error(`SuperLocalMemory: refusing unsafe runtime location: ${locationCheck.error}.`);
    console.error('Move that path aside manually, verify it contains no needed files, then run:');
    console.error('  npm rebuild superlocalmemory');
    return 1;
  }

  console.log(`  Python ${python.version.join('.')} (${[python.command, ...python.prefixArgs].join(' ')})`);

  const createVenv = spawnSync(
    python.command,
    [...python.prefixArgs, '-m', 'venv', venvRoot],
    { stdio: 'inherit', timeout: 120000, env: process.env },
  );
  if (createVenv.status !== 0) {
    console.error(`SuperLocalMemory: could not create ${venvRoot} (${failureDetail(createVenv)}).`);
    if (os.platform() === 'linux') {
      console.error('Install your distribution\'s Python venv package (for example python3-venv), then run:');
    } else {
      console.error('Repair the selected Python installation so the stdlib venv module is available, then run:');
    }
    console.error('  npm rebuild superlocalmemory');
    return 1;
  }

  const runtimePython = runtimePythonPath(packageRoot);
  const installPackage = spawnSync(
    runtimePython,
    [
      '-m', 'pip', 'install',
      '--disable-pip-version-check',
      '--no-input',
      '--upgrade',
      packageRoot,
    ],
    { stdio: 'inherit', timeout: INSTALL_TIMEOUT_MS, env: process.env },
  );
  if (installPackage.status !== 0) {
    console.error(`SuperLocalMemory: private-runtime installation failed (${failureDetail(installPackage)}).`);
    console.error('Check network access, available disk space, and Python build prerequisites, then run:');
    console.error('  npm rebuild superlocalmemory');
    console.error('No system Python packages or SLM durable data were modified.');
    return 1;
  }

  const verify = spawnSync(
    runtimePython,
    [
      '-c',
      "import importlib.metadata as m; print(m.version('superlocalmemory'))",
    ],
    { stdio: 'pipe', timeout: 15000, env: process.env },
  );
  const installedVersion = (verify.stdout || '').toString().trim();
  if (verify.status !== 0 || installedVersion !== packageVersion) {
    console.error(
      `SuperLocalMemory: runtime identity check failed (npm=${packageVersion}, python=${installedVersion || 'unavailable'}).`,
    );
    console.error('Run `npm rebuild superlocalmemory` to repair the package-owned runtime.');
    return 1;
  }

  console.log(`SuperLocalMemory ${packageVersion}: isolated runtime verified.`);
  console.log('No memory database, IDE hooks, daemon, configuration, or models were changed.');
  console.log('Run `slm setup` explicitly when you are ready to configure SLM.');
  console.log('');
  return 0;
}

if (require.main === module) {
  process.exit(main());
}

module.exports = {
  findSupportedPython,
  isSupportedPython,
  main,
  parsePythonVersion,
  pythonCandidates,
  runtimePythonPath,
  validateRuntimeLocation,
};
