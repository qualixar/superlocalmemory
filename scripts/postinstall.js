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
const { printWhatsNew, runMediaStep } = require('./postinstall/media-request.js');
const { nodeRunsTranslated } = require('./postinstall/node-arch.js');
const { runUpgradeStep } = require('./postinstall/engine-upgrade.js');

const MIN_PYTHON = Object.freeze([3, 12]);
const MAX_PYTHON_EXCLUSIVE = Object.freeze([3, 15]);
const INSTALL_TIMEOUT_MS = 15 * 60 * 1000;

// The official CPU-only wheel index (https://pytorch.org/get-started/locally/,
// fetched 2026-10-06). PyPI's plain `torch` wheel is CPU-only on macOS/Windows but
// a CUDA build on Linux (torch + triton + ~18 nvidia-* packages, ~2.6 GiB) even
// with no GPU present to use it.
const TORCH_CPU_INDEX_URL = 'https://download.pytorch.org/whl/cpu';

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
      ['python3'],
      ['python'],
    ];
  }
  // Debian 12, Ubuntu 22.04 (with deadsnakes) and RHEL 9 all ship a system
  // `python3` older than 3.12, with the supported versioned interpreter
  // installed alongside it, unused. Try the versioned names first so that
  // interpreter is found before falling back to the bare names and the
  // fixed absolute paths below.
  return [
    ['python3.14'],
    ['python3.13'],
    ['python3.12'],
    ['python3'],
    ['python'],
    ['/opt/homebrew/bin/python3'],
    ['/usr/local/bin/python3'],
    ['/usr/bin/python3'],
  ];
}

function probePython(command, prefixArgs) {
  try {
    const result = spawnSync(command, [...prefixArgs, '--version'], {
      stdio: 'pipe',
      timeout: 5000,
      env: process.env,
    });
    const output = `${(result.stdout || '').toString()} ${(result.stderr || '').toString()}`;
    return { status: result.status, version: parsePythonVersion(output) };
  } catch (error) {
    return { status: null, version: null, error };
  }
}

function findSupportedPython() {
  const override = String(process.env.SLM_PYTHON || '').trim();
  if (override) {
    // The user named this interpreter explicitly (docs/getting-started.md's
    // escape hatch for a system where no supported `python3` is on PATH, for
    // example a deadsnakes or pyenv install at a custom path). Honour it,
    // but still version-check it — and if it fails, say so plainly instead
    // of silently trying a different interpreter the user did not choose.
    const probe = probePython(override, []);
    if (probe.status === 0 && isSupportedPython(probe.version)) {
      return { command: override, prefixArgs: [], version: probe.version };
    }
    const detail = probe.status === 0 && probe.version
      ? `found Python ${probe.version.join('.')}, which is not supported (need 3.12-3.14)`
      : 'could not be run (check the path and that it is executable)';
    console.error(`SuperLocalMemory: SLM_PYTHON=${override} ${detail}.`);
    console.error('Point SLM_PYTHON at a Python 3.12-3.14 interpreter, or unset it to let the installer search automatically.');
    return null;
  }

  for (const candidate of pythonCandidates()) {
    const probe = probePython(candidate[0], candidate.slice(1));
    if (probe.status === 0 && isSupportedPython(probe.version)) {
      return { command: candidate[0], prefixArgs: candidate.slice(1), version: probe.version };
    }
  }
  return null;
}

/**
 * Is an NVIDIA GPU visible on this host? Linux-only signal: the three cheap,
 * side-effect-free checks pip itself has no equivalent for. False on any
 * error — a check that cannot tell is not evidence of a GPU.
 * @param {string} platform
 * @returns {boolean}
 */
function hasNvidiaGpu(platform = os.platform()) {
  if (platform !== 'linux') return false;
  if (fs.existsSync('/proc/driver/nvidia/version') || fs.existsSync('/dev/nvidia0')) {
    return true;
  }
  try {
    const probe = spawnSync('nvidia-smi', [], { stdio: 'ignore', timeout: 2000 });
    return probe.status === 0;
  } catch {
    return false;
  }
}

/**
 * Should this install pre-fetch the CPU-only torch wheel before the
 * main package? Mirrors plugin-src/scripts/torch-cpu-resolve.sh's
 * `torch_cpu_should_force` (same questions, same answers, separate runtime).
 * False covers: not Linux, a GPU is visible, or the user already steered pip
 * (their own index) or torch (SLM_TORCH_BACKEND set to anything but "cpu")
 * themselves.
 * @param {NodeJS.ProcessEnv} env
 * @param {string} platform
 * @param {(platform: string) => boolean} gpuCheck injected for testing
 * @returns {boolean}
 */
function shouldForceCpuTorch(env = process.env, platform = os.platform(), gpuCheck = hasNvidiaGpu) {
  if (platform !== 'linux') return false;

  const backend = String(env.SLM_TORCH_BACKEND || '').trim();
  if (backend === 'cpu') return true;
  if (backend) return false; // cuda / cuXXX / auto / ... — the user already decided

  if (env.PIP_INDEX_URL || env.PIP_EXTRA_INDEX_URL) return false;

  return !gpuCheck(platform);
}

/**
 * Read the `torch==X.Y.Z` pin shipped at plugin/requirements-cpu-torch.txt
 * (kept equal to pyproject.toml's torch pin by
 * tests/test_plugin_src/test_torch_cpu_pin_matches_pyproject.py). Returns
 * null — fail-open — when the file is missing or has no such line, so a
 * missing/stale pin file can never block an install that would otherwise
 * succeed; it only means this optimization is skipped for this run.
 * @param {string} packageRoot
 * @returns {string | null}
 */
function cpuTorchPin(packageRoot) {
  const pinFile = path.join(packageRoot, 'plugin', 'requirements-cpu-torch.txt');
  let content;
  try {
    content = fs.readFileSync(pinFile, 'utf8');
  } catch {
    return null;
  }
  const match = content.match(/^torch==\S+/m);
  return match ? match[0].trim() : null;
}

function runtimePythonPath(packageRoot, platform = os.platform()) {
  return platform === 'win32'
    ? path.join(packageRoot, '.slm-venv', 'Scripts', 'python.exe')
    : path.join(packageRoot, '.slm-venv', 'bin', 'python');
}

/**
 * Single source of truth for the Python payload (4.1.14 #134, Option A).
 *
 * The npm tarball carries NO Python sources. The package-owned venv is
 * populated from the pinned PyPI wheel `superlocalmemory==<npm version>`,
 * so there is exactly one copy of the package and patching any other
 * tree cannot shadow it. `SLM_LOCAL_WHEEL=/path/to/file.whl` overrides
 * the specifier for air-gapped installs: it must be an existing local
 * `.whl` file (directories, sdists and URLs are refused) and it still
 * passes the version identity check below. Note the boundary honestly:
 * the wheel itself comes from the file, but pip still resolves its
 * *dependencies* from the index unless PIP_FIND_LINKS/PIP_NO_INDEX say
 * otherwise — never a source tree, but not fully offline either.
 */
function pypiSpecifier(packageRoot) {
  const rawWheel = String(process.env.SLM_LOCAL_WHEEL || '').trim();
  if (rawWheel) {
    // 4.1.14 audit: resolve relative paths against the package root (not
    // the caller's cwd), require an existing FILE (a directory named
    // *.whl is refused), and require the .whl suffix (sdists, URLs and
    // directories cannot pass).
    const localWheel = path.isAbsolute(rawWheel)
      ? rawWheel
      : path.resolve(packageRoot, rawWheel);
    let isFile = false;
    try {
      isFile = fs.statSync(localWheel).isFile();
    } catch {
      isFile = false;
    }
    if (!localWheel.toLowerCase().endsWith('.whl') || !isFile) {
      throw new Error(
        `SLM_LOCAL_WHEEL must be an existing local .whl file, got: ${rawWheel}`,
      );
    }
    return localWheel;
  }
  const packageVersion = require(path.join(packageRoot, 'package.json')).version;
  return `superlocalmemory==${packageVersion}`;
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

function pythonGuidance(platform = os.platform()) {
  const lines = ['', 'SuperLocalMemory requires Python 3.12, 3.13, or 3.14.'];
  if (platform === 'darwin') {
    lines.push('Install it with Homebrew:  brew install python@3.13');
    lines.push('or the installer from https://www.python.org/downloads/macos/');
  } else if (platform === 'win32') {
    lines.push('Install it with:  winget install Python.Python.3.13');
    lines.push('or from https://www.python.org/downloads/windows/ (the py launcher is found automatically).');
  } else if (platform === 'linux') {
    lines.push('Debian 13 / Ubuntu 24.04 and later:  sudo apt install python3 python3-venv');
    lines.push('Ubuntu 22.04:  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv');
    lines.push('Fedora / RHEL 9:  sudo dnf install python3.12');
  } else {
    lines.push('Install it from https://www.python.org/downloads/');
  }
  lines.push('Then finish the install:  npm rebuild -g superlocalmemory');
  lines.push('(or set SLM_PYTHON to the interpreter if it is somewhere unusual).');
  lines.push('The npm installer creates a private virtual environment;');
  lines.push('it never installs packages into your system Python.');
  return lines;
}

// Machines SuperLocalMemory cannot run on: an Intel (x86_64) Python on macOS (the
// pinned security library, cryptography 50, has no Intel Mac build), a 32-bit Python
// on Windows or Linux, and Windows on ARM. pip would fail deep in dependency
// resolution instead. An Intel answer on a Mac can be false: a universal2 Python
// started by a Node that runs translated (Rosetta) reports x86_64 although the Mac
// is Apple Silicon, so `translated` (see nodeRunsTranslated) changes the advice.
function unsupportedMachine(platform, machine, is64, translated = false) {
  const m = String(machine || '').toLowerCase();
  if (platform === 'darwin' && m === 'x86_64') {
    if (translated === true) {
      return [
        'This Node.js is running as an Intel program on an Apple Silicon Mac (translated by Rosetta),',
        'so the Python it starts reports Intel too. SuperLocalMemory needs everything to run natively.',
        'Install the Apple Silicon (arm64) Node.js (nodejs.org installer, or Homebrew in /opt/homebrew),',
        'open a new terminal, and run Node natively, then: npm rebuild -g superlocalmemory',
      ].join('\n');
    }
    return [
      'This Python is an Intel (x86_64) build, and SuperLocalMemory needs an Apple Silicon (arm64) one.',
      'On an Apple Silicon Mac, install the arm64 Python (Homebrew in /opt/homebrew: brew install python@3.13,',
      'or the python.org installer run natively, not under Rosetta), then: npm rebuild -g superlocalmemory',
      'Intel Macs are not supported: the security library SuperLocalMemory pins has no Intel Mac build.',
    ].join('\n');
  }
  if (platform === 'win32' && m === 'arm64') {
    return 'Windows on ARM is not supported yet: the security library SuperLocalMemory pins has no '
      + 'Windows ARM build. Use a 64-bit Intel or AMD Windows computer, or WSL with an x86_64 Linux.';
  }
  if (platform === 'win32' && is64 === false) {
    return 'This Python is 32-bit. SuperLocalMemory needs 64-bit Python 3.12-3.14 on Windows '
      + '(winget install Python.Python.3.13), then: npm rebuild -g superlocalmemory';
  }
  if (platform === 'linux' && (is64 === false || ['i386', 'i486', 'i586', 'i686', 'armv6l', 'armv7l'].includes(m))) {
    return 'This computer or this Python is 32-bit. SuperLocalMemory needs a 64-bit Linux '
      + '(x86_64 or aarch64) with 64-bit Python 3.12-3.14, then: npm rebuild -g superlocalmemory';
  }
  return null;
}

function pythonMachine(python) {
  try {
    const result = spawnSync(python.command, [...python.prefixArgs, '-c',
      'import platform, sys; print(sys.platform, platform.machine(), sys.maxsize > 2**32)'], {
      stdio: 'pipe', timeout: 5000, env: process.env,
    });
    const [platform, machine, wide] = String(result.stdout || '').trim().split(/\s+/);
    return { platform: platform || '', machine: machine || '', is64: wide === 'True' ? true : wide === 'False' ? false : null };
  } catch (_) {
    return { platform: '', machine: '', is64: null };
  }
}

function printPythonGuidance() {
  for (const line of pythonGuidance()) console.error(line);
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
  const machine = pythonMachine(python);
  const refusal = unsupportedMachine(machine.platform, machine.machine, machine.is64, nodeRunsTranslated());
  if (refusal) {
    console.error('');
    console.error(refusal);
    return 1;
  }

  const locationCheck = validateRuntimeLocation(venvRoot);
  if (!locationCheck.ok) {
    console.error(`SuperLocalMemory: refusing unsafe runtime location: ${locationCheck.error}.`);
    console.error('Move that path aside manually, verify it contains no needed files, then run:');
    console.error('  npm rebuild -g superlocalmemory');
    return 1;
  }

  console.log(`  Python ${python.version.join('.')} (${[python.command, ...python.prefixArgs].join(' ')})`);

  // Fail fast on a bad package source BEFORE creating any venv, so a
  // misconfigured SLM_LOCAL_WHEEL never leaves a half-built runtime behind.
  let packageSource;
  try {
    packageSource = pypiSpecifier(packageRoot);
  } catch (error) {
    console.error(`SuperLocalMemory: ${error.message}`);
    console.error('Unset SLM_LOCAL_WHEEL to install from PyPI, or point it at a real wheel file.');
    return 1;
  }

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
    console.error('  npm rebuild -g superlocalmemory');
    return 1;
  }

  const runtimePython = runtimePythonPath(packageRoot);

  // Pre-install the pinned CPU-only torch wheel so the main install
  // below finds it already satisfied and never resolves the CUDA build.
  // Best-effort: a failure here just means this run keeps the default
  // resolution (and therefore the CUDA wheels on Linux) — it must never be
  // the reason the whole install fails.
  if (shouldForceCpuTorch()) {
    const torchPin = cpuTorchPin(packageRoot);
    if (torchPin) {
      console.log(`  Linux, no GPU detected — installing ${torchPin} from ${TORCH_CPU_INDEX_URL}`);
      console.log('  (set SLM_TORCH_BACKEND=cuda to opt out).');
      const installTorch = spawnSync(
        runtimePython,
        [
          '-m', 'pip', 'install',
          '--disable-pip-version-check',
          '--no-input',
          '--index-url', TORCH_CPU_INDEX_URL,
          torchPin,
        ],
        { stdio: 'inherit', timeout: INSTALL_TIMEOUT_MS, env: process.env },
      );
      if (installTorch.status !== 0) {
        console.error(
          `SuperLocalMemory: CPU-only torch pre-install failed (${failureDetail(installTorch)}); `
          + 'continuing with the default index.',
        );
      }
    }
  }

  const installPackage = spawnSync(
    runtimePython,
    [
      '-m', 'pip', 'install',
      '--disable-pip-version-check',
      '--no-input',
      '--upgrade',
      packageSource,
    ],
    { stdio: 'inherit', timeout: INSTALL_TIMEOUT_MS, env: process.env },
  );
  if (installPackage.status !== 0) {
    console.error(`SuperLocalMemory: private-runtime installation failed (${failureDetail(installPackage)}).`);
    console.error(`Tried Python source: ${packageSource}`);
    console.error('This install needs network access to PyPI (or set SLM_LOCAL_WHEEL=/path/to/superlocalmemory-<version>-py3-none-any.whl for air-gapped installs).');
    console.error('Check network access, available disk space, and Python build prerequisites, then run:');
    console.error('  npm rebuild -g superlocalmemory');
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
  // 4.1.14 audit: normalize across npm-semver and PEP 440 spellings
  // ("4.1.14-rc.1" vs "4.1.14rc1") before comparing — a strict strcmp
  // would fail a good wheel on pre-releases.
  const normalizeVersion = (value) => String(value || '')
    .trim().toLowerCase().replace(/^v/, '')
    .replace(/[-_.]+/g, '');
  if (
    verify.status !== 0
    || normalizeVersion(installedVersion) !== normalizeVersion(packageVersion)
  ) {
    console.error(
      `SuperLocalMemory: runtime identity check failed (npm=${packageVersion}, python=${installedVersion || 'unavailable'}).`,
    );
    console.error('Run `npm rebuild -g superlocalmemory` to repair the package-owned runtime.');
    return 1;
  }

  console.log(`SuperLocalMemory ${packageVersion}: isolated runtime verified.`);
  console.log(`Single source of truth for this npm installation: the PyPI wheel superlocalmemory==${packageVersion} in .slm-venv;`);
  console.log('the npm tarball carries no Python sources, so no second copy can shadow it.');
  console.log('(The Claude Code plugin keeps its own separate pinned runtime for its host;');
  console.log(' that scope is independent of this npm installation — see plugin/requirements.txt.)');
  console.log('No memory database, IDE hooks, daemon, configuration, or models were changed.');
  console.log('');
  console.log('  Your database will be automatically migrated on first run.');
  console.log('  A backup is created in ~/.superlocalmemory/pre-migration-snapshots/ before');
  console.log('  any migration runs. No action is needed from you.');
  console.log('');
  console.log('  Next step — run the guided setup (picks your mode, downloads');
  console.log('  models, connects your IDEs). Takes about a minute:');
  console.log('');
  console.log('      slm setup');
  console.log('');
  console.log('  If this updated an existing SLM installation, preview host integrations first:');
  console.log('');
  console.log('      slm upgrade-hosts');
  console.log('');
  console.log('  Prefer to tune performance profiles? Use:  slm reconfigure');
  printWhatsNew();
  console.log('');
  return 0;
}

if (require.main === module) {
  const code = main();
  const done = () => process.exit(code);
  if (code !== 0) done();
  else runMediaStep({ argv: process.argv.slice(2) }).then(() => runUpgradeStep()).then(done, done);
}

module.exports = {
  cpuTorchPin,
  findSupportedPython,
  hasNvidiaGpu,
  isSupportedPython,
  main,
  runMediaStep,
  parsePythonVersion,
  probePython,
  pypiSpecifier,
  pythonCandidates,
  pythonGuidance,
  unsupportedMachine,
  nodeRunsTranslated,
  runtimePythonPath,
  shouldForceCpuTorch,
  TORCH_CPU_INDEX_URL,
  validateRuntimeLocation,
};
