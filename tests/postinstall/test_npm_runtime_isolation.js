/**
 * NPM runtime isolation contract.
 *
 * User journey: as an npm user, I want npm to install SLM into a
 * package-owned Python virtual environment, so that installation never mutates
 * the system Python, durable memory, or external IDE configuration.
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const childProcess = require('node:child_process');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const POSTINSTALL = path.join(REPO_ROOT, 'scripts', 'postinstall.js');
const PREUNINSTALL = path.join(REPO_ROOT, 'scripts', 'preuninstall.js');
const INTERACTIVE_CONFIG = path.join(REPO_ROOT, 'scripts', 'postinstall-interactive.js');
const WRAPPER = path.join(REPO_ROOT, 'bin', 'slm-npm');
const PACKAGE_VERSION = require(path.join(REPO_ROOT, 'package.json')).version;

function clearModule(modulePath) {
  delete require.cache[require.resolve(modulePath)];
}

function capturePostinstall() {
  const calls = [];
  const mkdirs = [];
  const originalSpawnSync = childProcess.spawnSync;
  const originalExistsSync = fs.existsSync;
  const originalMkdirSync = fs.mkdirSync;

  childProcess.spawnSync = (command, args = [], options = {}) => {
    calls.push({ command, args: [...args], options });
    if (args.includes('--version')) {
      return { status: 0, stdout: Buffer.from('Python 3.12.8\n'), stderr: Buffer.from('') };
    }
    if (args.includes('-c')) {
      return { status: 0, stdout: Buffer.from(PACKAGE_VERSION + '\n'), stderr: Buffer.from('') };
    }
    return { status: 0, stdout: Buffer.from(''), stderr: Buffer.from('') };
  };
  fs.existsSync = () => false;
  fs.mkdirSync = (target, options) => mkdirs.push({ target, options });

  try {
    clearModule(POSTINSTALL);
    const moduleExports = require(POSTINSTALL);
    if (typeof moduleExports.main === 'function') {
      const result = moduleExports.main();
      assert.equal(result, 0);
    }
  } finally {
    childProcess.spawnSync = originalSpawnSync;
    fs.existsSync = originalExistsSync;
    fs.mkdirSync = originalMkdirSync;
    clearModule(POSTINSTALL);
  }

  return { calls, mkdirs };
}

function captureWrapper(platform) {
  const calls = [];
  const mkdirs = [];
  const originalSpawnSync = childProcess.spawnSync;
  const originalExistsSync = fs.existsSync;
  const originalMkdirSync = fs.mkdirSync;
  const originalPlatform = os.platform;
  const originalArgv = process.argv;
  const originalExit = process.exit;

  class ExitSignal extends Error {
    constructor(code) {
      super(`exit ${code}`);
      this.code = code;
    }
  }

  os.platform = () => platform;
  fs.existsSync = (target) => String(target).includes('.slm-venv');
  fs.mkdirSync = (target, options) => mkdirs.push({ target, options });
  childProcess.spawnSync = (command, args = [], options = {}) => {
    calls.push({ command, args: [...args], options });
    return { status: 0, stdout: Buffer.from(''), stderr: Buffer.from('') };
  };
  process.argv = [process.execPath, WRAPPER, 'status'];
  process.exit = (code) => { throw new ExitSignal(code); };

  try {
    clearModule(WRAPPER);
    assert.throws(
      () => require(WRAPPER),
      (error) => error instanceof ExitSignal && error.code === 0,
    );
  } finally {
    childProcess.spawnSync = originalSpawnSync;
    fs.existsSync = originalExistsSync;
    fs.mkdirSync = originalMkdirSync;
    os.platform = originalPlatform;
    process.argv = originalArgv;
    process.exit = originalExit;
    clearModule(WRAPPER);
  }

  return { calls, mkdirs };
}

test('postinstall creates only a package-owned venv and installs through its pip', () => {
  const { calls, mkdirs } = capturePostinstall();
  const packageVenv = path.join(REPO_ROOT, '.slm-venv');

  assert.equal(mkdirs.length, 0, 'postinstall must not create the durable data root');
  assert.ok(
    calls.some(({ args }) => args.includes('-m') && args.includes('venv') && args.includes(packageVenv)),
    'postinstall must create the package-owned venv',
  );

  const pipCalls = calls.filter(({ args }) => args.includes('-m') && args.includes('pip'));
  assert.equal(pipCalls.length, 1, 'postinstall should perform one isolated package install');
  assert.ok(
    String(pipCalls[0].command).includes('.slm-venv'),
    'pip must be invoked through the package-owned venv Python',
  );
  assert.equal(pipCalls[0].args.includes('--user'), false);
  assert.equal(pipCalls[0].args.includes('--break-system-packages'), false);
});

test('postinstall never auto-runs setup, hooks, daemon, or model downloads', () => {
  const { calls } = capturePostinstall();
  const flattened = calls.flatMap(({ args }) => args).join(' ');

  assert.doesNotMatch(flattened, /hooks\s+install/);
  assert.doesNotMatch(flattened, /\bsetup\b/);
  assert.doesNotMatch(flattened, /\bserve\b/);
  assert.doesNotMatch(flattened, /sentence_transformers|onnxruntime/);
});

test('npm wrapper executes the package-owned POSIX venv Python without data writes', () => {
  const { calls, mkdirs } = captureWrapper('darwin');

  assert.equal(mkdirs.length, 0);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, path.join(REPO_ROOT, '.slm-venv', 'bin', 'python'));
  assert.deepEqual(calls[0].args.slice(0, 2), ['-m', 'superlocalmemory.cli.main']);
  assert.equal(Object.hasOwn(calls[0].options.env, 'PYTHONPATH'), false);
});

test('npm wrapper selects the Windows venv interpreter', () => {
  const { calls, mkdirs } = captureWrapper('win32');

  assert.equal(mkdirs.length, 0);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, path.join(REPO_ROOT, '.slm-venv', 'Scripts', 'python.exe'));
});

test('shipped npm lifecycle has no protected-Python escape hatch', () => {
  const source = fs.readFileSync(POSTINSTALL, 'utf8');

  assert.doesNotMatch(source, /--break-system-packages/);
  assert.doesNotMatch(source, /['"]--user['"]/);
  assert.doesNotMatch(source, /hooks['"],\s*['"]install/);
  assert.doesNotMatch(source, /cli\.main['"],\s*['"]setup/);
});

test('postinstall rejects a symlinked private-runtime directory', (t) => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-npm-symlink-'));
  const externalDirectory = path.join(temporaryRoot, 'external');
  const runtimeLink = path.join(temporaryRoot, '.slm-venv');
  fs.mkdirSync(externalDirectory);
  fs.symlinkSync(externalDirectory, runtimeLink, 'dir');
  t.after(() => {
    fs.unlinkSync(runtimeLink);
    fs.rmdirSync(externalDirectory);
    fs.rmdirSync(temporaryRoot);
  });

  clearModule(POSTINSTALL);
  const { validateRuntimeLocation } = require(POSTINSTALL);
  const result = validateRuntimeLocation(runtimeLink);

  assert.equal(result.ok, false);
  assert.match(result.error, /symbolic link/i);
});

test('postinstall help is read-only and does not bootstrap the runtime', () => {
  const originalSpawnSync = childProcess.spawnSync;
  const calls = [];
  childProcess.spawnSync = (command, args) => {
    calls.push({ command, args });
    return { status: 0, stdout: Buffer.from(''), stderr: Buffer.from('') };
  };
  try {
    clearModule(POSTINSTALL);
    const { main } = require(POSTINSTALL);
    assert.equal(main(['--help']), 0);
  } finally {
    childProcess.spawnSync = originalSpawnSync;
    clearModule(POSTINSTALL);
  }
  assert.equal(calls.length, 0);
});

test('preuninstall removes code only and never inspects or advertises data deletion', () => {
  const source = fs.readFileSync(PREUNINSTALL, 'utf8');

  assert.doesNotMatch(source, /SLM_DATA_DIR|SL_MEMORY_PATH|SLM_HOME/);
  assert.doesNotMatch(source, /\.superlocalmemory/);
  assert.doesNotMatch(source, /rm\s+-rf|rmdir\s+\/s|Remove-Item/i);
  assert.match(source, /memory data is preserved/i);
});

test('interactive profile configuration does not silently install external integrations', () => {
  const source = fs.readFileSync(INTERACTIVE_CONFIG, 'utf8');

  assert.doesNotMatch(source, /plugin['"],\s*['"]marketplace/);
  assert.doesNotMatch(source, /plugin['"],\s*['"]install/);
  assert.doesNotMatch(source, /hooks['"],\s*['"]install/);
  assert.doesNotMatch(source, /Plugin auto-install on npm\/pip install/);
});
