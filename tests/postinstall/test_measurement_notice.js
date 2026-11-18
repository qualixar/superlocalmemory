/**
 * The install/upgrade measurement notice (D-28): printed to stderr when pending,
 * silent once acknowledged, never able to fail an install.
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const notice = require(path.join(REPO_ROOT, 'scripts', 'postinstall', 'measurement-notice.js'));

const run = (stdout, status = 0, error = null) => () => ({ stdout, status, error });

test('a pending notice is printed line for line', () => {
  const lines = [];
  const printed = notice.printMeasurementNotice('python', {
    log: (l) => lines.push(l), spawn: run('first line\nsecond line\n'),
  });
  assert.equal(printed, true);
  assert.deepEqual(lines, ['', 'first line\nsecond line']);
});

test('an acknowledged notice prints nothing', () => {
  const lines = [];
  assert.equal(notice.printMeasurementNotice('python', { log: (l) => lines.push(l), spawn: run('') }), false);
  assert.deepEqual(lines, []);
});

test('a runtime that cannot answer gets the one-line fallback, not a failure', () => {
  for (const spawn of [run('', 1), run('', 0, new Error('ENOENT')), () => { throw new Error('boom'); }]) {
    const lines = [];
    let printed;
    assert.doesNotThrow(() => { printed = notice.printMeasurementNotice('python', { log: (l) => lines.push(l), spawn }); });
    if (printed) assert.match(lines.join('\n'), /slm measurement off/);
  }
});

test('the install script calls it after the runtime is verified, and the package ships it', () => {
  const installer = fs.readFileSync(path.join(REPO_ROOT, 'scripts', 'postinstall.js'), 'utf8');
  assert.match(installer, /printMeasurementNotice\(runtimePython\)/);
  const pkg = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf8'));
  assert.ok(pkg.files.includes('scripts/postinstall/measurement-notice.js'));
});

test('the real runtime text comes from the Python disclosure module', () => {
  const { spawnSync } = require('node:child_process');
  const py = path.join(REPO_ROOT, '.venv', 'bin', 'python');
  if (!fs.existsSync(py)) return;  // dev venv only
  const dir = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'slm-notice-'));
  const env = Object.assign({}, process.env, { SLM_DATA_DIR: dir });
  const result = spawnSync(py, ['-c', notice.SNIPPET], { env, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /slm measurement off/);
});
