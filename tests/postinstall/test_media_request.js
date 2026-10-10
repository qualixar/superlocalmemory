/**
 * Images & documents at install time: the installer records a request, never
 * downloads, never prompts off a terminal, and the banner tells what is new.
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { spawnSync } = require('node:child_process');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const INSTALLER = path.join(REPO_ROOT, 'scripts', 'postinstall-interactive.js');
const MODULE = path.join(REPO_ROOT, 'scripts', 'postinstall', 'media-request.js');
const PREUNINSTALL = path.join(REPO_ROOT, 'scripts', 'preuninstall.js');
const media = require(MODULE);

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'slm-media-'));
}

function runInstaller(home, extraArgs = [], extraEnv = {}) {
  const env = Object.assign({}, process.env, {
    CI: 'true',
    SLM_INSTALL_FREE_RAM_MB: '8192',
    SLM_INSTALL_COLD_START_MS: '150',
    SLM_INSTALL_DISK_FREE_GB: '250',
  }, extraEnv);
  delete env.SLM_ENABLE_MEDIA;
  delete env.SLM_DATA_DIR;
  delete env.SL_MEMORY_PATH;
  delete env.SLM_HOME;
  Object.assign(env, extraEnv);
  return spawnSync('node', [INSTALLER, `--home=${home}`, '--home-outside-home', ...extraArgs],
    { encoding: 'utf8', env, timeout: 30000 });
}

const featuresFile = (home) => path.join(home, '.superlocalmemory', 'features.json');

test('off a terminal with no flag: nothing is written and nothing is asked', () => {
  const home = tmp();
  const result = runInstaller(home);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(featuresFile(home)), false);
  for (const line of result.stdout.split('\n')) {
    assert.ok(!line.trimEnd().endsWith('?'), `prompted: ${line}`);
    assert.ok(!/\[y\/N\]/i.test(line), `prompted: ${line}`);
  }
});

test('SLM_ENABLE_MEDIA=1 records a request merged into an existing features.json', () => {
  const home = tmp();
  fs.mkdirSync(path.join(home, '.superlocalmemory'), { recursive: true });
  fs.writeFileSync(featuresFile(home),
    JSON.stringify({ schema: 1, other: { keep: 1 }, media: { choice_source: 'cli', note: 'x' } }));
  const result = runInstaller(home, [], { SLM_ENABLE_MEDIA: '1' });
  assert.equal(result.status, 0, result.stderr);
  const saved = JSON.parse(fs.readFileSync(featuresFile(home), 'utf8'));
  assert.equal(saved.other.keep, 1);
  assert.equal(saved.media.requested, true);
  assert.equal(saved.media.choice_source, 'npm');
  assert.equal(saved.media.note, 'x');
  assert.ok(saved.media.requested_at);
  assert.ok(!saved.media.enabled);
  if (process.platform !== 'win32') {
    assert.equal(fs.statSync(featuresFile(home)).mode & 0o777, 0o600);
  }
});

test('--media records the request on a fresh data folder', () => {
  const home = tmp();
  const result = runInstaller(home, ['--media']);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(fs.readFileSync(featuresFile(home), 'utf8')).media.requested, true);
});

test('a dry run records nothing', () => {
  const home = tmp();
  const result = runInstaller(home, ['--media', '--dry-run']);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(featuresFile(home)), false);
});

test('an unreadable features.json is left exactly as it was', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'features.json'), '{broken');
  const result = media.recordMediaRequest(dir);
  assert.equal(result.ok, false);
  assert.equal(fs.readFileSync(path.join(dir, 'features.json'), 'utf8'), '{broken');
});

test('an already-enabled feature is not turned into a request', () => {
  const dir = tmp();
  const before = JSON.stringify({ schema: 1, media: { enabled: true } });
  fs.writeFileSync(path.join(dir, 'features.json'), before);
  assert.equal(media.recordMediaRequest(dir).ok, true);
  assert.equal(fs.readFileSync(path.join(dir, 'features.json'), 'utf8'), before);
});

test('the terminal question defaults to No and only a yes records', async () => {
  const dir = tmp();
  const quiet = () => {};
  const asked = [];
  const no = async (q) => { asked.push(q); return false; };
  const yes = async () => true;
  assert.equal(await media.handleMediaChoice({ args: {}, env: {}, slmDir: dir, interactive: true, ask: no, log: quiet }), false);
  assert.equal(fs.existsSync(path.join(dir, 'features.json')), false);
  assert.match(asked[0], /\[y\/N\]/);
  assert.equal(await media.handleMediaChoice({ args: {}, env: {}, slmDir: dir, interactive: false, ask: yes, log: quiet }), false);
  assert.equal(await media.handleMediaChoice({ args: {}, env: {}, slmDir: dir, interactive: true, ask: yes, log: quiet }), true);
  assert.equal(JSON.parse(fs.readFileSync(path.join(dir, 'features.json'), 'utf8')).media.requested, true);
});

test('the banner is about 4.1.25 and has no v3.4 wording', () => {
  const lines = [];
  media.printWhatsNew((l) => lines.push(l));
  const text = lines.join('\n');
  assert.match(text, /slm media enable/);
  assert.match(text, /slm features/);
  assert.match(text, /mesh/i);
  assert.doesNotMatch(text, /v?3\.4|session_init|Living/i);
  assert.ok(lines.filter((l) => l.trim()).length <= 4);
  assert.equal(typeof require(INSTALLER).printWhatsNew, 'function');
  assert.equal(require(INSTALLER).printLivingBrainDelta, undefined);
});

test('the installer module cannot download anything', () => {
  const source = fs.readFileSync(MODULE, 'utf8');
  assert.doesNotMatch(source, /require\(['"](https?|net|child_process)['"]\)|fetch\(/);
});

test('preuninstall names where the media files are, their size and the one command, and deletes nothing', () => {
  const root = tmp();
  const mediaDir = path.join(root, 'runtimes', 'media');
  fs.mkdirSync(mediaDir, { recursive: true });
  const payload = path.join(mediaDir, 'weights.bin');
  fs.writeFileSync(payload, Buffer.alloc(3 * 1024 * 1024));
  const env = Object.assign({}, process.env, { SLM_DATA_DIR: root });
  const result = spawnSync('node', [PREUNINSTALL], { encoding: 'utf8', env });
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.includes(mediaDir));
  assert.match(result.stdout, /3 MB/);
  assert.match(result.stdout, /slm media disable --remove-files/);
  assert.match(result.stdout, /memory data is preserved/i);
  assert.ok(fs.existsSync(payload));
});

test('preuninstall says nothing about media when there is none', () => {
  const root = tmp();
  const env = Object.assign({}, process.env, { SLM_DATA_DIR: root });
  const result = spawnSync('node', [PREUNINSTALL], { encoding: 'utf8', env });
  assert.equal(result.status, 0, result.stderr);
  assert.doesNotMatch(result.stdout, /remove-files/);
});

// ---- npm's real hook: scripts/postinstall.js ----
const NPM_HOOK = path.join(REPO_ROOT, 'scripts', 'postinstall.js');

test('the npm hook records a request for SLM_ENABLE_MEDIA=1, and for npm_config_media', async () => {
  const hook = require(NPM_HOOK);
  for (const env of [{ SLM_ENABLE_MEDIA: '1' }, { npm_config_media: 'true' }]) {
    const dir = tmp();
    const out = [];
    const ok = await hook.runMediaStep({ argv: [], env: Object.assign({ SLM_DATA_DIR: dir }, env), tty: false, log: (l) => out.push(l) });
    assert.equal(ok, true);
    assert.equal(JSON.parse(fs.readFileSync(path.join(dir, 'features.json'), 'utf8')).media.requested, true);
  }
});

test('the npm hook honours --media in argv', async () => {
  const dir = tmp();
  const ok = await require(NPM_HOOK).runMediaStep({ argv: ['--media'], env: { SLM_DATA_DIR: dir }, tty: false, log: () => {} });
  assert.equal(ok, true);
});

test('the npm hook writes nothing and never asks off a terminal or in CI', async () => {
  const hook = require(NPM_HOOK);
  for (const [tty, env] of [[false, {}], [true, { CI: 'true' }]]) {
    const dir = tmp();
    let asked = 0;
    const ok = await hook.runMediaStep({ argv: [], env: Object.assign({ SLM_DATA_DIR: dir }, env), tty, ask: async () => { asked += 1; return true; }, log: () => {} });
    assert.equal(ok, false);
    assert.equal(asked, 0);
    assert.equal(fs.existsSync(path.join(dir, 'features.json')), false);
  }
});

test('the npm hook asks on a terminal, default No', async () => {
  const dir = tmp();
  const hook = require(NPM_HOOK);
  assert.equal(await hook.runMediaStep({ argv: [], env: { SLM_DATA_DIR: dir }, tty: true, ask: async () => false, log: () => {} }), false);
  assert.equal(fs.existsSync(path.join(dir, 'features.json')), false);
  assert.equal(await hook.runMediaStep({ argv: [], env: { SLM_DATA_DIR: dir }, tty: true, ask: async () => true, log: () => {} }), true);
});

test('the npm hook prints the what is new banner, from the shared module', () => {
  const source = fs.readFileSync(NPM_HOOK, 'utf8');
  assert.match(source, /printWhatsNew\(\)/);
  assert.match(source, /require\('\.\/postinstall\/media-request\.js'\)/);
  assert.equal(require(INSTALLER).printWhatsNew, media.printWhatsNew);
});
