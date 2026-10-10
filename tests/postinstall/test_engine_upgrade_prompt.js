/**
 * "Upgrade memory engine" at install time: off by default behind a constant, never asks off a
 * terminal or in CI, defaults to No, only records a request (never downloads).
 */

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const MODULE = path.join(REPO_ROOT, 'scripts', 'postinstall', 'engine-upgrade.js');
const upgrade = require(MODULE);

function store({ provider = 'sentence-transformers', db = true, config = true } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-upgrade-'));
  if (db) fs.writeFileSync(path.join(dir, 'memory.db'), '');
  if (config) fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify({ embedding: { provider } }));
  return dir;
}
const featuresOf = (dir) => path.join(dir, 'features.json');
const quiet = () => {};
const base = (dir, over = {}) => Object.assign({
  enabled: true, env: {}, slmDir: dir, tty: true, ask: async () => true, log: quiet,
}, over);

test('the prompt is switched off by default', async () => {
  assert.equal(upgrade.UPGRADE_PROMPT_ENABLED, false);
  const dir = store();
  let asked = 0;
  const result = await upgrade.runUpgradeStep({ env: {}, slmDir: dir, tty: true, log: quiet,
    ask: async () => { asked += 1; return true; } });
  assert.equal(result, false);
  assert.equal(asked, 0);
  assert.equal(fs.existsSync(featuresOf(dir)), false);
});

test('enabled + existing store + terminal: a yes records both requests and nothing else', async () => {
  const dir = store();
  assert.equal(await upgrade.runUpgradeStep(base(dir)), true);
  const saved = JSON.parse(fs.readFileSync(featuresOf(dir), 'utf8'));
  assert.equal(saved.engine_upgrade.requested, true);
  assert.equal(saved.engine_upgrade.choice_source, 'npm');
  assert.ok(saved.engine_upgrade.requested_at);
  assert.equal(saved.media.requested, true);
  assert.ok(!saved.media.enabled);
  assert.deepEqual(fs.readdirSync(dir).sort(), ['config.json', 'features.json', 'memory.db']);
});

test('the question defaults to No: a no (or anything but yes) records nothing', async () => {
  const dir = store();
  const asked = [];
  const result = await upgrade.runUpgradeStep(base(dir, { ask: async (q) => { asked.push(q); return false; } }));
  assert.equal(result, false);
  assert.match(asked[0], /\[y\/N\]/);
  assert.equal(fs.existsSync(featuresOf(dir)), false);
});

test('off a terminal it never asks and records nothing', async () => {
  const dir = store();
  let asked = 0;
  assert.equal(await upgrade.runUpgradeStep(base(dir, { tty: false, ask: async () => { asked += 1; return true; } })), false);
  assert.equal(asked, 0);
  assert.equal(fs.existsSync(featuresOf(dir)), false);
});

test('in CI it never asks even on a terminal', async () => {
  for (const ci of ['true', '1']) {
    const dir = store();
    let asked = 0;
    const result = await upgrade.runUpgradeStep(base(dir, { env: { CI: ci }, ask: async () => { asked += 1; return true; } }));
    assert.equal(result, false);
    assert.equal(asked, 0);
  }
});

test('a new install (no memory database) is never asked', async () => {
  const dir = store({ db: false, config: false });
  let asked = 0;
  assert.equal(await upgrade.runUpgradeStep(base(dir, { ask: async () => { asked += 1; return true; } })), false);
  assert.equal(asked, 0);
});

test('a store that already uses the managed model is never asked', async () => {
  const dir = store({ provider: 'slm-media' });
  let asked = 0;
  assert.equal(await upgrade.runUpgradeStep(base(dir, { ask: async () => { asked += 1; return true; } })), false);
  assert.equal(asked, 0);
});

test('an unreadable config.json still counts as an existing store with another model', () => {
  const dir = store({ config: false });
  fs.writeFileSync(path.join(dir, 'config.json'), '{broken');
  assert.equal(upgrade.shouldOffer(dir), true);
});

test('an existing features.json is merged, not replaced; unreadable ones are left alone', async () => {
  const dir = store();
  fs.writeFileSync(featuresOf(dir), JSON.stringify({ schema: 1, sources: { enabled: true } }));
  assert.equal(await upgrade.runUpgradeStep(base(dir)), true);
  const saved = JSON.parse(fs.readFileSync(featuresOf(dir), 'utf8'));
  assert.equal(saved.sources.enabled, true);
  assert.equal(saved.engine_upgrade.requested, true);

  const bad = store();
  fs.writeFileSync(featuresOf(bad), '{broken');
  assert.equal(await upgrade.runUpgradeStep(base(bad)), false);
  assert.equal(fs.readFileSync(featuresOf(bad), 'utf8'), '{broken');
});

test('plain words, no download, no network in the module', () => {
  const source = fs.readFileSync(MODULE, 'utf8');
  assert.doesNotMatch(source, /require\(['"](https?|net|child_process)['"]\)|fetch\(/);
  const lines = [];
  upgrade.printUpgradeOffer((l) => lines.push(l));
  const text = lines.join('\n');
  assert.match(text, /recall keeps working/i);
  assert.match(text, /roll back/i);
  assert.match(text, /slm embedder upgrade/);
  assert.doesNotMatch(text, /re-?index/i);
});

test('the step is chained after the media step in the npm postinstall', () => {
  const source = fs.readFileSync(path.join(REPO_ROOT, 'scripts', 'postinstall.js'), 'utf8');
  assert.match(source, /engine-upgrade\.js/);
  assert.ok(source.indexOf('runMediaStep({ argv') < source.indexOf('runUpgradeStep()'));
});
