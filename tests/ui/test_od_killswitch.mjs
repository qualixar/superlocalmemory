/**
 * "Turn off everything new in 4.2.30" (D-23): the Settings card against a stubbed daemon. Plain
 * words, what it does and does not undo, one switch that acts at once, the bring-back button counts
 * first and asks, a refused change says why, nothing from the server parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-killswitch.js'), 'utf8');
const SETTINGS = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-settings.js'), 'utf8');
const INDEX = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 20));

const view = (on, source = on ? 'file' : 'off') => ({
  on, source,
  words: on ? 'Everything new in 4.2.30 is off.' : 'Everything new in 4.2.30 is working as set.' });

function body(over = {}) {
  return Object.assign({
    success: true, kill_switch: view(false), would_come_back: 2, profile_id: 'default',
    does: 'Stops everything new in 4.2.30 <img src=x onerror=alert(1)> and reads the way it did before.',
    does_not: 'It does not bring back memories that 4.2.30 already retired.' }, over);
}

function page({ initial = body(), refuse = 0, comeBack = 2 } = {}) {
  const dom = new JSDOM('<!DOCTYPE html><html><body><div id="card"></div></body></html>',
    { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  const w = dom.window;
  let current = initial;
  w.sent = [];
  w.toasts = [];
  w.showToast = (m) => w.toasts.push(m);
  w.fetch = (url, init) => {
    const method = (init && init.method) || 'GET';
    const reply = (ok, status, data) => Promise.resolve({ ok, status, json: () => Promise.resolve(data) });
    if (method === 'GET') return reply(true, 200, current);
    const sent = init.body ? JSON.parse(init.body) : {};
    w.sent.push({ url: String(url), method, body: sent });
    if (refuse) return reply(false, refuse, { detail: 'nope' });
    if (String(url).endsWith('/api/flags/kill')) {
      current = body({ ...current, kill_switch: view(sent.on) });
      return reply(true, 200, current);
    }
    if (sent.dry_run) return reply(true, 200, { success: true, dry_run: true, would_come_back: comeBack, came_back: 0 });
    current = body({ ...current, would_come_back: 0 });
    return reply(true, 200, { success: true, dry_run: false, would_come_back: comeBack, came_back: comeBack, refused: 0 });
  };
  w.eval(SRC);
  return w;
}

async function open(opts) {
  const w = page(opts);
  const card = w.document.getElementById('card');
  await w.SLMKillSwitch.renderInto(card);
  return { w, card };
}
const text = (c) => c.textContent;

describe('the kill switch card', () => {
  it('says what it does and does not undo, in words, and shows the state', async () => {
    const { card } = await open();
    assert.match(text(card), /Stops everything new in 4\.2\.30/);
    assert.match(text(card), /does not bring back memories that 4\.2\.30 already retired/);
    assert.match(card.querySelector('#od-kill-state').textContent, /working as set/);
    assert.equal(card.querySelector('#od-kill-switch').textContent, 'Turn off everything new in 4.2.30');
    assert.equal(card.querySelector('#od-kill-switch').getAttribute('aria-pressed'), 'false');
  });

  it('never parses a server string as HTML', async () => {
    const { card } = await open();
    assert.equal(card.querySelector('img'), null);
  });

  it('acts at once: one click turns it on, the button then offers to turn it back on', async () => {
    const { w, card } = await open();
    card.querySelector('#od-kill-switch').click();
    await flush();
    assert.deepEqual(w.sent, [{ url: '/api/flags/kill', method: 'POST', body: { on: true } }]);
    assert.match(card.querySelector('#od-kill-state').textContent, /is off/);
    assert.equal(card.querySelector('#od-kill-switch').textContent, 'Turn everything new back on');
    assert.equal(card.querySelector('#od-kill-switch').getAttribute('aria-pressed'), 'true');
    assert.match(card.querySelector('#od-kill-message').textContent, /before 4\.2\.30/);
    card.querySelector('#od-kill-switch').click();
    await flush();
    assert.deepEqual(w.sent[1].body, { on: false });
  });

  it('keeps it on and says why when the environment holds the switch', async () => {
    const held = body({ kill_switch: { on: true, source: 'environment',
      words: 'Everything new in 4.2.30 is off. The environment variable SLM_4230_OFF is set.' } });
    const { card } = await open({ initial: held });
    assert.match(card.querySelector('#od-kill-state').textContent, /SLM_4230_OFF/);
  });

  it('counts first, asks, then brings the memories back as one step', async () => {
    const { w, card } = await open();
    assert.match(text(card), /2 memories that 4\.2\.30 retired would come back/);
    card.querySelector('#od-kill-bring-back').click();
    await flush();
    assert.deepEqual(w.sent.map((s) => s.body), [{ dry_run: true }]);
    assert.match(text(card), /2 memories would come back\. Bring them back\?/);
    const yes = [...card.querySelectorAll('button')].find((b) => /Yes, bring them back/.test(b.textContent));
    yes.click();
    await flush();
    assert.deepEqual(w.sent.map((s) => s.body), [{ dry_run: true }, {}]);
    assert.match(card.querySelector('#od-kill-message').textContent, /Brought back 2 memories/);
    assert.match(card.querySelector('#od-kill-message').textContent, /Undo retires them again/);
    assert.equal(card.querySelector('#od-kill-bring-back').disabled, true);
  });

  it('Cancel changes nothing', async () => {
    const { w, card } = await open();
    card.querySelector('#od-kill-bring-back').click();
    await flush();
    [...card.querySelectorAll('button')].find((b) => b.textContent === 'Cancel').click();
    assert.deepEqual(w.sent.map((s) => s.body), [{ dry_run: true }]);
    assert.ok(card.querySelector('#od-kill-bring-back'));
  });

  it('has nothing to bring back and says so, with the button off', async () => {
    const { card } = await open({ initial: body({ would_come_back: 0 }) });
    assert.match(text(card), /Nothing that 4\.2\.30 retired is waiting to come back/);
    assert.equal(card.querySelector('#od-kill-bring-back').disabled, true);
  });

  it('says why a change is refused', async () => {
    const { card } = await open({ refuse: 403 });
    card.querySelector('#od-kill-switch').click();
    await flush();
    assert.match(card.querySelector('#od-kill-message').textContent, /Only the owner or an admin/);
    assert.match(card.querySelector('#od-kill-state').textContent, /working as set/);
  });

  it('asks to try again when the daemon cannot be read', async () => {
    const dom = new JSDOM('<div id="c"></div>', { runScripts: 'outside-only' });
    dom.window.fetch = () => Promise.reject(new Error('down'));
    dom.window.eval(SRC);
    const c = dom.window.document.getElementById('c');
    await dom.window.SLMKillSwitch.renderInto(c);
    assert.match(c.textContent, /Could not read the switch/);
  });
});

describe('the card is wired into Settings and the page', () => {
  it('Settings builds it and the page loads its script with a content stamp', () => {
    assert.match(SETTINGS, /buildKillSwitch\(\)/);
    assert.match(SETTINGS, /makeGrp\('Turn off everything new in 4\.2\.30'/);
    assert.match(INDEX, /static\/js\/od-killswitch\.js\?v=[0-9a-f]{8}/);
  });
});
