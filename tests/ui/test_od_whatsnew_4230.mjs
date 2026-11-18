/**
 * The 4.2.30 "What memory can do now" card (§8.10.3 l): one plain line for each new feature that is
 * not yet turned on, each with a link; a feature that is on is not offered; it hides when nothing is
 * left, when dismissed, or when the state cannot be read.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-whatsnew-4230.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));
const KEY = 'slm.whatsnew.4.2.30';

const FLAGS = { need_gate: 'delivery.need_gate', cues: 'delivery.cues', measurement: null,
  origin: 'trust.taint', firewall: 'trust.taint', secrets: 'trust.secret_screen', signing: 'trust.signing',
  why: null, proposals: 'compiler.dream' };
const state = (on = [], kill = false) => ({ success: true, kill_switch: kill,
  features: Object.fromEntries(Object.entries(FLAGS).map(([k, f]) => [k, { on: on.includes(k), flag: f }])) });

async function page(body, { status = 200, dismissed = false, nav = [] } = {}) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="od-whatsnew-4230"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  if (dismissed) d.window.localStorage.setItem(KEY, '1');
  d.window.requested = [];
  d.window.fetch = (url) => {
    d.window.requested.push(url);
    return Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });
  };
  d.window.slmNavigate = (p) => nav.push(p);
  d.window.eval(SRC);
  d.host = d.window.document.getElementById('od-whatsnew-4230');
  await d.window.odWhatsNew4230.mount(d.host);
  await flush();
  return d;
}
const entries = (d) => [...d.host.querySelectorAll('li')].map((li) => li.getAttribute('data-entry'));

describe('What memory can do now card (4.2.30)', () => {
  it('is in the page right after the older What\'s new card, with its script stamped', () => {
    assert.match(HTML, /<div id="od-whatsnew"><\/div>\s*(<!--[^>]*-->\s*)?<div id="od-whatsnew-4230"><\/div>/);
    assert.match(HTML, /static\/js\/od-whatsnew-4230\.js\?v=[0-9a-f]{8}/);
  });

  it('offers all nine entries when nothing is on, one line each', async () => {
    const d = await page(state());
    assert.deepEqual(entries(d), ['need_gate', 'cues', 'measurement', 'origin', 'firewall', 'secrets',
      'signing', 'why', 'proposals']);
    assert.ok(d.window.requested.length >= 1 && d.window.requested.every((u) => u === '/api/whatsnew/state'));
    for (const li of d.host.querySelectorAll('li')) {
      assert.ok(li.querySelector('strong').textContent.length > 3, 'a short title');
      assert.equal(li.querySelectorAll('a, button').length, 1, 'exactly one link');
    }
    assert.match(d.host.textContent, /Memory on every prompt\./);
    assert.match(d.host.textContent, /Secrets stored as placeholders\./);
    assert.match(d.host.textContent, /Why was this shown\? Open any memory/);  // no doubled full stop
    assert.match(d.host.textContent, /Proposed changes\./);
  });

  it('does not offer what is already on', async () => {
    const d = await page(state(['need_gate', 'origin', 'firewall', 'signing']));
    assert.deepEqual(entries(d), ['cues', 'measurement', 'secrets', 'why', 'proposals']);
  });

  it('shows the plain switch for a feature with a flag and none for one without', async () => {
    const d = await page(state());
    const how = (id) => d.host.querySelector(`[data-entry="${id}"] .od-whatsnew-how`);
    assert.equal(how('need_gate').textContent.trim(), 'slm flags set delivery.need_gate on');
    assert.equal(how('signing').textContent.trim(), 'slm flags set trust.signing on');
    assert.equal(how('why'), null);
    assert.equal(how('measurement'), null);
  });

  it('links to a guide that opens safely, or to the page of this dashboard that has the card', async () => {
    const nav = [];
    const d = await page(state(), { nav });
    const a = d.host.querySelector('[data-entry="signing"] a');
    assert.equal(a.href, 'https://github.com/qualixar/superlocalmemory/blob/main/docs/signed-memories.md');
    assert.equal(a.target, '_blank');
    assert.match(a.rel, /noopener/);
    d.host.querySelector('[data-entry="measurement"] button').click();
    d.host.querySelector('[data-entry="why"] button').click();
    assert.deepEqual(nav, ['settings-pane', 'memories-pane']);
  });

  it('hides itself when everything is on', async () => {
    const d = await page(state(Object.keys(FLAGS)));
    assert.equal(d.host.textContent, '');
  });

  it('says the kill switch is on, and why nothing works yet', async () => {
    const d = await page(state([], true));
    assert.match(d.host.textContent, /The kill switch is on, so every new feature is off/);
  });

  it('dismisses for good in this browser, and then asks the server nothing', async () => {
    const d = await page(state());
    d.host.querySelector('button[aria-label="Dismiss"]').click();
    assert.equal(d.host.textContent, '');
    assert.equal(d.window.localStorage.getItem(KEY), '1');
    const again = await page(state(), { dismissed: true });
    assert.equal(again.host.textContent, '');
    assert.deepEqual(again.window.requested, []);
  });

  it('draws nothing when the state cannot be read (never offers what may already be on)', async () => {
    const d = await page({ detail: 'x' }, { status: 500 });
    assert.equal(d.host.textContent, '');
  });

  it('ids match the server\'s entries', () => {
    const py = readFileSync(join(here, '../../src/superlocalmemory/server/routes/whatsnew.py'), 'utf8');
    for (const id of Object.keys(FLAGS)) assert.ok(py.includes(`"${id}"`), id);
  });
});
