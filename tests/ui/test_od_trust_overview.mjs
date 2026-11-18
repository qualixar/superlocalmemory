/**
 * The Health page's "Trust overview" card (4.2.30 WS-5, §8.10.3 h): where memories came from, what waits for
 * the person's OK (with a link to that card), signed-memory state, secrets held as placeholders and
 * the audit log, in plain words. On demand, read-only, no memory text or credential ever shown.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-trust-overview.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const HEALTH = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-health.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));

const REPORT = {
  success: true,
  firewall: { state: 'on', waiting: 3, approved: 1, unclassified: 0 },
  secret_screen: { state: 'on', by_type: { github_token: 2, aws_key: 1 }, memories: 3 },
  origins: [{ class: 'user_direct', label: 'you said this', count: 40 },
            { class: 'web', label: 'a web page <b>x</b>', count: 7 }],
  signatures: { key_words: 'Signing is ready: the key is in your keychain.', facts: 100, signed: 90,
                unsigned: 10, checked: 90, modified: 0, moved_by_a_fold: 2, ok: true,
                words: '90 of 100 memories are signed. 90 checked: none changed since signing.' },
  audit_chain: { present: true, ok: true, words: 'The audit log is intact: 12 entries, order and links verified.' },
};

function page(report, { fail = false, navigated = [] } = {}) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="health"></div><div id="dash-trust-card"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.requested = [];
  d.window.fetch = (url) => {
    d.window.requested.push(url);
    if (fail) return Promise.reject(new Error('down'));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(report) });
  };
  d.window.slmNavigate = (p) => navigated.push(p);
  d.window.eval(SRC);
  d.window.odMountTrustOverview(d.window.document.getElementById('health'));
  return d;
}
const check = (d) => [...d.window.document.querySelectorAll('button')].find((b) => b.textContent === 'Check now');
const text = (d) => d.window.document.body.textContent;

describe('Trust overview card', () => {
  it('is mounted by the Health page and its script is stamped', () => {
    assert.match(HTML, /static\/js\/od-trust-overview\.js\?v=[0-9a-f]{8}/);
    assert.match(HEALTH, /odMountTrustOverview/);
  });

  it('reads nothing until asked, then the one route', async () => {
    const d = page(REPORT);
    assert.deepEqual(d.window.requested, []);
    check(d).click();
    await flush();
    assert.deepEqual(d.window.requested, ['/api/trust/overview']);
  });

  it('counts memories by where they came from, in words', async () => {
    const d = page(REPORT);
    check(d).click();
    await flush();
    assert.match(text(d), /You said this40/);
    assert.match(text(d), /A web page <b>x<\/b>7/);
    assert.equal(d.window.document.querySelectorAll('b').length, 0);
  });

  it('says how many wait for OK and links to that card', async () => {
    const nav = [];
    const d = page(REPORT, { navigated: nav });
    check(d).click();
    await flush();
    assert.match(text(d), /3 memories are held back; 1 promoted\./);
    assert.match(text(d), /The instruction firewall is on/);
    [...d.window.document.querySelectorAll('button')].find((b) => b.textContent === 'Review them').click();
    assert.deepEqual(nav, ['dashboard-pane']);
  });

  it('has no review link while the firewall is off, and says nothing is held back', async () => {
    const d = page({ ...REPORT, firewall: { state: 'off', waiting: 5, approved: 0 } });
    check(d).click();
    await flush();
    assert.match(text(d), /The instruction firewall is off, so nothing is held back/);
    assert.ok(![...d.window.document.querySelectorAll('button')].some((b) => b.textContent === 'Review them'));
  });

  it('shows signed, unsigned, changed and waiting counts and the key state', async () => {
    const d = page(REPORT);
    check(d).click();
    await flush();
    const t = text(d);
    assert.match(t, /Signing is ready: the key is in your keychain\./);
    assert.match(t, /Signed90Not signed yet10Changed since signing0Waiting to be signed again2/);
    assert.match(t, /none changed since signing\.OK/);
  });

  it('calls a changed memory a problem, never OK', async () => {
    const d = page({ ...REPORT, signatures: { ...REPORT.signatures, modified: 2, ok: false,
      words: '2 of the 90 memories checked no longer match their signature. They were changed outside SuperLocalMemory.' } });
    check(d).click();
    await flush();
    assert.match(text(d), /no longer match their signature.*Problem found/);
    assert.ok(d.window.document.querySelector('.tone-warn'));
  });

  it('names secrets by type without a value', async () => {
    const d = page(REPORT);
    check(d).click();
    await flush();
    assert.match(text(d), /3 memories hold a credential as a placeholder: github token 2, aws key 1\./);
    assert.match(text(d), /the original is not kept anywhere/);
  });

  it('says the secret screen is off and how to turn it on', async () => {
    const d = page({ ...REPORT, secret_screen: { state: 'off', by_type: {}, memories: 0 } });
    check(d).click();
    await flush();
    assert.match(text(d), /The secret screen is off, so a password or key you save is stored as written/);
    assert.match(text(d), /No memory holds a credential placeholder\./);
  });

  it('reports the audit log: intact, failed, and none yet', async () => {
    let d = page(REPORT);
    check(d).click();
    await flush();
    assert.match(text(d), /The audit log is intact: 12 entries.*OK/);
    d = page({ ...REPORT, audit_chain: { present: true, ok: false, words: 'The audit log failed its check at entry 7: bad link.' } });
    check(d).click();
    await flush();
    assert.match(text(d), /failed its check at entry 7.*Problem found/);
    d = page({ ...REPORT, audit_chain: { present: false, ok: null, words: 'There is no audit log next to this store yet.' } });
    check(d).click();
    await flush();
    assert.match(text(d), /no audit log next to this store yet\.No log yet/);
  });

  it('says what is not worked out yet when there are no origins', async () => {
    const d = page({ ...REPORT, origins: [] });
    check(d).click();
    await flush();
    assert.match(text(d), /Not worked out yet/);
  });

  it('says it could not read when the route fails, and allows another try', async () => {
    const d = page(REPORT, { fail: true });
    check(d).click();
    await flush();
    assert.match(text(d), /could not be read/);
    assert.equal(check(d).disabled, false);
  });
});
