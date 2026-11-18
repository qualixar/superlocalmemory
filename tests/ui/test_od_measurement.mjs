/**
 * "Help SLM measure what works" (4.2.30 D-28): the Settings card and the
 * dashboard banner, against a stubbed daemon. Plain words, one switch that acts
 * at once, a refused change says why, nothing from the server parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-measurement.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 20));

const NOTICE = 'SuperLocalMemory now measures whether memory helps your agents.\n\n' +
  'About 1 in 10 session starts change one thing <img src=x onerror=alert(1)>.\n\n' +
  'To turn the withholding, swapping and usage detection off, run: slm measurement off';

function report(over = {}) {
  const flags = [
    { id: 'outcomes.receipts', label: 'Receipts', what: 'Writes a receipt.', state: 'on' },
    { id: 'outcomes.holdout', label: 'Withholding', what: 'Leaves one out.', state: 'on' },
    { id: 'outcomes.swaps', label: 'Swaps', what: 'Swaps two.', state: 'on' },
    { id: 'outcomes.usage', label: 'Usage check', what: 'Compares a reply.', state: 'off' },
    { id: 'delivery.need_gate', label: 'Need check', what: 'Per prompt.', state: 'off' },
  ];
  return Object.assign({ success: true, kill_switch: false, measuring: true, pending: true,
                         notice_version: '4.2.30-1', notice: NOTICE, flags }, over);
}

function page({ initial = report(), refuse = 0 } = {}) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><a class="nav-link" data-tab="settings-pane"></a>' +
    '<div id="od-measurement-banner"></div><div id="card"></div></body></html>',
    { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  const w = dom.window;
  let current = initial;
  w.sent = [];
  w.toasts = [];
  w.showToast = (m) => w.toasts.push(m);
  w.fetch = (url, init) => {
    const method = (init && init.method) || 'GET';
    if (method === 'GET') return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(current) });
    const body = init.body ? JSON.parse(init.body) : {};
    w.sent.push({ url: String(url), method, body });
    if (refuse) return Promise.resolve({ ok: false, status: refuse, json: () => Promise.resolve({ detail: 'nope' }) });
    if (String(url).endsWith('/acknowledge')) current = report({ ...current, pending: false });
    else {
      const on = body.on;
      current = report({ ...current, measuring: on, flags: current.flags.map((f) =>
        (f.id === 'outcomes.holdout' || f.id === 'outcomes.swaps') ? { ...f, state: on ? 'on' : 'off' } : f) });
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(current) });
  };
  w.eval(SRC);
  return w;
}

const byText = (w, label) => [...w.document.querySelectorAll('button')].find((b) => b.textContent === label);

describe('Settings card', () => {
  it('shows the notice in words, the state of each measure and one switch', async () => {
    const w = page();
    const mount = w.document.getElementById('card');
    await w.SLMMeasurement.renderInto(mount);
    const text = mount.textContent;
    assert.match(text, /now measures whether memory helps/);
    assert.match(text, /slm measurement off/);
    assert.match(text, /Withholding: Leaves one out\./);
    const sw = mount.querySelector('[role="switch"]');
    assert.equal(sw.getAttribute('aria-checked'), 'true');
    assert.equal(mount.querySelectorAll('[role="switch"]').length, 1);
    assert.equal(mount.querySelectorAll('img').length, 0);  // server text is never HTML
  });

  it('the switch turns measuring off at once and says so', async () => {
    const w = page();
    const mount = w.document.getElementById('card');
    await w.SLMMeasurement.renderInto(mount);
    mount.querySelector('[role="switch"]').click();
    await flush();
    assert.deepEqual(w.sent, [{ url: '/api/outcomes/measurement', method: 'POST', body: { on: false } }]);
    const sw = mount.querySelector('[role="switch"]');
    assert.equal(sw.getAttribute('aria-checked'), 'false');
    assert.match(mount.textContent, /Measuring is off/);
    sw.click();
    await flush();
    assert.equal(w.sent[1].body.on, true);
    assert.equal(mount.querySelector('[role="switch"]').getAttribute('aria-checked'), 'true');
  });

  it('a refused change says who can change it and leaves the switch as it was', async () => {
    const w = page({ refuse: 403 });
    const mount = w.document.getElementById('card');
    await w.SLMMeasurement.renderInto(mount);
    mount.querySelector('[role="switch"]').click();
    await flush();
    assert.match(mount.textContent, /Only the owner or an admin can change this/);
    assert.equal(mount.querySelector('[role="switch"]').getAttribute('aria-checked'), 'true');
  });

  it('says so when the kill switch is on, and when the daemon cannot be read', async () => {
    const w = page({ initial: report({ kill_switch: true, measuring: false }) });
    const mount = w.document.getElementById('card');
    await w.SLMMeasurement.renderInto(mount);
    assert.match(mount.textContent, /kill switch is on/);
    const w2 = page();
    w2.fetch = () => Promise.reject(new Error('down'));
    const m2 = w2.document.getElementById('card');
    await w2.SLMMeasurement.renderInto(m2);
    assert.match(m2.textContent, /Could not read the measurement settings/);
  });
});

describe('Dashboard banner', () => {
  it('appears while the notice is pending and offers Learn more, Turn off and OK', async () => {
    const w = page();
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    const host = w.document.getElementById('od-measurement-banner');
    assert.match(host.textContent, /SLM now measures whether memory helps your agents/);
    for (const label of ['Learn more', 'Turn off', 'OK']) assert.ok(byText(w, label), label);
  });

  it('is absent once the notice was read', async () => {
    const w = page({ initial: report({ pending: false }) });
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    assert.equal(w.document.getElementById('od-measurement-banner').textContent, '');
  });

  it('OK acknowledges, keeps measuring on, and removes the banner', async () => {
    const w = page();
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    byText(w, 'OK').click();
    await flush();
    assert.deepEqual(w.sent.map((s) => s.url), ['/api/outcomes/measurement/acknowledge']);
    assert.equal(w.document.getElementById('od-measurement-banner').textContent, '');
  });

  it('Turn off switches measuring off, then acknowledges', async () => {
    const w = page();
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    byText(w, 'Turn off').click();
    await flush();
    await flush();
    assert.deepEqual(w.sent.map((s) => [s.url, s.body.on]),
      [['/api/outcomes/measurement', false], ['/api/outcomes/measurement/acknowledge', undefined]]);
    assert.equal(w.document.getElementById('od-measurement-banner').textContent, '');
  });

  it('a refused Turn off keeps the banner and says why', async () => {
    const w = page({ refuse: 403 });
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    byText(w, 'Turn off').click();
    await flush();
    await flush();
    const host = w.document.getElementById('od-measurement-banner');
    assert.match(host.textContent, /Only the owner or an admin can change this/);
    assert.ok(byText(w, 'OK'));
    assert.deepEqual(w.sent.map((s) => s.url), ['/api/outcomes/measurement']);  // never acknowledged
  });

  it('Learn more opens Settings and does not acknowledge', async () => {
    const w = page();
    let opened = 0;
    w.document.querySelector('.nav-link').addEventListener('click', () => { opened += 1; });
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    byText(w, 'Learn more').click();
    assert.equal(opened, 1);
    assert.deepEqual(w.sent, []);
  });

  it('a banner when measuring is already off does not offer Turn off', async () => {
    const w = page({ initial: report({ measuring: false }) });
    await w.SLMMeasurement.mountBanner(w.document.getElementById('od-measurement-banner'));
    assert.equal(byText(w, 'Turn off'), undefined);
    assert.match(w.document.getElementById('od-measurement-banner').textContent, /turned off here/);
  });
});

describe('wiring', () => {
  it('index.html loads the script and has the banner anchor; Settings builds the card', () => {
    const html = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
    assert.match(html, /static\/js\/od-measurement\.js\?v=[0-9a-f]{8}/);
    assert.match(html, /id="od-measurement-banner"/);
    const settings = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-settings.js'), 'utf8');
    assert.match(settings, /Help SLM measure what works/);
    assert.match(settings, /SLMMeasurement\.renderInto/);
  });
});
