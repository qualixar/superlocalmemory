/**
 * Strict mode (SLM_REQUIRE_CREDENTIALS=1): the daemon refuses to hand the
 * install key to the page, so the dashboard asks the person to paste it once.
 * The pasted key stays in memory only and every write then carries it.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const coreSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/core.js'), 'utf8',
);

const REFUSAL = {
  error: 'key_required',
  message: 'This computer requires the SuperLocalMemory key. ' +
    'Run `slm token show` and paste it here.',
};

function build(tokenAnswer) {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'dangerously', url: 'http://localhost:8765/',
  });
  const { window } = dom;
  const calls = [];
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeListener() {} });
  window.fetch = function (input, init) {
    calls.push({ input: String(input), init: init || {} });
    if (String(input) === '/internal/token') {
      const a = tokenAnswer();
      return Promise.resolve({
        ok: a.status === 200, status: a.status,
        json: () => Promise.resolve(a.body),
        clone() { return { json: () => Promise.resolve(a.body) }; },
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) });
  };
  window.loadProfiles = () => {};
  window.loadStats = () => {};
  window.loadGraph = () => {};
  const script = window.document.createElement('script');
  script.textContent = coreSource;
  window.document.head.appendChild(script);
  return { window, calls };
}

function promptEl(window) {
  return window.document.querySelector('[data-slm-key-prompt]');
}

async function tick() { await new Promise((r) => setTimeout(r, 0)); }

function paste(window, value) {
  const el = promptEl(window);
  el.querySelector('input').value = value;
  el.querySelector('[data-slm-key-submit]').click();
}

describe('strict mode key prompt', () => {
  it('asks once, in plain words, when the daemon refuses the key', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const pending = h.window.slmInstallToken(false);
    await tick();
    const el = promptEl(h.window);
    assert.ok(el, 'a prompt is shown');
    assert.match(el.textContent, /requires the SuperLocalMemory key/);
    assert.match(el.textContent, /slm token show/);
    assert.equal(el.querySelector('input').type, 'password');
    paste(h.window, '  abc123  ');
    assert.equal(await pending, 'abc123');
    assert.equal(promptEl(h.window), null, 'the prompt closes');
  });

  it('keeps the key in memory only and reuses it without asking again', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const pending = h.window.slmInstallToken(false);
    await tick();
    paste(h.window, 'secretkey');
    await pending;
    assert.equal(await h.window.slmInstallToken(false), 'secretkey');
    assert.equal(promptEl(h.window), null);
    for (const store of [h.window.localStorage, h.window.sessionStorage]) {
      for (let i = 0; i < store.length; i += 1) {
        assert.ok(!String(store.getItem(store.key(i))).includes('secretkey'));
      }
    }
  });

  it('sends the pasted key with a write', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const write = h.window.fetch('/api/memories', { method: 'POST', body: '{}' });
    await tick();
    paste(h.window, 'secretkey');
    await write;
    const post = h.calls.find((c) => c.input === '/api/memories');
    assert.equal(new h.window.Headers(post.init.headers).get('X-Install-Token'), 'secretkey');
  });

  it('shows one prompt for several waiting callers', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const a = h.window.slmInstallToken(false);
    const b = h.window.slmInstallToken(true);
    await tick();
    assert.equal(h.window.document.querySelectorAll('[data-slm-key-prompt]').length, 1);
    paste(h.window, 'k1');
    assert.deepEqual([await a, await b], ['k1', 'k1']);
  });

  it('a page that reads /internal/token itself gets the pasted key as a normal answer', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const reading = h.window.fetch('/internal/token', { credentials: 'same-origin' });
    await tick();
    paste(h.window, 'pasted-key');
    const response = await reading;
    assert.equal(response.ok, true);
    assert.equal((await response.json()).token, 'pasted-key');
  });

  it('declining leaves the write blocked and does not nag straight away', async () => {
    const h = build(() => ({ status: 403, body: REFUSAL }));
    const first = h.window.slmInstallToken(false);
    await tick();
    h.window.document.querySelector('[data-slm-key-prompt] button').click();
    assert.equal(await first, '');
    const second = await h.window.slmInstallToken(false);
    assert.equal(second, '');
    assert.equal(promptEl(h.window), null, 'no second prompt right after declining');
    await assert.rejects(
      h.window.fetch('/api/memories', { method: 'POST', body: '{}' }),
      /local write credential unavailable/,
    );
  });

  it('a different refusal never opens the prompt', async () => {
    const h = build(() => ({ status: 403, body: { error: 'loopback only' } }));
    assert.equal(await h.window.slmInstallToken(false), '');
    assert.equal(promptEl(h.window), null);
  });

  it('normal mode is unchanged: the token is served and no prompt appears', async () => {
    const h = build(() => ({ status: 200, body: { token: 'served' } }));
    assert.equal(await h.window.slmInstallToken(false), 'served');
    assert.equal(promptEl(h.window), null);
  });
});
