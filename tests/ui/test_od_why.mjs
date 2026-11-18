/**
 * The memory detail's "Why was this shown?" (4.2.30, WS-5): nothing is fetched until the person
 * asks, the answer is plain words from the server, every string is set with textContent, a
 * failure is said plainly, and the real memory drawer mounts it.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-why.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const REPORT = {
  success: true,
  origin: { class: 'web', words: 'This memory is a web page.', rule: { row: 'web.page', why: 'It was read from a page.' } },
  firewall: { acting: true, below_system: true, words: 'The instruction firewall holds this memory back.',
              approvals: [{ wanted: 'rule', state: 'approved', decided_by: 'varun', decided_channel: 'cli' }] },
  signature: { state: 'ok', words: 'Signed, and unchanged since it was signed.' },
  deliveries: { shown: 2, considered: 3, recent: [
    { at: '2026-11-16T10:00:00+00:00', why: { words: 'Shown at session start, as a pinned memory.' } },
    { at: '2026-11-15T10:00:00+00:00', why: { words: 'Chosen on a cue, but left out on purpose.' } } ] },
};

function bare(handler) {
  const calls = [];
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="box"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url) => { calls.push(url); return handler(url); };
  d.window.eval(SRC);
  return { d, calls, box: d.window.document.getElementById('box') };
}

describe('Memory detail: Why was this shown?', () => {
  it('fetches nothing until the person asks', async () => {
    const { calls, d, box } = bare(() => reply(200, REPORT));
    await d.window.SLMWhy.mount('f1', box); await flush();
    assert.equal(calls.length, 0);
    assert.match(box.textContent, /Why was this shown\?/);
    assert.equal(box.querySelectorAll('button').length, 1);
  });

  it('sizes its heading like the drawer sections around it (the theme styles a bare h3 at 1.25rem)', async () => {
    const { d, box } = bare(() => reply(200, REPORT));
    await d.window.SLMWhy.mount('f1', box);
    assert.equal(box.querySelector('h3').style.fontSize, '13px');
  });

  it('shows the origin, the firewall, the signature and each delivery in words', async () => {
    const { calls, d, box } = bare(() => reply(200, REPORT));
    await d.window.SLMWhy.mount('f1', box);
    box.querySelector('button').click(); await flush();
    assert.deepEqual(calls, ['/api/trust/why/f1']);
    const text = box.textContent;
    assert.match(text, /This memory is a web page\./);
    assert.match(text, /Why it is counted that way: It was read from a page\./);
    assert.match(text, /holds this memory back/);
    assert.match(text, /It asked to be: rule \(approved by varun via cli\)\./);
    assert.match(text, /Signed, and unchanged since it was signed\./);
    assert.match(text, /Shown to your agents 2 times \(considered 3\)/);
    assert.match(text, /Shown at session start, as a pinned memory\./);
    assert.match(text, /left out on purpose/);
    assert.equal(box.querySelector('[data-origin]').getAttribute('data-origin'), 'web');
    assert.equal(box.querySelector('button').hidden, true);
  });

  it('says so when no agent has been shown the memory', async () => {
    const { d, box } = bare(() => reply(200, { ...REPORT, deliveries: { shown: 0, considered: 0, recent: [] } }));
    await d.window.SLMWhy.mount('f1', box);
    box.querySelector('button').click(); await flush();
    assert.match(box.textContent, /No agent has been shown this memory yet\./);
  });

  it('marks a modified signature as a warning', async () => {
    const { d, box } = bare(() => reply(200, { ...REPORT, signature: { state: 'modified', words: 'Signed, but the memory no longer matches its signature.' } }));
    await d.window.SLMWhy.mount('f1', box);
    box.querySelector('button').click(); await flush();
    assert.ok(box.querySelector('.tone-warn'));
  });

  it('never parses server text as HTML', async () => {
    const evil = '<img src=x onerror=alert(1)>';
    const { d, box } = bare(() => reply(200, { ...REPORT, origin: { class: 'web', words: evil },
      deliveries: { shown: 1, considered: 1, recent: [{ at: 'x', why: { words: evil } }] } }));
    await d.window.SLMWhy.mount('f1', box);
    box.querySelector('button').click(); await flush();
    assert.equal(box.querySelectorAll('img').length, 0);
    assert.match(box.textContent, /<img src=x/);
  });

  it('lists at most five deliveries', async () => {
    const many = Array.from({ length: 9 }, (_, i) => ({ at: `2026-11-${10 + i}T10:00:00+00:00`, why: { words: `delivery ${i}` } }));
    const { d, box } = bare(() => reply(200, { ...REPORT, deliveries: { shown: 9, considered: 9, recent: many } }));
    await d.window.SLMWhy.mount('f1', box);
    box.querySelector('button').click(); await flush();
    assert.equal((box.textContent.match(/delivery \d/g) || []).length, 5);
  });

  it('says plainly that it could not read, on an error or a refusal', async () => {
    for (const handler of [() => reply(500, {}), () => reply(404, {}), () => reply(200, { success: false }),
                           () => Promise.reject(new Error('down'))]) {
      const { d, box } = bare(handler);
      await d.window.SLMWhy.mount('f1', box);
      box.querySelector('button').click(); await flush();
      assert.match(box.textContent, /could not read why this memory was shown/);
    }
  });

  it('does nothing without a container or an id', async () => {
    const { d, calls } = bare(() => reply(200, REPORT));
    await d.window.SLMWhy.mount('f1', null);
    await d.window.SLMWhy.mount('', d.window.document.getElementById('box'));
    assert.equal(calls.length, 0);
  });

  it('encodes the id in the path', async () => {
    const { calls, d, box } = bare(() => reply(200, REPORT));
    await d.window.SLMWhy.mount('a/b c', box);
    box.querySelector('button').click(); await flush();
    assert.deepEqual(calls, ['/api/trust/why/a%2Fb%20c']);
  });
});

describe('The memory drawer mounts it', () => {
  it('opens with a Why section that loads on request', async () => {
    const memory = { id: 'f-why', fact_id: 'f-why', content: 'A memory.', scope: 'personal', category: 'episodic',
                     created_at: '2026-10-11T12:00:00Z', score: 0.9 };
    const requested = [];
    const h = buildHarness(['root'], (url) => {
      requested.push(url);
      const body = url.includes('/api/trust/why/') ? REPORT
        : url.includes('/api/memories') || url.includes('/api/search') ? { memories: [memory], results: [memory], total: 1 }
          : url.includes('kind-counts') ? { counts: {} } : {};
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
    });
    try {
      evalModule(h.window, 'od-memories.js');
      h.window.eval(SRC);
      const root = h.document.getElementById('root');
      h.window.odRenderMemories(root);
      await flushPromises();
      const row = root.querySelector('[data-od-act="drawer"]');
      assert.ok(row, 'a memory row');
      row.click(); await flushPromises();
      const slot = h.document.getElementById('od-drawer-why');
      assert.ok(slot, 'the drawer has the why slot');
      assert.match(slot.textContent, /Why was this shown\?/);
      assert.ok(!requested.some((u) => u.includes('/api/trust/why/')), 'not fetched on open');
      slot.querySelector('button').click(); await flushPromises(); await flush();
      assert.ok(requested.some((u) => u.startsWith('/api/trust/why/')));
      assert.match(slot.textContent, /This memory is a web page\./);
    } finally {
      h.dom.window.close();
    }
  });
});
