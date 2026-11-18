/**
 * The memory detail's "Where this came from" (4.2.30, WS-5): the origin class in plain
 * words from the server, a note when the taint is lower, an honest line for memories
 * saved before origins were kept. Text is only ever set with textContent.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-truth.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

function bare(body, status = 200) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="detail"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = () => reply(status, body);
  d.window.eval(SRC);
  return d;
}

async function show(body, status) {
  const d = bare(body, status);
  const box = d.window.document.getElementById('detail');
  await d.window.SLMTruth.detail('f1', box); await flush();
  return box;
}

const PLAIN = { success: true, slots: [], disputes: [], source: null };
const origin = (o) => ({ ...PLAIN, origin: o });

describe('Memory detail: Where this came from', () => {
  it('says it in plain words, with no slot section when the memory is in no slot', async () => {
    const box = await show(origin({ class: 'user_direct', taint: 'user_direct', label: 'you said this',
                                    taint_label: 'you said this' }));
    assert.match(box.textContent, /Where this came from/);
    assert.match(box.textContent, /You said this\./);
    assert.doesNotMatch(box.textContent, /When it was true/);
    assert.equal(box.querySelector('[data-origin]').getAttribute('data-origin'), 'user_direct');
  });

  it('sizes its heading like the drawer sections around it (the theme styles a bare h3 at 1.25rem)', async () => {
    const box = await show(origin({ class: 'user_direct', taint: 'user_direct', label: 'you said this',
                                    taint_label: 'you said this' }));
    assert.equal(box.querySelector('h3').style.fontSize, '13px');
  });

  it('adds a note when the memory was built from something less trusted', async () => {
    const box = await show(origin({ class: 'system', taint: 'web', label: 'built by SuperLocalMemory',
                                    taint_label: 'a web page' }));
    assert.match(box.textContent, /Built by SuperLocalMemory\./);
    assert.match(box.textContent, /built from something less trusted: a web page/);
  });

  it('is honest about a memory saved before origins were kept', async () => {
    const box = await show(origin({ class: 'unknown', taint: 'unknown',
                                    label: 'unknown (saved before 4.2.30)',
                                    taint_label: 'unknown (saved before 4.2.30)' }));
    assert.match(box.textContent, /Unknown \(saved before 4.2\.30\)\./);
    assert.match(box.textContent, /did not record where this came from/);
    assert.doesNotMatch(box.textContent, /less trusted/);
  });

  it('never parses the label as HTML', async () => {
    const box = await show(origin({ class: 'web', taint: 'web', label: '<img src=x onerror=alert(1)>',
                                    taint_label: 'x' }));
    assert.equal(box.querySelectorAll('img').length, 0);
    assert.match(box.textContent, /<img src=x/);
  });

  it('sits beside "When it was true" for a memory that fills a slot', async () => {
    const box = await show({ ...origin({ class: 'web', taint: 'web', label: 'a web page', taint_label: 'a web page' }),
      slots: [{ legacy: true, subject: null, relation: null, values: [] }] });
    const titles = [...box.querySelectorAll('h3')].map((h) => h.textContent);
    assert.deepEqual(titles, ['Where this came from', 'When it was true']);
  });

  it('shows nothing when the server has no origin or the call fails', async () => {
    assert.equal((await show(PLAIN)).textContent, '');
    assert.equal((await show({}, 500)).textContent, '');
  });
});

describe('A dispute the origin lattice held', () => {
  it('asks the person in its own words', async () => {
    const box = await show({ ...PLAIN, disputes: [{ dispute_id: 'd1', reason: 'origin_below_head', sides: [] }] });
    assert.match(box.textContent, /came from a source SuperLocalMemory trusts less/);
  });
});
