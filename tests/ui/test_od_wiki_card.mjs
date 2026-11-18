/**
 * The Dashboard home "Living wiki" card (4.2.30, WS-6, D-17, D-18; blueprint 8.10.3 i): pages
 * compiled from memory, each sentence shown with the memory it cites and a link to it, the
 * withheld count in words, and two person's buttons (Rebuild pages, Read my edits) that go to the
 * same routes `slm wiki` uses with the channel "dashboard". Memory text is never parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-wiki.js'), 'utf8');
const HTML = `<!DOCTYPE html><html><body>
  <a class="nav-link" data-tab="memories-pane" href="#">Memories</a>
  <input id="recall-lab-query"><button id="recall-lab-search">Search</button>
  <div class="card" id="dash-wiki-card" hidden><span id="dash-wiki-count"></span><div id="dash-wiki-body"></div></div>
</body></html>`;
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const PAGE = { page_id: 'p1', kind: 'topic', title: 'SuperLocalMemory: what is known <img src=x onerror=alert(1)>',
               version: 1, shown_count: 2, withheld_count: 3, built_at: '2026-11-18T10:00:00Z' };
const FULL = {
  page_id: 'p1', title: PAGE.title, origin: 'derived', stale: false,
  notice: 'Every sentence is a word-for-word span of a memory it cites, or a fixed template over its values. No model wrote any of it.',
  withheld: { count: 3, reasons: { over_limit: { count: 2, why: 'over the size of a page' },
                                    too_long: { count: 1, why: 'too long to be one sentence' } } },
  sentences: [
    { seq: 0, type: 'span', text: 'SLM ships every week.', facts: ['f1'],
      citations: [{ fact_id: 'f1', memory_id: 'm1', origin: 'you said this' }] },
    { seq: 1, type: 'template', text: 'slm cadence has had this value since 2026-03-01.', facts: ['f1'],
      citations: [{ fact_id: 'f1', memory_id: 'm1', origin: null }] },
  ],
};

function page({ flag = 'off', pages = [PAGE], pending = 0, fail = null, full = FULL, sync = null }, calls) {
  const d = new JSDOM(HTML, { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url, init) => {
    calls.push({ url, method: (init && init.method) || 'GET', body: init && init.body ? JSON.parse(init.body) : null });
    if (init && init.method === 'POST') {
      if (fail) return reply(fail.status, fail.body);
      return reply(200, url.endsWith('/sync') ? (sync || { success: true, found: 0, queued: [], deleted_lines: 2 })
                                              : { success: true, pages: 1 });
    }
    if (url.includes('/api/wiki/pages/')) return reply(200, { success: true, page: full });
    return reply(200, { success: true, flag, pages, status: { dir: '/w/default', pages: pages.length,
                                                              pending_sets: pending } });
  };
  d.window.eval(SRC);
  return d;
}

const buttons = (d, label) => [...d.window.document.querySelectorAll('button')]
  .filter((b) => b.textContent === label || b.textContent.startsWith(label));

describe('Living wiki card', () => {
  it('stays hidden when there are no pages and the Markdown wiki is off', async () => {
    const d = page({ pages: [] }, []);
    await d.window.SLMWiki.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-wiki-card').hidden, true);
  });

  it('shows itself with no pages once the Markdown wiki is on, and says how to start', async () => {
    const d = page({ pages: [], flag: 'on' }, []);
    await d.window.SLMWiki.reload(); await flush();
    const doc = d.window.document;
    assert.equal(doc.getElementById('dash-wiki-card').hidden, false);
    assert.match(doc.body.textContent, /No pages yet\. Markdown folder: \/w\/default/);
    assert.equal(buttons(d, 'Rebuild pages').length, 1);
  });

  it('lists the pages in plain words and never parses memory text as HTML', async () => {
    const d = page({ pending: 1 }, []);
    await d.window.SLMWiki.reload(); await flush();
    const doc = d.window.document;
    assert.equal(doc.getElementById('dash-wiki-card').hidden, false);
    assert.equal(doc.getElementById('dash-wiki-count').textContent, '1');
    assert.match(doc.body.textContent, /Topic\. 2 sentences, 3 withheld\./);
    assert.match(doc.body.textContent, /The Markdown folder is off/);
    assert.match(doc.body.textContent, /1 page change is waiting under Proposed changes/);
    assert.equal(doc.querySelector('img'), null);
    assert.match(doc.querySelector('.dash-card-lead').textContent, /<img src=x/);
    assert.equal(buttons(d, 'Read my edits').length, 0);  // only while the Markdown wiki is on
  });

  it('Read shows each sentence with its citation, how many were withheld and why', async () => {
    const calls = [];
    const d = page({}, calls);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Read')[0].click(); await flush();
    const doc = d.window.document;
    assert.ok(calls.some((c) => c.url === '/api/wiki/pages/p1'));
    assert.match(doc.body.textContent, /word-for-word span of a memory it cites/);
    assert.equal(doc.querySelectorAll('.wiki-page .truth-side').length, 2);
    assert.match(doc.body.textContent, /word for word from a memory/);
    assert.match(doc.body.textContent, /from slot values/);
    assert.match(doc.body.textContent, /3 sentences withheld \(2 over the size of a page; 1 too long to be one sentence\)\./);
    assert.equal(buttons(d, 'Open memory (you said this)').length, 1);
    buttons(d, 'Read')[0].click();                       // a second click closes it
    assert.equal(doc.querySelector('.wiki-page'), null);
  });

  it('Open memory takes you to the memory by searching for the sentence', async () => {
    const d = page({}, []);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Read')[0].click(); await flush();
    const doc = d.window.document;
    let clicked = 0, searched = 0;
    doc.querySelector('.nav-link[data-tab="memories-pane"]').addEventListener('click', () => clicked++);
    doc.getElementById('recall-lab-search').addEventListener('click', () => searched++);
    buttons(d, 'Open memory')[0].click();
    assert.equal(doc.getElementById('recall-lab-query').value, 'SLM ships every week.');
    assert.deepEqual([clicked, searched], [1, 1]);
  });

  it('a page whose sentences no longer match says so and asks for a rebuild', async () => {
    const d = page({ full: { ...FULL, stale: true } }, []);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Read')[0].click(); await flush();
    assert.match(d.window.document.body.textContent, /no longer match your memories and are hidden/);
  });

  it('Rebuild pages goes to the shared route with the dashboard channel and confirms', async () => {
    const calls = [];
    const d = page({}, calls);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Rebuild pages')[0].click(); await flush();
    const post = calls.find((c) => c.method === 'POST');
    assert.deepEqual([post.url, post.body], ['/api/wiki/rebuild', { channel: 'dashboard' }]);
    assert.match(d.window.document.body.textContent, /Compiled and approved 1 page\./);
  });

  it('Read my edits proposes, never writes, and says deleting a line forgets nothing', async () => {
    const calls = [];
    const d = page({ flag: 'on' }, calls);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Read my edits')[0].click(); await flush();
    const post = calls.find((c) => c.method === 'POST');
    assert.deepEqual([post.url, post.body], ['/api/wiki/sync', { channel: 'dashboard' }]);
    assert.match(d.window.document.body.textContent, /2 deleted lines ignored: deleting a line forgets nothing/);
    const d2 = page({ flag: 'on', sync: { success: true, found: 1, queued: [{ set_id: 's1' }], deleted_lines: 0 } }, []);
    await d2.window.SLMWiki.reload(); await flush();
    buttons(d2, 'Read my edits')[0].click(); await flush();
    assert.match(d2.window.document.body.textContent, /1 edit proposed; review them under Proposed changes/);
  });

  it('a refusal says why and keeps the button usable', async () => {
    const d = page({ fail: { status: 403, body: { detail: 'A person decides this.' } } }, []);
    await d.window.SLMWiki.reload(); await flush();
    buttons(d, 'Rebuild pages')[0].click(); await flush();
    assert.match(d.window.document.querySelector('.wiki-note').textContent, /A person decides this\./);
    assert.equal(buttons(d, 'Rebuild pages')[0].disabled, false);
  });

  it('hides itself when the pages cannot be read', async () => {
    const d = new JSDOM(HTML, { runScripts: 'outside-only', url: 'http://localhost:8765/' });
    d.window.fetch = () => reply(403, { detail: 'local only' });
    d.window.eval(SRC);
    await d.window.SLMWiki.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-wiki-card').hidden, true);
  });
});
