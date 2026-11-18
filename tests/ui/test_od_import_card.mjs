/**
 * The Dashboard home "Bring your memories from another app" card (4.2.30, WS-9, D-19): for people who
 * are not technical. Off -> a plain Turn on; on -> choose a file; a preview in words; Import; watch it;
 * done with an Undo; plain error words; no raw ids; server text never parsed as HTML.
 */
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-import.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const CARD = `<div class="card" id="dash-import-card"><div id="dash-import-body"></div></div>`;
const flush = (ms = 40) => new Promise((r) => setTimeout(r, ms));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const RUN = (extra = {}) => ({
  run_id: 'imp_0123456789abcdef0123', state: 'done', format: 'chatgpt', source_label: 'conversations.json',
  saved_items: 202, total_items: 214, progress: 1, skipped: {}, skipped_words: [],
  disputes_opened: 0, undoable: true, note: '', created_at: '2026-11-18T10:00:00+00:00', ...extra,
});
const PREVIEW = (extra = {}) => ({
  success: true, upload_id: 'a'.repeat(32), would_save: 202,
  summary: 'We found 214 conversations from ChatGPT, March 2024 to October 2026. 12 empty ones will be skipped.',
  ...extra,
});

const open = [];
afterEach(() => { while (open.length) open.pop().window.close(); });  // stops every poll timer

function page(handlers = {}, runs = [], flags = {}) {
  const calls = [];
  const dom = new JSDOM(`<!DOCTYPE html><html><body>${CARD}</body></html>`,
                        { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  dom.window.fetch = (url, init = {}) => {
    const method = init.method || 'GET';
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ method, url, body });
    const key = `${method} ${url.split('?')[0]}`;
    if (handlers[key]) return handlers[key](body, url);
    if (key === 'GET /api/import/runs') {
      return reply(200, { success: true, enabled: true, kill_switch: false, state: 'on',
                          limits: { max_upload_mb: 64 }, runs, ...flags });
    }
    return reply(404, { detail: 'Not found.' });
  };
  dom.window.eval(SRC);
  open.push(dom);
  return { dom, calls, doc: dom.window.document, text: () => dom.window.document.body.textContent };
}
const buttons = (doc) => [...doc.querySelectorAll('button')];
const button = (doc, label) => buttons(doc).find((b) => b.textContent === label);
const pick = (p, name = 'conversations.json', size = 100) => {
  const file = new p.dom.window.File(['{"a":1}'], name, { type: 'application/json' });
  Object.defineProperty(file, 'size', { value: size });
  p.dom.window.SLMImport._choose(file);
};

describe('Import card', () => {
  it('is on the home page with its script stamped', () => {
    assert.match(HTML, /id="dash-import-card"/);
    assert.match(HTML, /static\/js\/od-import\.js\?v=[0-9a-f]{8}/);
    assert.match(HTML, /<h3>Bring your memories from another app<\/h3>/);
  });

  it('while importing is off says so in plain words and offers one button to turn it on', async () => {
    const p = page({}, [], { enabled: false, state: 'off' });
    await p.dom.window.SLMImport.reload(); await flush();
    assert.match(p.text(), /Importing is turned off\./);
    assert.ok(button(p.doc, 'Turn on importing'));
    assert.doesNotMatch(p.text(), /portability|flag|flags set/);
    assert.equal(p.doc.querySelector('input[type=file]'), null);
  });

  it('turns importing on through the switch route and then shows the file chooser', async () => {
    let on = false;
    const p = page({ 'POST /api/import/switch': (b) => { on = b.state === 'on'; return reply(200, { success: true }); } },
                   [], { enabled: false });
    await p.dom.window.SLMImport.reload(); await flush();
    p.dom.window.fetch = ((orig) => (url, init) => (String(url).startsWith('/api/import/runs') && on
      ? reply(200, { success: true, enabled: true, kill_switch: false, limits: { max_upload_mb: 64 }, runs: [] })
      : orig(url, init)))(p.dom.window.fetch);
    button(p.doc, 'Turn on importing').click(); await flush(80);
    assert.ok(p.doc.querySelector('input[type=file]'));
    assert.match(p.text(), /Move your memory over from ChatGPT, Claude, Mem0, Letta/);
  });

  it('says plainly when the kill switch is on', async () => {
    const p = page({}, [], { enabled: false, kill_switch: true });
    await p.dom.window.SLMImport.reload(); await flush();
    assert.match(p.text(), /paused because the SuperLocalMemory kill switch is on/);
    assert.equal(button(p.doc, 'Turn on importing'), undefined);
  });

  it('shows the plain preview, never raw ids, and Import is disabled until there is something new', async () => {
    const p = page({ 'POST /api/import/preview': () => reply(200, PREVIEW()) });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p); await flush(80);
    assert.match(p.text(), /We found 214 conversations from ChatGPT, March 2024 to October 2026\. 12 empty ones will be skipped\./);
    assert.match(p.text(), /202 pieces will be added\. They are marked as imported/);
    assert.doesNotMatch(p.text(), /upload_id|aaaaaaaa|imp_|portability/);
    assert.equal(button(p.doc, 'Import').disabled, false);
    const none = page({ 'POST /api/import/preview': () => reply(200, PREVIEW({ would_save: 0 })) });
    await none.dom.window.SLMImport.reload(); await flush();
    pick(none); await flush(80);
    assert.match(none.text(), /Everything in this file is already in your memory/);
    assert.equal(button(none.doc, 'Import').disabled, true);
  });

  it('sends the file as base64 to the preview route and nothing else', async () => {
    const p = page({ 'POST /api/import/preview': () => reply(200, PREVIEW()) });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p, 'notes.md'); await flush(80);
    const sent = p.calls.find((c) => c.method === 'POST');
    assert.equal(sent.url, '/api/import/preview');
    assert.equal(sent.body.filename, 'notes.md');
    assert.equal(Buffer.from(sent.body.data_base64, 'base64').toString(), '{"a":1}');
    assert.deepEqual(p.calls.filter((c) => c.method === 'POST').length, 1);
  });

  it('a file over the limit is refused in words, pointing at the command line, without sending it', async () => {
    const p = page();
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p, 'big.json', 70 * 1024 * 1024); await flush();
    assert.match(p.text(), /bigger than 64 MB/);
    assert.match(p.text(), /slm import run/);
    assert.equal(p.calls.filter((c) => c.method === 'POST').length, 0);
  });

  it('shows the server\'s plain refusal and never parses it as HTML', async () => {
    const p = page({ 'POST /api/import/preview': () => reply(422, { detail: 'That file is not in a format <img src=x onerror=alert(1)> I know.' }) });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p); await flush(80);
    assert.match(p.text(), /That file is not in a format <img src=x onerror=alert\(1\)> I know\./);
    assert.equal(p.doc.querySelectorAll('img').length, 0);
  });

  it('Import starts the run with the staged upload and shows progress with a Stop', async () => {
    const posts = [];
    const p = page({
      'POST /api/import/preview': () => reply(200, PREVIEW()),
      'POST /api/import/run': (b) => { posts.push(b); return reply(200, { success: true, run: RUN({ state: 'running', saved_items: 40, progress: 0.2 }) }); },
      'GET /api/import/runs/imp_0123456789abcdef0123': () => reply(200, { success: true, run: RUN({ state: 'running', saved_items: 40, progress: 0.2 }) }),
    });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p, 'conversations.json'); await flush(80);
    button(p.doc, 'Import').click(); await flush(120);
    assert.deepEqual(posts, [{ upload_id: 'a'.repeat(32), filename: 'conversations.json' }]);
    assert.match(p.text(), /Importing \.\.\. 20%/);
    assert.match(p.text(), /40 pieces saved so far\. You can leave this page; it keeps going\./);
    assert.equal(p.doc.querySelector('progress').value, 20);
    assert.ok(button(p.doc, 'Stop'));
    assert.doesNotMatch(p.text(), /imp_0123/);
  });

  it('when done says what happened in words, with contradictions pointing at Needs your answer', async () => {
    const done = RUN({ disputes_opened: 2, skipped_words: [{ reason: 'empty_conversation', count: 12, words: 'empty conversations' }] });
    const p = page({}, [done]);
    await p.dom.window.SLMImport.reload(); await flush(120);
    const p2 = page({
      'POST /api/import/preview': () => reply(200, PREVIEW()),
      'POST /api/import/run': () => reply(200, { success: true, run: done }),
      'GET /api/import/runs/imp_0123456789abcdef0123': () => reply(200, { success: true, run: done }),
    });
    await p2.dom.window.SLMImport.reload(); await flush();
    pick(p2); await flush(80);
    button(p2.doc, 'Import').click(); await flush(120);
    assert.match(p2.text(), /202 pieces from conversations\.json are now in your memory\./);
    assert.match(p2.text(), /Left out: 12 empty conversations\./);
    assert.match(p2.text(), /2 things in this import disagree with what you already had\. They are waiting under "Needs your answer"; your earlier memory stays as it is\./);
    assert.ok(button(p2.doc, 'Undo this import'));
    assert.ok(button(p2.doc, 'Import another file'));
  });

  it('Undo takes the run back and says so; a second Undo is not offered', async () => {
    const posts = [];
    let state = 'done';
    const done = () => RUN({ state });
    const p = page({
      'POST /api/import/preview': () => reply(200, PREVIEW()),
      'POST /api/import/run': () => reply(200, { success: true, run: done() }),
      'GET /api/import/runs/imp_0123456789abcdef0123': () => reply(200, { success: true, run: done() }),
      'POST /api/import/runs/imp_0123456789abcdef0123/undo': () => { posts.push(1); state = 'rolled_back'; return reply(200, { success: true, run: done() }); },
    });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p); await flush(80);
    button(p.doc, 'Import').click(); await flush(120);
    button(p.doc, 'Undo this import').click(); await flush(160);
    assert.equal(posts.length, 1);
    assert.match(p.text(), /Undone\. Your memory is exactly as it was before this import\./);
    assert.equal(button(p.doc, 'Undo this import'), undefined);
  });

  it('Stop sends a cancel and the card says what was kept', async () => {
    const posts = [];
    let state = 'running';
    const run = () => RUN({ state, progress: 0.1, note: state === 'cancelled' ? 'Stopped. What was saved so far is kept.' : '' });
    const p = page({
      'POST /api/import/preview': () => reply(200, PREVIEW()),
      'POST /api/import/run': () => reply(200, { success: true, run: run() }),
      'GET /api/import/runs/imp_0123456789abcdef0123': () => reply(200, { success: true, run: run() }),
      'POST /api/import/runs/imp_0123456789abcdef0123/cancel': () => { posts.push(1); state = 'cancelled'; return reply(200, { success: true, run: run() }); },
    });
    await p.dom.window.SLMImport.reload(); await flush();
    pick(p); await flush(80);
    button(p.doc, 'Import').click(); await flush(120);
    button(p.doc, 'Stop').click(); await flush(160);
    assert.equal(posts.length, 1);
    assert.match(p.text(), /Stopped\. What was saved so far is kept\./);
  });

  it('lists earlier imports in words with an Undo, and never shows an id or a state code', async () => {
    const p = page({}, [RUN(), RUN({ run_id: 'imp_second', state: 'rolled_back', undoable: true, saved_items: 3 })]);
    await p.dom.window.SLMImport.reload(); await flush();
    assert.match(p.text(), /Earlier imports/);
    assert.match(p.text(), /202 pieces from conversations\.json/);
    assert.match(p.text(), /Taken back/);
    assert.doesNotMatch(p.text(), /imp_|rolled_back|rolling_back|done'|ChatGPT export format/);
    assert.equal(buttons(p.doc).filter((b) => b.textContent === 'Undo').length, 1);
  });

  it('shows a run that is still going when the page opens, and when the daemon does not answer it just says it is off', async () => {
    const live = RUN({ state: 'running', progress: 0.5, saved_items: 100 });
    const p = page({ 'GET /api/import/runs/imp_0123456789abcdef0123': () => reply(200, { success: true, run: live }) }, [live]);
    await p.dom.window.SLMImport.reload(); await flush(120);
    assert.match(p.text(), /Importing \.\.\. 50%/);
    const down = page({ 'GET /api/import/runs': () => reply(500, {}) });
    await down.dom.window.SLMImport.reload(); await flush();
    assert.match(down.text(), /Importing is turned off\./);
  });
});
