/**
 * The "Better pictures and documents" card on the Documents & Images pane (4.2.30, WS-8, D-19): for people
 * who are not technical. Sizes shown first, a plain confirmation, progress, plain error words, no raw ids,
 * server text never parsed as HTML.
 */
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-media-packs.js'), 'utf8');
const MEDIA = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-media.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const flush = (ms = 40) => new Promise((r) => setTimeout(r, ms));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const BASE = {
  captions: { pack: 'captions', label: 'Describe pictures', what: 'Writes a short description of each picture.',
    state: 'off', enabled: false, installed: false, killed: false, size_bytes: 468569086, size_words: '0.5 GB',
    download_words: 'Describe pictures: a 0.5 GB download', progress: 0, step: '', error: '', media_ready: true },
  documents: { pack: 'documents', label: 'Read scanned documents better', what: 'A larger model reads scanned pages.',
    state: 'off', enabled: false, installed: false, killed: false, size_bytes: 1930469491, size_words: '1.9 GB',
    download_words: 'Read scanned documents better: a 1.9 GB download', progress: 0, step: '', error: '', media_ready: true },
  regions: { pack: 'regions', label: 'Remember where text was found', what: 'Keeps the page and place.',
    state: 'off', enabled: false, installed: true, killed: false, size_bytes: 0, size_words: '0 GB',
    download_words: 'No download', progress: 1, step: '', error: '', media_ready: true },
};
const PACK = (pack, extra = {}) => ({ ...BASE[pack], ...extra });
const ALL = (over = {}) => ({ packs: ['captions', 'documents', 'regions'].map((n) => PACK(n, over[n] || {})) });

const open = [];
afterEach(() => { while (open.length) open.pop().window.close(); });

function page(handlers = {}, packs = ALL(), confirm = null) {
  const calls = [];
  const dom = new JSDOM('<!DOCTYPE html><html><body><div id="host"></div></body></html>',
                        { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  dom.window.fetch = (url, init = {}) => {
    const method = init.method || 'GET';
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ method, url, body });
    const key = `${method} ${url}`;
    if (handlers[key]) return handlers[key](body);
    if (key === 'GET /api/v3/features/media/packs') return reply(200, packs);
    return reply(404, { detail: 'Not found.' });
  };
  if (confirm) dom.window.confirmDestructive = confirm;
  dom.window.eval(SRC);
  open.push(dom);
  const host = dom.window.document.getElementById('host');
  return { dom, calls, host, doc: dom.window.document, text: () => host.textContent,
           mount: () => dom.window.odMediaPacks.mount(host) };
}
const buttons = (doc) => [...doc.querySelectorAll('button')];
const button = (doc, label) => buttons(doc).find((b) => b.textContent === label);
const row = (p, name) => p.doc.querySelector(`[data-pack="${name}"]`);

describe('Better pictures and documents card', () => {
  it('is wired into the pane and its script is stamped', () => {
    assert.match(HTML, /static\/js\/od-media-packs\.js\?v=[0-9a-f]{8}/);
    assert.match(MEDIA, /odMediaPacks\.mount\(packs\)/);
  });

  it('shows the three parts, off, with the download size on the button before anything is downloaded', async () => {
    const p = page();
    await p.mount(); await flush();
    assert.match(p.text(), /Better pictures and documents/);
    assert.match(p.text(), /nothing is downloaded without your yes/);
    assert.match(row(p, 'captions').textContent, /Describe pictures: a 0\.5 GB download/);
    assert.ok(button(p.doc, 'Turn on (0.5 GB download)'));
    assert.ok(button(p.doc, 'Turn on (1.9 GB download)'));
    assert.deepEqual(buttons(p.doc).filter((b) => b.textContent === 'Turn on').length, 1); // the one with no download
    assert.equal(p.calls.filter((c) => c.method === 'POST').length, 0);
  });

  it('asks for a yes with the size in words, and only then downloads', async () => {
    let asked = null;
    const p = page({ 'POST /api/v3/features/media/packs/enable': (b) => reply(202, { pack: PACK('captions', { state: 'downloading' }) }) },
      ALL(), (o) => { asked = o; return Promise.resolve(true); });
    await p.mount(); await flush();
    button(p.doc, 'Turn on (0.5 GB download)').click(); await flush(60);
    assert.match(asked.consequence, /a 0\.5 GB download\. It stays on this computer/);
    const sent = p.calls.find((c) => c.method === 'POST');
    assert.deepEqual(sent.body, { pack: 'captions', yes: true, source: 'dashboard' });
  });

  it('a declined confirmation downloads nothing', async () => {
    const p = page({}, ALL(), () => Promise.resolve(false));
    await p.mount(); await flush();
    button(p.doc, 'Turn on (0.5 GB download)').click(); await flush(60);
    assert.equal(p.calls.filter((c) => c.method === 'POST').length, 0);
  });

  it('shows progress while downloading and no buttons to start it again', async () => {
    const p = page({}, ALL({ captions: { state: 'downloading', progress: 0.4, step: 'Downloading the model (180 MB)' } }));
    await p.mount(); await flush();
    const r = row(p, 'captions');
    assert.equal(r.querySelector('[role=progressbar]').getAttribute('aria-valuenow'), '40');
    assert.match(r.textContent, /Downloading the model \(180 MB\)/);
    assert.equal([...r.querySelectorAll('button')].length, 0);
  });

  it('says in plain words when it did not finish and offers to try again', async () => {
    const p = page({}, ALL({ captions: { state: 'failed', error: 'The download did not finish. Check your connection and try again.' } }));
    await p.mount(); await flush();
    assert.match(row(p, 'captions').textContent, /The download did not finish\. Check your connection/);
    assert.ok(button(row(p, 'captions').ownerDocument, 'Turn on (0.5 GB download)'));
  });

  it('a part that is on offers Turn off, with the choice to remove the files', async () => {
    const p = page({ 'POST /api/v3/features/media/packs/disable': (b) => reply(200, { pack: PACK('captions') }) },
      ALL({ captions: { state: 'on', enabled: true, installed: true } }));
    await p.mount(); await flush();
    const r = row(p, 'captions');
    assert.match(r.textContent, /On/);
    r.querySelector('input[type=checkbox]').checked = true;
    [...r.querySelectorAll('button')].find((b) => b.textContent === 'Turn off').click(); await flush(60);
    assert.deepEqual(p.calls.find((c) => c.method === 'POST').body, { pack: 'captions', remove_files: true });
  });

  it('Turn off without the box ticked keeps the downloaded files', async () => {
    const p = page({ 'POST /api/v3/features/media/packs/disable': () => reply(200, { pack: PACK('captions') }) },
      ALL({ captions: { state: 'on', enabled: true, installed: true } }));
    await p.mount(); await flush();
    [...row(p, 'captions').querySelectorAll('button')].find((b) => b.textContent === 'Turn off').click(); await flush(60);
    assert.deepEqual(p.calls.find((c) => c.method === 'POST').body, { pack: 'captions', remove_files: false });
  });

  it('the places switch offers no "remove the downloaded files" box because it has no files', async () => {
    const p = page({}, ALL({ regions: { state: 'on', enabled: true } }));
    await p.mount(); await flush();
    assert.equal(row(p, 'regions').querySelector('input[type=checkbox]'), null);
    assert.ok([...row(p, 'regions').querySelectorAll('button')].find((b) => b.textContent === 'Turn off'));
  });

  it('while a download runs it asks again every couple of seconds and shows the finished part', async () => {
    let reads = 0;
    const p = page({ 'GET /api/v3/features/media/packs': () => {
      reads += 1;
      return reply(200, reads === 1 ? ALL({ captions: { state: 'downloading', progress: 0.5 } })
        : ALL({ captions: { state: 'on', enabled: true, installed: true } }));
    } });
    await p.mount(); await flush();
    assert.equal(reads, 1);
    await flush(2400);
    assert.equal(reads, 2);
    assert.match(row(p, 'captions').textContent, /On/);
    assert.equal(row(p, 'captions').querySelector('[role=progressbar]'), null);
  });

  it('the places switch has no download and turns on without a question', async () => {
    let asked = false;
    const p = page({ 'POST /api/v3/features/media/packs/enable': () => reply(200, { pack: PACK('regions', { state: 'on' }) }) },
      ALL(), () => { asked = true; return Promise.resolve(true); });
    await p.mount(); await flush();
    button(row(p, 'regions').ownerDocument, 'Turn on').click(); await flush(60);
    assert.equal(asked, false);
    assert.deepEqual(p.calls.find((c) => c.method === 'POST').body, { pack: 'regions', yes: true, source: 'dashboard' });
  });

  it('cannot be turned on until images and documents are on, and says why', async () => {
    const p = page({}, ALL({ captions: { media_ready: false } }));
    await p.mount(); await flush();
    assert.equal(button(p.doc, 'Turn on (0.5 GB download)').disabled, true);
    assert.match(row(p, 'captions').textContent, /Turn on Images & documents first/);
  });

  it('shows the daemon refusal in its own words and the kill switch plainly', async () => {
    const refusal = 'Not enough free disk space: this needs about 1.0 GB free and 0.2 GB is free.';
    const p = page({ 'POST /api/v3/features/media/packs/enable': () => reply(409, { detail: refusal }) },
      ALL(), () => Promise.resolve(true));
    await p.mount(); await flush();
    button(p.doc, 'Turn on (0.5 GB download)').click(); await flush(60);
    assert.match(row(p, 'captions').textContent, /Not enough free disk space/);
    const killed = page({}, ALL({ captions: { killed: true } }));
    await killed.mount(); await flush();
    assert.match(row(killed, 'captions').textContent, /switched off for now/);
    assert.equal(row(killed, 'captions').querySelectorAll('button').length, 0);
  });

  it('draws nothing on an older daemon, never shows raw ids, and never parses server text as HTML', async () => {
    const older = page({ 'GET /api/v3/features/media/packs': () => reply(404, {}) });
    await older.mount(); await flush();
    assert.equal(older.text(), '');
    const hostile = page({}, ALL({ captions: { what: '<img src=x onerror=alert(1)>' } }));
    await hostile.mount(); await flush();
    assert.equal(hostile.doc.querySelector('img'), null);
    assert.doesNotMatch(hostile.text(), /florence|PaddlePaddle|[0-9a-f]{32}|media\.captions|flag/i);
  });

  it('a daemon that cannot be reached says so in words', async () => {
    const p = page({ 'GET /api/v3/features/media/packs': () => reply(500, {}) });
    await p.mount(); await flush();
    assert.match(p.text(), /Could not read these settings/);
  });
});
