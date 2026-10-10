/**
 * Documents & Images pane (js/od-media.js): upload, thumbnails, documents, lint.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, features, writes, flushPromises, TOKEN } from './features_helpers.mjs';

const MID = 'a'.repeat(32);
const DID = 'b'.repeat(32);
const JID = 'c'.repeat(32);
const XSS = '<img src=x onerror="window.__pwn=1">';
const MB = 1024 * 1024;
import { readFileSync } from 'node:fs';
const require_sources = () => readFileSync(new URL('../../src/superlocalmemory/ui/js/od-sources.js', import.meta.url), 'utf8');

function setup({ on = true, routes = [], docs = [], ram } = {}) {
    const media = on ? { enabled: true, env_state: 'ready', restart_required: false, ...(ram ? { ram } : {}) }
                     : { enabled: false, env_state: 'not_installed' };
    const state = { docs, jobs: [] };
    const base = [
        ['GET', '/api/v3/features', () => ({ json: features(media) })],
        ['GET', '/api/v3/documents/lint', () => ({ json: { empty_pages: [], no_entities: [1, 2], duplicate_pages: [], contradicted: null } })],
        ['GET', '/api/v3/documents', () => ({ json: { documents: state.docs, next_cursor: null } })],
        ['GET', /^\/api\/v3\/jobs\//, () => ({ json: state.jobs.shift() })],
    ];
    const h = makeEnv([...routes, ...base], { modules: ['od-features.js', 'od-media.js'] });
    h.state = state;
    h.pane = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderMedia(h.pane); await flushPromises(); };
    h.pick = async (file) => {
        const input = h.pane.querySelector('input[type=file]');
        Object.defineProperty(input, 'files', { value: [file], configurable: true });
        input.dispatchEvent(new h.window.Event('change', { bubbles: true }));
        await flushPromises(); await flushPromises();
    };
    h.click_turn_on = async () => {
        Array.from(h.pane.querySelectorAll('button')).find(b => b.textContent === 'Turn on').click();
        await flushPromises(); await flushPromises(); await flushPromises();
    };
    h.file = (name, type, size) => {
        const f = new h.window.File(['hello'], name, { type });
        if (size) Object.defineProperty(f, 'size', { value: size });
        return f;
    };
    return h;
}

describe('Documents & Images pane', () => {
    it('off: only the turn-on card, no upload', async () => {
        const h = setup({ on: false });
        await h.open();
        assert.match(h.pane.textContent, /Turn on images and documents/);
        assert.equal(h.pane.querySelectorAll('input[type=file]').length, 0);
        assert.equal(h.calls.filter(c => c.url.startsWith('/api/v3/documents')).length, 0);
    });

    it('on: upload control, documents and lint are requested', async () => {
        const h = setup();
        await h.open();
        assert.ok(h.pane.querySelector('input[type=file]'));
        assert.ok(h.calls.some(c => c.url.startsWith('/api/v3/documents') && !c.url.includes('lint')));
        assert.ok(h.calls.some(c => c.url === '/api/v3/documents/lint'));
        assert.match(h.pane.textContent, /2 .*without people or places/i);
    });

    it('image upload: base64 JSON with the credential, thumbnail at the right URL, preview as text', async () => {
        const preview = 'Invoice ' + XSS;
        const h = setup({ routes: [['POST', '/api/v3/media/remember', () => ({
            json: { status: 'stored', media_id: MID, memory_id: 'm1', extracted_text_preview: preview } })]] });
        await h.open();
        await h.pick(h.file('scan.png', 'image/png'));
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].url, '/api/v3/media/remember');
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
        const body = JSON.parse(w[0].body);
        assert.equal(body.base64, Buffer.from('hello').toString('base64'));
        const imgs = h.pane.querySelectorAll('img');
        assert.equal(imgs.length, 1, 'only the thumbnail, nothing built from the preview');
        assert.equal(imgs[0].getAttribute('src'), `/api/v3/media/${MID}/thumb`);
        assert.ok(h.pane.textContent.includes(preview), 'preview shown as text');
        assert.equal(h.window.__pwn, undefined);
    });

    it('an id that is not 32 hex characters gets no thumbnail request', async () => {
        const h = setup({ routes: [['POST', '/api/v3/media/remember', () => ({
            json: { status: 'stored', media_id: '../../x' } })]] });
        await h.open();
        await h.pick(h.file('a.png', 'image/png'));
        assert.equal(h.pane.querySelectorAll('img').length, 0);
    });

    it('refuses an oversized image and an oversized PDF before any request', async () => {
        const h = setup();
        await h.open();
        await h.pick(h.file('big.png', 'image/png', 25 * MB + 1));
        await h.pick(h.file('big.pdf', 'application/pdf', 100 * MB + 1));
        assert.equal(writes(h).length, 0);
        assert.match(h.pane.textContent, /25 MB/);
        assert.match(h.pane.textContent, /100 MB/);
    });

    it('refuses a file type that is neither image nor PDF', async () => {
        const h = setup();
        await h.open();
        await h.pick(h.file('x.exe', 'application/octet-stream'));
        assert.equal(writes(h).length, 0);
        assert.match(h.pane.textContent, /image or a PDF/);
    });

    it('a refused receipt shows the server reason as text', async () => {
        const h = setup({ routes: [['POST', '/api/v3/media/remember', () => ({
            status: 422, json: { status: 'refused', reason: 'bad ' + XSS } })]] });
        await h.open();
        await h.pick(h.file('a.png', 'image/png'));
        assert.ok(h.pane.textContent.includes('bad ' + XSS));
        assert.equal(h.pane.querySelectorAll('img').length, 0);
    });

    it('PDF upload: file name sent, job polled until done, then the list reloads and polling stops', async () => {
        const h = setup({ routes: [['POST', '/api/v3/documents', () => ({
            status: 202, json: { status: 'processing', document_id: DID, job_id: JID } })]] });
        h.state.jobs = [
            { job_id: JID, state: 'running', done: 1, total: 4 },
            { job_id: JID, state: 'done', done: 4, total: 4, document_id: DID },
        ];
        await h.open();
        await h.pick(h.file('paper.pdf', 'application/pdf'));
        const body = JSON.parse(writes(h)[0].body);
        assert.equal(body.file_name, 'paper.pdf');
        assert.equal(writes(h)[0].url, '/api/v3/documents');
        await h.tick();
        assert.match(h.pane.textContent, /1 of 4/);
        const lists = () => h.calls.filter(c => c.url.startsWith('/api/v3/documents?')).length;
        const before = lists();
        await h.tick();
        assert.ok(lists() > before, 'list reloaded after the job finished');
        assert.equal(h.timers.length, 0);
        assert.ok(h.calls.some(c => c.url === `/api/v3/jobs/${JID}`));
    });

    it('document titles are text, and removing one confirms then DELETEs', async () => {
        const h = setup({ docs: [{ document_id: DID, title: XSS, state: 'ready', page_count: 3,
            pages_text_layer: 3, pages_ocr: 0, pages_empty: 0, created_at: 1, entities: [{ name: XSS, facts: 2 }] }],
            routes: [['DELETE', `/api/v3/documents/${DID}`, () => ({ json: { removed: true } })]] });
        await h.open();
        assert.equal(h.pane.querySelectorAll('img').length, 0);
        assert.ok(h.pane.textContent.includes(XSS));
        const rm = [...h.pane.querySelectorAll('button')].find(b => /Remove/.test(b.textContent));
        rm.click();
        await flushPromises();
        assert.equal(h.confirms.length, 1);
        const w = writes(h);
        assert.equal(w[0].method, 'DELETE');
        assert.equal(w[0].url, `/api/v3/documents/${DID}`);
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
    });

    it('declining the remove confirm sends nothing', async () => {
        const h = setup({ docs: [{ document_id: DID, title: 't', state: 'ready', page_count: 1, entities: [] }] });
        h.confirmAnswer = false;
        await h.open();
        [...h.pane.querySelectorAll('button')].find(b => /Remove/.test(b.textContent)).click();
        await flushPromises();
        assert.equal(writes(h).length, 0);
    });

    it('a 404 documents route says not available in this build and the pane still works', async () => {
        const h = setup();
        h.calls.length = 0;
        const orig = h.window.fetch;
        h.window.fetch = (u, i) => (String(u).startsWith('/api/v3/documents')
            ? Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) })
            : orig(u, i));
        await h.open();
        assert.match(h.pane.textContent, /not available in this build/);
        assert.ok(h.pane.querySelector('input[type=file]'));
    });

    it('Folders show below the turn-on card while images are off', async () => {
        const h = setup({ on: false, routes: [['GET', '/api/v3/sources', () => ({ json: { sources: [] } })]] });
        h.window.eval(require_sources());
        await h.open();
        assert.match(h.pane.textContent, /Turn on images and documents/);
        assert.equal(h.pane.querySelectorAll('input[type=file]').length, 0);
        assert.equal(h.pane.textContent.split('No folders connected').length - 1, 1);
        assert.ok(h.calls.some(c => c.url === '/api/v3/sources'));
    });

    it('Folders show with the body while images are on', async () => {
        const h = setup({ routes: [['GET', '/api/v3/sources', () => ({ json: { sources: [] } })]] });
        h.window.eval(require_sources());
        await h.open();
        assert.ok(h.pane.querySelector('input[type=file]'));
        assert.equal(h.pane.textContent.split('No folders connected').length - 1, 1);
    });

    it('turning images on leaves exactly one Folders section', async () => {
        let media = { enabled: false, env_state: 'not_installed' };
        const h = setup({ on: false, routes: [
            ['GET', '/api/v3/sources', () => ({ json: { sources: [] } })],
            ['GET', '/api/v3/features', () => ({ json: features(media) })],
            ['POST', '/api/v3/features/media/enable', () => {
                media = { enabled: true, env_state: 'ready', restart_required: false };
                return { json: { media } };
            }]] });
        h.window.eval(require_sources());
        await h.open();
        await h.click_turn_on();
        assert.ok(h.pane.querySelector('input[type=file]'), 'body appeared');
        assert.equal(h.pane.querySelectorAll('h3').length &&
            Array.from(h.pane.querySelectorAll('h3')).filter(x => x.textContent === 'Folders').length, 1);
        assert.equal(h.calls.filter(c => c.url === '/api/v3/sources').length, 1);
    });
});

describe('Saved images grid', () => {
    const idOf = n => String(n).repeat(32).slice(0, 32);
    const item = (n, extra = {}) => ({ media_id: idOf(n), has_thumb: true, created_at: 't' + n, ...extra });
    const listRoute = (pages) => ['GET', '/api/v3/media', (call) => {
        const cursor = new URL(call.url, 'http://x').searchParams.get('cursor') || '';
        return { json: pages[cursor] };
    }];
    const thumbs = h => Array.from(h.pane.querySelectorAll('img')).map(i => i.getAttribute('src'));
    const more = h => Array.from(h.pane.querySelectorAll('button')).find(b => b.textContent === 'Show more');

    it('shows the images saved earlier when the pane opens', async () => {
        const h = setup({ routes: [listRoute({ '': { items: [item(1), item(2)], next_cursor: null } })] });
        await h.open();
        assert.ok(h.calls.some(c => c.url.startsWith('/api/v3/media?limit=60')));
        assert.deepEqual(thumbs(h), [`/api/v3/media/${idOf(1)}/thumb`, `/api/v3/media/${idOf(2)}/thumb`]);
        assert.equal(more(h), undefined);
    });

    it('skips images that have no thumbnail', async () => {
        const h = setup({ routes: [listRoute({ '': { items: [item(1, { has_thumb: false }), item(2)], next_cursor: null } })] });
        await h.open();
        assert.deepEqual(thumbs(h), [`/api/v3/media/${idOf(2)}/thumb`]);
    });

    it('Show more appears only with a next cursor and appends the next page', async () => {
        const h = setup({ routes: [listRoute({
            '': { items: [item(1)], next_cursor: 'abc' },
            abc: { items: [item(2)], next_cursor: null } })] });
        await h.open();
        assert.equal(thumbs(h).length, 1);
        more(h).click();
        await flushPromises(); await flushPromises();
        assert.ok(h.calls.some(c => c.url.includes('cursor=abc')));
        assert.equal(thumbs(h).length, 2);
        assert.equal(more(h), undefined, 'no more pages, no button');
    });

    it('an image already on screen is not added twice', async () => {
        const h = setup({ routes: [
            ['POST', '/api/v3/media/remember', () => ({ json: { status: 'stored', media_id: idOf(1) } })],
            listRoute({ '': { items: [item(1), item(2)], next_cursor: null } })] });
        await h.open();
        await h.pick(h.file('a.png', 'image/png'));
        assert.deepEqual(thumbs(h), [`/api/v3/media/${idOf(1)}/thumb`, `/api/v3/media/${idOf(2)}/thumb`]);
    });

    it('a failed load leaves the grid empty and the rest of the pane working', async () => {
        const h = setup({ routes: [['GET', '/api/v3/media', () => ({ status: 500, json: { detail: 'boom' } })]] });
        await h.open();
        assert.equal(thumbs(h).length, 0);
        assert.equal(more(h), undefined);
        assert.ok(h.pane.querySelector('input[type=file]'));
    });

    it('an id with markup is never parsed as HTML', async () => {
        const h = setup({ routes: [listRoute({ '': { items: [item(1, { media_id: XSS })], next_cursor: null } })] });
        await h.open();
        assert.equal(thumbs(h).length, 0);
        assert.equal(h.window.__pwn, undefined);
        assert.ok(!h.pane.innerHTML.includes('onerror'));
    });
});

describe('Documents & Images pane: picture model memory line', () => {
    const line = h => h.pane.querySelector('[data-od-ram]');

    it('shows what the picture model uses and its limit', async () => {
        const h = setup({ ram: { system_total_mb: 16000, worker_rss_mb: 2400, worker_cap_mb: 4500, model: 'google/embeddinggemma-2' } });
        await h.open();
        assert.equal(line(h).textContent, 'Picture model: 2.4 GB in use, limit 4.5 GB');
        assert.ok(line(h).className.includes('muted'));
    });

    it('says not running when the worker is idle', async () => {
        const h = setup({ ram: { system_total_mb: 16000, worker_rss_mb: null, worker_cap_mb: 4500, model: 'google/embeddinggemma-2' } });
        await h.open();
        assert.equal(line(h).textContent, 'Picture model: not running');
    });

    it('shows no line when the daemon sends no memory block', async () => {
        const h = setup({});
        await h.open();
        assert.equal(line(h).textContent, '');
        assert.equal(line(h).hidden, true);
    });

    it('asks nothing extra: one features read to draw the pane', async () => {
        const h = setup({ ram: { system_total_mb: 1, worker_rss_mb: null, worker_cap_mb: 1, model: '' } });
        await h.open();
        assert.equal(h.calls.filter(c => c.url === '/api/v3/features').length, 1);
    });
});
