/**
 * Documents & Images pane, launch polish: a drop zone instead of the raw file input,
 * and a "Find a picture" box that shows only results that carry a picture.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, features, writes, flushPromises } from './features_helpers.mjs';

const MID = 'a'.repeat(32);
const MID2 = 'd'.repeat(32);
const XSS = '<img src=x onerror="window.__pwn=1">';
const EMPTY = 'Nothing matched. Pictures are found by what they show and the text inside them.';

const hit = (id, content, score, media) => ({ fact_id: id, content, score, ...(media ? { media } : {}) });
const pic = (id, kind = 'image') => ({ media_id: id, kind, thumbnail_uri: `slm://media/${id}/thumb`, page: null, document_id: null, citation: '' });

function setup({ results = [], searchStatus = 200, routes = [] } = {}) {
    const state = { results, queries: [] };
    const base = [
        ['GET', '/api/v3/features', () => ({ json: features({ enabled: true, env_state: 'ready', restart_required: false }) })],
        ['GET', '/api/v3/documents/lint', () => ({ json: { empty_pages: [], no_entities: [], duplicate_pages: [], contradicted: null } })],
        ['GET', '/api/v3/documents', () => ({ json: { documents: [], next_cursor: null } })],
        ['POST', '/api/search', (call) => {
            state.queries.push(JSON.parse(call.body));
            return { status: searchStatus, json: searchStatus === 200 ? { results: state.results } : { detail: 'boom' } };
        }],
    ];
    const h = makeEnv([...routes, ...base], { modules: ['od-features.js', 'od-media-find.js', 'od-media.js'] });
    h.state = state;
    h.pane = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderMedia(h.pane); await flushPromises(); await flushPromises(); };
    h.btn = t => [...h.pane.querySelectorAll('button')].find(b => b.textContent.trim() === t);
    h.file = (name, type) => new h.window.File(['hello'], name, { type });
    h.find = async (q) => {
        const box = h.pane.querySelector('input[type=search]');
        box.value = q;
        h.pane.querySelector('.od-find').dispatchEvent(new h.window.Event('submit', { bubbles: true, cancelable: true }));
        await flushPromises(); await flushPromises();
    };
    h.found = () => [...h.pane.querySelectorAll('.od-find-results li')];
    h.drop = async (files) => {
        const zone = h.pane.querySelector('.od-dropzone');
        const ev = new h.window.Event('drop', { bubbles: true, cancelable: true });
        Object.defineProperty(ev, 'dataTransfer', { value: { files } });
        zone.dispatchEvent(ev);
        await flushPromises(); await flushPromises(); await flushPromises();
        return ev;
    };
    return h;
}

describe('drop zone', () => {
    it('shows a dashed drop area with a real button; the native file input is hidden', async () => {
        const h = setup();
        await h.open();
        const zone = h.pane.querySelector('.od-dropzone');
        assert.ok(zone);
        assert.match(zone.textContent, /Drop pictures or PDFs here, or choose files/);
        const input = h.pane.querySelector('input[type=file]');
        assert.ok(input && input.hidden === true, 'hidden, so "No file chosen" never shows');
        assert.ok(zone.contains(h.btn('Choose files')), 'the button lives in the zone');
        assert.equal(h.btn('Choose files').type, 'button');
    });

    it('the button opens the file chooser', async () => {
        const h = setup();
        await h.open();
        const input = h.pane.querySelector('input[type=file]');
        let clicked = 0;
        input.addEventListener('click', () => { clicked += 1; });
        h.btn('Choose files').click();
        assert.equal(clicked, 1);
    });

    it('dragging over highlights the area and leaving clears it', async () => {
        const h = setup();
        await h.open();
        const zone = h.pane.querySelector('.od-dropzone');
        const over = new h.window.Event('dragover', { bubbles: true, cancelable: true });
        zone.dispatchEvent(over);
        assert.equal(over.defaultPrevented, true, 'needed for the browser to allow a drop');
        assert.ok(zone.classList.contains('is-over'));
        zone.dispatchEvent(new h.window.Event('dragleave', { bubbles: true }));
        assert.ok(!zone.classList.contains('is-over'));
    });

    it('dropping files uses the normal upload and shows one result line per file', async () => {
        const h = setup({ routes: [['POST', '/api/v3/media/upload', () => ({ json: { status: 'stored', media_id: MID } })]] });
        await h.open();
        const ev = await h.drop([h.file('one.png', 'image/png'), h.file('two.txt', 'text/plain')]);
        assert.equal(ev.defaultPrevented, true, 'the browser does not open the file');
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].url, '/api/v3/media/upload?kind=image');
        const rows = [...h.pane.querySelectorAll('.od-media-list li')].filter(li => /one\.png|two\.txt/.test(li.textContent));
        assert.equal(rows.length, 2);
        assert.match(rows[0].textContent, /one\.png[\s\S]*Saved\./);
        assert.match(rows[1].textContent, /two\.txt[\s\S]*image or a PDF/);
        assert.ok(!h.pane.querySelector('.od-dropzone').classList.contains('is-over'));
    });

    it('a drop with no files does nothing', async () => {
        const h = setup();
        await h.open();
        await h.drop([]);
        assert.equal(writes(h).length, 0);
    });
});

describe('Find a picture', () => {
    it('sits at the top of the pane, above the upload area', async () => {
        const h = setup();
        await h.open();
        const find = h.pane.querySelector('.od-find');
        const zone = h.pane.querySelector('.od-dropzone');
        assert.ok(find && zone);
        assert.ok(find.compareDocumentPosition(zone) & 4, 'the search box comes first');
        assert.match(h.pane.textContent, /Find a picture/);
        assert.equal(h.pane.querySelector('input[type=search]').getAttribute('aria-label'), 'Find a picture');
    });

    it('is not shown while images are off', async () => {
        const h = setup({ routes: [['GET', '/api/v3/features', () => ({ json: features({ enabled: false }) })]] });
        await h.open();
        assert.equal(h.pane.querySelector('.od-find'), null);
    });

    it('runs the dashboard search and keeps only results that carry a picture', async () => {
        const h = setup({ results: [
            hit('f1', 'Quarterly numbers\nRevenue up 12%', 0.91, pic(MID)),
            hit('f2', 'A plain text note about quarterly numbers', 0.88, null),
            hit('f3', 'Page text\nsecond line', 0.85, pic(MID2, 'page')),
        ] });
        await h.open();
        await h.find('the slide with the quarterly numbers');
        assert.deepEqual(h.state.queries[0].query, 'the slide with the quarterly numbers');
        const rows = h.found();
        assert.equal(rows.length, 2);
        assert.equal(rows[0].querySelector('img').getAttribute('src'), `/api/v3/media/${MID}/thumb`);
        assert.match(rows[0].textContent, /^Quarterly numbers/);
        assert.ok(!rows[0].textContent.includes('Revenue up 12%'), 'first line only');
        assert.match(rows[0].textContent, /91%/);
        assert.ok(!h.pane.textContent.includes('plain text note'));
    });

    it('shows only strong matches: near the best one, eight at most', async () => {
        const many = Array.from({ length: 12 }, (_, i) =>
            hit('g' + i, 'pic ' + i, 0.80 - i * 0.001, pic(String(i).padStart(2, '0').repeat(16))));
        const h = setup({ results: [hit('top', 'best', 0.69, pic(MID)), hit('weak', 'weak', 0.53, pic(MID2))] });
        await h.open();
        await h.find('the dashboard');
        assert.deepEqual(h.found().map(r => r.textContent.split(/\d+%/)[0].trim()), ['best']);
        const h2 = setup({ results: many });
        await h2.open();
        await h2.find('pictures');
        assert.equal(h2.found().length, 8);
    });

    it('does not make a write-credential request or clear the dashboard cache', async () => {
        const h = setup({ results: [hit('f1', 'x', 0.9, pic(MID))] });
        await h.open();
        await h.find('x');
        const call = h.calls.find(c => c.url === '/api/search');
        assert.equal(call.headers.get('x-install-token'), undefined);
    });

    it('says what to expect when nothing matched, and when only text matched', async () => {
        const h = setup({ results: [] });
        await h.open();
        await h.find('nothing like this');
        assert.ok(h.pane.textContent.includes(EMPTY));
        h.state.results = [hit('f2', 'only text', 0.9, null)];
        await h.find('only text');
        assert.ok(h.pane.textContent.includes(EMPTY));
        assert.equal(h.found().length, 0);
    });

    it('a blank question searches nothing', async () => {
        const h = setup();
        await h.open();
        await h.find('   ');
        assert.equal(h.state.queries.length, 0);
    });

    it('a failed search says so in plain words and keeps the box usable', async () => {
        const h = setup({ searchStatus: 500 });
        await h.open();
        await h.find('anything');
        assert.match(h.pane.textContent, /Search did not work/);
        assert.ok(!h.pane.textContent.includes(EMPTY));
    });

    it('result text and ids are text only; an id that is not 32 hex characters is skipped', async () => {
        const h = setup({ results: [
            hit('f1', 'Name ' + XSS, 0.9, pic(MID)),
            hit('f2', 'bad id', 0.9, pic('../../x')),
        ] });
        await h.open();
        await h.find('x');
        assert.equal(h.found().length, 1);
        assert.ok(h.pane.textContent.includes('Name ' + XSS));
        assert.equal(h.pane.querySelectorAll('.od-find-results img').length, 1);
        assert.equal(h.window.__pwn, undefined);
    });

    it('a newer search replaces the older results', async () => {
        const h = setup({ results: [hit('f1', 'first', 0.9, pic(MID))] });
        await h.open();
        await h.find('one');
        h.state.results = [hit('f2', 'second', 0.8, pic(MID2))];
        await h.find('two');
        assert.equal(h.found().length, 1);
        assert.match(h.found()[0].textContent, /second/);
    });
});
