/**
 * A failed PDF can be tried again from the copy already saved: no file is dropped again.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, features, writes, flushPromises, TOKEN } from './features_helpers.mjs';

const DID = 'b'.repeat(32);
const JID = 'c'.repeat(32);

function setup({ state = 'failed', retry } = {}) {
    const jobs = [];
    const docs = { list: [{ document_id: DID, title: 'Report', state, page_count: 3, pages_text_layer: 3,
        pages_ocr: 0, pages_empty: 0, created_at: 1, entities: [] }] };
    const routes = [
        ['GET', '/api/v3/features', () => ({ json: features({ enabled: true, env_state: 'ready' }) })],
        ['GET', '/api/v3/documents/lint', () => ({ json: { empty_pages: [], no_entities: [], duplicate_pages: [], contradicted: null } })],
        ['GET', '/api/v3/documents', () => ({ json: { documents: docs.list, next_cursor: null } })],
        ['GET', /^\/api\/v3\/jobs\//, () => ({ json: jobs.shift() })],
        ['POST', `/api/v3/documents/${DID}/retry`, retry || (() => ({
            status: 202, json: { status: 'processing', document_id: DID, job_id: JID } }))],
    ];
    const h = makeEnv(routes, { modules: ['od-features.js', 'od-media.js'] });
    h.jobs = jobs;
    h.docs = docs;
    h.pane = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderMedia(h.pane); await flushPromises(); };
    h.button = (re) => [...h.pane.querySelectorAll('button')].find(b => re.test(b.textContent));
    return h;
}

describe('trying a failed document again', () => {
    it('a failed document offers Try again next to Remove', async () => {
        const h = setup();
        await h.open();
        assert.ok(h.button(/^Try again$/));
        assert.ok(h.button(/^Remove$/));
    });

    it('a finished or still-reading document does not', async () => {
        for (const state of ['ready', 'processing']) {
            const h = setup({ state });
            await h.open();
            assert.equal(h.button(/^Try again$/), undefined, state);
            assert.ok(h.button(/^Remove$/));
        }
    });

    it('Try again POSTs the retry for that id, with no file, then follows the job to the end', async () => {
        const h = setup();
        h.jobs.push({ job_id: JID, state: 'running', done: 1, total: 4 },
                    { job_id: JID, state: 'done', done: 4, total: 4 });
        await h.open();
        h.button(/^Try again$/).click();
        await flushPromises(); await flushPromises();
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].method, 'POST');
        assert.equal(w[0].url, `/api/v3/documents/${DID}/retry`);
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
        assert.equal(w[0].rawBody, undefined, 'no file is sent again');
        await h.tick();
        assert.match(h.pane.textContent, /Reading page 1 of 4/);
        h.docs.list = [{ ...h.docs.list[0], state: 'ready' }];
        await h.tick();
        assert.match(h.pane.textContent, /Report ready/);
        assert.equal(h.button(/^Try again$/), undefined, 'the list reloaded with the finished document');
    });

    it('a refusal is shown in plain words and the button works again', async () => {
        const h = setup({ retry: () => ({ status: 422, json: {
            status: 'refused', detail: 'The saved copy of this document is missing. Add the document again.' } }) });
        await h.open();
        const b = h.button(/^Try again$/);
        b.click();
        await flushPromises(); await flushPromises();
        assert.match(h.pane.textContent, /saved copy of this document is missing/);
        assert.equal(b.disabled, false);
    });

    it('while the request is out the button cannot be pressed twice', async () => {
        const h = setup();
        await h.open();
        const b = h.button(/^Try again$/);
        b.click();
        assert.equal(b.disabled, true);
        await flushPromises(); await flushPromises();
    });
});
