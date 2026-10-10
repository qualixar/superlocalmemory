/**
 * Folders section: "Choose folder" (the computer's own picker), one-click suggestions, and the
 * text box kept as "or type a path". A picked path still goes through the preview and confirm steps.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeEnv, flushPromises } from './features_helpers.mjs';

const PICK = '/api/v3/sources/pick-folder';
const SUGG = '/api/v3/sources/suggestions';
const preview = { source_id: 'b'.repeat(32), root: '/notes/new', kind: 'folder', files_by_type: { '.md': 3 },
    skipped_by_rule: {}, est_bytes: 1, quarantined_count: 0, est_seconds: 1, estimate_note: '', capped: false, warnings: [] };

function setup({ pick, suggestions = [], suggStatus = 200 } = {}) {
    const h = makeEnv([
        ['GET', '/api/v3/sources', () => ({ json: { sources: [] } })],
        ['GET', SUGG, () => suggStatus === 200 ? { json: { suggestions } } : { status: suggStatus, json: { detail: { code: 'remote_access_on' } } }],
        ['POST', PICK, () => pick || { json: { path: '/Users/me/Notes' } }],
        ['POST', '/api/v3/sources', () => ({ json: Object.assign({}, preview, { root: h.lastPath }) })],
    ], { modules: ['od-features.js', 'od-sources.js'] });
    h.host = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderSources(h.host); await flushPromises(); await flushPromises(); };
    h.button = label => Array.from(h.host.querySelectorAll('button')).find(b => b.textContent === label);
    h.click = async b => { b.click(); for (let i = 0; i < 4; i++) await flushPromises(); };
    h.posts = () => h.calls.filter(c => c.method === 'POST');
    return h;
}

describe('Choose folder', () => {
    it('calls the pick route, fills the path and opens the existing preview (never connects)', async () => {
        const h = setup();
        await h.open();
        await h.click(h.button('Choose folder'));
        assert.equal(h.posts()[0].url, PICK);
        const input = h.host.querySelector('input[type=text]');
        assert.equal(input.value, '/Users/me/Notes');
        const second = h.posts()[1];
        assert.equal(second.url, '/api/v3/sources');
        assert.deepEqual(JSON.parse(second.body), { path: '/Users/me/Notes' });
        assert.ok(h.button('Connect this folder'), 'preview offers the confirm step');
        assert.equal(h.posts().length, 2, 'nothing is confirmed on its own');
        assert.equal(h.confirms.length, 0);
    });

    it('a cancelled dialog changes nothing', async () => {
        const h = setup({ pick: { json: { cancelled: true } } });
        await h.open();
        await h.click(h.button('Choose folder'));
        assert.equal(h.posts().length, 1);
        assert.equal(h.host.querySelector('input[type=text]').value, '');
    });

    it('picker_unavailable removes the button and leaves the text box', async () => {
        const h = setup({ pick: { status: 501, json: { detail: { code: 'picker_unavailable', message: 'x' } } } });
        await h.open();
        await h.click(h.button('Choose folder'));
        assert.equal(h.button('Choose folder'), undefined);
        assert.ok(h.host.querySelector('input[type=text]'));
        assert.ok(h.button('Check folder'));
    });

    it('a busy dialog or refusal is said in words and the button stays', async () => {
        const h = setup({ pick: { status: 409, json: { detail: { code: 'picker_busy', message: 'A folder dialog is already open.' } } } });
        await h.open();
        await h.click(h.button('Choose folder'));
        assert.match(h.host.textContent, /already open/);
        assert.ok(h.button('Choose folder'));
    });

    it('the text box still works, labelled "or type a path"', async () => {
        const h = setup();
        await h.open();
        assert.match(h.host.textContent, /or type a path/i);
        h.host.querySelector('input[type=text]').value = '/typed/here';
        h.lastPath = '/typed/here';
        await h.click(h.button('Check folder'));
        assert.deepEqual(JSON.parse(h.posts()[0].body), { path: '/typed/here' });
        assert.equal(h.posts()[0].url, '/api/v3/sources');
    });
});

describe('Suggestions', () => {
    const items = [{ path: '/Users/me/Vault', kind: 'obsidian', name: 'Vault' },
        { path: '/Users/me/Documents', kind: 'documents', name: 'Documents' },
        { path: '/Users/me/Desktop', kind: 'desktop', name: 'Desktop' }];

    it('shows a chip per folder; one click fills the path and opens the preview', async () => {
        const h = setup({ suggestions: items });
        await h.open();
        assert.ok(h.button('Vault') && h.button('Documents') && h.button('Desktop'));
        h.lastPath = '/Users/me/Documents';
        await h.click(h.button('Documents'));
        assert.equal(h.host.querySelector('input[type=text]').value, '/Users/me/Documents');
        assert.deepEqual(JSON.parse(h.posts()[0].body), { path: '/Users/me/Documents' });
        assert.ok(h.button('Connect this folder'));
        assert.equal(h.confirms.length, 0);
    });

    it('no chips when the route refuses or finds nothing', async () => {
        for (const h of [setup({ suggStatus: 409 }), setup({ suggestions: [] })]) {
            await h.open();
            assert.equal(h.host.querySelectorAll('.od-folder-chip').length, 0);
            assert.ok(h.button('Choose folder'));
        }
    });

    it('names and paths are written as text, never markup', async () => {
        const evil = '<img src=x onerror="window.__pwn=1">';
        const h = setup({ suggestions: [{ path: '/x/' + evil, kind: 'obsidian', name: evil }] });
        await h.open();
        assert.equal(h.host.querySelector('img'), null);
        assert.ok(h.host.textContent.includes(evil));
        assert.equal(h.window.__pwn, undefined);
    });
});

describe('source', () => {
    it('uses textContent only', () => {
        const js = readFileSync(new URL('../../src/superlocalmemory/ui/js/od-sources.js', import.meta.url), 'utf8');
        assert.doesNotMatch(js, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
    });
});
