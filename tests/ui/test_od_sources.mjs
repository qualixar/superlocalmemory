/**
 * Folders section (js/od-sources.js): list, add with preview, rescan, report, release, remove.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeEnv, features, writes, flushPromises, TOKEN } from './features_helpers.mjs';

const SID = 'a'.repeat(32);
const XSS = '<img src=x onerror="window.__pwn=1">';

function src(over = {}) {
    return Object.assign({ source_id: SID, profile_id: 'default', kind: 'folder', root_path: '/notes/work',
        display_name: 'work', state: 'active', include_types: ['.md'], files: { indexed: 7, quarantined: 1 },
        last_scan_at: '2026-10-01T10:00:00Z', offline_reason: null }, over);
}

function setup({ sources = [src()], routes = [], report = null, listStatus = 200 } = {}) {
    const state = { sources, report: report || { source_id: SID, state: 'active', counts: { indexed: 7 },
        skipped_by_rule: { too_big: 2 }, quarantined: [], cloud_only: [], errors: [], watch: 1 } };
    const base = [
        ['GET', '/api/v3/sources', () => listStatus === 200 ? { json: { sources: state.sources } }
            : { status: listStatus, json: { detail: 'Not found.' } }],
        ['GET', /^\/api\/v3\/sources\/[^/]+\/report$/, () => ({ json: state.report })],
    ];
    const h = makeEnv([...routes, ...base], { modules: ['od-features.js', 'od-sources.js'] });
    h.state = state;
    h.host = h.document.getElementById('pane');
    h.open = async () => { h.window.odRenderSources(h.host); await flushPromises(); };
    h.click = async (label, root) => {
        const b = Array.from((root || h.host).querySelectorAll('button')).find(x => x.textContent === label);
        assert.ok(b, 'button ' + label);
        b.click(); await flushPromises(); await flushPromises(); await flushPromises();
    };
    return h;
}

const preview = (over = {}) => Object.assign({ source_id: 'b'.repeat(32), root: '/notes/new', kind: 'obsidian',
    files_by_type: { '.md': 12, '.pdf': 2 }, skipped_by_rule: { hidden: 3 }, est_bytes: 1000,
    quarantined_count: 1, est_seconds: 4.5, estimate_note: 'Estimate only.', capped: false, warnings: [] }, over);

describe('Folders section', () => {
    it('lists name, path, kind, state, last scan and counts', async () => {
        const h = setup({ sources: [src({ kind: 'obsidian' })] });
        await h.open();
        const t = h.host.textContent;
        assert.match(t, /work/); assert.match(t, /\/notes\/work/); assert.match(t, /Obsidian/);
        assert.match(t, /7 indexed/); assert.match(t, /2026-10-01/);
    });

    it('says paused means remote access is on, and gives plain words for offline reasons', async () => {
        const h = setup({ sources: [src({ state: 'paused' }),
            src({ source_id: 'c'.repeat(32), state: 'offline', offline_reason: 'disk_changed' }),
            src({ source_id: 'd'.repeat(32), state: 'offline', offline_reason: 'unreachable' }),
            src({ source_id: 'e'.repeat(32), state: 'offline', offline_reason: 'empty_folder' }),
            src({ source_id: 'f'.repeat(32), state: 'offline', offline_reason: 'root_moved' }),
            src({ source_id: '1'.repeat(32), state: 'offline', offline_reason: 'home_directory' })] });
        await h.open();
        const t = h.host.textContent;
        assert.match(t, /Paused because remote access is on/);
        assert.match(t, /different disk/i);
        assert.match(t, /cannot be reached/i);
        assert.match(t, /empty/i);
        assert.match(t, /moved/i);
        assert.match(t, /home_directory/);
    });

    it('empty list says so; 404 says not available in this build', async () => {
        const a = setup({ sources: [] }); await a.open();
        assert.match(a.host.textContent, /No folders connected/);
        const b = setup({ listStatus: 404 }); await b.open();
        assert.match(b.host.textContent, /not available in this build/);
        assert.equal(b.host.querySelectorAll('input').length, 0);
    });

    it('add: preview, counts, refusals as words, confirm modal, then confirm call', async () => {
        const h = setup({ sources: [], routes: [
            ['POST', '/api/v3/sources', () => ({ json: preview({ warnings: ['over the limit'] }) })],
            ['POST', /\/confirm$/, () => ({ status: 202, json: { confirmed: true } })]] });
        await h.open();
        h.host.querySelector('input[type=text]').value = '/notes/new';
        await h.click('Check folder');
        const w = writes(h);
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
        assert.deepEqual(JSON.parse(w[0].body), { path: '/notes/new' });
        assert.equal(w.length, 1, 'preview saves nothing');
        const t = h.host.textContent;
        assert.match(t, /14 files/); assert.match(t, /Estimate only/); assert.match(t, /1 .*held back/i);
        assert.match(t, /over the limit/);
        await h.click('Connect this folder');
        assert.equal(h.confirms.length, 1);
        assert.equal(writes(h).length, 2);
        assert.equal(writes(h)[1].url, `/api/v3/sources/${'b'.repeat(32)}/confirm`);
    });

    it('cancelling the modal sends no confirm', async () => {
        const h = setup({ sources: [], routes: [['POST', '/api/v3/sources', () => ({ json: preview() })]] });
        h.confirmAnswer = false;
        await h.open();
        h.host.querySelector('input[type=text]').value = '/x';
        await h.click('Check folder');
        await h.click('Connect this folder');
        assert.equal(writes(h).length, 1);
    });

    it('a refused path and a 409 remote_access_on are shown as sentences', async () => {
        const h = setup({ sources: [], routes: [['POST', '/api/v3/sources', () => ({
            status: 422, json: { detail: { code: 'home_directory', message: 'Your whole home folder cannot be a source.' } } })]] });
        await h.open();
        h.host.querySelector('input[type=text]').value = '~';
        await h.click('Check folder');
        assert.match(h.host.textContent, /whole home folder/);
        const g = setup({ sources: [], routes: [
            ['POST', '/api/v3/sources', () => ({ json: preview() })],
            ['POST', /\/confirm$/, () => ({ status: 409, json: { detail: { code: 'remote_access_on', message: 'x' } } })]] });
        await g.open();
        g.host.querySelector('input[type=text]').value = '/x';
        await g.click('Check folder'); await g.click('Connect this folder');
        assert.match(g.host.textContent, /Turn remote access off/);
    });

    it('an empty path sends nothing', async () => {
        const h = setup(); await h.open();
        await h.click('Check folder');
        assert.equal(writes(h).length, 0);
    });

    it('rescan posts to the rescan route', async () => {
        const h = setup({ routes: [['POST', /\/rescan$/, () => ({ status: 202, json: { job_id: 'j' } })]] });
        await h.open();
        await h.click('Rescan');
        assert.equal(writes(h)[0].url, `/api/v3/sources/${SID}/rescan`);
        assert.match(h.host.textContent, /Scan queued/);
    });

    it('report: quarantined files with hit kinds, skipped and error counts, release per file', async () => {
        const h = setup({ report: { source_id: SID, state: 'active', counts: {}, skipped_by_rule: { too_big: 2 },
            quarantined: [{ relpath: 'a/key.md', reason: 'secret' }], cloud_only: ['c.md'],
            errors: [{ relpath: 'bad.pdf', reason: 'unreadable' }], watch: 0 },
            routes: [['POST', /quarantine\/release$/, () => ({ json: { released: true } })]] });
        await h.open();
        await h.click('Report');
        const t = h.host.textContent;
        assert.match(t, /Watching for changes: no/);
        assert.match(t, /a\/key\.md/); assert.match(t, /secret/); assert.match(t, /too_big/);
        assert.match(t, /1 .*error/i);
        await h.click('Release');
        const w = writes(h).pop();
        assert.equal(w.url, `/api/v3/sources/${SID}/quarantine/release`);
        assert.deepEqual(JSON.parse(w.body), { relpath: 'a/key.md' });
    });

    it('remove: modal, default keeps memories; ticking erase adds purge=1', async () => {
        const h = setup({ routes: [['DELETE', /^\/api\/v3\/sources\//, () => ({ json: { removed: true } })]] });
        await h.open();
        await h.click('Remove');
        assert.equal(writes(h)[0].url, `/api/v3/sources/${SID}`);
        const g = setup({ routes: [['DELETE', /^\/api\/v3\/sources\//, () => ({ json: { removed: true } })]] });
        await g.open();
        const box = g.host.querySelector('input[type=checkbox]');
        assert.equal(box.checked, false, 'unticked by default');
        box.checked = true;
        await g.click('Remove');
        assert.equal(writes(g)[0].url, `/api/v3/sources/${SID}?purge=1`);
        assert.match(g.confirms[0].consequence, /erase/i);
    });

    it('cancelled remove sends nothing', async () => {
        const h = setup(); h.confirmAnswer = false;
        await h.open(); await h.click('Remove');
        assert.equal(writes(h).length, 0);
    });

    it('a source id that is not 32 hex gets no action buttons', async () => {
        const h = setup({ sources: [src({ source_id: '../x' })] });
        await h.open();
        assert.equal(Array.from(h.host.querySelectorAll('button')).filter(b => b.textContent === 'Remove').length, 0);
    });
});

describe('Folders section: forget the files of an emptied folder', () => {
    const empty = (over = {}) => src(Object.assign({ state: 'offline', offline_reason: 'empty_folder' }, over));
    const forgetUrl = `/api/v3/sources/${SID}/forget-empty`;
    const labels = h => Array.from(h.host.querySelectorAll('button')).map(b => b.textContent);
    const okRoute = ['POST', /\/forget-empty$/, () => ({ json: { source_id: SID, forgotten: 5, state: 'active' } })];

    it('the button shows only for an offline folder that is empty', async () => {
        const h = setup({ sources: [empty()] });
        await h.open();
        assert.ok(labels(h).includes('Forget its files'));
        assert.match(h.host.textContent, /If you emptied it on purpose, use Forget its files\./);
        for (const s of [src(), src({ state: 'offline', offline_reason: 'unreachable' }),
            src({ state: 'offline', offline_reason: 'disk_changed' })]) {
            const g = setup({ sources: [s] });
            await g.open();
            assert.equal(labels(g).includes('Forget its files'), false);
        }
    });

    it('click asks first, then posts to the forget route, and says how many were forgotten', async () => {
        const h = setup({ sources: [empty()], routes: [okRoute] });
        await h.open();
        await h.click('Forget its files');
        assert.equal(h.confirms.length, 1);
        assert.equal(h.confirms[0].title, 'Forget the files of this folder');
        assert.equal(h.confirms[0].confirmLabel, 'Forget');
        assert.equal(writes(h).length, 1);
        assert.equal(writes(h)[0].method, 'POST');
        assert.equal(writes(h)[0].url, forgetUrl);
        assert.match(h.host.textContent, /Forgot 5 file\(s\); the folder is active again\./);
    });

    it('cancelling the confirmation sends nothing', async () => {
        const h = setup({ sources: [empty()], routes: [okRoute] });
        h.confirmAnswer = false;
        await h.open(); await h.click('Forget its files');
        assert.equal(writes(h).length, 0);
    });

    it('the daemon message is shown when it refuses', async () => {
        const h = setup({ sources: [empty()], routes: [['POST', /\/forget-empty$/, () => ({ status: 409,
            json: { detail: { code: 'folder_not_empty', message: 'The folder has files again; they are read on the next scan.' } } })]] });
        await h.open(); await h.click('Forget its files');
        assert.match(h.host.textContent, /The folder has files again/);
    });

    it('a hostile folder name and message stay text', async () => {
        const h = setup({ sources: [empty({ display_name: XSS })], routes: [['POST', /\/forget-empty$/, () => ({ status: 409,
            json: { detail: { code: 'x', message: XSS } } })]] });
        await h.open(); await h.click('Forget its files');
        assert.equal(h.host.querySelectorAll('img').length, 0);
        assert.equal(h.window.__pwn, undefined);
        assert.ok(h.host.textContent.includes(XSS));
        assert.equal(h.confirms[0].target, XSS.slice(0, 80));
    });
});

describe('Folders section XSS', () => {
    function hostile() {
        return setup({
            sources: [src({ display_name: XSS, root_path: '/p/' + XSS, state: 'offline', offline_reason: XSS })],
            report: { source_id: SID, state: 'active', counts: {}, skipped_by_rule: { [XSS]: 1 },
                quarantined: [{ relpath: XSS, reason: XSS }], cloud_only: [XSS],
                errors: [{ relpath: XSS, reason: XSS }], watch: 0 },
            routes: [['POST', '/api/v3/sources', () => ({ json: preview({ root: XSS, warnings: [XSS],
                files_by_type: { [XSS]: 1 }, skipped_by_rule: { [XSS]: 1 } }) })]] });
    }
    async function drive(h) {
        await h.open();
        await h.click('Report');
        h.host.querySelector('input[type=text]').value = XSS;
        await h.click('Check folder');
    }

    it('hostile folder name, relpath and reason render as text, never as elements', async () => {
        const h = hostile();
        await drive(h);
        assert.equal(h.host.querySelectorAll('img').length, 0);
        assert.equal(h.window.__pwn, undefined);
        assert.ok(h.host.textContent.includes(XSS), 'shown literally as text');
        assert.equal(h.host.innerHTML.includes('<img'), false);
    });

    it('mutation check: the same payload via innerHTML would create an img', async () => {
        const h = hostile();
        await drive(h);
        const probe = h.document.createElement('div');
        probe.innerHTML = h.host.textContent;
        assert.equal(probe.querySelectorAll('img').length > 0, true, 'test payload is live HTML');
    });

    it('the module never uses innerHTML, outerHTML or insertAdjacentHTML', () => {
        const code = readFileSync(new URL('../../src/superlocalmemory/ui/js/od-sources.js', import.meta.url), 'utf8');
        assert.doesNotMatch(code, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
        assert.ok(code.split('\n').length < 400);
    });
});
