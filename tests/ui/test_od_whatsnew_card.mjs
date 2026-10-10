/**
 * What's-new card on the Dashboard pane (js/od-features.js): two equal tiles that
 * say what is on and what is not, a small dismiss icon, no version number.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { makeEnv, features, flushPromises } from './features_helpers.mjs';

const KEY = 'slm.whatsnew.4.1.25';
const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const css = readFileSync(join(here, '../../src/superlocalmemory/ui/css/design-system.css'), 'utf8');
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-features.js'), 'utf8');

const ON = { enabled: true, env_state: 'ready', restart_required: false };
const MESSAGE = 'Images and documents need a computer with at least 16 GB of memory; this one has 8.0 GB. '
    + 'Your text memories keep working.';

async function setup({ media, apps = 0, status } = {}) {
    const fetched = [
        ['GET', '/api/v3/features', () => status
            ? { status, json: {} }
            : { json: { ...features(media), mesh: { apps_with_mesh: apps } } }],
    ];
    const h = makeEnv(fetched, { ids: ['od-whatsnew', 'pane'], modules: ['od-features.js'] });
    h.host = h.document.getElementById('od-whatsnew');
    h.seen = [];
    h.window.slmNavigate = p => h.seen.push(p);
    await flushPromises();
    h.btn = t => [...h.host.querySelectorAll('button')].find(b => b.textContent.trim() === t);
    h.tiles = () => [...h.host.querySelectorAll('.od-whatsnew-tile')];
    h.dismiss = () => h.host.querySelector('button[aria-label="Dismiss"]');
    return h;
}

describe("what's-new card layout", () => {
    it('has a title with no version number, two tiles, and a small dismiss icon button', async () => {
        const h = await setup({ media: { enabled: false } });
        assert.equal(h.host.querySelector('h3').textContent, 'New in SuperLocalMemory');
        assert.ok(!/4\.1\.25/.test(h.host.textContent));
        const tiles = h.tiles();
        assert.equal(tiles.length, 2);
        for (const t of tiles) {
            assert.ok(t.querySelector('h4').textContent.length > 0, 'a short title');
            assert.equal(t.querySelectorAll('p').length >= 1, true, 'a plain sentence');
            assert.equal(t.querySelectorAll('button').length, 1, 'exactly one button');
        }
        const x = h.dismiss();
        assert.ok(x);
        assert.equal(x.textContent, '×');
        assert.ok(!h.host.textContent.includes('Dismiss'), 'the word is only the accessible name');
    });

    it('the tiles sit in an equal two-column grid that stacks on narrow screens', () => {
        assert.match(css, /\.od-whatsnew-tiles\s*\{[^}]*display:\s*grid[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
        assert.match(css, /@media \(max-width: 640px\)[\s\S]*?\.od-whatsnew-tiles\s*\{[^}]*grid-template-columns:\s*1fr/);
        assert.match(css, /\.od-whatsnew-close\s*\{[^}]*position:\s*absolute/);
    });
});

describe("what's-new card follows the real state", () => {
    it('images off: offers to turn on, with the developer line', async () => {
        const h = await setup({ media: { enabled: false } });
        assert.ok(h.btn('Turn on images & documents'));
        assert.equal(h.btn('Open Documents & Images'), undefined);
        assert.equal(h.host.querySelector('.badge.ok'), null);
        assert.match(h.host.textContent, /For developers: slm media enable/);
    });

    it('images on: an On chip and an Open button; no turn-on button and no developer line', async () => {
        const h = await setup({ media: ON });
        assert.equal(h.btn('Turn on images & documents'), undefined);
        assert.equal(h.host.querySelector('.badge.ok').textContent, 'On');
        assert.ok(h.btn('Open Documents & Images'));
        assert.ok(!/slm media enable/.test(h.host.textContent));
        h.btn('Open Documents & Images').click();
        assert.deepEqual(h.seen, ['media-pane']);
    });

    it('setting up: shows the progress text, never the turn-on button, and keeps the developer line off', async () => {
        const h = await setup({ media: { enabled: true, env_state: 'installing', progress: 0.42, step: 'Downloading models' } });
        assert.match(h.tiles()[0].textContent, /Setting up/);
        assert.match(h.tiles()[0].textContent, /Downloading models/);
        assert.match(h.tiles()[0].textContent, /42%/);
        assert.equal(h.btn('Turn on images & documents'), undefined);
        assert.ok(!/slm media enable/.test(h.host.textContent));
    });

    it('setting up: polls and flips to On when the install finishes', async () => {
        const state = { media: { enabled: true, env_state: 'installing', progress: 0.1, step: 'Working' } };
        const h = makeEnv([['GET', '/api/v3/features', () => ({ json: features(state.media) })]],
            { ids: ['od-whatsnew'], modules: ['od-features.js'] });
        await flushPromises();
        const host = h.document.getElementById('od-whatsnew');
        assert.match(host.textContent, /Setting up/);
        state.media = ON;
        await h.tick();
        assert.equal(host.querySelector('.badge.ok').textContent, 'On');
        assert.equal(h.timers.length, 0);
    });

    it('under 16 GB: the refusal message shows and the turn-on button is disabled', async () => {
        const h = await setup({ media: { enabled: false, ram_ok: false, ram_message: MESSAGE } });
        assert.ok(h.host.textContent.includes(MESSAGE));
        assert.equal(h.btn('Turn on images & documents').disabled, true);
        h.btn('Turn on images & documents').click();
        await flushPromises();
        assert.deepEqual(h.seen, []);
        assert.equal(h.confirms.length, 0);
    });

    it('setup that failed offers the turn-on button again', async () => {
        const h = await setup({ media: { enabled: true, env_state: 'failed', step: 'No network' } });
        assert.ok(h.btn('Turn on images & documents'));
    });

    it('turned on but needing a restart: points to the pane, which has the restart button', async () => {
        const h = await setup({ media: { enabled: true, env_state: 'ready', restart_required: true } });
        assert.match(h.tiles()[0].textContent, /restart/i);
        assert.ok(h.btn('Open Documents & Images'));
        assert.equal(h.host.querySelector('.badge.ok'), null);
    });

    it('no app with bot messages: offers to set them up in Connected apps', async () => {
        const h = await setup({ media: { enabled: false }, apps: 0 });
        h.btn('Set up bot messages').click();
        assert.deepEqual(h.seen, ['apps-pane']);
        assert.equal(h.btn('Open Bot messages'), undefined);
    });

    it('some apps with bot messages: says how many and opens the Bot messages pane', async () => {
        const h = await setup({ media: ON, apps: 3 });
        assert.match(h.tiles()[1].textContent, /3 apps can message each other/);
        assert.equal(h.btn('Set up bot messages'), undefined);
        h.btn('Open Bot messages').click();
        assert.deepEqual(h.seen, ['botmsg-pane']);
    });

    it('one app is not described as many', async () => {
        const h = await setup({ media: ON, apps: 1 });
        assert.ok(!/1 apps/.test(h.host.textContent));
        assert.match(h.tiles()[1].textContent, /1 app/);
    });

    it('a features route that fails still shows a usable card (turn-on and set-up)', async () => {
        const h = await setup({ status: 500 });
        assert.ok(h.btn('Turn on images & documents'));
        assert.ok(h.btn('Set up bot messages'));
    });

    it('server text is placed as text, never as markup', async () => {
        const evil = '<img src=x onerror="window.__pwn=1">';
        const h = await setup({ media: { enabled: true, env_state: 'installing', progress: 0.2, step: evil } });
        assert.ok(h.host.textContent.includes(evil));
        assert.equal(h.host.querySelectorAll('img').length, 0);
    });
});

describe("what's-new card actions", () => {
    it('"Turn on images & documents" opens the pane and starts the pane\'s own turn-on flow', async () => {
        const e = makeEnv([
            ['GET', '/api/v3/features', () => ({ json: features() })],
            ['POST', '/api/v3/features/media/enable', () => ({ json: { media: features({ enabled: true, env_state: 'installing' }).media } })],
        ], { ids: ['od-whatsnew', 'pane'], modules: ['od-features.js'] });
        await flushPromises();
        const host = e.document.getElementById('od-whatsnew');
        const seen = [];
        const pane = e.document.getElementById('pane');
        e.window.slmNavigate = p => { seen.push(p); e.window.odFeatures.mountMediaCard(pane, {}); };
        [...host.querySelectorAll('button')].find(b => b.textContent === 'Turn on images & documents').click();
        await flushPromises(); await flushPromises();
        assert.deepEqual(seen, ['media-pane']);
        assert.equal(e.confirms.length, 1);
        assert.match(e.confirms[0].title, /Turn on images and documents/);
        assert.ok(e.calls.some(c => c.method === 'POST' && c.url === '/api/v3/features/media/enable'));
    });
});

describe("what's-new card dismiss", () => {
    it('hides it and remembers the choice; a later mount stays hidden and reads nothing', async () => {
        const h = await setup({ media: { enabled: false } });
        h.dismiss().click();
        assert.equal(h.host.textContent, '');
        assert.equal(h.window.localStorage.getItem(KEY), '1');
        const before = h.calls.length;
        await h.window.odFeatures.mountWhatsNew(h.host);
        assert.equal(h.host.textContent, '');
        assert.equal(h.calls.length, before, 'no request for a card nobody will see');
    });

    it('a poll that is still pending does not bring the card back after dismiss', async () => {
        const state = { media: { enabled: true, env_state: 'installing', progress: 0.1, step: 'x' } };
        const h = makeEnv([['GET', '/api/v3/features', () => ({ json: features(state.media) })]],
            { ids: ['od-whatsnew'], modules: ['od-features.js'] });
        await flushPromises();
        const host = h.document.getElementById('od-whatsnew');
        host.querySelector('button[aria-label="Dismiss"]').click();
        await h.tick();
        assert.equal(host.textContent, '');
    });

    it('storage that throws on read still shows the card', async () => {
        const h = makeEnv([], { ids: ['od-whatsnew'] });
        Object.defineProperty(h.window, 'localStorage', { configurable: true, get() { throw new Error('blocked'); } });
        h.window.eval(SRC);
        await flushPromises();
        assert.match(h.document.getElementById('od-whatsnew').textContent, /New in SuperLocalMemory/);
    });

    it('storage that throws on write: dismiss still hides it without an error', async () => {
        const h = await setup({ media: { enabled: false } });
        h.window.localStorage.setItem = () => { throw new Error('quota'); };
        assert.doesNotThrow(() => h.dismiss().click());
        assert.equal(h.host.textContent, '');
    });
});

describe("what's-new card placement", () => {
    it('lives inside the Dashboard pane in index.html', () => {
        const start = html.indexOf('id="dashboard-pane"');
        const at = html.indexOf('id="od-whatsnew"');
        assert.ok(start > 0 && at > start);
        assert.ok(at < html.indexOf('class="kpi-strip"'));
    });
});
