/**
 * The turn-on card for images and documents (js/od-features.js).
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, features, writes, flushPromises, TOKEN } from './features_helpers.mjs';

const MEDIA = (state, extra = {}) => ({ enabled: true, env_state: state, ...extra });

function setup(initial, extraRoutes = []) {
    const state = { media: initial };
    const routes = [
        ...extraRoutes,  // first, so a test can answer a route differently
        ['GET', '/api/v3/features', () => ({ json: features(state.media) })],
        ['POST', '/api/v3/features/media/enable', () => {
            state.media = { ...state.media, enabled: true, env_state: 'installing', progress: 0.1, step: 'Starting' };
            return { status: 202, json: { media: state.media } };
        }],
        ['POST', '/api/v3/features/media/disable', () => {
            state.media = { ...state.media, enabled: false, env_state: 'not_installed' };
            return { json: { media: state.media } };
        }],
    ];
    const h = makeEnv(routes, { modules: ['od-features.js'] });
    h.state = state;
    h.host = h.document.getElementById('pane');
    h.mount = async (opts) => { await h.window.odFeatures.mountMediaCard(h.host, opts || {}); await flushPromises(); };
    return h;
}

const btn = (h, label) => [...h.host.querySelectorAll('button')].find(b => b.textContent.includes(label));

describe('turn-on card', () => {
    it('off: shows only the turn-on card with size and locality', async () => {
        const h = setup({ enabled: false });
        await h.mount();
        const t = h.host.textContent;
        assert.match(t, /Turn on images and documents/);
        assert.match(t, /1\.5 GB/);
        assert.match(t, /stays on this computer/);
        assert.ok(btn(h, 'Turn on'));
        assert.equal(h.host.querySelectorAll('input[type=file]').length, 0);
    });

    it('confirm then POST {"yes": true} with the write credential', async () => {
        const h = setup({ enabled: false });
        await h.mount();
        btn(h, 'Turn on').click();
        await flushPromises();
        assert.equal(h.confirms.length, 1);
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].url, '/api/v3/features/media/enable');
        assert.deepEqual(JSON.parse(w[0].body), { yes: true });
        assert.equal(w[0].headers.get('x-install-token'), TOKEN);
        assert.match(h.host.textContent, /Starting/);
    });

    it('declined confirm sends nothing', async () => {
        const h = setup({ enabled: false });
        h.confirmAnswer = false;
        await h.mount();
        btn(h, 'Turn on').click();
        await flushPromises();
        assert.equal(writes(h).length, 0);
    });

    it('installing: progress bar and step update on each poll, then stop after ready', async () => {
        const h = setup(MEDIA('installing', { progress: 0.25, step: 'Downloading models' }));
        await h.mount();
        const bar = () => h.host.querySelector('[role=progressbar]');
        assert.equal(bar().getAttribute('aria-valuenow'), '25');
        assert.match(h.host.textContent, /Downloading models/);
        h.state.media = MEDIA('installing', { progress: 0.6, step: 'Checking' });
        await h.tick();
        assert.equal(bar().getAttribute('aria-valuenow'), '60');
        assert.match(h.host.textContent, /Checking/);
        h.state.media = MEDIA('ready', { restart_required: true });
        await h.tick();
        assert.equal(h.timers.length, 0, 'no further poll is scheduled once ready');
        const gets = h.calls.filter(c => c.url === '/api/v3/features').length;
        await h.tick();
        assert.equal(h.calls.filter(c => c.url === '/api/v3/features').length, gets);
    });

    it('ready + restart required: the button calls the existing restart function', async () => {
        const h = setup(MEDIA('ready', { restart_required: true }));
        let got = null;
        h.window.odRestartDaemon = (b, s) => { got = [b, s]; };
        await h.mount();
        const b = btn(h, 'Restart SuperLocalMemory');
        assert.ok(b);
        b.click();
        assert.ok(got && got[0] === b);
    });

    it('failed: the server message is shown as text only', async () => {
        const evil = 'Setup failed <img src=x onerror="window.__pwn=1">';
        const h = setup(MEDIA('failed', { step: evil, error: 'network' }));
        await h.mount();
        assert.ok(h.host.textContent.includes(evil));
        assert.equal(h.host.querySelectorAll('img').length, 0);
        assert.ok(btn(h, 'Try again'));
    });

    it('on and ready: calls onReady and offers turn-off with the remove-files box unticked', async () => {
        const h = setup(MEDIA('ready'));
        let ready = 0;
        await h.mount({ onReady: () => { ready += 1; } });
        assert.equal(ready, 1);
        const box = h.host.querySelector('input[type=checkbox]');
        assert.ok(box && box.checked === false);
        btn(h, 'Turn off').click();
        await flushPromises();
        assert.deepEqual(JSON.parse(writes(h)[0].body), { remove_files: false });
        assert.equal(writes(h)[0].url, '/api/v3/features/media/disable');
    });

    it('turn-off with the box ticked asks to remove the files', async () => {
        const h = setup(MEDIA('ready'));
        await h.mount({ onReady: () => {} });
        h.host.querySelector('input[type=checkbox]').checked = true;
        btn(h, 'Turn off').click();
        await flushPromises();
        assert.deepEqual(JSON.parse(writes(h)[0].body), { remove_files: true });
    });

    it('a 404 from the features route says not available in this build', async () => {
        const h = makeEnv([], { modules: ['od-features.js'] });
        const host = h.document.getElementById('pane');
        await h.window.odFeatures.mountMediaCard(host, {});
        await flushPromises();
        assert.match(host.textContent, /not available in this build/);
    });
});

describe('the 16 GB gate on the turn-on card', () => {
    const MESSAGE = 'Images and documents need a computer with at least 16 GB of memory; this one has 4.0 GB. '
        + 'Your text memories keep working.';
    const SMALL = { enabled: false, ram_ok: false, ram_message: MESSAGE };

    it('shows the message, keeps the rest of the card, and disables Turn on', async () => {
        const h = setup(SMALL);
        await h.mount();
        assert.ok(h.host.textContent.includes(MESSAGE));
        assert.match(h.host.textContent, /Turn on images and documents/);
        assert.match(h.host.textContent, /1\.5 GB/);
        assert.equal(btn(h, 'Turn on').disabled, true);
    });

    it('a click on the disabled button asks and sends nothing', async () => {
        const h = setup(SMALL);
        await h.mount();
        btn(h, 'Turn on').click();
        await flushPromises();
        assert.equal(h.confirms.length, 0);
        assert.equal(writes(h).length, 0);
    });

    it('a machine with enough memory (or an older daemon that says nothing) can turn it on', async () => {
        for (const media of [{ enabled: false, ram_ok: true, ram_message: '' }, { enabled: false }]) {
            const h = setup(media);
            await h.mount();
            assert.equal(btn(h, 'Turn on').disabled, false);
            btn(h, 'Turn on').click();
            await flushPromises();
            assert.equal(h.confirms.length, 1);
            assert.ok(!/16 GB of memory/.test(h.host.textContent));
        }
    });

    it('a refusal from the daemon (409) puts the message on the card instead of an error', async () => {
        const refusal = { enabled: false, ram_ok: false, ram_message: MESSAGE, error: MESSAGE, refused: 'low_ram' };
        const h = setup({ enabled: false }, [['POST', '/api/v3/features/media/enable',
            () => ({ status: 409, json: { media: features(refusal).media, detail: MESSAGE } })]]);
        await h.mount();
        btn(h, 'Turn on').click();
        await flushPromises();
        assert.ok(h.host.textContent.includes(MESSAGE));
        assert.equal(btn(h, 'Turn on').disabled, true);
    });
});
