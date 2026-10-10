/**
 * What's-new card on the Dashboard pane (js/od-features.js).
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { makeEnv, features, flushPromises } from './features_helpers.mjs';

const KEY = 'slm.whatsnew.4.1.25';
const html = readFileSync(join(dirname(fileURLToPath(import.meta.url)),
    '../../src/superlocalmemory/ui/index.html'), 'utf8');

function setup() {
    const h = makeEnv([], { ids: ['od-whatsnew'], modules: ['od-features.js'] });
    h.host = h.document.getElementById('od-whatsnew');
    return h;
}
const btn = (h, t) => [...h.host.querySelectorAll('button')].find(b => b.textContent.trim() === t);
const dismiss = h => [...h.host.querySelectorAll('button')].find(b => /Dismiss/.test(b.textContent));

describe("what's-new card", () => {
    it('is shown on first load with two chips, two buttons and a small developer line', () => {
        const h = setup();
        assert.match(h.host.textContent, /New in SuperLocalMemory 4\.1\.25/);
        const chips = [...h.host.querySelectorAll('.badge')].map(b => b.textContent);
        assert.deepEqual(chips, ['Now remembers images and documents', 'Your bots can talk to each other']);
        assert.ok(btn(h, 'Turn on images & documents'));
        assert.ok(btn(h, 'Set up bot messages'));
        assert.match(h.host.querySelector('p.muted').textContent, /^For developers: slm media enable$/);
    });

    it('"Set up bot messages" opens Connected apps', () => {
        const h = setup();
        const seen = [];
        h.window.slmNavigate = p => seen.push(p);
        btn(h, 'Set up bot messages').click();
        assert.deepEqual(seen, ['apps-pane']);
    });

    it('"Turn on images & documents" opens the pane and starts the pane\'s own turn-on flow', async () => {
        const e = makeEnv([
            ['GET', '/api/v3/features', () => ({ json: features() })],
            ['POST', '/api/v3/features/media/enable', () => ({ json: { media: features({ enabled: true, env_state: 'installing' }).media } })],
        ], { ids: ['od-whatsnew', 'pane'], modules: ['od-features.js'] });
        e.host = e.document.getElementById('od-whatsnew');
        const seen = [];
        const pane = e.document.getElementById('pane');
        e.window.slmNavigate = p => { seen.push(p); e.window.odFeatures.mountMediaCard(pane, {}); };
        btn(e, 'Turn on images & documents').click();
        await flushPromises(); await flushPromises();
        assert.deepEqual(seen, ['media-pane']);
        assert.equal(e.confirms.length, 1);
        assert.match(e.confirms[0].title, /Turn on images and documents/);
        assert.ok(e.calls.some(c => c.method === 'POST' && c.url === '/api/v3/features/media/enable'));
    });

    it('dismiss hides it and remembers the choice; a later mount stays hidden', () => {
        const h = setup();
        dismiss(h).click();
        assert.equal(h.host.textContent, '');
        assert.equal(h.window.localStorage.getItem(KEY), '1');
        h.window.odFeatures.mountWhatsNew(h.host);
        assert.equal(h.host.textContent, '');
    });

    it('storage that throws on read still shows the card', () => {
        const h = makeEnv([], { ids: ['od-whatsnew'] });
        Object.defineProperty(h.window, 'localStorage', { configurable: true, get() { throw new Error('blocked'); } });
        h.window.eval(readFileSync(join(dirname(fileURLToPath(import.meta.url)),
            '../../src/superlocalmemory/ui/js/od-features.js'), 'utf8'));
        assert.match(h.document.getElementById('od-whatsnew').textContent, /New in SuperLocalMemory/);
    });

    it('storage that throws on write: dismiss still hides it without an error', () => {
        const h = setup();
        h.window.localStorage.setItem = () => { throw new Error('quota'); };
        assert.doesNotThrow(() => dismiss(h).click());
        assert.equal(h.host.textContent, '');
    });

    it('lives inside the Dashboard pane in index.html', () => {
        const start = html.indexOf('id="dashboard-pane"');
        const at = html.indexOf('id="od-whatsnew"');
        assert.ok(start > 0 && at > start);
        assert.ok(at < html.indexOf('class="kpi-strip"'));
    });
});
