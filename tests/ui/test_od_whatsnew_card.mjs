/**
 * What's-new card on the Dashboard pane (js/od-features.js).
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { makeEnv } from './features_helpers.mjs';

const KEY = 'slm.whatsnew.4.1.25';
const html = readFileSync(join(dirname(fileURLToPath(import.meta.url)),
    '../../src/superlocalmemory/ui/index.html'), 'utf8');

function setup() {
    const h = makeEnv([], { ids: ['od-whatsnew'], modules: ['od-features.js'] });
    h.host = h.document.getElementById('od-whatsnew');
    return h;
}
const dismiss = h => [...h.host.querySelectorAll('button')].find(b => /Dismiss/.test(b.textContent));

describe("what's-new card", () => {
    it('is shown on first load with the three lines', () => {
        const h = setup();
        assert.match(h.host.textContent, /What's new in 4\.1\.25/);
        assert.equal(h.host.querySelectorAll('li').length, 3);
        assert.match(h.host.textContent, /Images & documents/);
        assert.match(h.host.textContent, /Bots on the web/);
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
        assert.match(h.document.getElementById('od-whatsnew').textContent, /What's new/);
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
