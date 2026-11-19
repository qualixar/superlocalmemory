/**
 * The first-run Welcome card on the Dashboard pane (js/od-welcome.js, window.odWelcome):
 * three rows (memory working, AI agents, pictures and documents) and a Done button, shown
 * only on a new store (fewer than 20 memories) that the person has not dismissed.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { makeEnv, features, flushPromises } from './features_helpers.mjs';
import { evalModule } from './harness.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const KEY = 'slm.welcome.dismissed';
const PROMPT = 'Install SuperLocalMemory on this computer for me. Follow '
    + 'https://github.com/qualixar/superlocalmemory/blob/main/SETUP_FOR_AGENTS.md step by step and show me '
    + 'what each step printed. Ask me only one question, before turning on anything optional: whether I also '
    + 'want to find my pictures, screenshots and PDFs by describing them. Connect SuperLocalMemory to the '
    + 'agent I am using. When you finish, run slm doctor and tell me in plain words what works.';

const stats = (n) => ({ json: { overview: { total_memories: n, total_facts: n * 3 } } });
const hosts = (configured = []) => ({ json: { configured_hosts: configured, available_hosts: [], apps: [] } });
const ON = { enabled: true, env_state: 'ready', restart_required: false };

function setup({ memories = 3, stat, local, feat, throwingStorage = false, dismissed = false } = {}) {
    const routes = [
        ['GET', '/api/stats', stat || (() => stats(memories))],
        ['GET', '/api/v3/local-apps', local || (() => hosts([{ id: 'claude-code', name: 'Claude Code', live: 1 },
            { id: 'cursor', name: 'Cursor', live: 0 }]))],
        ['GET', '/api/v3/features', feat || (() => ({ json: features({ enabled: false }) }))],
    ];
    const h = makeEnv(routes, { ids: ['pane'], modules: ['od-features.js'] });
    h.store = {};
    if (dismissed) h.store[KEY] = '1';
    const storage = throwingStorage
        ? { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } }
        : { getItem: (k) => (k in h.store ? h.store[k] : null), setItem: (k, v) => { h.store[k] = String(v); } };
    Object.defineProperty(h.window, 'localStorage', { value: storage, configurable: true });
    h.copied = [];
    Object.defineProperty(h.window.navigator, 'clipboard',
        { value: { writeText: async (t) => { h.copied.push(t); } }, configurable: true });
    h.nav = [];
    h.window.slmNavigate = (p) => h.nav.push(p);
    evalModule(h.window, 'od-welcome.js');
    h.host = h.document.getElementById('pane');
    h.mount = async () => { await h.window.odWelcome.mount(h.host); await flushPromises(); };
    h.btn = (t) => [...h.host.querySelectorAll('button')].find((b) => b.textContent.trim() === t);
    h.row = (name) => h.host.querySelector('[data-row="' + name + '"]');
    h.gets = (url) => h.calls.filter((c) => c.url === url);
    return h;
}

describe('Welcome card visibility', () => {
    it('is hidden on a store with 20 or more memories', async () => {
        for (const n of [20, 500]) {
            const h = setup({ memories: n });
            await h.mount();
            assert.equal(h.host.children.length, 0, n + ' memories');
        }
    });

    it('is shown on a new store, with its title, three rows and a Done button', async () => {
        const h = setup({ memories: 19 });
        await h.mount();
        assert.equal(h.host.querySelector('h3').textContent, 'Welcome to SuperLocalMemory');
        assert.deepEqual([...h.host.querySelectorAll('[data-row]')].map((r) => r.dataset.row),
            ['memory', 'agents', 'media']);
        assert.match(h.row('memory').textContent, /Your memory is working/);
        assert.match(h.row('agents').textContent, /Your AI agents/);
        assert.match(h.row('media').textContent, /Pictures and documents \(optional\)/);
        assert.ok(h.btn('Done'));
        for (const r of h.host.querySelectorAll('[data-row]')) {
            assert.ok(r.querySelectorAll('button').length <= 1, 'at most one button per row');
        }
    });

    it('Done dismisses it and the choice survives a reload through storage', async () => {
        const h = setup({ memories: 2 });
        await h.mount();
        h.btn('Done').click();
        assert.equal(h.host.children.length, 0);
        assert.equal(h.store[KEY], '1');
        const before = h.calls.length;
        await h.mount();                                    // "reload": same storage, fresh mount
        assert.equal(h.host.children.length, 0);
        assert.equal(h.calls.length, before, 'a dismissed card costs no request');
    });

    it('a storage that throws still shows the card, and Done still closes it this visit', async () => {
        const h = setup({ memories: 2, throwingStorage: true });
        await h.mount();
        assert.ok(h.host.querySelector('h3'));
        assert.doesNotThrow(() => h.btn('Done').click());
        assert.equal(h.host.children.length, 0);
    });

    it('is mounted at the top of the dashboard pane, before the key numbers, and its script is stamped', () => {
        const card = html.indexOf('id="od-welcome"');
        assert.ok(card > html.indexOf('id="dashboard-pane"'));
        assert.ok(card < html.indexOf('class="dash-cols"'));
        assert.ok(card < html.indexOf('class="kpi-strip'));
        assert.match(html, /static\/js\/od-welcome\.js\?v=[0-9a-f]{8}"/);
        assert.ok(html.indexOf('static/js/od-welcome.js') > html.indexOf('static/js/od-features.js'), 'loads after od-features.js');
    });
});

describe('Row 1: your memory is working', () => {
    it('shows a tick when the daemon answers', async () => {
        const h = setup({ memories: 4 });
        await h.mount();
        const status = h.row('memory').querySelector('.od-welcome-status');
        assert.match(status.textContent, /✓/);
        assert.ok(status.classList.contains('ok'));
        assert.equal(h.gets('/api/stats').length, 1);
    });

    it('says why when the service cannot be reached, and still shows the card', async () => {
        const h = setup();
        h.window.fetch = async () => { throw new TypeError('offline'); };
        await h.mount();
        assert.ok(h.host.querySelector('h3'), 'card is still shown');
        assert.match(h.row('memory').textContent, /Could not reach the local service/);
        assert.doesNotMatch(h.row('memory').textContent, /✓/);
    });

    it('says why when the endpoint answers 404 or an error', async () => {
        for (const status of [404, 500]) {
            const h = setup({ stat: () => ({ status, json: { detail: 'nope' } }) });
            await h.mount();
            assert.ok(h.host.querySelector('h3'));
            assert.doesNotMatch(h.row('memory').textContent, /✓/);
            assert.match(h.row('memory').textContent, /memory|service|answer/i);
        }
    });
});

describe('Row 2: your AI agents', () => {
    it('lists the connected agents and shows the exact copyable prompt', async () => {
        const h = setup();
        await h.mount();
        const text = h.row('agents').textContent;
        assert.match(text, /Claude Code/);
        assert.match(text, /Cursor/);
        assert.match(text, /To connect another agent, paste this into it:/);
        assert.equal(h.row('agents').querySelector('.od-welcome-prompt').textContent, PROMPT);
        assert.ok(h.btn('Copy'));
        assert.equal(h.gets('/api/v3/local-apps').length, 1);
    });

    it('Copy puts the exact prompt on the clipboard', async () => {
        const h = setup();
        await h.mount();
        h.btn('Copy').click();
        await flushPromises();
        assert.deepEqual(h.copied, [PROMPT]);
        assert.match(h.row('agents').textContent, /Copied/);
    });

    it('without clipboard access the prompt is selected for the person to copy', async () => {
        const h = setup();
        Object.defineProperty(h.window.navigator, 'clipboard', { value: undefined, configurable: true });
        await h.mount();
        h.btn('Copy').click();
        await flushPromises();
        assert.match(h.row('agents').textContent, /Ctrl\+C|⌘C|Cmd\+C/);
        assert.equal(h.window.getSelection().toString(), PROMPT);
    });

    it('says so when no agent is connected yet, and still offers the prompt', async () => {
        const h = setup({ local: () => hosts([]) });
        await h.mount();
        assert.match(h.row('agents').textContent, /No agent is connected yet/);
        assert.ok(h.row('agents').querySelector('.od-welcome-prompt'));
    });

    it('says why when the list cannot be read (404, 403, daemon down), never blank', async () => {
        for (const status of [404, 403]) {
            const h = setup({ local: () => ({ status, json: { detail: 'Local dashboard required' } }) });
            await h.mount();
            assert.match(h.row('agents').textContent, /Could not read|not available/i);
            assert.ok(h.row('agents').querySelector('.od-welcome-prompt'), 'the prompt still helps');
        }
    });
});

describe('Row 3: pictures and documents (optional)', () => {
    it('off: one sentence and one button that opens the Documents & Images pane', async () => {
        const h = setup({ feat: () => ({ json: features({ enabled: false }) }) });
        await h.mount();
        const row = h.row('media');
        assert.match(row.textContent, /Find your pictures, screenshots and PDFs by describing them\. Adds a second local model: about 1\.5 GB\. It sets up by itself on a computer with 16 GB of memory or more\./);
        assert.doesNotMatch(row.textContent, /✓/);
        h.btn('Set up pictures and documents').click();
        assert.deepEqual(h.nav, ['media-pane']);
        assert.equal(h.calls.filter((c) => c.method !== 'GET').length, 0, 'the card itself changes nothing');
    });

    it('on: a tick and "On", and no button', async () => {
        const h = setup({ feat: () => ({ json: features(ON) }) });
        await h.mount();
        const row = h.row('media');
        assert.match(row.querySelector('.od-welcome-status').textContent, /✓/);
        assert.match(row.textContent, /\bOn\b/);
        assert.equal(row.querySelectorAll('button').length, 0);
    });

    it('a computer with too little memory is warned in plain words and can still start the setup', async () => {
        const msg = 'This computer has 8.0 GB of memory; pictures and documents may be slow and the models are released when memory runs low.';
        const h = setup({ feat: () => ({ json: features({ enabled: false, ram_ok: true, ram_message: '', ram_warning: msg }) }) });
        await h.mount();
        assert.match(h.row('media').textContent, /this computer has 8\.0 GB/i);
        assert.equal(h.btn('Set up pictures and documents').disabled, false);
    });

    it('while it is being set up, it says so and points at the pane', async () => {
        const h = setup({ feat: () => ({ json: features({ enabled: true, env_state: 'installing', progress: 0.4 }) }) });
        await h.mount();
        assert.match(h.row('media').textContent, /being set up/i);
        h.btn('Open Documents & Images').click();
        assert.deepEqual(h.nav, ['media-pane']);
    });

    it('endpoint failures give a reason and no button', async () => {
        for (const status of [404, 500]) {
            const h = setup({ feat: () => ({ status, json: {} }) });
            await h.mount();
            assert.ok(h.row('media').textContent.length > 30);
            assert.equal(h.row('media').querySelectorAll('button').length, 0);
        }
    });

    it('never offers moving text to the picture model', async () => {
        const h = setup();
        await h.mount();
        assert.doesNotMatch(h.host.textContent, /upgrade|move your (text|memories)|re-?index/i);
    });
});

describe('Safety and tone', () => {
    it('server text is never parsed as HTML', async () => {
        const evil = '<img src=x onerror=alert(1)>';
        const h = setup({
            local: () => hosts([{ id: 'x', name: evil, live: 0 }]),
            feat: () => ({ json: features({ enabled: false, ram_ok: true, ram_message: '', ram_warning: evil }) }),
        });
        await h.mount();
        assert.equal(h.host.querySelectorAll('img').length, 0);
        assert.match(h.host.textContent, /<img src=x/);
    });

    it('shows no developer command, apart from the copyable agent prompt', async () => {
        const h = setup();
        await h.mount();
        const clone = h.host.cloneNode(true);
        clone.querySelector('.od-welcome-prompt').remove();
        assert.doesNotMatch(clone.textContent, /\bslm [a-z]+|pip install|npm |curl /i);
    });

    it('the script never builds markup from strings', () => {
        const src = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-welcome.js'), 'utf8');
        assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML/);
    });
});
