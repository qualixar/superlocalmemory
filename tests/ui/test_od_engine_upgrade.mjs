/**
 * The "Upgrade memory engine" card (js/od-engine-upgrade.js), in the Embeddings group of Settings.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, writes, flushPromises } from './features_helpers.mjs';
import { buildHarness, evalModule } from './harness.mjs';

const API = '/api/v3/embedding/reindex';
const PLAN = {
    available: true, reason: '', already: false, needs_media: false, turn_on_command: '',
    media_enabled: true, env_state: 'ready',
    from: { provider: 'sentence-transformers', model: 'nomic-ai/nomic-embed-text-v1.5', dimension: 768 },
    to: { provider: 'slm-media', model: 'google/embeddinggemma-2', dimension: 768 },
    memories: 1200, ram_mb: 3000, disk_new_mb: 4, disk_kept_mb: 4, disk_mb: 8, minutes: 10,
    minutes_label: 'about 10 minutes', rollback: true, label: 'upgrade',
    explain: 'Your memories are re-read with the new engine in the background. Recall keeps working '
        + 'on the current engine until it finishes, nothing is deleted, and you can roll back '
        + 'to the previous engine afterwards.',
};
const JOB = { job_id: 4, kind: 'switch', state: 'queued', from: 'a::768', to: 'b::768', done: 0, total: 1200 };
const STATUS = { job: null, live: 'nomic-ai/nomic-embed-text-v1.5::768', previous: null, previous_vectors_kept: false };

function setup(initial = {}, extra = []) {
    const state = { plan: { ...PLAN, ...initial.plan }, status: { ...STATUS, ...initial.status },
        planStatus: initial.planStatus || 200, tracked: [], conflicts: [] };
    const routes = [
        ['GET', API + '/upgrade', () => ({ status: state.planStatus, json: state.plan })],
        ['GET', API, () => ({ json: state.status })],
        ['POST', API + '/upgrade', () => ({ status: 202, json: { accepted: true, label: 'upgrade', job: JOB, detail: 'Upgrading' } })],
        ['POST', API + '/rollback', () => ({ status: 202, json: { accepted: true, job: { ...JOB, kind: 'rollback' } } })],
        ['POST', API + '/forget-previous', () => ({ json: { success: true, freed_vectors: 1200 } })],
        ['POST', '/api/v3/features/media/enable', () => {
            state.plan = { ...state.plan, media_enabled: true, env_state: 'installing', available: false,
                needs_media: false, reason: 'Images and documents are still being set up. Try again when they finish.' };
            return { status: 202, json: { media: { enabled: true } } };
        }],
        ...extra,
    ];
    const h = makeEnv(routes, { modules: [] });
    h.state = state;
    h.window.odReindex = { track: (b) => state.tracked.push(b), conflict: (b) => state.conflicts.push(b), onFinish: () => {} };
    h.window.fetch0 = h.window.fetch;
    h.post = (url, body) => h.window.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body) }).then(r => r.ok ? r : r.json().then(d => {
        const e = new Error((d && (typeof d.detail === 'string' ? d.detail : d.error)) || 'failed');
        e.status = r.status; e.body = d; throw e; }));
    evalModule(h.window, 'od-engine-upgrade.js');
    h.mount = async () => {
        const panel = h.window.odEngineUpgrade.panel(h.post);
        h.document.getElementById('pane').appendChild(panel);
        await h.window.odEngineUpgrade.refresh();
        await flushPromises();
        return panel;
    };
    h.btn = (label) => [...h.document.querySelectorAll('#od-engine-upgrade button')].find(b => b.textContent.includes(label));
    h.text = () => h.document.getElementById('od-engine-upgrade').textContent;
    return h;
}

describe('upgrade memory engine card', () => {
    it('available: New badge, plain explanation, the numbers, an enabled button', async () => {
        const h = setup();
        await h.mount();
        const card = h.document.getElementById('od-engine-upgrade');
        assert.ok(card.querySelector('.badge'), 'a New badge like the what\'s-new card');
        assert.equal(card.querySelector('.badge').textContent, 'Preview');  // matches the README: a preview in 4.1.25
        const t = h.text();
        assert.match(t, /Upgrade memory engine/);
        assert.match(t, /Recall keeps working/);
        assert.match(t, /1200 memories/);
        assert.match(t, /about 10 minutes/);
        assert.match(t, /2\.9 GB/);
        assert.doesNotMatch(t, /re-?index/i);
        assert.equal(h.btn('Upgrade memory engine').disabled, false);
        assert.equal(h.btn('Turn on images'), undefined);
    });

    it('declined confirm sends nothing; confirm shows RAM, disk and time', async () => {
        const h = setup();
        await h.mount();
        h.confirmAnswer = false;
        h.btn('Upgrade memory engine').click();
        await flushPromises();
        assert.equal(h.confirms.length, 1);
        assert.match(h.confirms[0].consequence, /2\.9 GB/);
        assert.match(h.confirms[0].consequence, /about 10 minutes/);
        assert.equal(writes(h).length, 0);
    });

    it('confirm posts the upgrade and hands the 202 body to the shared progress panel', async () => {
        const h = setup();
        await h.mount();
        h.btn('Upgrade memory engine').click();
        await flushPromises();
        const w = writes(h);
        assert.equal(w.length, 1);
        assert.equal(w[0].url, API + '/upgrade');
        assert.equal(h.state.tracked.length, 1);
        assert.equal(h.state.tracked[0].job.job_id, 4);
        assert.equal(h.btn('Upgrade memory engine').disabled, true);
    });

    it('images off: the upgrade is disabled, the reason is plain words, and one button opens the images pane', async () => {
        const h = setup({ plan: { available: false, needs_media: true, media_enabled: false, env_state: 'not_installed',
            turn_on_command: 'slm media enable',
            reason: 'Turn on images and documents first (about 1.5 GB): slm media enable' } });
        await h.mount();
        assert.equal(h.btn('Upgrade memory engine').disabled, true);
        assert.match(h.text(), /Turn on images and documents first \(about 1\.5 GB\)\./);
        assert.ok(!/slm media enable/.test(h.text()), 'no developer command in the card');
        const seen = [];
        h.window.slmNavigate = p => seen.push(p);
        h.btn('Turn on images & documents first').click();
        await flushPromises();
        assert.deepEqual(seen, ['media-pane']);
        assert.equal(writes(h).length, 0, 'the pane owns the turn-on flow, with its memory check and confirmation');
        assert.equal(h.confirms.length, 0);
    });

    it('not set up yet: the "Run:" developer hint is dropped too', async () => {
        const h = setup({ plan: { available: false, needs_media: true, media_enabled: false, env_state: 'not_installed',
            turn_on_command: 'slm media enable',
            reason: 'Images and documents are not set up yet. Run: slm media enable' } });
        await h.mount();
        assert.match(h.text(), /Images and documents are not set up yet\./);
        assert.ok(!/slm media enable|Run:/.test(h.text()));
    });

    it('while images are being set up: shows the reason, no turn-on button, and stops polling when ready', async () => {
        const h = setup({ plan: { available: false, needs_media: false, media_enabled: true, env_state: 'installing',
            reason: 'Images and documents are still being set up. Try again when they finish.' } });
        await h.mount();
        assert.match(h.text(), /still being set up/);
        assert.equal(h.btn('Turn on images'), undefined);
        assert.equal(h.timers.length, 1, 'polls while it installs');
        h.state.plan = { ...PLAN };
        await h.tick();
        assert.equal(h.btn('Upgrade memory engine').disabled, false, 'offered once the environment is ready');
        assert.equal(h.timers.length, 0, 'polling stops when ready');
    });

    it('already upgraded: says so, no upgrade button', async () => {
        const h = setup({ plan: { available: false, already: true, reason: 'Your memories already use the new engine.' } });
        await h.mount();
        assert.match(h.text(), /already use the new engine/);
        assert.equal(h.btn('Upgrade memory engine'), undefined);
    });

    it('roll back and free-the-old-data appear only on the new engine with the previous one kept', async () => {
        const h = setup({ plan: { available: false, already: true, reason: 'Your memories already use the new engine.' },
            status: { live: 'google/embeddinggemma-2::768', previous: 'nomic::768', previous_vectors_kept: true } });
        await h.mount();
        assert.ok(h.btn('Roll back'));
        h.btn('Roll back').click();
        await flushPromises();
        assert.equal(writes(h)[0].url, API + '/rollback');
        assert.equal(h.state.tracked.length, 1);

        const g = setup({ status: { live: 'nomic::768', previous: 'old::8', previous_vectors_kept: true } });
        await g.mount();
        assert.equal(g.btn('Roll back'), undefined);
    });

    it('free the old data asks first, then calls forget-previous and refreshes', async () => {
        const h = setup({ status: { live: 'google/embeddinggemma-2::768', previous: 'nomic::768', previous_vectors_kept: true } });
        await h.mount();
        h.confirmAnswer = false;
        h.btn('Free the old').click();
        await flushPromises();
        assert.equal(writes(h).length, 0);
        h.confirmAnswer = true;
        h.btn('Free the old').click();
        await flushPromises();
        assert.equal(writes(h)[0].url, API + '/forget-previous');
    });

    it('a refusal shows the daemon\'s words; a running job goes to the shared panel', async () => {
        const h = setup({}, []);
        h.calls.length = 0;
        await h.mount();
        h.state.plan = { ...PLAN };
        const routes = h.calls;
        // swap the upgrade route to refuse
        h.window.fetch = (url, init) => (init && init.method === 'POST' && String(url).endsWith('/upgrade'))
            ? Promise.resolve({ ok: false, status: 409, json: async () => ({ error: 'reindex_running', detail: 'Another change is running' }) })
            : h.window.fetch0(url, init);
        h.btn('Upgrade memory engine').click();
        await flushPromises();
        assert.match(h.text(), /Another change is running/);
        assert.equal(h.state.conflicts.length, 1);
        assert.ok(routes);
    });

    it('server text is never parsed as HTML', async () => {
        const h = setup({ plan: { available: false, reason: '<img src=x onerror=alert(1)>' } });
        await h.mount();
        assert.equal(h.document.querySelectorAll('#od-engine-upgrade img').length, 0);
        assert.match(h.text(), /<img src=x/);
    });

    it('no such route (older daemon) hides the card; a stopped daemon explains', async () => {
        const h = setup({ planStatus: 404 });
        const panel = await h.mount();
        assert.equal(panel.style.display, 'none');
        const g = setup({ planStatus: 409, plan: { error: 'daemon_required', detail: 'start it with: slm serve' } });
        const p2 = await g.mount();
        assert.notEqual(p2.style.display, 'none');
        assert.match(g.text(), /slm serve/);
        assert.equal(g.btn('Upgrade memory engine'), undefined);
    });

    it('the Settings Embeddings group carries the card and refreshes it when it opens', async () => {
        const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
        const urls = [];
        h.window.fetch = async (url, o = {}) => {
            urls.push(String(url));
            const ok = (json) => ({ ok: true, status: 200, json: async () => json });
            if (url === '/internal/token') return ok({ token: 't' });
            if (String(url) === API + '/upgrade') return ok(PLAN);
            if (String(url) === API) return ok(STATUS);
            return ok({});
        };
        h.window.showToast = () => {};
        for (const m of ['od-reindex.js', 'od-engine-upgrade.js', 'od-settings.js']) evalModule(h.window, m);
        h.window.odRenderSettings(h.document.getElementById('mount'));
        await flushPromises();
        await flushPromises();
        assert.ok(h.document.getElementById('od-engine-upgrade'), 'card in the Embeddings group');
        assert.ok(urls.includes(API + '/upgrade'), 'the plan was read');
    });
});
