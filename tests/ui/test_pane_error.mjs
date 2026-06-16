/**
 * tests/ui/test_pane_error.mjs — WP-12 pane error-state tests.
 * Runner: node --test tests/ui/
 * Requires: jsdom (npm install --save-dev jsdom)
 *
 * AC coverage:
 *   AC1 503 → .pane-error with "503"
 *   AC2 network reject → .pane-error
 *   AC3 .pane-error-retry button re-invokes loader
 *   AC4 success path unchanged (no .pane-error)
 *   AC5 empty data → .empty-state
 *   AC6 XSS-safe via textContent (not innerHTML)
 *   AC7 no silent return on non-2xx (verified by AC1/AC2 not being blank)
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

// ============================================================================
// 1. paneErrorMessage() unit tests
// ============================================================================

describe('paneErrorMessage', function() {
    let win;

    before(function() {
        const h = buildHarness([], { ok: true, status: 200, json: {} });
        win = h.window;
    });

    it('returns network message for status 0', function() {
        const msg = win.paneErrorMessage(0);
        assert.ok(msg.length > 0, 'should return non-empty string');
        assert.ok(/network|connect|unavailable/i.test(msg), 'should describe network failure');
    });

    it('returns server message for status 503', function() {
        const msg = win.paneErrorMessage(503);
        assert.ok(/503/.test(msg), 'should include status code 503');
    });

    it('returns server message for status 500', function() {
        const msg = win.paneErrorMessage(500);
        assert.ok(/500/.test(msg), 'should include status code 500');
    });

    it('returns client message for status 404', function() {
        const msg = win.paneErrorMessage(404);
        assert.ok(/404/.test(msg), 'should include status code 404');
    });
});

// ============================================================================
// 2. showPaneError() — container mode (slotMode=false)
// ============================================================================

describe('showPaneError — container mode', function() {
    let win, doc;

    before(function() {
        const h = buildHarness(['test-container'], { ok: true, status: 200, json: {} });
        win = h.window;
        doc = h.document;
    });

    it('AC1 — renders .pane-error with status 503 in textContent', function() {
        win.showPaneError('test-container', win.paneErrorMessage(503), null, false);
        const el = doc.getElementById('test-container');
        const errDiv = el.querySelector('.pane-error');
        assert.ok(errDiv, '.pane-error element must exist');
        assert.ok(/503/.test(errDiv.textContent), 'textContent must include 503');
    });

    it('has role=alert for a11y', function() {
        const el = doc.getElementById('test-container');
        const errDiv = el.querySelector('.pane-error');
        assert.equal(errDiv.getAttribute('role'), 'alert');
    });

    it('AC6 — XSS: message set via textContent, not innerHTML', function() {
        win.showPaneError('test-container', '<img src=x onerror=alert(1)>', null, false);
        const el = doc.getElementById('test-container');
        // The img tag must NOT have been created as DOM element
        const imgs = el.querySelectorAll('img');
        assert.equal(imgs.length, 0, 'no img element should be created (XSS blocked)');
        // But the raw string is preserved as text
        assert.ok(el.textContent.includes('<img'), 'raw text must be preserved safely');
    });

    it('AC3 — retry button invokes onRetry callback', function() {
        let retryCalled = false;
        function onRetry() { retryCalled = true; }
        win.showPaneError('test-container', 'Error 503', onRetry, false);
        const el = doc.getElementById('test-container');
        const btn = el.querySelector('.pane-error-retry');
        assert.ok(btn, '.pane-error-retry button must exist when onRetry is given');
        btn.click();
        assert.equal(retryCalled, true, 'onRetry must be called on click');
    });

    it('no retry button when onRetry is null', function() {
        win.showPaneError('test-container', 'Error 503', null, false);
        const el = doc.getElementById('test-container');
        const btn = el.querySelector('.pane-error-retry');
        assert.equal(btn, null, 'no retry button when onRetry is null');
    });
});

// ============================================================================
// 3. showPaneError() — slot mode (slotMode=true)
// ============================================================================

describe('showPaneError — slot mode', function() {
    let win, doc;

    before(function() {
        const h = buildHarness(['slot-anchor'], { ok: true, status: 200, json: {} });
        win = h.window;
        doc = h.document;
    });

    it('dedup: calling twice leaves exactly one error slot', function() {
        win.showPaneError('slot-anchor', 'Error 503', null, true);
        win.showPaneError('slot-anchor', 'Error 503', null, true);
        const el = doc.getElementById('slot-anchor');
        const slots = el.querySelectorAll('#slot-anchor-error-slot');
        assert.equal(slots.length, 1, 'exactly one error slot after two showPaneError calls');
    });
});

// ============================================================================
// 4. clearPaneError() — removes error slot
// ============================================================================

describe('clearPaneError', function() {
    let win, doc;

    before(function() {
        const h = buildHarness(['clear-anchor'], { ok: true, status: 200, json: {} });
        win = h.window;
        doc = h.document;
    });

    it('removes the error slot after showPaneError in slot mode', function() {
        win.showPaneError('clear-anchor', 'Error 503', null, true);
        const el = doc.getElementById('clear-anchor');
        assert.ok(el.querySelector('#clear-anchor-error-slot'), 'slot must exist first');
        win.clearPaneError('clear-anchor', true);
        assert.equal(el.querySelector('#clear-anchor-error-slot'), null, 'slot must be removed');
    });

    it('clears container in container mode', function() {
        win.showPaneError('clear-anchor', 'Error 503', null, false);
        const el = doc.getElementById('clear-anchor');
        assert.ok(el.querySelector('.pane-error'), '.pane-error must exist first');
        win.clearPaneError('clear-anchor', false);
        assert.equal(el.querySelector('.pane-error'), null, '.pane-error must be removed');
    });
});

// ============================================================================
// 5. math-health.js loader — representative per-loader test
// ============================================================================

describe('loadMathHealth — math-health.js', function() {
    it('AC1 — 503 response → .pane-error in #math-health-cards, no .card', async function() {
        const h = buildHarness(['math-health-cards'], {
            ok: false, status: 503, json: {}
        });
        evalModule(h.window, 'math-health.js');
        await h.window.loadMathHealth();
        await flushPromises();

        const container = h.document.getElementById('math-health-cards');
        assert.ok(container.querySelector('.pane-error'), 'AC1: .pane-error must exist on 503');
        assert.ok(/503/.test(container.textContent), 'AC1: 503 must appear in textContent');
        assert.equal(container.querySelector('.card'), null, 'AC1: no .card rendered on error');
    });

    it('AC2 — network reject → .pane-error', async function() {
        const h = buildHarness(['math-health-cards'], {
            ok: true, status: 200, json: {}
        });
        h.window.__setFetchReject(new Error('Network failure'));
        evalModule(h.window, 'math-health.js');
        await h.window.loadMathHealth();
        await flushPromises();

        const container = h.document.getElementById('math-health-cards');
        assert.ok(container.querySelector('.pane-error'), 'AC2: .pane-error must exist on network error');
    });

    it('AC4 — success path: .card rendered, no .pane-error', async function() {
        const h = buildHarness(['math-health-cards'], {
            ok: true, status: 200,
            json: {
                health: {
                    fisher: { status: 'active', description: 'Fisher-Rao', mode: 'online' },
                }
            }
        });
        evalModule(h.window, 'math-health.js');
        await h.window.loadMathHealth();
        await flushPromises();

        const container = h.document.getElementById('math-health-cards');
        assert.equal(container.querySelector('.pane-error'), null, 'AC4: no .pane-error on success');
        assert.ok(container.querySelector('.card'), 'AC4: .card must be rendered on success');
    });

    it('AC5 — empty health object → .empty-state, no .card', async function() {
        const h = buildHarness(['math-health-cards'], {
            ok: true, status: 200,
            json: { health: {} }
        });
        evalModule(h.window, 'math-health.js');
        await h.window.loadMathHealth();
        await flushPromises();

        const container = h.document.getElementById('math-health-cards');
        assert.ok(container.querySelector('.empty-state'), 'AC5: .empty-state must appear for empty data');
        assert.equal(container.querySelector('.card'), null, 'AC5: no .card for empty data');
    });
});

// ============================================================================
// 6. loadDashboard() — field-scatter pane
// ============================================================================

describe('loadDashboard — dashboard.js', function() {
    it('AC1 — 503 response → .pane-error in #dashboard-pane (slot mode)', async function() {
        const h = buildHarness([
            'dashboard-pane',
            'dashboard-mode', 'dashboard-mode-desc', 'dashboard-memory-count',
            'dashboard-provider', 'dashboard-model', 'dashboard-profile',
            'dashboard-basedir'
        ], { ok: false, status: 503, json: {} });
        evalModule(h.window, 'dashboard.js');
        await h.window.loadDashboard();
        await flushPromises();

        const anchor = h.document.getElementById('dashboard-pane');
        assert.ok(anchor.querySelector('.pane-error'), 'AC1: .pane-error in dashboard-pane on 503');
    });

    it('AC2 — network reject → .pane-error', async function() {
        const h = buildHarness([
            'dashboard-pane',
            'dashboard-mode', 'dashboard-mode-desc', 'dashboard-memory-count',
            'dashboard-provider', 'dashboard-model', 'dashboard-profile',
            'dashboard-basedir'
        ], { ok: true, status: 200, json: {} });
        h.window.__setFetchReject(new Error('Network failure'));
        evalModule(h.window, 'dashboard.js');
        await h.window.loadDashboard();
        await flushPromises();

        const anchor = h.document.getElementById('dashboard-pane');
        assert.ok(anchor.querySelector('.pane-error'), 'AC2: .pane-error on network reject');
    });
});

// ============================================================================
// 7. loadTrustDashboard() — IIFE, Retry fires 2nd fetch (CRIT-1)
// ============================================================================

describe('loadTrustDashboard — trust-dashboard.js', function() {
    it('AC1 — 503 response → .pane-error in operations-pane (slot mode)', async function() {
        const h = buildHarness([
            'operations-pane',
            'trust-agents-body', 'trust-agent-count', 'trust-avg-score',
            'trust-burst-count'
        ], { ok: false, status: 503, json: {} });
        evalModule(h.window, 'trust-dashboard.js');
        await h.window.loadTrustDashboard();
        await flushPromises();

        const anchor = h.document.getElementById('operations-pane');
        assert.ok(anchor.querySelector('.pane-error'), 'AC1: .pane-error on 503');
        assert.ok(/503/.test(anchor.textContent), 'AC1: 503 in error text');
    });

    it('AC3/CRIT-1 — Retry button fires 2nd fetch', async function() {
        let fetchCount = 0;
        const h = buildHarness([
            'operations-pane',
            'trust-agents-body', 'trust-agent-count', 'trust-avg-score',
            'trust-burst-count'
        ], { ok: false, status: 503, json: {} });

        // Intercept all fetch calls to count them
        h.window.fetch = function() {
            fetchCount++;
            return Promise.resolve({
                ok: false,
                status: 503,
                json: function() { return Promise.resolve({}); },
            });
        };

        evalModule(h.window, 'trust-dashboard.js');
        await h.window.loadTrustDashboard();
        await flushPromises();
        assert.equal(fetchCount, 1, 'first fetch on initial load');

        const anchor = h.document.getElementById('operations-pane');
        const btn = anchor.querySelector('.pane-error-retry');
        assert.ok(btn, 'retry button must exist');
        btn.click();
        await flushPromises();
        assert.equal(fetchCount, 2, 'CRIT-1: Retry must fire a 2nd fetch');
    });
});

// ============================================================================
// 8. loadIDEStatus() — container pane
// ============================================================================

describe('loadIDEStatus — ide-status.js', function() {
    it('AC1 — 503 → .pane-error in #ide-list-body', async function() {
        const h = buildHarness(['ide-list-body'], {
            ok: false, status: 503, json: {}
        });
        evalModule(h.window, 'ide-status.js');
        await h.window.loadIDEStatus();
        await flushPromises();

        const container = h.document.getElementById('ide-list-body');
        assert.ok(container.querySelector('.pane-error'), 'AC1: .pane-error on 503');
    });

    it('AC5 — empty ides array → .empty-state', async function() {
        const h = buildHarness(['ide-list-body'], {
            ok: true, status: 200, json: { ides: [] }
        });
        evalModule(h.window, 'ide-status.js');
        await h.window.loadIDEStatus();
        await flushPromises();

        const container = h.document.getElementById('ide-list-body');
        assert.ok(container.querySelector('.empty-state'), 'AC5: .empty-state for empty ides');
    });
});

// ============================================================================
// 9. optimize.js — _loadOptimizeConfig and _loadSavings
// ============================================================================

describe('optimize.js — _loadOptimizeConfig', function() {
    it('AC1 — 503 → .pane-error in #optimize-config-card', async function() {
        const h = buildHarness([
            'optimize-config-card', 'optimize-savings-card',
            'opt-enabled', 'opt-proxy-enabled', 'opt-cache-enabled',
            'opt-semantic-enabled', 'opt-compress-enabled',
            'opt-compress-mode', 'opt-compress-prose', 'opt-config-version'
        ], { ok: false, status: 503, json: {} });
        evalModule(h.window, 'optimize.js');
        // initOptimizeTab triggers both loaders
        await h.window.initOptimizeTab();
        await flushPromises();

        const card = h.document.getElementById('optimize-config-card');
        assert.ok(card.querySelector('.pane-error'), 'AC1: .pane-error in config card on 503');
    });
});

describe('optimize.js — _loadSavings', function() {
    it('AC1 — 503 → .pane-error in #optimize-savings-card (no retry button per D-3)', async function() {
        const h = buildHarness([
            'optimize-config-card', 'optimize-savings-card',
            'opt-tokens-saved', 'opt-usd-saved', 'opt-inr-saved',
            'opt-hit-rate', 'opt-cache-entries', 'opt-cache-size',
            'opt-compression-ratio', 'opt-pricing-date', 'opt-stale-warning'
        ], { ok: false, status: 503, json: {} });
        evalModule(h.window, 'optimize.js');
        await h.window.initOptimizeTab();
        await flushPromises();

        const card = h.document.getElementById('optimize-savings-card');
        assert.ok(card.querySelector('.pane-error'), 'AC1: .pane-error in savings card on 503');
        // D-3: _loadSavings has no Retry (auto-heals on next poll)
        assert.equal(card.querySelector('.pane-error-retry'), null, 'D-3: no retry button on savings card');
    });
});
