import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule } from './harness.mjs';

function profileHarness(response) {
    const harness = buildHarness(
        ['profile-select', 'profiles-table'],
        { ok: true, status: 200, json: {} },
    );
    const calls = [];
    harness.window.fetch = async function(url, options) {
        calls.push({ url, options });
        return {
            ok: response.ok,
            status: response.status,
            json: async function() { return response.json; },
        };
    };
    evalModule(harness.window, 'profiles.js');
    // profiles.js declares loadProfiles/switchProfile globally, so install
    // observability stubs after evaluating the module under test.
    harness.window.showToast = function(message) { calls.push({ toast: message }); };
    harness.window.loadProfiles = function() { calls.push({ reload: 'profiles' }); };
    harness.window.loadStats = function() { calls.push({ reload: 'stats' }); };
    return { ...harness, calls };
}

describe('dashboard profile switching', function() {
    it('acknowledges an exact daemon ACK and drops piecemeal refresh', async function() {
        const h = profileHarness({
            ok: true,
            status: 200,
            json: { success: true, active_profile: 'beta', generation: 7 },
        });
        await new Promise(resolve => setTimeout(resolve, 10));
        h.calls.length = 0;

        const result = await h.window.switchProfile('beta');

        assert.equal(result, true);
        assert.equal(h.calls[0].url, '/api/profiles/beta/switch');
        assert.equal(h.calls[0].options.method, 'POST');
        // v3.8.0: an acknowledged switch does ONE full window reload (not
        // observable under jsdom), so the legacy piecemeal per-view refreshers
        // must NOT run — that's the behavior change.
        assert.ok(!h.calls.some(call => call.reload === 'stats'));
        assert.ok(!h.calls.some(call => call.reload === 'profiles'));
        assert.ok(h.calls.some(call => /switched to profile/i.test(call.toast || '')));
    });

    it('rejects HTTP errors even if their body contains active_profile', async function() {
        const h = profileHarness({
            ok: false,
            status: 500,
            json: { success: false, active_profile: 'beta', detail: 'rebind failed' },
        });
        await new Promise(resolve => setTimeout(resolve, 10));
        h.calls.length = 0;

        const result = await h.window.switchProfile('beta');

        assert.equal(result, false);
        assert.ok(!h.calls.some(call => call.reload === 'stats'));
        assert.ok(h.calls.some(call => /rebind failed/i.test(call.toast || '')));
    });

    it('rejects an ACK for a different profile and reloads runtime truth', async function() {
        const h = profileHarness({
            ok: true,
            status: 200,
            json: { success: true, active_profile: 'alpha', generation: 7 },
        });
        await new Promise(resolve => setTimeout(resolve, 10));
        h.calls.length = 0;

        const result = await h.window.switchProfile('beta');

        assert.equal(result, false);
        assert.ok(h.calls.some(call => call.reload === 'profiles'));
        assert.ok(!h.calls.some(call => call.reload === 'stats'));
    });
});
