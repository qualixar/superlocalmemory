import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule } from './harness.mjs';

describe('dashboard scope settings', function() {
    it('sends explicit booleans and waits for daemon acknowledgement', async function() {
        const h = buildHarness([], { ok: true, status: 200, json: {} });
        h.document.body.innerHTML = `
            <select id="settings-default-scope">
                <option value="personal">Personal</option>
                <option value="shared">Shared</option>
            </select>
            <input type="checkbox" id="settings-recall-shared">
            <input type="checkbox" id="settings-recall-global">
            <div id="settings-scope-status"></div>
        `;
        h.document.getElementById('settings-default-scope').value = 'personal';
        h.document.getElementById('settings-recall-shared').checked = true;
        const calls = [];
        h.window.fetch = async function(url, options) {
            calls.push({ url, options });
            return {
                ok: true,
                status: 200,
                json: async function() {
                    return {
                        success: true,
                        default_scope: 'personal',
                        recall_include_shared: true,
                        recall_include_global: false,
                    };
                },
            };
        };
        evalModule(h.window, 'auto-settings.js');

        const result = await h.window.saveScopeSettings();
        const payload = JSON.parse(calls[0].options.body);

        assert.equal(result, true);
        assert.equal(calls[0].url, '/api/v3/scope/config');
        assert.deepEqual(payload, {
            default_scope: 'personal',
            recall_include_shared: true,
            recall_include_global: false,
        });
        assert.match(
            h.document.getElementById('settings-scope-status').textContent,
            /applied/i,
        );
    });
});
