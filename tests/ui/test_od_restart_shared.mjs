/**
 * The restart function in od-operations.js is shared, not copied: the
 * images-and-documents card calls window.odRestartDaemon.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { makeEnv, flushPromises } from './features_helpers.mjs';

describe('shared daemon restart', () => {
    it('od-operations exposes odRestartDaemon, which confirms and then POSTs the restart', async () => {
        const h = makeEnv([['POST', '/api/daemon/restart', () => ({ json: { success: true } })],
                           ['GET', '/health', () => ({ json: { ready: false } })]],
                          { modules: ['od-operations.js'] });
        assert.equal(typeof h.window.odRestartDaemon, 'function');
        const b = h.document.createElement('button');
        b.textContent = 'Restart SuperLocalMemory';
        h.window.odRestartDaemon(b, h.document.createElement('p'));
        await flushPromises();
        assert.equal(h.confirms.length, 1, 'the existing confirm modal is used');
        assert.ok(h.calls.some(c => c.method === 'POST' && c.url === '/api/daemon/restart'));
    });
});
