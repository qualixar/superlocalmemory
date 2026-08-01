/**
 * tests/ui/test_confirm_modal.mjs — shared confirmDestructive modal logic.
 * Runner: node --test tests/ui/test_confirm_modal.mjs
 * Requires: jsdom (devDependency)
 *
 * Because Bootstrap is not available in jsdom, window.bootstrap is stubbed so
 * confirmDestructive() takes the modal-show path rather than a fallback path.
 * The tests exercise DOM construction and Promise resolution; visual rendering
 * is not tested here.
 *
 * Acceptance criterion: every destructive dashboard action routes through
 * confirmDestructive(), which populates the shared modal with the exact target +
 * consequence text before issuing any network request. Cancelling issues no
 * request.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule } from './harness.mjs';

function cdHarness() {
    const h = buildHarness([], { ok: true, status: 200, json: {} });
    // Stub bootstrap.Modal so confirmDestructive() takes the full modal path.
    h.window.bootstrap = {
        Modal: {
            getOrCreateInstance: function(el) {
                return { show: function() {}, hide: function() {} };
            },
        },
    };
    evalModule(h.window, 'modal.js');
    return h;
}

describe('confirmDestructive — shared destructive-action modal', function() {

    it('exposes confirmDestructive on window after modal.js loads', function() {
        const h = cdHarness();
        assert.equal(
            typeof h.window.confirmDestructive, 'function',
            'confirmDestructive must be a function on window',
        );
    });

    it('creates the modal element with the supplied title, target, and consequence', function() {
        const h = cdHarness();
        // Start the promise but do not await — we inspect the DOM synchronously.
        h.window.confirmDestructive({
            title: 'Delete profile',
            target: 'my-project',
            consequence: 'Its memories will be moved to the default profile.',
        });

        const modal = h.document.getElementById('slm-confirm-destructive-modal');
        assert.ok(modal, 'modal element must exist in document after call');

        const titleText = modal.querySelector('.slm-cd-title').textContent;
        assert.ok(
            titleText.includes('Delete profile'),
            'title element must contain "Delete profile", got: ' + titleText,
        );

        const targetText = modal.querySelector('.slm-cd-target').textContent;
        assert.ok(
            targetText.includes('my-project'),
            'target element must contain "my-project", got: ' + targetText,
        );

        const conseqText = modal.querySelector('.slm-cd-consequence').textContent;
        assert.ok(
            conseqText.includes('moved to the default profile'),
            'consequence element must contain the consequence text, got: ' + conseqText,
        );
    });

    it('resolves true when the confirm button is clicked', async function() {
        const h = cdHarness();
        const p = h.window.confirmDestructive({
            title: 'Delete profile',
            target: 'my-project',
            consequence: 'Its memories will be moved to the default profile.',
        });

        const confirmBtn = h.document.querySelector('[data-slm-cd-action="confirm"]');
        assert.ok(confirmBtn, 'confirm button must exist in modal');
        confirmBtn.click();

        const result = await p;
        assert.equal(result, true, 'promise must resolve true on confirm');
    });

    it('resolves false when the cancel button is clicked', async function() {
        const h = cdHarness();
        const p = h.window.confirmDestructive({
            title: 'Delete profile',
            target: 'my-project',
            consequence: 'Its memories will be moved to the default profile.',
        });

        const cancelBtn = h.document.querySelector('[data-slm-cd-action="cancel"]');
        assert.ok(cancelBtn, 'cancel button must exist in modal');
        cancelBtn.click();

        const result = await p;
        assert.equal(result, false, 'promise must resolve false on cancel');
    });

    it('does NOT issue a network request until confirmed', async function() {
        const h = cdHarness();
        const calls = [];
        h.window.fetch = async function(url, opts) {
            calls.push({ url, opts });
            return { ok: true, status: 200, json: async function() { return {}; } };
        };

        // Simulate a callsite: gate the fetch behind confirmDestructive.
        async function doDestructiveAction() {
            const confirmed = await h.window.confirmDestructive({
                title: 'Delete profile',
                target: 'my-project',
                consequence: 'Its memories will be moved to the default profile.',
            });
            if (!confirmed) return;
            await h.window.fetch('/api/profiles/my-project', { method: 'DELETE' });
        }

        // Start — awaits modal confirmation.
        const action = doDestructiveAction();

        // Modal is shown; verify no request has been issued yet.
        assert.equal(calls.length, 0, 'no fetch before confirmation');

        // Cancel — request must NOT be issued.
        const cancelBtn = h.document.querySelector('[data-slm-cd-action="cancel"]');
        cancelBtn.click();
        await action;

        assert.equal(calls.length, 0, 'no fetch after cancel');
    });

    it('issues the network request after explicit confirmation', async function() {
        const h = cdHarness();
        const calls = [];
        h.window.fetch = async function(url, opts) {
            calls.push({ url, opts });
            return { ok: true, status: 200, json: async function() { return {}; } };
        };

        async function doDestructiveAction() {
            const confirmed = await h.window.confirmDestructive({
                title: 'Delete profile',
                target: 'my-project',
                consequence: 'Its memories will be moved to the default profile.',
            });
            if (!confirmed) return;
            await h.window.fetch('/api/profiles/my-project', { method: 'DELETE' });
        }

        const action = doDestructiveAction();
        assert.equal(calls.length, 0, 'no fetch before confirmation');

        const confirmBtn = h.document.querySelector('[data-slm-cd-action="confirm"]');
        confirmBtn.click();
        await action;

        assert.equal(calls.length, 1, 'exactly one fetch after confirmation');
        assert.equal(calls[0].url, '/api/profiles/my-project', 'fetch must hit the correct endpoint');
        assert.equal(calls[0].opts.method, 'DELETE', 'fetch must use DELETE method');
    });

    it('reuses the same modal element on repeated calls', function() {
        const h = cdHarness();
        h.window.confirmDestructive({ title: 'First', target: 'foo', consequence: 'c1' });
        h.window.confirmDestructive({ title: 'Second', target: 'bar', consequence: 'c2' });
        const modals = h.document.querySelectorAll('#slm-confirm-destructive-modal');
        assert.equal(modals.length, 1, 'must not create duplicate modal elements');
    });

    it('confirm button label defaults to "Confirm"', function() {
        const h = cdHarness();
        h.window.confirmDestructive({ title: 'T', target: 'x', consequence: 'c' });
        const confirmBtn = h.document.querySelector('[data-slm-cd-action="confirm"]');
        assert.equal(confirmBtn.textContent.trim(), 'Confirm', 'default label must be "Confirm"');
    });

    it('confirm button label can be overridden via confirmLabel', function() {
        const h = cdHarness();
        h.window.confirmDestructive({
            title: 'Remove user', target: 'alice', consequence: 'Access deleted.',
            confirmLabel: 'Remove',
        });
        const confirmBtn = h.document.querySelector('[data-slm-cd-action="confirm"]');
        assert.equal(confirmBtn.textContent.trim(), 'Remove', 'label must match confirmLabel option');
    });
});
