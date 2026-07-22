/**
 * tests/ui/test_memories_render.mjs — memories.js loader tests.
 * Runner: node --test tests/ui/
 * Requires: jsdom
 *
 * Regression target (P1f): the legacy loadMemories()/renderMemoriesTable()
 * loader is still invoked on the OD dashboard (od-shell.js:282) where the
 * legacy #memories-list table does NOT exist. Before the fix,
 * renderMemoriesTable() fell through to `container.textContent = ''` on a null
 * container and threw "Cannot set properties of null (setting 'textContent')",
 * which surfaced as a caught console.error("Error loading memories:", …) on
 * every Memories-tab open. The fix null-guards the DOM render while preserving
 * the window._slmMemories cache that OD features read.
 *
 * AC coverage:
 *   AC-OD  container absent + non-empty data → no throw / no error log; cache set
 *   AC5    container present + empty data     → .empty-state, no error
 *   AC4    container present + non-empty data → table rendered; cache set
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function memHarness(containerIds, json) {
    const h = buildHarness(containerIds, { ok: true, status: 200, json });
    const errors = [];
    // Spy on console.error AFTER core.js eval so we capture the loader's catch.
    h.window.console.error = function() {
        errors.push(Array.prototype.map.call(arguments, String).join(' '));
    };
    evalModule(h.window, 'memories.js');
    return { ...h, errors };
}

const NON_EMPTY = {
    memories: [
        { memory_id: 'm1', content: 'first memory', importance: 7, category: 'semantic' },
        { memory_id: 'm2', content: 'second memory', importance: 9, category: 'episodic' },
    ],
    total: 2, limit: 50, offset: 0, has_more: false,
};

describe('loadMemories — OD-pane null-container regression', function() {
    it('AC-OD — no #memories-list + non-empty data → no error, cache still set', async function() {
        // No container ids at all → mirrors the OD dashboard, which renders
        // via odRenderMemories and has no legacy #memories-list table.
        const h = memHarness([], NON_EMPTY);

        await h.window.loadMemories();
        await flushPromises();

        const memErrs = h.errors.filter(e => /Error loading memories/i.test(e));
        assert.equal(
            memErrs.length, 0,
            'loadMemories must not error when #memories-list is absent (got: ' + memErrs.join(' | ') + ')',
        );
        // Side-effect OD features depend on must survive the early return.
        assert.ok(Array.isArray(h.window._slmMemories), 'window._slmMemories must be an array');
        assert.equal(h.window._slmMemories.length, 2, 'cache must hold the returned memories');
    });
});

describe('loadMemories — legacy #memories-list present', function() {
    it('AC5 — empty data → .empty-state, no error', async function() {
        const h = memHarness(['memories-list'], { memories: [], total: 0, limit: 50, offset: 0 });

        await h.window.loadMemories();
        await flushPromises();

        const container = h.document.getElementById('memories-list');
        assert.ok(container.querySelector('.empty-state'), 'AC5: .empty-state must render for empty data');
        assert.equal(
            h.errors.filter(e => /Error loading/i.test(e)).length, 0,
            'AC5: no error on empty data',
        );
    });

    it('AC4 — non-empty data → memory table rendered, cache set, no error', async function() {
        const h = memHarness(['memories-list'], NON_EMPTY);

        await h.window.loadMemories();
        await flushPromises();

        const container = h.document.getElementById('memories-list');
        assert.ok(container.querySelector('table.memory-table'), 'AC4: memory table must be rendered');
        assert.equal(container.querySelectorAll('tbody tr').length, 2, 'AC4: one row per memory');
        assert.equal(h.window._slmMemories.length, 2, 'AC4: cache set on success');
        assert.equal(
            h.errors.filter(e => /Error loading/i.test(e)).length, 0,
            'AC4: no error on success',
        );
    });
});
