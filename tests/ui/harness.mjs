/**
 * tests/ui/harness.mjs — jsdom harness for WP-12 pane-error tests.
 *
 * CRITICAL: window.fetch MUST be stubbed before core.js is eval'd because
 * core.js:25 self-patches window.fetch at eval time. If fetch isn't present
 * before eval, the patch captures undefined.
 */

import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC_JS = join(__dirname, '../../src/superlocalmemory/ui/js');

/**
 * Build a JSDOM environment with anchor elements for the given pane IDs.
 *
 * @param {string[]} containerIds  - IDs to create as <div> in document body
 * @param {object}   fetchStub     - initial fetch stub { ok, status, json }
 * @returns {{ dom, window, document, mockFetch }}
 */
export function buildHarness(containerIds, fetchStub) {
    const html = containerIds
        .map(id => `<div id="${id}"></div>`)
        .join('\n');

    const dom = new JSDOM(`<!DOCTYPE html><html><body>${html}</body></html>`, {
        runScripts: 'dangerously',
        resources: 'usable',
    });

    const { window } = dom;
    const { document } = window;

    // Provide fetch stub BEFORE core.js eval so the patch captures it.
    let _stub = fetchStub;
    window.fetch = function mockFetch() {
        return Promise.resolve({
            ok: _stub.ok,
            status: _stub.status,
            json: function() { return Promise.resolve(_stub.json); },
        });
    };

    // Expose fetch-rejection helper for network error simulation
    window.__setFetchReject = function(reason) {
        window.fetch = function() { return Promise.reject(reason || new Error('NetworkError')); };
    };

    // Allow tests to swap stub mid-test
    window.__setFetchStub = function(stub) {
        _stub = stub;
        window.fetch = function mockFetch() {
            return Promise.resolve({
                ok: stub.ok,
                status: stub.status,
                json: function() { return Promise.resolve(stub.json); },
            });
        };
    };

    // Stub requestAnimationFrame (used by animateCounter)
    window.requestAnimationFrame = function(cb) { return setTimeout(cb, 0); };

    // Stub setInterval to prevent polling timers from keeping the event loop alive
    // in tests. Returns a fake timer ID; clearInterval is also stubbed as no-op.
    window.setInterval = function() { return 0; };
    window.clearInterval = function() {};

    // Stub localStorage — jsdom already provides one, but it may not be writable.
    // Define via Object.defineProperty to handle both getter-only and absent cases.
    const _store = {};
    const localStorageStub = {
        getItem: function(k) { return _store[k] !== undefined ? _store[k] : null; },
        setItem: function(k, v) { _store[k] = String(v); },
        removeItem: function(k) { delete _store[k]; },
        clear: function() { Object.keys(_store).forEach(function(k) { delete _store[k]; }); },
    };
    try {
        Object.defineProperty(window, 'localStorage', {
            value: localStorageStub,
            writable: true,
            configurable: true,
        });
    } catch (e) { /* already writable or not accessible */ }

    // Stub sessionStorage
    const _sstore = {};
    const sessionStorageStub = {
        getItem: function(k) { return _sstore[k] !== undefined ? _sstore[k] : null; },
        setItem: function(k, v) { _sstore[k] = String(v); },
        removeItem: function(k) { delete _sstore[k]; },
    };
    try {
        Object.defineProperty(window, 'sessionStorage', {
            value: sessionStorageStub,
            writable: true,
            configurable: true,
        });
    } catch (e) { /* already writable or not accessible */ }

    // Stub matchMedia
    window.matchMedia = function() {
        return { matches: false, addEventListener: function() {}, removeListener: function() {} };
    };

    // Evaluate core.js so helpers (showPaneError etc.) land on window.
    // NOTE: core.js defines loadStats etc., which will overwrite any stubs set before eval.
    const coreJs = readFileSync(join(SRC_JS, 'core.js'), 'utf8');
    const script = dom.window.document.createElement('script');
    script.textContent = coreJs;
    dom.window.document.head.appendChild(script);

    // Stub global functions that core.js's DOMContentLoaded tries to call.
    // Must be done AFTER core.js eval because function declarations in core.js
    // overwrite any pre-stubs on the window object.
    window.loadProfiles = function() {};
    window.loadStats = function() { return Promise.resolve(); };
    window.loadGraph = function() {};
    window.initEventStream = function() {};
    window.loadEventStats = function() {};
    window.loadAgents = function() {};
    window.loadCloudDestinations = function() {};
    window.populateFilters = function() {};

    return { dom, window, document };
}

/**
 * Evaluate a UI module file in the given jsdom window context.
 *
 * @param {object} window  - jsdom window
 * @param {string} filename - basename, e.g. 'math-health.js'
 */
export function evalModule(window, filename) {
    const code = readFileSync(join(SRC_JS, filename), 'utf8');
    const script = window.document.createElement('script');
    script.textContent = code;
    window.document.head.appendChild(script);
}

/**
 * Return a function that, when called, resolves the next microtask queue flush.
 * Used to await async loader functions in jsdom.
 */
export function flushPromises() {
    return new Promise(function(resolve) { setTimeout(resolve, 10); });
}
