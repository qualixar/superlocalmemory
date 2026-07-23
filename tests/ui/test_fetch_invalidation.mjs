/**
 * Successful mutations invalidate in-memory OD pane snapshots.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { buildHarness } from './harness.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const coreSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/core.js'),
  'utf8',
);

function buildCoreHarness() {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'dangerously',
    url: 'http://localhost:8765/',
  });
  const { window } = dom;
  const calls = [];
  window.matchMedia = function () {
    return { matches: false, addEventListener: function () {}, removeListener: function () {} };
  };
  window.fetch = function (input, init) {
    calls.push({ input, init });
    return Promise.resolve({
      ok: true,
      status: 200,
      json: function () { return Promise.resolve({ token: 'test-install-token' }); },
    });
  };
  const script = window.document.createElement('script');
  script.textContent = coreSource;
  window.document.head.appendChild(script);
  window.loadProfiles = function () {};
  window.loadStats = function () {};
  window.loadGraph = function () {};
  return { dom, window, calls };
}

describe('global fetch mutation invalidation', function () {
  it('invalidates mounted panes after a successful local mutation', async function () {
    const h = buildHarness([], {
      ok: true,
      status: 200,
      json: { token: 'test-install-token', success: true },
    });
    let invalidations = 0;
    h.window.slmInvalidatePanes = function () { invalidations += 1; };

    const response = await h.window.fetch('/remember', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: 'test memory' }),
    });

    assert.equal(response.ok, true);
    assert.equal(invalidations, 1);
  });

  it('treats read-only search POSTs as neither writes nor cache invalidations', async function () {
    const h = buildCoreHarness();
    let invalidations = 0;
    h.window.slmInvalidatePanes = function () { invalidations += 1; };

    await h.window.fetch('/api/search', {
      method: 'POST',
      slmInvalidatesCache: false,
      slmRequiresWriteAuth: false,
    });

    assert.equal(h.calls.length, 1);
    assert.equal(h.calls[0].input, '/api/search');
    assert.equal(h.calls[0].init.slmInvalidatesCache, undefined);
    assert.equal(h.calls[0].init.slmRequiresWriteAuth, undefined);
    assert.equal(invalidations, 0);
    h.dom.window.close();
  });

  it('never sends local credentials to protocol-relative external URLs', async function () {
    const h = buildCoreHarness();
    let invalidations = 0;
    h.window.slmInvalidatePanes = function () { invalidations += 1; };

    await h.window.fetch('//untrusted.example/mutate', { method: 'POST' });

    assert.equal(h.calls.length, 1);
    assert.equal(h.calls[0].input, '//untrusted.example/mutate');
    assert.equal(h.calls[0].init.headers, undefined);
    assert.equal(invalidations, 0);
    h.dom.window.close();
  });

  it('does not mistake an external URL object for an empty local path', async function () {
    const h = buildCoreHarness();

    await h.window.fetch(new h.window.URL('https://untrusted.example/mutate'), {
      method: 'POST',
    });

    assert.equal(h.calls.length, 1);
    assert.equal(h.calls[0].input.href, 'https://untrusted.example/mutate');
    h.dom.window.close();
  });
});
