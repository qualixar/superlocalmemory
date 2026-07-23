/** Dashboard browsing race and state contracts for OD panes. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function deferred() {
  let resolve;
  const promise = new Promise(function (done) { resolve = done; });
  return { promise, resolve };
}

function response(body) {
  return { ok: true, status: 200, json: function () { return Promise.resolve(body); } };
}

describe('OD entity browsing', function () {
  it('resets the page and ignores a late unfiltered response after a type/search change', async function () {
    const h = buildHarness(['entity-test-root'], { ok: true, status: 200, json: {} });
    const pending = [];
    h.window.fetch = function (url) {
      const next = deferred();
      pending.push({ url: String(url), next });
      return next.promise;
    };
    evalModule(h.window, 'od-entities.js');
    h.window.odRenderEntities(h.document.getElementById('entity-test-root'));

    const person = h.document.querySelector('[data-od-act="type-filter"][data-type="person"]');
    person.click();
    const search = h.document.querySelector('[data-od-act="ent-search"]');
    search.value = 'needle';
    search.dispatchEvent(new h.window.Event('input', { bubbles: true }));
    await new Promise(function (resolve) { setTimeout(resolve, 280); });

    const current = pending[pending.length - 1];
    assert.match(current.url, /type=person/);
    assert.match(current.url, /search=needle/);
    assert.match(current.url, /offset=0/);
    current.next.resolve(response({
      entities: [{ name: 'Needle Person', type: 'person', fact_count: 1 }],
      total: 1, limit: 50, offset: 0,
    }));
    await flushPromises();

    pending[0].next.resolve(response({
      entities: [{ name: 'Wrong old page', type: 'concept', fact_count: 99 }],
      total: 99, limit: 50, offset: 0,
    }));
    await flushPromises();

    assert.match(h.document.getElementById('entity-test-root').textContent, /Needle Person/);
    assert.doesNotMatch(h.document.getElementById('entity-test-root').textContent, /Wrong old page/);
  });
});

describe('OD memory browsing', function () {
  it('clears an active search, resets page, and ignores its late response after a category change', async function () {
    const h = buildHarness(['memory-test-root'], { ok: true, status: 200, json: {} });
    const pending = [];
    h.window.fetch = function (url, opts) {
      const next = deferred();
      pending.push({ url: String(url), opts: opts || {}, next });
      return next.promise;
    };
    evalModule(h.window, 'od-memories.js');
    h.window.odRenderMemories(h.document.getElementById('memory-test-root'));
    pending.slice(1).forEach(function (request) {
      request.next.resolve(response({ total: request.url.includes('semantic') ? 1 : 0 }));
    });
    await flushPromises();

    const search = h.document.querySelector('[data-od-act="search"]');
    search.value = 'old query';
    search.dispatchEvent(new h.window.Event('input', { bubbles: true }));
    await new Promise(function (resolve) { setTimeout(resolve, 360); });
    const searchRequest = pending[pending.length - 1];
    assert.equal(searchRequest.url, '/api/search');
    assert.equal(searchRequest.opts.slmInvalidatesCache, false);
    assert.equal(searchRequest.opts.slmRequiresWriteAuth, false);

    const category = h.document.querySelector('[data-od-act="cat"][data-cat="semantic"]');
    category.click();
    const current = pending[pending.length - 1];
    assert.match(current.url, /category=semantic/);
    assert.match(current.url, /offset=0/);
    current.next.resolve(response({
      memories: [{ id: 'new', content: 'semantic truth', category: 'semantic' }],
      total: 1, limit: 50, offset: 0,
    }));
    await flushPromises();

    searchRequest.next.resolve(response({
      results: [{ fact_id: 'old', content: 'late search result', category: 'opinion' }],
    }));
    await flushPromises();

    assert.match(h.document.getElementById('memory-test-root').textContent, /semantic truth/);
    assert.doesNotMatch(h.document.getElementById('memory-test-root').textContent, /late search result/);
  });
});
