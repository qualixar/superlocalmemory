/**
 * OD shell navigation must preserve mounted panes and suppress legacy loaders.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const shellSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/od-shell.js'),
  'utf8',
);

function harness(options = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <aside id="sidebar"></aside>
    <main id="main-content">
      <div id="dashboard-pane" class="tab-pane active"></div>
      <div id="memories-pane" class="tab-pane"></div>
    </main>
    <span id="topbar-crumb"></span>
    <span id="topbar-heading"></span>
    <div class="topbar"><button data-theme-icon></button></div>
    <button id="dashboard-tab"></button>
    <button id="memories-tab"></button>
  </body></html>`, { runScripts: 'dangerously', url: 'http://localhost:8765/' });
  const { window } = dom;
  window.scrollTo = function () {};
  window.HTMLElement.prototype.scrollTo = function () {};
  window.HTMLElement.prototype.scrollIntoView = function () {};
  window.fetch = function () {
    return Promise.resolve({ ok: true, json: function () { return Promise.resolve({}); } });
  };
  if (options.cacheTtlMs !== undefined) {
    window.SLM_PANE_CACHE_TTL_MS = options.cacheTtlMs;
  }
  if (options.refreshQuietMs !== undefined) {
    window.SLM_PANE_REFRESH_QUIET_MS = options.refreshQuietMs;
  }
  const script = window.document.createElement('script');
  script.textContent = shellSource;
  window.document.head.appendChild(script);
  return window;
}

describe('OD shell navigation lifecycle', function () {
  it('mounts an OD pane once and never invokes its legacy tab loader', function () {
    const window = harness();
    let odRenders = 0;
    let legacyLoads = 0;
    window.odRenderMemories = function (pane) {
      odRenders += 1;
      pane.innerHTML = '<div data-mounted="memories">ready</div>';
    };
    window.document.getElementById('memories-tab').addEventListener(
      'shown.bs.tab',
      function () { legacyLoads += 1; },
    );

    window.slmShell({ active: 'memories-pane' });
    const dashboard = window.document.querySelector('[data-tab="dashboard-pane"]');
    const memories = window.document.querySelector('[data-tab="memories-pane"]');
    dashboard.click();
    memories.click();

    assert.equal(odRenders, 1, 'returning to a mounted pane must preserve its state');
    assert.equal(legacyLoads, 0, 'OD-owned panes must not dispatch legacy loaders');
  });

  it('keeps the GitHub CTA actionable without a hard-coded star count', function () {
    const window = harness();
    window.slmShell({ active: 'dashboard-pane' });

    const cta = window.document.querySelector('.star-cta');
    assert.ok(cta, 'the GitHub CTA must remain present');
    assert.equal(cta.querySelector('.star-count'), null, 'counts must not be shipped as static UI data');
    assert.doesNotMatch(cta.textContent, /2,431|197/, 'the CTA must not imply a live repository count');
  });

  it('serves a mounted pane during the TTL, then performs one stale refresh', async function () {
    const window = harness({ cacheTtlMs: 1000, refreshQuietMs: 0 });
    let now = 100;
    let requests = 0;
    let releaseRefresh;
    window.Date.now = function () { return now; };
    window.fetch = function (path) {
      if (path !== '/api/memories') {
        return Promise.resolve({
          ok: true,
          json: function () { return Promise.resolve({}); },
        });
      }
      requests += 1;
      if (requests === 1) {
        return Promise.resolve({
          ok: true,
          json: function () { return Promise.resolve({ value: 'initial' }); },
        });
      }
      return new Promise(function (resolve) {
        releaseRefresh = function () {
          resolve({
            ok: true,
            json: function () { return Promise.resolve({ value: 'fresh' }); },
          });
        };
      });
    };
    window.odRenderMemories = function (pane) {
      pane.innerHTML = '<div data-state="loading">loading</div>';
      window.fetch('/api/memories').then(function (response) {
        return response.json();
      }).then(function (data) {
        pane.innerHTML = '<div data-state="ready">' + data.value + '</div>';
      });
    };

    window.slmShell({ active: 'memories-pane' });
    await new Promise(function (resolve) { window.setTimeout(resolve, 0); });
    assert.equal(requests, 1);
    assert.equal(window.document.getElementById('memories-pane').textContent, 'initial');

    now = 1099;
    window.document.querySelector('[data-tab="dashboard-pane"]').click();
    window.document.querySelector('[data-tab="memories-pane"]').click();
    assert.equal(requests, 1, 'navigation inside the TTL must not query again');

    now = 1100;
    window.document.querySelector('[data-tab="dashboard-pane"]').click();
    window.document.querySelector('[data-tab="memories-pane"]').click();
    await new Promise(function (resolve) { window.setTimeout(resolve, 0); });
    assert.equal(requests, 2, 'the first stale visit must start one refresh');

    window.document.querySelector('[data-tab="dashboard-pane"]').click();
    window.document.querySelector('[data-tab="memories-pane"]').click();
    await new Promise(function (resolve) { window.setTimeout(resolve, 0); });
    assert.equal(requests, 2, 'an in-flight stale refresh must be coalesced');

    const snapshot = window.document.querySelector('[data-slm-stale-for="memories-pane"]');
    const replacement = window.document.getElementById('memories-pane');
    assert.ok(snapshot, 'the mounted pane must remain available during refresh');
    assert.equal(snapshot.textContent, 'initial');
    assert.equal(snapshot.classList.contains('active'), true);
    assert.equal(replacement.style.display, 'none');

    releaseRefresh();
    await new Promise(function (resolve) { window.setTimeout(resolve, 5); });
    assert.equal(
      window.document.querySelector('[data-slm-stale-for="memories-pane"]'),
      null,
      'the visual snapshot must be removed after the refresh settles',
    );
    assert.equal(window.document.getElementById('memories-pane').textContent, 'fresh');
    assert.equal(requests, 2);
    window.close();
  });
});
