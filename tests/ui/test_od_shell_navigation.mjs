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
    <div class="scrim" id="scrim"></div>
    <button id="menuBtn" aria-label="Open menu"></button>
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
  it('closes the phone menu once a page is chosen (audit 4.1.20 L7)', function () {
    const window = harness();
    window.slmShell({ active: 'dashboard-pane' });
    const doc = window.document;
    const sidebar = doc.getElementById('sidebar');
    const scrim = doc.getElementById('scrim');
    const menuBtn = doc.getElementById('menuBtn');

    menuBtn.click();
    assert.ok(sidebar.classList.contains('open'), 'the menu opens');
    assert.ok(scrim.classList.contains('on'));
    assert.equal(menuBtn.getAttribute('aria-expanded'), 'true');

    doc.querySelector('[data-tab="memories-pane"]').click();
    assert.ok(!sidebar.classList.contains('open'), 'choosing a page closes the menu');
    assert.ok(!scrim.classList.contains('on'), 'and removes the scrim');
    assert.equal(menuBtn.getAttribute('aria-expanded'), 'false');

    menuBtn.click();
    doc.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
    assert.ok(!sidebar.classList.contains('open'), 'Escape closes it too');
  });

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
    // The refresh settles through several promise and timer hops; on a busy machine 5 ms is
    // not enough. Wait for the snapshot to go (or 2 s), then check.
    for (let waited = 0; waited < 2000; waited += 10) {
      if (!window.document.querySelector('[data-slm-stale-for="memories-pane"]')) break;
      await new Promise(function (resolve) { window.setTimeout(resolve, 10); });
    }
    assert.equal(
      window.document.querySelector('[data-slm-stale-for="memories-pane"]'),
      null,
      'the visual snapshot must be removed after the refresh settles',
    );
    assert.equal(window.document.getElementById('memories-pane').textContent, 'fresh');
    assert.equal(requests, 2);
    window.close();
  });

  it('changing page never scrolls the page itself, only the sidebar list', async function () {
    // scrollIntoView on the sidebar link also scrolled the page, hiding each
    // pane's first line under the sticky top bar.
    const window = harness();
    const scrolled = [];
    window.HTMLElement.prototype.scrollIntoView = function () { scrolled.push(this); };
    window.slmShell({ active: 'dashboard-pane' });
    const doc = window.document;
    const link = doc.querySelector('[data-tab="memories-pane"]');
    const list = link.closest('.nav');
    assert.ok(list, 'the sidebar links sit in their own scrolling list');

    link.getBoundingClientRect = () => ({ top: 900, bottom: 940 });
    list.getBoundingClientRect = () => ({ top: 100, bottom: 800 });
    link.click();

    assert.deepEqual(scrolled, [], 'nothing asked the browser to scroll the page');
    assert.equal(list.scrollTop, 140, 'the list moved just enough to show the link');
    await new Promise((resolve) => setTimeout(resolve, 600));  // the shell's empty-pane retry
    window.close();
  });

  it('changing page brings the page itself back to the top', async function () {
    // The page scrolls on <body>, which window.scrollTo does not move.
    const window = harness();
    window.slmShell({ active: 'dashboard-pane' });
    const doc = window.document;
    doc.body.scrollTop = 86;
    doc.querySelector('[data-tab="memories-pane"]').click();
    assert.equal(doc.body.scrollTop, 0, 'the page starts at its first line');
    await new Promise((resolve) => setTimeout(resolve, 600));  // the shell's empty-pane retry
    window.close();
  });
});
