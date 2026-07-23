/** Dashboard navigation must coalesce and cache expensive read queries. */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dashboardSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/dashboard.js'),
  'utf8',
);
const coreSource = readFileSync(
  join(__dirname, '../../src/superlocalmemory/ui/js/core.js'),
  'utf8',
);

function responseFor(path) {
  if (path === '/api/stats') {
    return { overview: {}, ingestion_sources: [] };
  }
  if (path.startsWith('/api/timeline')) return { timeline: [] };
  if (path.startsWith('/api/memories')) return { memories: [] };
  return {
    mode: 'a',
    mode_name: 'Local',
    provider: 'none',
    profile: 'default',
    version: '3.8.1',
  };
}

describe('dashboard read cache', function () {
  it('does not query the daemon again during ordinary tab navigation', async function () {
    const dom = new JSDOM(`<!doctype html><html><body>
      <button id="dashboard-tab"></button>
      <div id="dashboard-pane"></div>
      <div id="src"></div>
      <div id="feed"></div>
      <div id="sp-big"></div>
    </body></html>`, {
      runScripts: 'dangerously',
      url: 'http://localhost:8765/',
    });
    const { window } = dom;
    const calls = [];
    window.clearPaneError = function () {};
    window.showPaneError = function () {};
    window.paneErrorMessage = function () { return 'unavailable'; };
    window.fetch = function (path) {
      calls.push(path);
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.resolve(responseFor(path)); },
      });
    };
    const script = window.document.createElement('script');
    script.textContent = dashboardSource;
    window.document.head.appendChild(script);

    await window.refreshDashboard({ force: true });
    const initialCalls = calls.length;
    await window.refreshDashboard();

    assert.equal(initialCalls, 4);
    assert.equal(calls.length, initialCalls);

    await window.refreshDashboard({ force: true });
    assert.equal(calls.length, initialCalls * 2);
    dom.window.close();
  });

  it('re-queries after a successful mutation invalidates the dashboard snapshot', async function () {
    const dom = new JSDOM(`<!doctype html><html><body>
      <button id="dashboard-tab"></button>
      <div id="dashboard-pane"></div>
      <div id="src"></div>
      <div id="feed"></div>
      <div id="sp-big"></div>
    </body></html>`, {
      runScripts: 'dangerously',
      url: 'http://localhost:8765/',
    });
    const { window } = dom;
    const calls = [];
    window.matchMedia = function () {
      return {
        matches: false,
        addEventListener: function () {},
        removeEventListener: function () {},
      };
    };
    window.fetch = function (path) {
      calls.push(path);
      const payload = path === '/internal/token'
        ? { token: 'test-install-token' }
        : responseFor(path);
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.resolve(payload); },
      });
    };
    let paneInvalidations = 0;
    window.slmInvalidatePanes = function () { paneInvalidations += 1; };
    const originalAddEventListener = window.addEventListener.bind(window);
    window.addEventListener = function (type, listener, options) {
      if (type !== 'DOMContentLoaded') {
        originalAddEventListener(type, listener, options);
      }
    };
    const coreScript = window.document.createElement('script');
    coreScript.textContent = coreSource;
    window.document.head.appendChild(coreScript);
    window.addEventListener = originalAddEventListener;
    const dashboardScript = window.document.createElement('script');
    dashboardScript.textContent = dashboardSource;
    window.document.head.appendChild(dashboardScript);

    await window.refreshDashboard({ force: true });
    const readsBeforeMutation = calls.filter((path) => path !== '/internal/token').length;
    await window.fetch('/api/behavioral/report-outcome', { method: 'POST' });
    await window.refreshDashboard();

    const readsAfterMutation = calls.filter(
      (path) => path !== '/internal/token' && path !== '/api/behavioral/report-outcome',
    ).length;
    assert.equal(paneInvalidations, 1);
    assert.equal(readsAfterMutation, readsBeforeMutation + 4);
    dom.window.close();
  });

  it('queues a trailing refresh when invalidated during an in-flight read', async function () {
    const dom = new JSDOM(`<!doctype html><html><body>
      <button id="dashboard-tab"></button>
      <div id="dashboard-pane"></div>
      <div id="src"></div>
      <div id="feed"></div>
      <div id="sp-big"></div>
    </body></html>`, {
      runScripts: 'dangerously',
      url: 'http://localhost:8765/',
    });
    const { window } = dom;
    const calls = [];
    const firstWave = [];
    window.clearPaneError = function () {};
    window.showPaneError = function () {};
    window.paneErrorMessage = function () { return 'unavailable'; };
    window.fetch = function (path) {
      calls.push(path);
      if (calls.length <= 4) {
        return new Promise(function (resolve) {
          firstWave.push(function () {
            resolve({
              ok: true,
              json: function () { return Promise.resolve(responseFor(path)); },
            });
          });
        });
      }
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.resolve(responseFor(path)); },
      });
    };
    const script = window.document.createElement('script');
    script.textContent = dashboardSource;
    window.document.head.appendChild(script);

    const refresh = window.refreshDashboard({ force: true });
    window.slmInvalidateDashboardCache();
    firstWave.forEach(function (release) { release(); });
    await refresh;

    assert.equal(calls.length, 8);
    dom.window.close();
  });

  it('does not cache a failed refresh, so the next navigation retries immediately', async function () {
    const dom = new JSDOM(`<!doctype html><html><body>
      <button id="dashboard-tab"></button>
      <div id="dashboard-pane"></div>
      <div id="src"></div>
      <div id="feed"></div>
      <div id="sp-big"></div>
    </body></html>`, {
      runScripts: 'dangerously',
      url: 'http://localhost:8765/',
    });
    const { window } = dom;
    const calls = [];
    let unavailable = true;
    window.clearPaneError = function () {};
    window.showPaneError = function () {};
    window.paneErrorMessage = function () { return 'unavailable'; };
    window.fetch = function (path) {
      calls.push(path);
      return Promise.resolve({
        ok: !unavailable,
        status: unavailable ? 503 : 200,
        json: function () { return Promise.resolve(responseFor(path)); },
      });
    };
    const script = window.document.createElement('script');
    script.textContent = dashboardSource;
    window.document.head.appendChild(script);

    await window.refreshDashboard({ force: true });
    unavailable = false;
    await window.refreshDashboard();

    assert.equal(calls.length, 8);
    dom.window.close();
  });
});
