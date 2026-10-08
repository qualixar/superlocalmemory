/**
 * tests/ui/test_connected_apps.mjs — the Connected apps pane (4.1.23).
 *
 * Runner: node scripts/run-ui-tests.mjs   (needs jsdom: see the symlink note in that script's README)
 *
 * Covers the contract the founder asked for:
 *   - "Your connected apps" renders one row per app, and a hostile third-party
 *     app name is shown as text, never as markup;
 *   - empty state, remove-access success (row disappears + toast), 409 refresh,
 *     503 message, cancel sends nothing;
 *   - stale answers are fenced out;
 *   - the pane separation: MCP & Tools no longer contains the internet connect
 *     flow (only a pointer), Connected apps does;
 *   - the sign-in flow keeps its guards (consent required, saving not
 *     pre-checked, controls blocked until status arrives, intent persisted);
 *   - the shell lists the new pane and routes a sign-in return to it.
 *
 * The backend endpoints are mocked here; the contract is:
 *   GET  /api/v3/connections/{id}/apps
 *   POST /api/v3/connections/{id}/apps/{authorization_id}/revoke {profile_id, expected_version}
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI = join(__dirname, '../../src/superlocalmemory/ui');
const source = (name) => readFileSync(join(UI, 'js', name), 'utf8');

const CID = 'a'.repeat(32);
const HOSTILE = '<img src=x onerror=alert(1)>';
const DAY = 86400e3;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
function makeApi(handlers) {
  const calls = [];
  function fetchImpl(path, init) {
    const method = (init && init.method) || 'GET';
    calls.push({ path, method, init: init || {}, body: init && init.body ? JSON.parse(init.body) : null });
    for (const [m, pattern, handler] of handlers) {
      const hit = method === m && pattern.exec(path);
      if (hit) {
        return Promise.resolve(handler(hit, init)).then((out) => ({
          ok: out.status >= 200 && out.status < 300,
          status: out.status,
          json: () => Promise.resolve(out.body),
        }));
      }
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) });
  }
  return { fetchImpl, calls };
}

const settle = async (window, ticks = 8) => {
  for (let i = 0; i < ticks; i += 1) await new Promise((r) => window.setTimeout(r, 0));
};

function app(overrides) {
  return Object.assign({
    authorization_id: 'auth-1', name: 'ChatGPT', client_host: 'chatgpt.com',
    permissions: { read: true, save: false, session: false }, version: 3,
    connected_at_ms: Date.now() - 3 * DAY, last_used_at_ms: Date.now() - 2 * 3600e3,
  }, overrides || {});
}

function statusBody(extra) {
  return Object.assign({
    available: true, installation_id: 'inst-1', current_profile: 'default',
    hosts: ['muse', 'chatgpt', 'claude_web', 'claude_code_web', 'composio', 'other_mcp'],
    connections: [{
      connection_id: CID, host: 'composio', state: 'ready_for_client', version: 4, verified: true,
      mcp_url: 'https://mcp.superlocalmemory.com/mcp', intent_key: 'b'.repeat(32), cleanup_pending: false,
    }],
  }, extra || {});
}

async function boot({ html, url, fetchImpl, scripts, beforeScripts }) {
  const dom = new JSDOM(html || '<!doctype html><html><body><div id="mcp-pane"></div><div id="apps-pane"></div></body></html>', {
    runScripts: 'dangerously', url: url || 'http://localhost:8765/',
  });
  const { window } = dom;
  await new Promise((resolve) => (window.document.readyState === 'complete' ? resolve() : window.addEventListener('load', resolve)));
  window.fetch = fetchImpl;
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  if (beforeScripts) beforeScripts(window);
  for (const name of scripts) {
    const el = window.document.createElement('script');
    el.textContent = source(name);
    window.document.head.appendChild(el);
  }
  return dom;
}

const LIST_SCRIPTS = ['od-apps-ui.js', 'od-apps-list.js'];
const PANE_SCRIPTS = ['od-apps-ui.js', 'od-apps-list.js', 'od-connections.js', 'od-apps.js', 'od-mcp.js'];

async function mountList(appsBody, extra) {
  const api = makeApi([
    ['GET', new RegExp(`/api/v3/connections/${CID}/apps$`), () => ({ status: 200, body: { connection_id: CID, apps: appsBody } })],
  ].concat((extra && extra.handlers) || []));
  const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
  const { window } = dom;
  const list = window.odCreateConnectedAppsList();
  window.document.body.appendChild(list);
  return { dom, window, document: window.document, list, api };
}

const rowNames = (document) => Array.from(document.querySelectorAll('[data-apps-list] .apps-row .apps-name')).map((n) => n.textContent);

// ---------------------------------------------------------------------------
// Your connected apps
// ---------------------------------------------------------------------------
describe('Your connected apps: rendering', () => {
  it('renders one row per app with host, permission chips and relative times', async () => {
    const { dom, window, document, list } = await mountList([
      app(),
      app({ authorization_id: 'auth-2', name: 'Composio', client_host: 'backend.composio.dev',
        permissions: { read: true, save: true, session: true }, last_used_at_ms: null, connected_at_ms: Date.now() - 9 * DAY }),
    ]);
    await list.odSetConnections([CID], 'default');
    await settle(window);

    const rows = document.querySelectorAll('[data-apps-list] .apps-row');
    assert.equal(rows.length, 2);
    const [first, second] = Array.from(rows);
    assert.match(first.textContent, /ChatGPT/);
    assert.match(first.textContent, /chatgpt\.com/);
    assert.match(first.textContent, /Connected 3 days ago/);
    assert.match(first.textContent, /Last used 2 hours ago/);
    const chips = (row) => Array.from(row.querySelectorAll('.apps-chip')).map((c) => c.textContent);
    assert.deepEqual(chips(first), ['Read']);
    assert.deepEqual(chips(second), ['Read', 'Save', 'Session tools']);
    assert.match(second.textContent, /Not used yet/);
    assert.match(second.textContent, /Connected 1 week ago/);
    assert.equal(first.querySelector('[data-remove-access]').getAttribute('aria-label'), 'Remove access for ChatGPT');
    assert.equal(document.querySelector('.apps-empty').hidden, true);
    dom.window.close();
  });

  it('omits null dates gracefully and never prints Invalid Date', async () => {
    const { dom, window, document, list } = await mountList([app({ connected_at_ms: null, last_used_at_ms: null, client_host: null })]);
    await list.odSetConnections([CID], 'default');
    await settle(window);
    const row = document.querySelector('.apps-row');
    assert.ok(!/Connected/.test(row.textContent));
    assert.match(row.textContent, /Not used yet/);
    assert.ok(!/Invalid Date|NaN|null|undefined/.test(row.textContent));
    assert.equal(row.querySelector('.apps-host'), null);
    dom.window.close();
  });

  it('shows a hostile app name and host as plain text, never as markup', async () => {
    const hostile = app({ name: HOSTILE, client_host: '<script>alert(2)</script>.example' });
    const { dom, window, document, list } = await mountList([hostile]);
    let alerted = false;
    window.alert = () => { alerted = true; };
    await list.odSetConnections([CID], 'default');
    await settle(window);

    const row = document.querySelector('.apps-row');
    assert.equal(row.querySelector('.apps-name').textContent, HOSTILE);
    assert.equal(row.querySelector('.apps-host').textContent, '<script>alert(2)</script>.example');
    assert.equal(document.querySelectorAll('img, script[src], [onerror]').length, 0);
    assert.equal(row.querySelector('.apps-avatar').textContent.length <= 2, true);
    assert.equal(alerted, false);
    assert.equal(row.querySelector('[data-remove-access]').getAttribute('aria-label'), `Remove access for ${HOSTILE}`);
    dom.window.close();
  });

  it('shows the friendly empty state, and makes no request when nothing is connected', async () => {
    const { dom, window, document, list, api } = await mountList([]);
    await list.odSetConnections([], 'default');
    await settle(window);
    const empty = document.querySelector('.apps-empty');
    assert.equal(empty.hidden, false);
    assert.match(empty.textContent, /No apps connected yet/);
    assert.equal(api.calls.length, 0);

    await list.odSetConnections([CID], 'default');
    await settle(window);
    assert.equal(api.calls.length, 1);
    assert.equal(document.querySelector('.apps-empty').hidden, false);
    dom.window.close();
  });

  it('tells the user when the list cannot load (503) instead of claiming it is empty', async () => {
    const api = makeApi([['GET', /\/apps$/, () => ({ status: 503, body: { detail: 'apps_unavailable' } })]]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
    const list = dom.window.odCreateConnectedAppsList();
    dom.window.document.body.appendChild(list);
    await list.odSetConnections([CID], 'default');
    await settle(dom.window);
    const notice = dom.window.document.querySelector('.apps-notice');
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /couldn’t load your connected apps/i);
    assert.equal(dom.window.document.querySelector('.apps-empty').hidden, true);
    dom.window.close();
  });

  it('drops an answer that arrives after a fence (stale-data fencing)', async () => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const api = makeApi([['GET', /\/apps$/, () => gate.then(() => ({ status: 200, body: { apps: [app()] } }))]]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
    const { window } = dom;
    const list = window.odCreateConnectedAppsList();
    window.document.body.appendChild(list);
    const pending = list.odSetConnections([CID], 'default');
    list.odFence();
    release();
    await pending; await settle(window);
    assert.deepEqual(rowNames(window.document), []);
    dom.window.close();
  });
});

// ---------------------------------------------------------------------------
// Remove access
// ---------------------------------------------------------------------------
describe('Remove access', () => {
  const revokeRoute = (status, body) => ['POST', /\/apps\/auth-1\/revoke$/, () => ({ status, body })];

  it('asks first with an accessible dialog and sends nothing on Cancel', async () => {
    const { dom, window, document, list, api } = await mountList([app()], { handlers: [revokeRoute(200, { revoked: true })] });
    await list.odSetConnections([CID], 'default'); await settle(window);
    document.querySelector('[data-remove-access]').click();

    const dialog = document.querySelector('[role="dialog"]');
    assert.ok(dialog);
    assert.equal(dialog.getAttribute('aria-modal'), 'true');
    assert.equal(document.getElementById(dialog.getAttribute('aria-labelledby')).textContent, 'Remove access for ChatGPT?');
    assert.equal(document.getElementById(dialog.getAttribute('aria-describedby')).textContent,
      'ChatGPT will no longer be able to read or save memories. You can connect it again any time.');
    const buttons = Array.from(dialog.querySelectorAll('button')).map((b) => b.textContent);
    assert.deepEqual(buttons, ['Cancel', 'Remove access']);
    assert.equal(document.activeElement.textContent, 'Cancel', 'a destructive prompt starts on Cancel');

    dialog.querySelector('button').click();
    assert.equal(document.querySelector('[role="dialog"]'), null);
    assert.equal(api.calls.filter((c) => c.method === 'POST').length, 0);
    assert.equal(rowNames(document).length, 1);
    dom.window.close();
  });

  it('closes on Escape and returns focus to the button that opened it', async () => {
    const { dom, window, document, list } = await mountList([app()]);
    await list.odSetConnections([CID], 'default'); await settle(window);
    const opener = document.querySelector('[data-remove-access]');
    opener.focus(); opener.click();
    document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    assert.equal(document.querySelector('[role="dialog"]'), null);
    assert.equal(document.activeElement, opener);
    dom.window.close();
  });

  it('traps Tab inside the dialog', async () => {
    const { dom, window, document, list } = await mountList([app()]);
    await list.odSetConnections([CID], 'default'); await settle(window);
    document.querySelector('[data-remove-access]').click();
    const [cancel, confirm] = Array.from(document.querySelectorAll('[role="dialog"] button'));
    confirm.focus();
    const forward = new window.KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.dispatchEvent(forward);
    assert.equal(forward.defaultPrevented, true);
    assert.equal(document.activeElement, cancel, 'Tab from the last button wraps to the first');
    const back = new window.KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true });
    document.dispatchEvent(back);
    assert.equal(document.activeElement, confirm, 'Shift+Tab from the first button wraps to the last');
    dom.window.close();
  });

  it('removes the row, toasts "Access removed" and posts the version it listed (success)', async () => {
    const { dom, window, document, list, api } = await mountList(
      [app(), app({ authorization_id: 'auth-2', name: 'Composio', version: 1, last_used_at_ms: Date.now() - 5 * 3600e3 })],
      { handlers: [revokeRoute(200, { revoked: true })] });
    await list.odSetConnections([CID], 'default'); await settle(window);
    assert.deepEqual(rowNames(document), ['ChatGPT', 'Composio']);

    document.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);

    const post = api.calls.find((c) => c.method === 'POST');
    assert.equal(post.path, `/api/v3/connections/${CID}/apps/auth-1/revoke`);
    assert.deepEqual(post.body, { profile_id: 'default', expected_version: 3 });
    assert.equal(post.init.headers['Content-Type'], 'application/json');
    assert.equal(post.init.credentials, 'same-origin');
    assert.deepEqual(rowNames(document), ['Composio']);
    assert.equal(document.querySelector('[role="dialog"]'), null);
    assert.equal(document.querySelector('#apps-toast-region .apps-toast').textContent, 'Access removed');
    assert.equal(document.getElementById('apps-toast-region').getAttribute('aria-live'), 'polite');
    dom.window.close();
  });

  it('shows the empty state after the last app is removed', async () => {
    const { dom, window, document, list } = await mountList([app()], { handlers: [revokeRoute(200, { revoked: true })] });
    await list.odSetConnections([CID], 'default'); await settle(window);
    document.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);
    assert.equal(document.querySelector('.apps-empty').hidden, false);
    dom.window.close();
  });

  it('on 409 says the list changed and reloads it', async () => {
    let listCalls = 0;
    const api = makeApi([
      ['GET', /\/apps$/, () => { listCalls += 1; return { status: 200, body: { apps: listCalls === 1 ? [app()] : [app({ version: 4 })] } }; }],
      revokeRoute(409, { detail: 'version_conflict' }),
    ]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
    const { window } = dom; const { document } = window;
    const list = window.odCreateConnectedAppsList(); document.body.appendChild(list);
    await list.odSetConnections([CID], 'default'); await settle(window);

    document.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);

    assert.equal(listCalls, 2, 'the list is reloaded after a 409');
    assert.equal(document.querySelector('[role="dialog"]'), null);
    assert.equal(rowNames(document).length, 1, 'the row is still there: the server did not revoke it');
    assert.equal(document.querySelector('#apps-toast-region'), null, 'no success toast on a conflict');
    dom.window.close();
  });

  it('on 409 shows "This list changed. Refreshing…" while the reload is in flight', async () => {
    let releaseReload;
    const reload = new Promise((resolve) => { releaseReload = resolve; });
    let listCalls = 0;
    const api = makeApi([
      ['GET', /\/apps$/, () => { listCalls += 1; return listCalls === 1 ? { status: 200, body: { apps: [app()] } } : reload.then(() => ({ status: 200, body: { apps: [] } })); }],
      revokeRoute(409, { detail: 'profile_changed' }),
    ]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
    const { window } = dom; const { document } = window;
    const list = window.odCreateConnectedAppsList(); document.body.appendChild(list);
    await list.odSetConnections([CID], 'default'); await settle(window);
    document.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);
    const notice = document.querySelector('.apps-notice');
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /This list changed\. Refreshing…/);
    releaseReload(); await settle(window);
    assert.equal(document.querySelector('.apps-notice').hidden, true);
    assert.equal(rowNames(document).length, 0);
    dom.window.close();
  });

  it('on 503 keeps the dialog open with a clear message and leaves the row in place', async () => {
    const { dom, window, document, list } = await mountList([app()], { handlers: [revokeRoute(503, { detail: 'apps_unavailable' })] });
    await list.odSetConnections([CID], 'default'); await settle(window);
    document.querySelector('[data-remove-access]').click();
    const confirm = Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access');
    confirm.click(); await settle(window);

    const error = document.querySelector('[role="dialog"] [role="alert"]');
    assert.equal(error.hidden, false);
    assert.equal(error.textContent, 'Couldn’t reach the connection service. Try again.');
    assert.equal(confirm.disabled, false, 'the user can try again');
    assert.equal(rowNames(document).length, 1);
    dom.window.close();
  });

  it('survives a network failure without losing the row', async () => {
    const api = makeApi([['GET', /\/apps$/, () => ({ status: 200, body: { apps: [app()] } })]]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: LIST_SCRIPTS });
    const { window } = dom; const { document } = window;
    const list = window.odCreateConnectedAppsList(); document.body.appendChild(list);
    await list.odSetConnections([CID], 'default'); await settle(window);
    window.fetch = () => Promise.reject(new Error('offline'));
    document.querySelector('[data-remove-access]').click();
    Array.from(document.querySelectorAll('[role="dialog"] button')).find((b) => b.textContent === 'Remove access').click();
    await settle(window);
    assert.match(document.querySelector('[role="dialog"] [role="alert"]').textContent, /Couldn’t reach the connection service/);
    assert.equal(rowNames(document).length, 1);
    dom.window.close();
  });

  it('blocks removal while the pane is being refreshed for a profile change', async () => {
    const { dom, window, document, list } = await mountList([app()]);
    await list.odSetConnections([CID], 'default'); await settle(window);
    list.odFence();
    assert.equal(document.querySelector('[data-remove-access]').disabled, true);
    await list.odSetConnections([CID], 'default'); await settle(window);
    assert.equal(document.querySelector('[data-remove-access]').disabled, false);
    dom.window.close();
  });
});

// ---------------------------------------------------------------------------
// Pane separation + the connection flow
// ---------------------------------------------------------------------------
function paneApi(extra) {
  return makeApi([
    ['GET', /\/api\/v3\/connections\/status$/, () => ({ status: 200, body: statusBody(extra && extra.status) })],
    ['GET', /\/api\/v3\/mcp\/profiles$/, () => ({ status: 200, body: { current: 'core', total_tools: 18, profiles: { core: { count: 18, tools: ['recall'], description: 'Minimal' } } } })],
    ['GET', new RegExp(`/api/v3/connections/${CID}/apps$`), () => ({ status: 200, body: { apps: (extra && extra.apps) || [app()] } })],
  ].concat((extra && extra.handlers) || []));
}

async function bootPanes(extra) {
  const api = paneApi(extra);
  const dom = await boot({ fetchImpl: api.fetchImpl, scripts: PANE_SCRIPTS });
  const { window } = dom; const { document } = window;
  const mcp = document.getElementById('mcp-pane'); const apps = document.getElementById('apps-pane');
  await window.odRenderMcp(mcp);
  await window.odRenderApps(apps);
  await settle(window, 12);
  return { dom, window, document, mcp, apps, api };
}

describe('Pane separation', () => {
  it('MCP & Tools no longer contains the connect flow, only a pointer to Connected apps', async () => {
    const { dom, mcp } = await bootPanes();
    assert.equal(mcp.querySelector('#od-ai-connections'), null);
    assert.equal(mcp.querySelector('form'), null);
    for (const gone of ['Connect your AI', 'Add an app', 'Past connections', 'OAuth metadata URL', 'Musebot', 'Link this computer']) {
      assert.ok(!mcp.textContent.includes(gone), `MCP & Tools must not say "${gone}"`);
    }
    assert.match(mcp.textContent, /Want ChatGPT, Claude on the web, Muse or bots to use your memory\?/);
    const link = mcp.querySelector('[data-apps-pointer] a');
    assert.equal(link.getAttribute('href'), '#apps-pane');
    assert.match(mcp.textContent, /Active MCP profile/, 'local-agent profile content is intact');
    dom.window.close();
  });

  it('Connected apps carries the whole flow, in order, with the renamed apps', async () => {
    const { dom, apps } = await bootPanes();
    assert.ok(apps.querySelector('#od-ai-connections'));
    assert.equal(apps.querySelector('h2').textContent, 'Connected apps');
    assert.match(apps.textContent, /Free during beta/);
    for (const trust of ['Your memory stays on this computer', 'You approve every app', 'Read-only unless you allow saving']) {
      assert.ok(apps.textContent.includes(trust), trust);
    }
    assert.match(apps.textContent, /Results an app reads are shared with that app\. This computer must be on and online for apps to reach your memory\./);

    const titles = Array.from(apps.querySelectorAll('.apps-section-title')).map((n) => n.textContent);
    assert.deepEqual(titles, ['This computer', 'Your connected apps', 'Add an app']);
    assert.ok(apps.querySelector('details.apps-tech summary').textContent.includes('Technical details'));

    const names = Array.from(apps.querySelectorAll('.apps-app-name')).map((n) => n.textContent);
    assert.deepEqual(names, ['ChatGPT', 'Claude (web)', 'Claude Code (web)', 'Composio', 'Muse', 'Other app (MCP)']);
    assert.ok(!apps.textContent.includes('Musebot'));
    assert.ok(!apps.textContent.includes('Other MCP client'));
    dom.window.close();
  });

  it('puts the MCP server URL, OAuth metadata URL and Past connections inside Technical details', async () => {
    const { dom, apps } = await bootPanes({ status: { connections: [
      { connection_id: CID, host: 'composio', state: 'ready_for_client', version: 4, verified: true, mcp_url: 'https://mcp.superlocalmemory.com/mcp', intent_key: 'b'.repeat(32), cleanup_pending: false },
      { connection_id: 'c'.repeat(32), host: 'muse', state: 'cancelled', version: 5, verified: false, cleanup_pending: false, intent_key: 'd'.repeat(32) },
    ] } });
    const tech = apps.querySelector('details.apps-tech');
    assert.equal(tech.open, false, 'collapsed by default');
    const values = Array.from(tech.querySelectorAll('input[readonly]')).map((i) => i.value);
    assert.deepEqual(values, ['https://mcp.superlocalmemory.com/mcp', 'https://auth.superlocalmemory.com/.well-known/oauth-authorization-server']);
    assert.equal(Array.from(tech.querySelectorAll('button')).filter((b) => /Copy/.test(b.textContent)).length, 2);
    const history = tech.querySelector('details.apps-history-details');
    assert.ok(history, 'cancelled history lives in Technical details');
    assert.equal(history.open, false, 'cancelled history stays collapsed');
    assert.match(history.querySelector('summary').textContent, /Past connections \(1\)/);
    assert.equal(apps.querySelectorAll('.apps-conn-row').length, 2, 'one live row on the computer card + one history row');
    assert.equal(apps.querySelector('.apps-conn-rows').querySelectorAll(':scope > .apps-conn-row').length, 1);
    dom.window.close();
  });

  it('lists the apps of the live connection and shows the computer as Ready', async () => {
    const { dom, apps } = await bootPanes();
    assert.equal(apps.querySelector('[data-computer-status]').textContent, 'Ready');
    assert.match(apps.textContent, /GitHub sign-in\s*Linked/);
    assert.deepEqual(rowNames(apps.ownerDocument), ['ChatGPT']);
    dom.window.close();
  });

  it('names the computer link "Web access", not the app chosen at setup', async () => {
    const { dom, apps } = await bootPanes();
    const live = apps.querySelector('.apps-conn-rows').querySelector(':scope > .apps-conn-row');
    assert.equal(live.querySelector('.apps-name').textContent, 'Web access');
    assert.match(live.textContent, /Apps you approve can reach your memory/);
    assert.doesNotMatch(live.textContent, /Add it to your app to finish/);
    dom.window.close();
  });

  it('asks before turning off web access and sends nothing until confirmed', async () => {
    const cancelled = { connection_id: CID, state: 'cancelled', verified: false, cleanup_pending: false };
    const { dom, window, apps, api } = await bootPanes({ handlers: [['POST', new RegExp(`/api/v3/connections/${CID}/cancel$`), () => ({ status: 200, body: cancelled })]] });
    const doc = window.document;
    const turnOff = apps.querySelector('button[aria-label="Turn off web access for this computer"]');
    assert.ok(turnOff, 'Turn off button present for a live link');
    const posts = () => api.calls.filter((c) => c.method === 'POST' && /\/cancel$/.test(c.path));
    turnOff.click(); await settle(window);
    let dialog = doc.querySelector('[role="dialog"]');
    assert.ok(dialog, 'confirmation dialog opens');
    assert.match(dialog.textContent, /Turn off web access\?/);
    assert.match(dialog.textContent, /Every connected app will lose access/);
    assert.equal(posts().length, 0, 'nothing sent before confirming');
    [...dialog.querySelectorAll('button')].find((b) => b.textContent === 'Keep it on').click(); await settle(window);
    assert.equal(doc.querySelector('[role="dialog"]'), null);
    assert.equal(posts().length, 0, 'Keep it on sends nothing');
    turnOff.click(); await settle(window);
    dialog = doc.querySelector('[role="dialog"]');
    [...dialog.querySelectorAll('button')].find((b) => b.textContent === 'Turn off web access').click(); await settle(window, 12);
    assert.equal(posts().length, 1);
    assert.deepEqual(posts()[0].body, { profile_id: 'default', expected_version: 4 });
    dom.window.close();
  });

  it('keeps the mounted pane (form, focus) when the shell re-renders it', async () => {
    const { dom, window, apps } = await bootPanes();
    const root = apps.querySelector('#od-apps-root');
    const write = apps.querySelector('[data-permission="write"]');
    write.checked = true;
    await window.odRenderApps(apps, { preserveConnectionScope: true });
    assert.equal(apps.querySelector('#od-apps-root'), root);
    assert.equal(apps.querySelector('[data-permission="write"]'), write);
    assert.equal(write.checked, true);
    dom.window.close();
  });
});

describe('Setting up an app (guards moved from the old card)', () => {
  const initiateRoute = (calls) => ['POST', /\/api\/v3\/connections\/initiate$/, (_hit, init) => {
    calls.push(init.headers);
    return { status: 200, body: { state: 'pending', connection_id: 'e'.repeat(32), authorization_url: `https://auth.superlocalmemory.com/owner-login?connection_id=${'e'.repeat(32)}` } };
  }];
  const none = { status: { connections: [] } };

  it('blocks every control until the status call returns', async () => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const api = makeApi([['GET', /\/status$/, () => gate.then(() => ({ status: 200, body: statusBody({ connections: [] }) }))]]);
    const dom = await boot({ fetchImpl: api.fetchImpl, scripts: PANE_SCRIPTS });
    const { window } = dom; const { document } = window;
    const pane = document.getElementById('apps-pane');
    await window.odRenderApps(pane);
    assert.ok(Array.from(pane.querySelectorAll('[data-client]')).every((b) => b.disabled), 'Set up buttons are disabled');
    assert.equal(pane.querySelector('form button[type="submit"]').disabled, true);
    assert.equal(pane.querySelector('[data-remote-opt-in]').disabled, true);
    release(); await settle(window, 12);
    assert.ok(Array.from(pane.querySelectorAll('[data-client]')).every((b) => !b.disabled));
    dom.window.close();
  });

  it('opens the four-step panel from a card; saving and session tools are not pre-checked', async () => {
    const { dom, document, apps } = await bootPanes(none);
    assert.equal(apps.querySelector('.apps-flow').hidden, true);
    apps.querySelector('[data-client="chatgpt"]').click();
    const flow = apps.querySelector('.apps-flow');
    assert.equal(flow.hidden, false);
    assert.equal(document.getElementById('apps-flow-title').textContent, 'Set up ChatGPT');
    assert.deepEqual(Array.from(flow.querySelectorAll('.apps-step-label')).map((n) => n.textContent),
      ['Choose app', 'Link this computer', 'Check connection', 'Add to your app']);
    assert.equal(apps.querySelector('[data-permission="write"]').checked, false);
    assert.equal(apps.querySelector('[data-permission="session"]').checked, false);
    assert.equal(apps.querySelector('[data-remote-opt-in]').checked, false);
    assert.match(flow.textContent, /Allow saving memories \(recommended\)/);
    assert.match(flow.textContent, /Allow session tools/);
    dom.window.close();
  });

  it('requires consent before anything is sent', async () => {
    const calls = [];
    const { dom, window, apps, api } = await bootPanes({ status: { connections: [] }, handlers: [initiateRoute(calls)] });
    apps.querySelector('[data-client="muse"]').click();
    apps.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(window);
    assert.equal(api.calls.filter((c) => c.method === 'POST').length, 0);
    assert.match(apps.querySelector('.apps-status').textContent, /turn on internet access/i);
    dom.window.close();
  });

  it('sends the chosen permissions with an idempotency key and persists the intent first', async () => {
    const heads = [];
    const { dom, window, apps, api } = await bootPanes({ status: { connections: [] }, handlers: [initiateRoute(heads)] });
    window.open = () => null;
    apps.querySelector('[data-client="composio"]').click();
    apps.querySelector('[data-remote-opt-in]').checked = true;
    apps.querySelector('[data-permission="write"]').checked = true;
    apps.querySelector('[data-permission="session"]').checked = true;
    apps.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(window);

    const post = api.calls.find((c) => c.method === 'POST');
    assert.deepEqual(post.body, {
      host: 'composio', profile_id: 'default', remote_opt_in: true,
      permissions: { read: true, write: true, correction: false, session: true },
    });
    assert.match(post.init.headers['Idempotency-Key'], /^[a-f0-9]{32}$/);
    const stored = window.sessionStorage.getItem('slm-ai-intent-v1:inst-1:default');
    assert.ok(stored, 'the request is remembered so a lost response can be retried');
    assert.ok(!/token|secret|authorization_url/i.test(stored), 'no secrets are stored');
    dom.window.close();
  });

  it('leaves saving off by default in the request', async () => {
    const { dom, window, apps, api } = await bootPanes({ status: { connections: [] }, handlers: [initiateRoute([])] });
    window.open = () => null;
    apps.querySelector('[data-client="chatgpt"]').click();
    apps.querySelector('[data-remote-opt-in]').checked = true;
    apps.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(window);
    const post = api.calls.find((c) => c.method === 'POST');
    assert.equal(post.body.permissions.write, false);
    assert.equal(post.body.permissions.session, false);
    dom.window.close();
  });

  it('shows the add-to-your-app steps and copy buttons for a ready connection', async () => {
    const { dom, apps } = await bootPanes();
    Array.from(apps.querySelectorAll('.apps-conn-row button')).find((b) => b.textContent === 'How to add an app').click();
    const flow = apps.querySelector('.apps-flow');
    assert.equal(flow.hidden, false);
    assert.equal(flow.querySelector('form').hidden, true);
    assert.equal(flow.querySelector('[aria-current="step"] .apps-step-label').textContent, 'Add to your app');
    assert.match(flow.textContent, /Add SuperLocalMemory to Composio/);
    assert.equal(flow.querySelector('input[aria-label="MCP server URL"]').value, 'https://mcp.superlocalmemory.com/mcp');
    assert.equal(flow.querySelector('input[aria-label="OAuth metadata URL"]').value, 'https://auth.superlocalmemory.com/.well-known/oauth-authorization-server');
    dom.window.close();
  });
});

// ---------------------------------------------------------------------------
// Shell: navigation and the sign-in return
// ---------------------------------------------------------------------------
function indexHtmlWithoutAssets() {
  return readFileSync(join(UI, 'index.html'), 'utf8')
    .replace(/<script\b[^>]*><\/script>/g, '')
    .replace(/<link\b[^>]*>/g, '');
}

async function bootShell({ url, performanceType, status }) {
  const api = makeApi([
    ['GET', /\/api\/v3\/connections\/status$/, () => ({ status: 200, body: status || statusBody() })],
    ['GET', /\/api\/v3\/mcp\/profiles$/, () => ({ status: 200, body: { current: 'core', profiles: {}, total_tools: 0 } })],
    ['GET', /\/apps$/, () => ({ status: 200, body: { apps: [] } })],
  ]);
  const dom = await boot({
    html: indexHtmlWithoutAssets(), url, fetchImpl: api.fetchImpl, scripts: [],
    beforeScripts: (window) => {
      window.scrollTo = () => {};
      window.HTMLElement.prototype.scrollTo = () => {};
      if (performanceType) {
        Object.defineProperty(window, 'performance', { configurable: true, value: { getEntriesByType: () => [{ type: performanceType }] } });
      }
    },
  });
  const { window } = dom;
  for (const name of ['od-shell.js', ...PANE_SCRIPTS]) {
    const el = window.document.createElement('script');
    el.textContent = source(name);
    window.document.head.appendChild(el);
  }
  window.document.dispatchEvent(new window.Event('DOMContentLoaded'));
  await settle(window, 14);
  return { dom, window, document: window.document, api };
}

const activePane = (document) => Array.from(document.querySelectorAll('.tab-pane.active')).map((p) => p.id);

describe('Shell: navigation', () => {
  it('lists Connected apps under Integrations, right after MCP & Tools, tagged Optional', async () => {
    const { dom, document } = await bootShell({ url: 'http://localhost:8765/#dashboard-pane' });
    const links = Array.from(document.querySelectorAll('.nav-link[data-tab]'));
    const keys = links.map((l) => l.getAttribute('data-tab'));
    assert.equal(keys[keys.indexOf('mcp-pane') + 1], 'apps-pane');
    const link = document.querySelector('.nav-link[data-tab="apps-pane"]');
    assert.match(link.textContent, /Connected apps/);
    assert.equal(link.querySelector('.tag').textContent, 'Optional');
    assert.ok(document.getElementById('apps-pane').classList.contains('tab-pane'), 'index.html ships the pane');
    dom.window.close();
  });

  it('deep-links: /#apps-pane opens Connected apps', async () => {
    const { dom, document } = await bootShell({ url: 'http://localhost:8765/#apps-pane', performanceType: 'reload' });
    assert.deepEqual(activePane(document), ['apps-pane']);
    assert.equal(document.querySelector('.nav-link[data-tab="apps-pane"]').classList.contains('active'), true);
    dom.window.close();
  });

  it('opens Connected apps from the sidebar and from the pointer card on MCP & Tools', async () => {
    const { dom, window, document } = await bootShell({ url: 'http://localhost:8765/#mcp-pane', performanceType: 'reload' });
    assert.deepEqual(activePane(document), ['mcp-pane']);
    assert.equal(document.querySelector('#mcp-pane #od-ai-connections'), null);
    document.querySelector('#mcp-pane [data-apps-pointer] a').click();
    await settle(window);
    assert.deepEqual(activePane(document), ['apps-pane']);
    assert.ok(document.querySelector('#apps-pane #od-ai-connections'));
    assert.equal(window.location.hash, '#apps-pane');
    assert.equal(document.getElementById('topbar-heading').textContent, 'Connected apps');
    dom.window.close();
  });
});

describe('Shell: returning from sign-in (server redirects to /#mcp-pane)', () => {
  it('lands a fresh navigation to /#mcp-pane on Connected apps when a connection exists', async () => {
    const { dom, window, document } = await bootShell({ url: 'http://localhost:8765/#mcp-pane', performanceType: 'navigate' });
    assert.deepEqual(activePane(document), ['apps-pane']);
    assert.equal(window.location.hash, '#apps-pane');
    assert.ok(document.querySelector('#apps-pane #od-ai-connections'));
    dom.window.close();
  });

  it('treats a connection that was already ready on arrival as new and opens its steps', async () => {
    const { dom, document } = await bootShell({ url: 'http://localhost:8765/#mcp-pane', performanceType: 'navigate' });
    assert.equal(document.querySelector('#apps-pane .apps-flow').hidden, false);
    assert.match(document.querySelector('#apps-pane .apps-flow').textContent, /Add SuperLocalMemory to Composio/);
    dom.window.close();
  });

  it('falls back to MCP & Tools when there is no live connection', async () => {
    const { dom, document } = await bootShell({
      url: 'http://localhost:8765/#mcp-pane', performanceType: 'navigate',
      status: statusBody({ connections: [{ connection_id: 'c'.repeat(32), host: 'muse', state: 'cancelled', version: 5, verified: false, cleanup_pending: false }] }),
    });
    assert.deepEqual(activePane(document), ['mcp-pane']);
    dom.window.close();
  });

  it('leaves a reload of MCP & Tools alone', async () => {
    const { dom, document } = await bootShell({ url: 'http://localhost:8765/#mcp-pane', performanceType: 'reload' });
    assert.deepEqual(activePane(document), ['mcp-pane']);
    dom.window.close();
  });

  it('leaves /#mcp-pane alone when the URL carries any other intent', async () => {
    const { dom, document } = await bootShell({ url: 'http://localhost:8765/?tab=tools#mcp-pane', performanceType: 'navigate' });
    assert.deepEqual(activePane(document), ['mcp-pane']);
    dom.window.close();
  });
});

// ---------------------------------------------------------------------------
// Static guarantees: CSP-safe, no markup from data, no secrets
// ---------------------------------------------------------------------------
describe('Connected apps source is CSP-safe', () => {
  const FILES = ['od-apps-ui.js', 'od-apps-list.js', 'od-connections.js', 'od-apps.js'];
  for (const name of FILES) {
    it(`${name} builds the DOM with textContent only`, () => {
      // Comments may describe what is banned; only code is judged.
      const text = source(name).replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
      assert.ok(!/\.innerHTML\s*=|insertAdjacentHTML|outerHTML|document\.write/.test(text), 'no HTML string sinks');
      assert.ok(!/\bon[a-z]+\s*=\s*["']/.test(text), 'no inline event handler attributes');
      assert.ok(!/\beval\s*\(|new Function\(/.test(text), 'no eval');
      assert.ok(!/createElement\(['"]style['"]\)|<style/.test(text), 'no injected <style>');
      assert.ok(!/<script/i.test(text), 'no script injection');
    });
  }

  it('loads nothing from outside the dashboard', () => {
    for (const name of FILES) {
      const urls = source(name).match(/https?:\/\/[^\s'")]+/g) || [];
      for (const url of urls) {
        assert.match(url, /^(https:\/\/(mcp|auth)\.superlocalmemory\.com\b|http:\/\/www\.w3\.org\/2000\/svg$)/, `${name}: unexpected URL ${url}`);
      }
    }
    const css = readFileSync(join(UI, 'css/od-apps.css'), 'utf8');
    assert.ok(!/@import|url\(\s*['"]?https?:/.test(css), 'the stylesheet pulls in nothing external');
  });

  it('index.html wires the new pane, scripts and stylesheet', () => {
    const html = readFileSync(join(UI, 'index.html'), 'utf8');
    assert.match(html, /id="apps-pane"/);
    for (const name of ['od-apps-ui.js', 'od-apps-list.js', 'od-connections.js', 'od-apps.js', 'od-mcp.js']) {
      assert.match(html, new RegExp(`static/js/${name.replace('.', '\\.')}\\?v=[0-9a-f]{8}`), name);
    }
    assert.match(html, /static\/css\/od-apps\.css\?v=[0-9a-f]{8}/);
    // order matters: the toolkit and list load before the controller that uses them
    const at = (name) => html.indexOf(`static/js/${name}`);
    assert.ok(at('od-apps-ui.js') < at('od-apps-list.js') && at('od-apps-list.js') < at('od-connections.js') && at('od-connections.js') < at('od-apps.js'));
  });
});
