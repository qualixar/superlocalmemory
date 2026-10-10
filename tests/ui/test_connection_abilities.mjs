/**
 * tests/ui/test_connection_abilities.mjs — the second yes as a dashboard click (4.1.25, package D).
 *
 * The Web access row shows two switches: let its apps message your other bots (mesh),
 * and let them save and read pictures and documents (media). Before this, the only way
 * was `slm remote keys allow web-<id> mesh|media` in a terminal.
 *
 *   GET  /api/v3/connections/{id}/abilities            -> {connection_id, mesh, media}
 *   POST /api/v3/connections/{id}/abilities {profile_id, ability, allow}
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
const SCRIPTS = ['od-apps-ui.js', 'od-apps-list.js', 'od-connections.js', 'od-apps.js'];

const settle = async (window, ticks = 12) => {
  for (let i = 0; i < ticks; i += 1) await new Promise((r) => window.setTimeout(r, 0));
};

function status(state) {
  return {
    available: true, installation_id: 'inst-1', current_profile: 'default',
    hosts: ['chatgpt', 'claude_web', 'composio', 'muse', 'other_mcp'],
    connections: [{
      connection_id: CID, host: 'chatgpt', state, version: 4, verified: true,
      mcp_url: 'https://mcp.superlocalmemory.com/mcp', intent_key: 'b'.repeat(32), cleanup_pending: false,
    }],
  };
}

async function boot({ state = 'ready_for_client', abilities = { mesh: false, media: true }, failPost = false, failGet = false } = {}) {
  const calls = [];
  let current = { connection_id: CID, ...abilities };
  const fetchImpl = (path, init) => {
    const method = (init && init.method) || 'GET';
    const body = init && init.body ? JSON.parse(init.body) : null;
    calls.push({ path, method, body });
    const reply = (code, json) => Promise.resolve({ ok: code < 300, status: code, json: () => Promise.resolve(json) });
    if (path.endsWith('/api/v3/connections/status')) return reply(200, status(state));
    if (path.endsWith(`/api/v3/connections/${CID}/apps`)) return reply(200, { connection_id: CID, apps: [] });
    if (path.endsWith(`/api/v3/connections/${CID}/abilities`) && method === 'GET') {
      return failGet ? reply(503, { detail: 'apps_unavailable' }) : reply(200, current);
    }
    if (path.endsWith(`/api/v3/connections/${CID}/abilities`) && method === 'POST') {
      if (failPost) return reply(503, { detail: 'apps_unavailable' });
      current = { ...current, [body.ability]: body.allow };
      return reply(200, current);
    }
    return reply(200, {});
  };
  const dom = new JSDOM('<!doctype html><html><body><div id="apps-pane"></div></body></html>', {
    runScripts: 'dangerously', url: 'http://localhost:8765/',
  });
  const { window } = dom;
  window.fetch = fetchImpl;
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  for (const name of SCRIPTS) {
    const el = window.document.createElement('script');
    el.textContent = source(name);
    window.document.head.appendChild(el);
  }
  await window.odRenderApps(window.document.getElementById('apps-pane'));
  await settle(window);
  const box = () => window.document.querySelector('[data-abilities]');
  const input = (name) => window.document.querySelector(`[data-ability="${name}"]`);
  return { dom, window, calls, box, input };
}

describe('The second yes on the Web access row', () => {
  it('shows both switches with their current state, in plain words', async () => {
    const { dom, box, input } = await boot();
    assert.ok(box() && !box().hidden);
    assert.equal(input('mesh').checked, false);
    assert.equal(input('media').checked, true);
    assert.match(box().textContent, /message your other bots/);
    assert.match(box().textContent, /pictures and documents/);
    assert.match(box().textContent, /also be approved/i, 'says the per-app approval is still needed');
    dom.window.close();
  });

  it('a click sends the change for the current profile and keeps the answer', async () => {
    const { dom, window, calls, input } = await boot();
    input('mesh').click();
    await settle(window);
    const post = calls.find((c) => c.method === 'POST' && c.path.endsWith('/abilities'));
    assert.deepEqual(post.body, { profile_id: 'default', ability: 'mesh', allow: true });
    assert.equal(input('mesh').checked, true);
    dom.window.close();
  });

  it('a refused change puts the switch back and says so', async () => {
    const { dom, window, input, box } = await boot({ failPost: true });
    input('media').click();
    await settle(window);
    assert.equal(input('media').checked, true, 'restored to what the computer says');
    assert.match(box().textContent, /could not be changed/i);
    dom.window.close();
  });

  it('stays hidden when the state cannot be read, and for a connection that is not on', async () => {
    const a = await boot({ failGet: true });
    assert.ok(!a.box() || a.box().hidden);
    a.dom.window.close();
    const b = await boot({ state: 'pending' });
    assert.equal(b.box(), null);
    b.dom.window.close();
  });
});
