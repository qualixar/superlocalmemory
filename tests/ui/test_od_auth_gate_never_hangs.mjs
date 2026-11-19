/**
 * The access gate never leaves the owner on "Checking access..." (19 Nov 2026, found on the owner's
 * Mac): a dashboard tab open while SLM restarts (every upgrade) can be left with a check that never
 * answers. The gate gives each check a deadline and then fails open, as it already does on an error;
 * the daemon still enforces access on every request.
 * Runner: npm test (scripts/run-ui-tests.mjs). Requires jsdom.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function mount(fetchImpl, timeoutMs) {
  const h = buildHarness(['main'], { ok: true, status: 200, json: {} });
  const w = h.window;
  if (timeoutMs !== undefined) w.SLM_AUTH_CHECK_TIMEOUT_MS = timeoutMs;
  w.fetch = fetchImpl;
  evalModule(w, 'od-auth-gate.js');
  return { w, d: w.document };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const gate = (t) => t.d.getElementById('slm-auth-gate');

describe('access gate deadline', function () {
  it('a status check that never answers still lets the owner in', async function () {
    const t = mount(() => new Promise(() => {}), 30);
    assert.ok(gate(t), 'the checking screen is shown first');
    await sleep(120);
    for (let i = 0; i < 6; i++) await flushPromises();
    assert.equal(gate(t), null, 'the gate is gone after its deadline');
    t.w.close();
  });

  it('a whoami that never answers (status fine) still lets the owner in', async function () {
    const t = mount((url) => String(url).endsWith('/api/rbac/status')
      ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ require_login: false }) })
      : new Promise(() => {}), 30);
    await sleep(120);
    for (let i = 0; i < 6; i++) await flushPromises();
    assert.equal(gate(t), null);
    t.w.close();
  });

  it('answers that come in time behave exactly as before (personal mode: no gate)', async function () {
    const t = mount((url) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(
      String(url).endsWith('/api/rbac/status') ? { require_login: false } : { kind: 'owner', session_invalid: false }) }));
    for (let i = 0; i < 8; i++) await flushPromises();
    assert.equal(gate(t), null);
    t.w.close();
  });
});
