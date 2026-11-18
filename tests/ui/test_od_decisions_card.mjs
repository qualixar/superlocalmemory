/**
 * The Dashboard home "Recent decisions by your agents" card (4.2.30, D-49): who decided what,
 * through which channel and when, newest first, with an Undo for each decision that can be taken
 * back, through POST /api/decisions/{id}/undo. Hidden when nothing was decided; server text is
 * never parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-decisions.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const CARD = `<div class="card" id="dash-decisions-card" hidden><span id="dash-decisions-count"></span>
  <div id="dash-decisions-body"></div></div>`;
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const ROW = (id, extra = {}) => ({
  decision_id: id, kind: 'promotion', action: 'promote', subject_id: 'f1',
  summary: 'Promoted a memory that was a tool\'s output', actor: 'agent:claude-code',
  channel: 'mcp', decided_at: '2026-11-18T10:00:00+00:00', undone: false, can_undo: true,
  preview: 'Run the tests <img src=x onerror=alert(1)>', ...extra,
});

function page({ rows = [ROW('d1'), ROW('d2', { kind: 'dispute', action: 'settle', actor: 'client:local',
                                               channel: 'http', preview: undefined })],
                fail = null }, posts) {
  const d = new JSDOM(`<!DOCTYPE html><html><body>${CARD}</body></html>`,
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: JSON.parse(init.body) });
      return fail ? reply(fail.status, fail.body) : reply(200, { success: true });
    }
    return reply(200, { success: true, decisions: rows,
                        counts: { total: rows.length, undone: 0, open: rows.length } });
  };
  d.window.eval(SRC);
  return d;
}
const undoButtons = (d) => [...d.window.document.querySelectorAll('button')].filter((b) => b.textContent === 'Undo');

describe('Recent decisions by your agents card', () => {
  it('is in the page with its script stamped', () => {
    assert.match(HTML, /id="dash-decisions-card" hidden/);
    assert.match(HTML, /static\/js\/od-decisions\.js\?v=[0-9a-f]{8}/);
    assert.match(HTML, /<h3>Recent decisions by your agents<\/h3>/);
  });

  it('stays hidden when nothing was decided', async () => {
    const d = page({ rows: [] }, []);
    await d.window.SLMDecisions.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-decisions-card').hidden, true);
  });

  it('names who decided, through which channel, and says nothing is waiting for you', async () => {
    const d = page({}, []);
    await d.window.SLMDecisions.reload(); await flush();
    const text = d.window.document.body.textContent;
    assert.equal(d.window.document.getElementById('dash-decisions-card').hidden, false);
    assert.match(text, /agent:claude-code through an agent over MCP/);
    assert.match(text, /client:local through HTTP/);
    assert.match(text, /nothing here is waiting for you/);
    assert.equal(d.window.document.getElementById('dash-decisions-count').textContent, '2');
  });

  it('never parses a memory preview as HTML', async () => {
    const d = page({}, []);
    await d.window.SLMDecisions.reload(); await flush();
    assert.equal(d.window.document.querySelectorAll('img').length, 0);
    assert.match(d.window.document.body.textContent, /<img src=x onerror=alert\(1\)>/);
  });

  it('each decision that can be undone has an Undo, sent to the one undo route', async () => {
    const posts = [];
    const d = page({}, posts);
    await d.window.SLMDecisions.reload(); await flush();
    assert.equal(undoButtons(d).length, 2);
    undoButtons(d)[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/decisions/d1/undo', body: {} }]);
    assert.match(d.window.document.body.textContent, /Undone\. It is recorded below\./);
  });

  it('an undone decision says who took it back and has no second Undo', async () => {
    const d = page({ rows: [ROW('d1', { undone: true, can_undo: false, undone_by: 'dashboard',
                                        undone_channel: 'dashboard', undone_at: '2026-11-18T11:00:00+00:00' })] }, []);
    await d.window.SLMDecisions.reload(); await flush();
    assert.equal(undoButtons(d).length, 0);
    assert.match(d.window.document.body.textContent, /Withdrawn by dashboard through the dashboard/);
  });

  it('a decision that cannot be undone (a rollback is itself an undo) has no button', async () => {
    const d = page({ rows: [ROW('d1', { action: 'withdraw', can_undo: false })] }, []);
    await d.window.SLMDecisions.reload(); await flush();
    assert.equal(undoButtons(d).length, 0);
  });

  it('a refused undo is shown and the button comes back', async () => {
    const d = page({ fail: { status: 409, body: { detail: 'The value changed since. Nothing was changed.' } } }, []);
    await d.window.SLMDecisions.reload(); await flush();
    undoButtons(d)[0].click(); await flush();
    assert.match(d.window.document.body.textContent, /The value changed since/);
    assert.equal(undoButtons(d)[0].disabled, false);
  });
});
