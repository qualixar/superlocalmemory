/**
 * The Dashboard home "Proposed changes" card (4.2.30, WS-6, I-10): a set of changes
 * SuperLocalMemory suggests, in plain words, answered with one click (Apply, Reject)
 * and, once applied, undone with one click, through the same routes `slm proposals`
 * uses. Hidden when there is nothing to show; memory text is never parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-proposals.js'), 'utf8');
const CARD = `<div class="card" id="dash-proposals-card" hidden><span id="dash-proposals-count"></span>
  <div id="dash-proposals-body"></div></div>`;
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const PENDING = {
  set_id: 's1', state: 'proposed', title: 'Tidy up the release notes', created_at: '2026-11-17T10:00:00Z',
  proposals: [
    { proposal_id: 'p1', kind: 'retire', state: 'proposed',
      summary: 'Retire “weekly releases <img src=x onerror=alert(1)>” because “one big release” replaces it' },
    { proposal_id: 'p2', kind: 'edit_derived', state: 'proposed', summary: 'Edit the persona summary: “a corrected text”' },
  ],
};
const APPLIED = {
  set_id: 's0', state: 'applied', applied_at: '2026-11-16T10:00:00Z',
  proposals: [{ proposal_id: 'p0', kind: 'merge', state: 'applied', summary: 'Keep “a” and retire 1 copy of it' }],
};

function page({ pending = [PENDING], applied = [], fail = null }, posts) {
  const d = new JSDOM(`<!DOCTYPE html><html><body>${CARD}</body></html>`,
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: JSON.parse(init.body) });
      return fail ? reply(fail.status, fail.body) : reply(200, { success: true, set: {} });
    }
    const sets = url.includes('state=applied') ? applied : pending;
    return reply(200, { success: true, count: pending.length, sets });
  };
  d.window.eval(SRC);
  return d;
}

const buttons = (d, label) => [...d.window.document.querySelectorAll('button')]
  .filter((b) => b.textContent === label);

describe('Proposed changes card', () => {
  it('stays hidden when nothing is proposed or applied', async () => {
    const d = page({ pending: [], applied: [] }, []);
    await d.window.SLMProposals.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-proposals-card').hidden, true);
  });

  it('lists each proposal in plain words and never parses memory text as HTML', async () => {
    const d = page({}, []);
    await d.window.SLMProposals.reload(); await flush();
    const doc = d.window.document;
    assert.equal(doc.getElementById('dash-proposals-card').hidden, false);
    assert.equal(doc.getElementById('dash-proposals-count').textContent, '1');
    assert.match(doc.body.textContent, /Tidy up the release notes/);
    assert.match(doc.body.textContent, /Nothing changes until you apply it, and you can undo it afterwards/);
    assert.match(doc.body.textContent, /Retire a memory/);
    assert.match(doc.body.textContent, /Rewrite a summary/);
    assert.equal(doc.querySelectorAll('.truth-side').length, 2);
    assert.equal(doc.querySelector('img'), null);
    assert.match(doc.querySelector('.truth-text').textContent, /<img src=x/);
    assert.equal(buttons(d, 'Apply').length, 1);
    assert.equal(buttons(d, 'Reject').length, 1);
  });

  it('Apply goes to the shared route (the daemon records the dashboard channel) and confirms', async () => {
    const posts = [];
    const d = page({}, posts);
    await d.window.SLMProposals.reload(); await flush();
    buttons(d, 'Apply')[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/proposals/s1/apply', body: {} }]);
    assert.match(d.window.document.body.textContent, /Applied\. You can undo it below\./);
  });

  it('Reject changes nothing and says so', async () => {
    const posts = [];
    const d = page({}, posts);
    await d.window.SLMProposals.reload(); await flush();
    buttons(d, 'Reject')[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/proposals/s1/reject', body: {} }]);
    assert.match(d.window.document.body.textContent, /Rejected\. Nothing was changed\./);
  });

  it('a recently applied set offers Undo', async () => {
    const posts = [];
    const d = page({ pending: [], applied: [APPLIED] }, posts);
    await d.window.SLMProposals.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-proposals-card').hidden, false);
    assert.match(d.window.document.body.textContent, /Undo puts back exactly what it replaced/);
    assert.equal(buttons(d, 'Apply').length, 0);
    buttons(d, 'Undo')[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/proposals/s0/rollback', body: {} }]);
    assert.match(d.window.document.body.textContent, /Everything it replaced is back as it was/);
  });

  it('a refusal says why and keeps the buttons usable', async () => {
    const d = page({ fail: { status: 409, body: { detail: 'This set is applied, not waiting for a decision. Nothing was changed.' } } }, []);
    await d.window.SLMProposals.reload(); await flush();
    buttons(d, 'Apply')[0].click(); await flush();
    assert.match(d.window.document.querySelector('.truth-error').textContent, /not waiting for a decision/);
    assert.equal(buttons(d, 'Apply')[0].disabled, false);
  });

  it('hides itself when the list cannot be read', async () => {
    const d = new JSDOM(`<!DOCTYPE html><html><body>${CARD}</body></html>`,
                        { runScripts: 'outside-only', url: 'http://localhost:8765/' });
    d.window.fetch = () => reply(403, { detail: 'local only' });
    d.window.eval(SRC);
    await d.window.SLMProposals.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-proposals-card').hidden, true);
  });
});
