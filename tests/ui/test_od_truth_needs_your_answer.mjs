/**
 * The Dashboard home "Needs your answer" card (4.2.30, D-37..D-39): disputes and
 * changed source files in plain words, answered with one click through the same
 * routes agents and the CLI use. Hidden when nothing needs an answer; memory
 * text is never parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-truth.js'), 'utf8');
const CARD = `<div class="card" id="dash-attention-card" hidden><span id="dash-attention-count"></span>
  <div id="dash-attention-body"></div></div>`;
const flush = () => new Promise((r) => setTimeout(r, 30));
const reply = (status, body) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });

const DISPUTE = {
  dispute_id: 'd1', reason: 'pattern_key', question: 'Which is true now?', next: 'ask_user',
  sides: [
    { fact_id: 'new', said_at: '2026-11-14T00:00:00Z', current: false, visible: true,
      content: 'SLM ships one big release <img src=x onerror=alert(1)>' },
    { fact_id: 'old', said_at: '2026-03-01T00:00:00Z', current: true, visible: true,
      content: 'SLM ships every week' },
  ],
};
const CHANGED = { fact_id: 'f9', state: 'changed', file: '/Users/me/project/config.yaml' };

function page(attention, posts) {
  const d = new JSDOM(`<!DOCTYPE html><html><body>${CARD}</body></html>`,
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url, init) => {
    if (init && init.method === 'POST') {
      posts.push({ url, body: JSON.parse(init.body) });
      return reply(200, { success: true, retired: url.includes('settle') && !/both|unrelated/.test(init.body) ? ['old'] : [] });
    }
    return reply(200, { success: true, ...attention });
  };
  d.window.eval(SRC);
  return d;
}

const buttons = (d, label) => [...d.window.document.querySelectorAll('button')]
  .filter((b) => b.textContent === label);

describe('Needs your answer card', () => {
  it('stays hidden when nothing needs an answer', async () => {
    const d = page({ disputes: [], changed_sources: [] }, []);
    await d.window.SLMTruth.reload(); await flush();
    assert.equal(d.window.document.getElementById('dash-attention-card').hidden, true);
  });

  it('shows both sides of a dispute in plain words, text never parsed as HTML', async () => {
    const d = page({ disputes: [DISPUTE], changed_sources: [] }, []);
    await d.window.SLMTruth.reload(); await flush();
    const doc = d.window.document;
    assert.equal(doc.getElementById('dash-attention-card').hidden, false);
    assert.equal(doc.getElementById('dash-attention-count').textContent, '1');
    assert.match(doc.body.textContent, /These two memories disagree\. Which is true now\?/);
    assert.equal(doc.querySelectorAll('.truth-side').length, 2);
    assert.equal(doc.querySelector('img'), null);
    assert.match(doc.querySelector('.truth-text').textContent, /<img src=x/);
    assert.equal(buttons(d, 'Keep this one').length, 2);
  });

  it('Keep this one settles through the shared route with the side and the channel', async () => {
    const posts = [];
    const d = page({ disputes: [DISPUTE], changed_sources: [] }, posts);
    await d.window.SLMTruth.reload(); await flush();
    buttons(d, 'Keep this one')[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/disputes/d1/settle', body: { keep: 'new', channel: 'dashboard' } }]);
    assert.match(d.window.document.body.textContent, /The other memory was retired; you can undo it/);
  });

  it('Both are true and Different things send both and unrelated', async () => {
    for (const [label, keep] of [['Both are true', 'both'], ['They are about different things', 'unrelated']]) {
      const posts = [];
      const d = page({ disputes: [DISPUTE], changed_sources: [] }, posts);
      await d.window.SLMTruth.reload(); await flush();
      buttons(d, label)[0].click(); await flush();
      assert.equal(posts[0].body.keep, keep);
      assert.match(d.window.document.body.textContent, /Both memories stay/);
    }
  });

  it('Not sure hides the item here and changes nothing', async () => {
    const posts = [];
    const d = page({ disputes: [DISPUTE], changed_sources: [] }, posts);
    await d.window.SLMTruth.reload(); await flush();
    buttons(d, 'Not sure')[0].click(); await flush();
    assert.equal(posts.length, 0);
    assert.equal(d.window.document.getElementById('dash-attention-card').hidden, true);
  });

  it('a changed file offers Still true; a missing one does not', async () => {
    const posts = [];
    const d = page({ disputes: [], changed_sources: [CHANGED, { ...CHANGED, fact_id: 'f8', state: 'missing' }] }, posts);
    await d.window.SLMTruth.reload(); await flush();
    const doc = d.window.document;
    assert.match(doc.body.textContent, /came from a file that changed since it was saved/);
    assert.match(doc.body.textContent, /came from a file that is gone/);
    assert.equal(buttons(d, 'Still true').length, 1);
    buttons(d, 'Still true')[0].click(); await flush();
    assert.deepEqual(posts, [{ url: '/api/changed-sources/f9/still-true', body: {} }]);
  });

  it('a refused answer says why and keeps the buttons usable', async () => {
    const d = page({ disputes: [DISPUTE], changed_sources: [] }, []);
    d.window.fetch = (url, init) => init && init.method === 'POST'
      ? reply(409, { detail: 'The value changed since this question was asked, so it was closed; nothing else was changed.' })
      : reply(200, { success: true, disputes: [DISPUTE], changed_sources: [] });
    await d.window.SLMTruth.reload(); await flush();
    buttons(d, 'Keep this one')[0].click(); await flush();
    assert.match(d.window.document.querySelector('.truth-error').textContent, /value changed since/);
    assert.equal(buttons(d, 'Keep this one')[0].disabled, false);
  });
});
