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

const FACT = {
  success: true, fact_id: 'new', disputes: [], source: { state: 'changed', file: '/p/config.yaml', next: 'reread_source' },
  slots: [
    { legacy: false, subject: 'slm', relation: 'release_cadence', qualifier: null, values: [
      { fact_id: 'new', state: 'head', began: '2026-11-14T00:00:00Z', current: true, this_memory: true, visible: true, content: 'One big release a month' },
      { fact_id: 'old', state: 'superseded', began: '2026-03-01T00:00:00Z', current: false, this_memory: false, visible: false, content: null } ] },
    { legacy: true, subject: null, relation: null, values: [] },
  ],
};

function bare(routes) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="detail"></div><div id="health"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = (url) => {
    const key = Object.keys(routes).find((k) => url.startsWith(k));
    return key ? reply(200, routes[key]) : reply(404, {});
  };
  d.window.eval(SRC);
  return d;
}

describe('Memory detail: When it was true', () => {
  it('shows the slot, each value with its date and state, the changed file, never hidden text', async () => {
    const d = bare({ '/api/truth/facts/new': FACT });
    const box = d.window.document.getElementById('detail');
    await d.window.SLMTruth.detail('new', box); await flush();
    const text = box.textContent;
    assert.match(text, /When it was true/);
    assert.match(text, /It states the value of: slm · release cadence/);
    assert.match(text, /Shown now/);
    assert.match(text, /Replaced/);
    assert.match(text, /This memory/);
    assert.match(text, /You cannot see this memory/);
    assert.match(text, /Part of a correction/);
    assert.match(text, /came from a file that changed/);
  });

  it('adds nothing for a memory that is in no slot', async () => {
    const d = bare({ '/api/truth/facts/plain': { success: true, slots: [], disputes: [], source: null } });
    const box = d.window.document.getElementById('detail');
    await d.window.SLMTruth.detail('plain', box); await flush();
    assert.equal(box.textContent, '');
  });
});

describe('Health page: Truth checks', () => {
  it('runs only when asked and says the result in plain words', async () => {
    let asked = 0;
    const d = bare({});
    d.window.fetch = (url) => {
      asked++;
      return reply(200, { success: true, results: [
        { invariant: 'I-01', title: 't', status: 'ok', count: 0 },
        { invariant: 'I-05', title: 't', status: 'violated', count: 3 },
        { invariant: 'I-09', title: 'Raw title for I-09', status: 'not_checked', count: 0 } ] });
    };
    const host = d.window.document.getElementById('health');
    d.window.odMountTruthChecks(host);
    d.window.odMountTruthChecks(host);  // mounting twice adds one card
    assert.equal(d.window.document.querySelectorAll('#od-truth-checks').length, 1);
    assert.equal(asked, 0);
    buttons(d, 'Run checks')[0].click(); await flush();
    const text = host.textContent;
    assert.equal(asked, 1);
    assert.match(text, /1 check found a problem/);
    assert.match(text, /Values follow each other in time without overlapping\..*Problem found \(3\)/);
    assert.match(text, /Raw title for I-09.*Not checked yet/);
  });
});
