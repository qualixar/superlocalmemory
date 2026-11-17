/**
 * The Health page's "What memory did for you" card (4.2.30 W3-5, §8.10.3 g):
 * on demand, read-only, plain words, a claim only from the report's own
 * claim field, and nothing from the server parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-outcomes.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));

const REPORT = {
  success: true,
  ledger: { receipts: 'on', withholding: 'on', recent_days: 30, deliveries: { mcp_recall: 7, session_init: 3 } },
  use: { rate: 0.42, facts_delivered: 9, never_used: 4,
         most_used: [{ fact_id: 'f-star<img src=x onerror=alert(1)>', kind: 'rule', deliveries: 6, uses: 5, rate: 0.7, low: 0.4, high: 0.9 }],
         rarely_used: [{ fact_id: 'f-dull', kind: 'episodic', deliveries: 9, uses: 0, rate: 0.05, low: 0, high: 0.2 }] },
  effects: [
    { path: 'session_init', contrast: 'memory', outcome: 'used', n: 40, planned: 7536, aipw: 0.05, claim: null },
    { path: 'session_context', contrast: 'memory', outcome: 'used', n: 5000, planned: 4175, aipw: 0.12, claim: 'helps' },
    { path: 'session_init', contrast: 'channel:bm25', outcome: 'used', n: 3, planned: 48099, aipw: null, claim: null },
    { path: 'session_init', contrast: 'channel', outcome: 'reask', n: 0, planned: 7536, aipw: null, claim: null },
  ],
  certificates: [{ voided_at: null }, { voided_at: '2026-11-10' }],
  second_opinions: [],
};

function page(report, opened) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="health"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.requested = [];
  d.window.fetch = (url) => {
    d.window.requested.push(url);
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(report) });
  };
  d.window.odOpenMemory = (id) => opened.push(id);
  d.window.eval(SRC);
  d.window.odMountMemoryOutcomes(d.window.document.getElementById('health'));
  return d;
}

const show = (d) => [...d.window.document.querySelectorAll('button')].find((b) => b.textContent === 'Show');

describe('What memory did for you card', () => {
  it('reads nothing until asked, then the one report route', async () => {
    const d = page(REPORT, []);
    assert.deepEqual(d.window.requested, []);
    show(d).click();
    await flush();
    assert.deepEqual(d.window.requested, ['/api/outcomes/health']);
  });

  it('says it in plain words and claims only what the report claims', async () => {
    const d = page(REPORT, []);
    show(d).click();
    await flush();
    const text = d.window.document.body.textContent;
    assert.match(text, /memories 10 times in the last 30 days/);
    assert.match(text, /Measuring is on/);
    assert.match(text, /acts on it 42% of the time\. 4 of 9 memories given were never used/);
    assert.match(text, /Helps: \+12\.0 points/);
    assert.match(text, /No clear effect yet \(40 of about 7536\)/);
    assert.doesNotMatch(text, /bm25/);               // an unclaimed per-channel row is not shown
    assert.match(text, /Not enough data yet \(0 of about 7536\)/);
    assert.match(text, /act on their own: 1\. Second opinions that disagreed: 0/);
    assert.equal(d.window.document.querySelectorAll('img').length, 0);
  });

  it('opens a listed memory', async () => {
    const opened = [];
    const d = page(REPORT, opened);
    show(d).click();
    await flush();
    const open = [...d.window.document.querySelectorAll('button')].filter((b) => b.textContent === 'Open');
    assert.equal(open.length, 2);
    open[1].click();
    assert.deepEqual(opened, ['f-dull']);
  });

  it('says measuring is off and how to turn it on', async () => {
    const d = page({ ...REPORT, ledger: { ...REPORT.ledger, withholding: 'off' }, use: { rate: null } }, []);
    show(d).click();
    await flush();
    const text = d.window.document.body.textContent;
    assert.match(text, /Measuring is off/);
    assert.match(text, /Not enough deliveries yet to say/);
  });
});
