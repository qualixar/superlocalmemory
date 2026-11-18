/**
 * The Health page's "Shown to your agents" card (4.2.30 W3, D-41, §8.10.3): on demand, read-only,
 * deliveries grouped by conversation, each memory with its reason in plain words, and the two
 * gate numbers. Nothing from the server is parsed as HTML.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-delivered.js'), 'utf8');
const HTML = readFileSync(join(here, '../../src/superlocalmemory/ui/index.html'), 'utf8');
const HEALTH = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-health.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));

const MEMORY = (id, extra = {}) => ({
  fact_id: id, role: 'recall', preview: 'The backup runs at night <img src=x onerror=alert(1)>',
  why: { how: 'Found by search (rank 1)', standing: 'current', standing_words: 'Still current',
         said: '3 days ago', origin: null,
         words: 'Found by search (rank 1). Still current. Saved 3 days ago.' }, ...extra,
});
const REPORT = {
  success: true, deliveries: 3, memories: 3,
  groups: [
    { session_id: 's-new', client: 'claude', ago: '5 minutes ago', deliveries: [
      { kind: 'cue', kind_words: 'On a cue: a file, command, error, repository or date', ago: '5 minutes ago',
        left_out: 0, more: 0, memories: [MEMORY('f-cue')] },
      { kind: 'prompt', kind_words: 'With a prompt or before a web search (automatic)', ago: '20 minutes ago',
        left_out: 1, more: 2, memories: [MEMORY('f-prompt', { why: { standing: 'disputed',
          words: 'Found by search (rank 2). Disputed: another memory says something different.' } })] }] },
    { session_id: 's-old', client: 'codex', ago: '5 hours ago', deliveries: [
      { kind: 'session', kind_words: 'At the start of a conversation', ago: '5 hours ago', left_out: 0, more: 0,
        memories: [MEMORY('f-old')] }] },
  ],
  need_gate: { turns: 40, skipped: 12, rules_bound: 0.03, alpha: 0.05, days: 30 },
  harm_gate: { state: 'off', acting: false, tests: 4, would_withhold_count: 1,
               would_withhold: [{ fact_id: 'f-harm', preview: 'A memory that hurt' }] },
};

function page(report, opened = [], { fail = false } = {}) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="health"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.requested = [];
  d.window.fetch = (url) => {
    d.window.requested.push(url);
    if (fail) return Promise.reject(new Error('down'));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(report) });
  };
  d.window.odOpenMemory = (id) => opened.push(id);
  d.window.eval(SRC);
  d.window.odMountDelivered(d.window.document.getElementById('health'));
  return d;
}
const show = (d) => [...d.window.document.querySelectorAll('button')].find((b) => b.textContent === 'Show');
const text = (d) => d.window.document.body.textContent;

describe('Shown to your agents card', () => {
  it('is mounted by the Health page and its script is stamped', () => {
    assert.match(HTML, /static\/js\/od-delivered\.js\?v=[0-9a-f]{8}/);
    assert.match(HEALTH, /odMountDelivered/);
  });

  it('reads nothing until asked, then only the one read route', async () => {
    const d = page(REPORT);
    assert.deepEqual(d.window.requested, []);
    show(d).click();
    await flush();
    assert.deepEqual(d.window.requested, ['/api/delivered/recent?limit=20']);
  });

  it('groups by conversation, newest first, the first one open', async () => {
    const d = page(REPORT);
    show(d).click();
    await flush();
    const groups = [...d.window.document.querySelectorAll('details.delivered-group')];
    assert.equal(groups.length, 2);
    assert.match(groups[0].querySelector('summary').textContent, /claude · 5 minutes ago · 2 memories in 2 deliveries/);
    assert.match(groups[1].querySelector('summary').textContent, /codex · 5 hours ago · 1 memory in 1 delivery/);
    assert.equal(groups[0].open, true);
    assert.equal(groups[1].open, false);
    assert.match(text(d), /In the last 3 deliveries your agents were shown 3 memories, in 2 conversations\./);
  });

  it('names the kind of each delivery and every memory\'s reason in words', async () => {
    const d = page(REPORT);
    show(d).click();
    await flush();
    const t = text(d);
    assert.match(t, /At the start of a conversation · 5 hours ago/);
    assert.match(t, /With a prompt or before a web search \(automatic\) · 20 minutes ago/);
    assert.match(t, /On a cue: a file, command, error, repository or date/);
    assert.match(t, /Found by search \(rank 1\)\. Still current\. Saved 3 days ago\./);
    const warn = [...d.window.document.querySelectorAll('.delivered-why')].filter((n) => n.className.includes('tone-warn'));
    assert.equal(warn.length, 1);
    assert.match(warn[0].textContent, /Disputed/);
  });

  it('says what was left out on purpose and what was not listed', async () => {
    const d = page(REPORT);
    show(d).click();
    await flush();
    assert.match(text(d), /1 memory was left out on purpose: a random check of whether memory helps/);
    assert.match(text(d), /2 more memories not listed/);
  });

  it('shows the two numbers with their bound, in words', async () => {
    const d = page(REPORT);
    show(d).click();
    await flush();
    const t = text(d);
    assert.match(t, /12 prompts of 40 in the last 30 days did not need memory, so none was added/);
    assert.match(t, /rules skipped at most 3% \(measured, not assumed\)/);
    assert.match(t, /1 memory would be withheld if the harm check acted/);
    assert.match(t, /It only watches: nothing is withheld/);
    assert.match(t, /A memory that hurt/);
  });

  it('says so when no prompt was checked and nothing would be withheld', async () => {
    const d = page({ ...REPORT, need_gate: { turns: null }, harm_gate: { would_withhold_count: 0, would_withhold: [] } });
    show(d).click();
    await flush();
    assert.match(text(d), /No prompts have been checked yet/);
    assert.match(text(d), /No memory has been measured as making an agent worse/);
  });

  it('says so when nothing has been shown yet', async () => {
    const d = page({ ...REPORT, groups: [], deliveries: 0, memories: 0 });
    show(d).click();
    await flush();
    assert.match(text(d), /Nothing has been shown to an agent yet\./);
  });

  it('never parses memory text as HTML and opens a memory on click', async () => {
    const opened = [];
    const d = page(REPORT, opened);
    show(d).click();
    await flush();
    assert.equal(d.window.document.querySelectorAll('img').length, 0);
    assert.match(text(d), /<img src=x onerror=alert\(1\)>/);
    const open = [...d.window.document.querySelectorAll('button')].filter((b) => b.textContent === 'Open');
    assert.ok(open.length >= 3);
    open[0].click();  // a conversation reads oldest first: the prompt came before the cue
    assert.deepEqual(opened, ['f-prompt']);
  });

  it('says it could not read when the route fails, and lets the person try again', async () => {
    const d = page(REPORT, [], { fail: true });
    show(d).click();
    await flush();
    assert.match(text(d), /could not be read/);
    assert.equal(show(d).disabled, false);
  });

  it('mounts once', () => {
    const d = page(REPORT);
    d.window.odMountDelivered(d.window.document.getElementById('health'));
    assert.equal(d.window.document.querySelectorAll('#od-delivered').length, 1);
  });
});
