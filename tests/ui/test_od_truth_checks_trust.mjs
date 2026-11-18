/**
 * The Truth checks card (4.2.30 WS-5, D-16): signed memories and the audit log appear in words,
 * a modified memory or a failed audit log counts as a problem, an older server that sends neither
 * changes nothing, and no memory text is ever shown.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '../../src/superlocalmemory/ui/js/od-truth.js'), 'utf8');
const flush = () => new Promise((r) => setTimeout(r, 30));

function run(extra) {
  const d = new JSDOM('<!DOCTYPE html><html><body><div id="health"></div></body></html>',
                      { runScripts: 'outside-only', url: 'http://localhost:8765/' });
  d.window.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({
    success: true, results: [{ invariant: 'I-01', title: 't', status: 'ok', count: 0 }],
    certificates: [], second_opinions: [], ...extra }) });
  d.window.eval(SRC);
  const host = d.window.document.getElementById('health');
  d.window.odMountTruthChecks(host);
  [...host.querySelectorAll('button')].find((b) => b.textContent === 'Run checks').click();
  return flush().then(() => host.textContent);
}

describe('Truth checks: signatures and the audit log', () => {
  it('reports both as OK in words when they are', async () => {
    const t = await run({
      signatures: { ok: true, modified: 0, words: '90 of 100 memories are signed. 90 checked: none changed since signing.' },
      audit_chain: { present: true, ok: true, words: 'The audit log is intact: 12 entries, order and links verified.' } });
    assert.match(t, /Everything that was checked is consistent\./);
    assert.match(t, /Signed memories and the audit log/);
    assert.match(t, /none changed since signing\.OK/);
    assert.match(t, /order and links verified\.OK/);
  });

  it('counts a modified memory as a problem', async () => {
    const t = await run({
      signatures: { ok: false, modified: 2, words: '2 of the 90 memories checked no longer match their signature.' },
      audit_chain: { present: true, ok: true, words: 'The audit log is intact.' } });
    assert.match(t, /1 check found a problem/);
    assert.match(t, /no longer match their signature\.Problem found/);
  });

  it('counts a failed audit log as a problem and two problems as two', async () => {
    const t = await run({
      signatures: { ok: false, modified: 1, words: 'One memory changed.' },
      audit_chain: { present: true, ok: false, words: 'The audit log failed its check at entry 7: bad link.' } });
    assert.match(t, /2 checks found a problem/);
    assert.match(t, /failed its check at entry 7: bad link\.Problem found/);
  });

  it('does not call an unchecked signature or a missing log a pass or a problem', async () => {
    const t = await run({
      signatures: { ok: null, modified: 0, words: 'No memory is signed yet. Signing is off (feature flag trust.signing).' },
      audit_chain: { present: false, ok: null, words: 'There is no audit log next to this store yet.' } });
    assert.match(t, /Everything that was checked is consistent\./);
    assert.match(t, /Signing is off \(feature flag trust\.signing\)\.Not checked/);
    assert.match(t, /no audit log next to this store yet\.Not used here/);
  });

  it('is unchanged for an older server that sends neither', async () => {
    const t = await run({});
    assert.doesNotMatch(t, /Signed memories and the audit log/);
    assert.match(t, /Judges allowed to act on their own/);
  });
});
