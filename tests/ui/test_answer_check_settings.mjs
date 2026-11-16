/**
 * tests/ui/test_answer_check_settings.mjs — Settings › Answer check (4.1.20).
 * Runner: npm test (scripts/run-ui-tests.mjs). Requires jsdom.
 *
 * Every control a non-technical user can touch, against a stubbed daemon:
 * the choices are always clickable and say what to finish; an open section
 * survives the 2-second poll; one Save sends the whole form and shows the
 * outcome next to it; unsaved changes are marked and leaving warns; tests
 * show a persistent result; no internal words reach the page.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildHarness, evalModule, flushPromises } from './harness.mjs';

function status(over = {}) {
  const base = {
    mode: 'auto', active: 'off', apple_silicon: true, setup_running: false,
    laya_test_running: false, tests: {},
    laya: { state: 'failed', managed: false, python: '/Users/alice/.local/share/laya-venv/bin/python',
            hf_home: '', model_path: '/Users/alice/model', step: 'Needs a check — choose Check this install.',
            error: "SLM found this install but hasn't checked it yet. Checking downloads nothing.",
            action: 'check', progress: 0 },
    adopt: { running: false, state: 'not_installed', error: '' },
    jev: { provider: 'typesafe', has_key: false, key_hint: '', key_problem: '', consent: false,
           rerank: { enabled: false, active: false, k: 20 } },
  };
  const out = Object.assign({}, base, over);
  out.laya = Object.assign({}, base.laya, over.laya || {});
  out.jev = Object.assign({}, base.jev, over.jev || {});
  return out;
}

async function mount(initial, replies = {}) {
  const h = buildHarness(['mount'], { ok: true, status: 200, json: {} });
  const w = h.window;
  const sent = [];
  let current = initial;
  w.__setStatus = (s) => { current = s; };
  w.fetch = function (url, init) {
    const u = String(url);
    const method = (init && init.method) || 'GET';
    if (method !== 'GET') sent.push({ url: u, method, body: init.body ? JSON.parse(init.body) : null });
    let reply;
    if (u.endsWith('/api/rbac/whoami')) reply = { status: 200, body: { permissions: ['read', 'manage'] } };
    else if (method === 'GET') reply = { status: 200, body: current };
    else reply = (replies[u] && replies[u](init)) || { status: 200, body: current };
    return Promise.resolve({ ok: reply.status < 400, status: reply.status,
                             json: () => Promise.resolve(reply.body) });
  };
  const confirms = [];
  w.confirm = (m) => { confirms.push(m); return true; };
  const toasts = [];
  w.showToast = (m) => toasts.push(m);
  evalModule(w, 'answer-check.js');
  const root = w.document.getElementById('mount');
  w.SLMAnswerCheck.renderInto(root);
  for (let i = 0; i < 4; i++) await flushPromises();
  const ac = (id) => root.querySelector(`[data-ac="${id}"]`);
  const text = (sel) => (root.querySelector(sel) || { textContent: '' }).textContent;
  return { w, d: w.document, root, sent, confirms, toasts, ac, text };
}

async function click(p, el) { el.click(); for (let i = 0; i < 4; i++) await flushPromises(); }
async function choose(p, mode) {
  const r = p.ac('mode-' + mode);
  r.checked = true;
  r.dispatchEvent(new p.w.Event('change', { bubbles: true }));
  await flushPromises();
}

describe('Answer check settings', function () {
  it('every choice is clickable, and a not-ready one says what to do — no request sent', async function () {
    const p = await mount(status());
    for (const m of ['laya', 'jev', 'off']) assert.equal(p.ac('mode-' + m).disabled, false);
    await choose(p, 'laya');
    assert.equal(p.ac('mode-laya').checked, true);
    assert.match(p.text('.ac-guidance'), /Check this install/);
    assert.match(p.text('.ac-guidance'), /nothing is downloaded/);
    assert.equal(p.sent.length, 0, 'choosing must not send anything before Save');
    await choose(p, 'jev');
    assert.match(p.text('.ac-guidance'), /Needs your Jev key/);
    assert.match(p.text('.ac-save-state'), /Unsaved changes/);
  });

  it('"Check this install" checks the found install — never starts a download', async function () {
    const p = await mount(status());
    await choose(p, 'laya');
    await click(p, p.root.querySelector('.ac-guidance button'));
    assert.deepEqual(p.sent.map((s) => s.url), ['/api/v3/answer-check/laya/adopt']);
    assert.equal(p.sent[0].body.python, '/Users/alice/.local/share/laya-venv/bin/python');
    assert.equal(p.sent[0].body.model_path, '/Users/alice/model');
  });

  it('an open section stays open when the page refreshes itself', async function () {
    const p = await mount(status({ laya: { state: 'not_installed', action: '', error: '', step: '' } }));
    const existing = p.root.querySelector('.ac-existing');
    existing.open = true;
    existing.dispatchEvent(new p.w.Event('toggle'));
    p.ac('existing-python').value = '/Users/alice/env/bin/python';
    p.ac('existing-python').dispatchEvent(new p.w.Event('input'));
    await choose(p, 'off');                       // any re-render
    assert.equal(p.root.querySelector('.ac-existing').open, true);
    assert.equal(p.ac('existing-python').value, '/Users/alice/env/bin/python');
  });

  it('a running setup shows real progress and a Cancel that works', async function () {
    const p = await mount(status({ setup_running: true,
      laya: { state: 'installing', progress: 0.4, step: 'Downloading the model weights (73 of 807 MB)', action: '', error: '' } }));
    assert.match(p.text('.ac-laya-state'), /73 of 807 MB.*40%/);
    assert.equal(p.root.querySelector('[role="progressbar"]').getAttribute('aria-valuenow'), '40');
    await click(p, p.ac('laya-cancel'));
    assert.equal(p.sent[0].url, '/api/v3/answer-check/laya/cancel');
  });

  it('a stopped setup offers Repair and Remove, never an endless download', async function () {
    const p = await mount(status({ laya: { state: 'failed', managed: true, action: 'setup',
      error: "Can't reach the model download server — the download stopped making progress." } }));
    assert.ok(!p.root.querySelector('[role="progressbar"]'));
    assert.ok(p.ac('laya-repair') && p.ac('laya-remove'));
    assert.match(p.text('.ac-laya-state'), /Can't reach/);
    await click(p, p.ac('laya-repair'));
    assert.equal(p.sent[0].url, '/api/v3/answer-check/laya/setup');
  });

  it('one Save sends the whole form; a refusal is shown next to Save', async function () {
    const p = await mount(status(), {
      '/api/v3/answer-check/save': () => ({ status: 400,
        body: { error: 'Tick the notice about what is sent to Jev, then Save.' } }) });
    await choose(p, 'jev');
    p.ac('jev-key').value = 'sk-test-fake';
    p.ac('jev-key').dispatchEvent(new p.w.Event('input'));
    assert.match(p.text('.ac-key-unsaved'), /Not saved yet/);
    await click(p, p.ac('ac-save'));
    const save = p.sent.find((s) => s.url.endsWith('/save'));
    assert.deepEqual(save.body, { provider: 'typesafe', key: 'sk-test-fake', consent: false,
                                  rerank: false, mode: 'jev' });
    assert.match(p.text('.ac-save-state'), /Tick the notice/);
  });

  it('a successful Save says Saved, clears the key box and the unsaved mark', async function () {
    const after = status({ mode: 'jev', active: 'jev',
      jev: { has_key: true, key_hint: '****fake', consent: true } });
    const p = await mount(status(), {
      '/api/v3/answer-check/save': () => ({ status: 200, body: Object.assign({}, after, { message: 'Saved. Answer check is now online with Jev.' }) }) });
    await choose(p, 'jev');
    p.ac('jev-key').value = 'sk-test-fake';
    p.ac('jev-key').dispatchEvent(new p.w.Event('input'));
    p.ac('jev-consent').click();
    await flushPromises();
    await click(p, p.ac('ac-save'));
    assert.match(p.text('.ac-save-state'), /Saved/);
    assert.equal(p.ac('jev-key').value, '');
    assert.match(p.text('.ac-running'), /Online with Jev/);
    assert.equal(p.w.SLMAnswerCheck.isDirty(), false);
  });

  it('leaving with unsaved changes warns; without them it does not', async function () {
    const p = await mount(status());
    const leave = () => { const e = new p.w.Event('beforeunload', { cancelable: true });
                          p.w.dispatchEvent(e); return e.defaultPrevented; };
    assert.equal(leave(), false);
    await choose(p, 'jev');
    assert.equal(leave(), true);
  });

  it('switching away from Jev says reordering pauses — before saving', async function () {
    const p = await mount(status({ mode: 'jev', active: 'jev', laya: { state: 'ready', action: '', error: '' },
      jev: { has_key: true, key_hint: '****abcd', consent: true, rerank: { enabled: true, active: true, k: 20 } } }));
    await choose(p, 'laya');
    assert.match(p.text('.ac-rerank-paused'), /Reordering with Jev is off while On this Mac is chosen/);
    await click(p, p.ac('ac-save'));
    assert.match(p.confirms[0], /Reordering with Jev turns off while On this Mac is selected/);
    const save = p.sent.find((s) => s.url.endsWith('/save'));
    assert.equal(save.body.rerank, true, 'the choice is kept, not reset');
  });

  it('background judges: need Jev first, then agree once, show watch-only, and stop', async function () {
    const NOTICE = 'In the background, SuperLocalMemory sends pairs of your memories to TypeSafe, one pair per request.';
    let p = await mount(status());
    assert.match(p.text('.ac-ladder-needs'), /Turn on the online answer check with Jev/);
    assert.equal(p.ac('ac-ladder-allow'), null, 'no way to agree before Jev is on');

    const on = { mode: 'jev', active: 'jev', laya: { state: 'ready', action: '', error: '' },
                 jev: { has_key: true, key_hint: '****abcd', consent: true,
                        ladder: { consent: false, since: null, flag: 'shadow', notice: NOTICE } } };
    p = await mount(status(on));
    assert.equal(p.text('.ac-ladder-notice'), NOTICE);
    await click(p, p.ac('ac-ladder-allow'));
    const agreed = p.sent.find((s) => s.url.endsWith('/jev/ladder'));
    assert.deepEqual(agreed.body, { accepted: true, provider: 'typesafe' });

    const given = JSON.parse(JSON.stringify(on));
    given.jev.ladder = { consent: true, since: '2026-11-16T22:00:00+00:00', flag: 'shadow', notice: NOTICE };
    p = await mount(status(given));
    assert.match(p.text('.ac-ladder-state'), /Allowed since 2026-11-16/);
    assert.match(p.text('.ac-ladder-watch'), /only watch for now/);
    await click(p, p.ac('ac-ladder-stop'));
    assert.deepEqual(p.sent.find((s) => s.url.endsWith('/jev/ladder')).body,
                     { accepted: false, provider: 'typesafe' });

    given.jev.ladder.flag = 'on';
    p = await mount(status(given));
    assert.equal(p.text('.ac-ladder-watch'), '', 'no watch-only note once they are on');
  });

  it('test results stay on the page, with when and how long', async function () {
    const p = await mount(status({ laya: { state: 'ready', action: '', error: '' },
      jev: { has_key: true, key_hint: '****abcd' },
      tests: { laya: { ok: true, at: '2026-10-04T08:42:00+00:00', seconds: 1.2, message: 'Answered correctly.' },
               jev: { ok: false, at: '2026-10-04T08:43:00+00:00', seconds: 0.4, message: 'The key was refused.' } } }));
    assert.match(p.text('.ac-laya-test'), /Tested ✓ at .* — answered in 1\.2 s/);
    assert.match(p.text('.ac-jev-test'), /Test failed at .*The key was refused/);
    await click(p, p.ac('jev-test'));
    assert.equal(p.sent[0].url, '/api/v3/answer-check/jev/test');
  });

  it('Test with an unsaved key explains instead of testing the old one', async function () {
    const p = await mount(status({ jev: { has_key: true, key_hint: '****abcd' } }));
    p.ac('jev-key').value = 'sk-test-new';
    p.ac('jev-key').dispatchEvent(new p.w.Event('input'));
    await click(p, p.ac('jev-test'));
    assert.equal(p.sent.length, 0);
    assert.match(p.toasts.join(' '), /Save the key first/);
  });

  it('reordering needs the notice, and says so', async function () {
    const p = await mount(status({ jev: { has_key: true } }));
    p.ac('jev-rerank').click();
    await flushPromises();
    assert.equal(p.ac('jev-rerank').checked, false);
    assert.match(p.root.textContent, /Reordering needs the notice above ticked first/);
  });

  it('ticking Jev reordering while On this Mac is chosen asks to switch — Yes switches and keeps the tick', async function () {
    const p = await mount(status({ mode: 'laya', active: 'laya', laya: { state: 'ready', action: '', error: '' },
      jev: { has_key: true, key_hint: '****abcd', consent: true } }));
    assert.equal(p.ac('mode-laya').checked, true);
    p.ac('jev-rerank').click();
    await flushPromises();
    assert.match(p.confirms[0], /Switch to Jev\?/);
    assert.equal(p.ac('mode-jev').checked, true, 'the judge switched to Jev');
    assert.equal(p.ac('jev-rerank').checked, true, 'the ticked option is kept');
    assert.match(p.text('.ac-save-state'), /Unsaved changes/);
  });

  it('ticking Jev reordering while On this Mac is chosen asks to switch — No leaves Laya chosen and unticks', async function () {
    const p = await mount(status({ mode: 'laya', active: 'laya', laya: { state: 'ready', action: '', error: '' },
      jev: { has_key: true, key_hint: '****abcd', consent: true } }));
    p.w.confirm = (m) => { p.confirms.push(m); return false; };
    p.ac('jev-rerank').click();
    await flushPromises();
    assert.match(p.confirms[0], /Switch to Jev\?/);
    assert.equal(p.ac('mode-laya').checked, true, 'Laya stays selected');
    assert.equal(p.ac('jev-rerank').checked, false, 'the option is unticked, never silently on');
  });

  it('no internal words outside Details', async function () {
    const p = await mount(status());
    const clone = p.root.cloneNode(true);
    clone.querySelectorAll('.ac-details').forEach((d) => d.remove());
    assert.doesNotMatch(clone.textContent, /sufficiency|adopt|managed|venv\b|hf_home/i);
  });

  it('Details opens', async function () {
    const p = await mount(status());
    const d = p.root.querySelector('.ac-details');
    d.querySelector('summary').click();
    assert.equal(d.open, true);
    assert.match(d.textContent, /Python program: ~\/\.local\/share\/laya-venv\/bin\/python/);
    assert.doesNotMatch(d.textContent, /\/Users\/v\//);
  });

  describe('each choice says what state it is in', function () {
    const label = (p, m) => p.ac('mode-' + m).parentElement.textContent.replace(/\s+/g, ' ').trim();
    const cases = [
      ['laya ready', status({ laya: { state: 'ready', action: '', error: '' } }), 'laya',
       'On this Mac — private (recommended) ✓ Ready'],
      ['laya not set up', status({ laya: { state: 'not_installed', action: '', error: '' } }), 'laya',
       'On this Mac — private (recommended) — set it up below'],
      ['laya found, unchecked', status(), 'laya', 'On this Mac — private (recommended) — check it below'],
      ['laya half-made', status({ laya: { state: 'failed', managed: true, action: 'setup' } }), 'laya',
       'On this Mac — private (recommended) — repair it below'],
      ['laya installing', status({ setup_running: true, laya: { state: 'installing', action: '', error: '' } }), 'laya',
       'On this Mac — private (recommended) — being set up below'],
      ['laya no Apple Silicon', status({ apple_silicon: false, laya: { state: 'unsupported', action: '' } }), 'laya',
       'On this Mac — private (recommended) — needs a Mac with Apple Silicon'],
      ['jev no key', status(), 'jev', 'Online with Jev — add your key to use it'],
      ['jev key, no notice', status({ jev: { has_key: true, key_hint: '****abcd' } }), 'jev',
       'Online with Jev — tick the notice below to use it'],
      ['jev ready', status({ jev: { has_key: true, key_hint: '****abcd', consent: true } }), 'jev',
       'Online with Jev ✓ Ready'],
      ['off', status(), 'off', 'Off'],
    ];
    for (const [name, s, mode, expected] of cases) {
      it(name, async function () {
        const p = await mount(s);
        assert.equal(label(p, mode), expected);
      });
    }
    it('jev with a pasted, unsaved key and the notice ticked says to Save', async function () {
      const p = await mount(status());
      p.ac('jev-key').value = 'sk-test-fake';
      p.ac('jev-key').dispatchEvent(new p.w.Event('input'));
      p.ac('jev-consent').click();
      await flushPromises();
      assert.equal(label(p, 'jev'), 'Online with Jev — Save to use it');
    });
  });
});

describe('Automatic — the never-chosen state (audit 4.1.20 L11)', function () {
  it('shows Automatic, not a checked "Off" beside "Not chosen yet"', async function () {
    const p = await mount(status({ mode: 'auto', active: 'off' }));
    for (const m of ['laya', 'jev', 'off']) {
      assert.equal(p.ac('mode-' + m).checked, false, m + ' must not look chosen');
    }
    assert.match(p.text('.ac-auto-state'), /^Automatic — not checking yet/);
    assert.ok(!/Not chosen yet/.test(p.root.textContent));
    assert.equal(p.text('.ac-save-state').includes('Unsaved'), false);
  });

  it('says when Automatic has turned the on-device check on', async function () {
    const p = await mount(status({ mode: 'auto', active: 'laya', laya: { state: 'ready', action: '' } }));
    assert.match(p.text('.ac-auto-state'), /Automatic — checking on this Mac/);
    assert.equal(p.ac('mode-laya').checked, false);
  });

  it('saving without choosing keeps Automatic; choosing Off sends Off', async function () {
    const p = await mount(status({ mode: 'auto', active: 'off' }));
    await click(p, p.ac('ac-save'));
    assert.equal('mode' in p.sent[p.sent.length - 1].body, false, 'Automatic stays Automatic');
    await choose(p, 'off');
    assert.match(p.text('.ac-save-state'), /Unsaved changes/);
    await click(p, p.ac('ac-save'));
    assert.equal(p.sent[p.sent.length - 1].body.mode, 'off');
  });

  it('an explicit choice is shown as that choice, with no Automatic line', async function () {
    const p = await mount(status({ mode: 'off', active: 'off' }));
    assert.equal(p.ac('mode-off').checked, true);
    assert.equal(p.root.querySelector('.ac-auto-state'), null);
  });
});

it('successful local calibration starts polling and renders progress', async () => {
  const initial = status({ mode: 'laya', active: 'laya', laya: { state: 'ready' },
                          execution: { state: 'ready', detail: 'Ready' } });
  const p = await mount(initial, {
    '/api/v3/answer-check/calibration': () => ({ status: 202,
      body: { state: 'running', done: 0, total: 48, detail: 'Building local labels' } }),
  });
  let polls = 0;
  p.w.setInterval = () => { polls += 1; return 42; };
  await click(p, p.ac('calibrate'));
  assert.equal(polls, 1);
  assert.match(p.root.textContent, /Calibration running: 0\/48/);
  assert.equal(p.ac('calibrate').disabled, true);
});

it('an installed but warming judge never displays a Ready headline', async () => {
  const p = await mount(status({ mode: 'laya', active: 'laya', laya: { state: 'ready' },
    execution: { state: 'warming', detail: 'The on-device model is warming' } }));
  assert.match(p.text('.ac-laya-state'), /warming/);
});

it('running local calibration offers a working per-run cancel action', async () => {
  const p = await mount(status({ mode: 'laya', active: 'laya', laya: { state: 'ready' },
    execution: { state: 'ready' }, calibration: { state: 'running', done: 12, total: 48 } }), {
    '/api/v3/answer-check/calibration/cancel': () => ({ status: 202,
      body: { state: 'cancelling', done: 12, total: 48, detail: 'Cancelling calibration' } }),
  });
  assert.ok(p.ac('calibration-cancel'));
  assert.equal(p.ac('calibration-cancel').disabled, false);
  await click(p, p.ac('calibration-cancel'));
  assert.deepEqual(p.sent.map(s => s.url), ['/api/v3/answer-check/calibration/cancel']);
  assert.match(p.root.textContent, /Calibration cancelling/);
});
