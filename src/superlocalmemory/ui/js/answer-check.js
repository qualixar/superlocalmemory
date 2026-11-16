/* answer-check.js — Settings › "Answer check" (4.1.20)
 *
 * Lets a non-technical user choose how their memory checks an answer before
 * giving it: on this Mac (the on-device check), online with Jev, or off — and
 * get there without ever being stuck.
 *
 * How it is built
 * - A FORM with one Save. The choice, the Jev provider, a new key, the notice
 *   and Jev's reordering are edited locally (`draft`) and sent together to
 *   POST /api/v3/answer-check/save, which applies them live (no restart) or
 *   refuses the whole form with a plain reason shown next to Save. Unsaved
 *   changes are marked, and leaving the page with them asks first.
 * - ACTIONS run at once and do not wait for Save: set up / repair / cancel the
 *   on-device check, check an install the person already has, test each
 *   option, remove. They are background jobs the page polls.
 * - Every choice is always clickable. Choosing something that is not ready
 *   yet shows, right under the choices, what to do to finish it, with the
 *   button that does it — never a greyed-out control with no reason.
 * - "Running now" always says what actually runs (status.active), separately
 *   from what is chosen.
 * - Re-rendering keeps what the person was doing: open sections, typed paths,
 *   the typed key, focus. (In 4.1.19 the 2-second poll rebuilt the panel and
 *   snapped "Advanced" shut, which looked like it never opened.)
 *
 * od-settings.js builds the group card and calls renderInto(mount). Auth:
 * core.js's fetch patch adds the install token to every change; the server
 * is the real permission gate.
 *
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0-or-later
 */
(function () {
  'use strict';

  var API = '/api/v3/answer-check';
  var POLL_MS = 2000;
  var POLL_MAX_FAILURES = 5;

  var canManage = true;
  var mountEl = null;
  var status = null;          // last status read from the server
  var draft = null;           // the form, as the person has edited it
  var formError = '';         // why the last Save was refused
  var savedNote = '';         // "Saved" confirmation
  var guidanceFor = '';       // a choice the person clicked that isn't ready yet
  var ui = { existingOpen: false, detailsOpen: false, python: '', model: '', cache: '' };
  var pollTimer = null, pollFailures = 0, contactLost = false;
  var pendingAdopt = false;

  /* ── tiny DOM helpers ─────────────────────────────────────────── */
  function el(tag, attrs, css) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]);
    });
    if (css) Object.assign(e.style, css);
    return e;
  }
  function btn(text, primary, id) {
    var b = el('button', { type: 'button', 'data-ac': id || '' });
    b.className = 'btn sm ' + (primary ? 'primary' : 'ghost');
    b.textContent = text;
    if (!canManage) b.disabled = true;
    return b;
  }
  function input(id, placeholder, type) {
    var i = el('input', { type: type || 'text', placeholder: placeholder || '', 'data-ac': id,
                          'aria-label': placeholder || id });
    Object.assign(i.style, { height: '34px', borderRadius: 'var(--r-md)',
      border: '1px solid var(--border)', background: 'var(--card-2)', color: 'var(--fg)',
      padding: '0 10px', fontSize: '13px', minWidth: '220px', maxWidth: '420px',
      flex: '1 1 260px', outline: 'none' });
    i.disabled = !canManage;
    return i;
  }
  function row() {
    var d = el('div', null, { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap',
                              marginTop: '8px' });
    Array.prototype.forEach.call(arguments, function (c) { if (c) d.appendChild(c); });
    return d;
  }
  function note(text, tone, cls) {
    var colors = { ok: 'var(--ok-text)', bad: 'var(--danger-text)', dim: 'var(--fg-3)' };
    return el('div', { text: text, class: cls || '' },
      { fontSize: '12.5px', color: colors[tone] || 'var(--fg-2)', marginTop: '6px' });
  }
  function card(title, cls) {
    var box = el('div', { class: cls }, { border: '1px solid var(--border)',
      borderRadius: 'var(--r-md)', padding: '12px 14px', marginBottom: '14px' });
    box.appendChild(el('h4', { text: title }, { fontSize: '13px', fontWeight: '640', margin: '0 0 6px' }));
    return box;
  }
  function toast(message) {
    if (typeof window.showToast === 'function') window.showToast(message);
  }

  /* ── server calls ─────────────────────────────────────────────── */
  function getStatus() {
    return fetch(API, { credentials: 'same-origin' }).then(function (r) {
      return r.ok ? r.json().catch(function () { return null; }) : null;
    }).catch(function () { return null; });
  }
  function send(path, method, body) {
    return fetch(API + path, { method: method, credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (data) {
          return { ok: r.ok, status: r.status, data: data };
        });
      }).catch(function () {
        return { ok: false, status: 0, data: { error: 'Could not reach SLM. Try again in a moment.' } };
      });
  }
  function why(result) {
    if (result.status === 403) return 'Ask an admin to change this.';
    var d = result.data || {};
    return (typeof d.error === 'string' && d.error) ||
      (typeof d.detail === 'string' && d.detail) || 'That did not work. Try again in a moment.';
  }

  /* ── what the saved settings are, and what the form says ──────── */
  function effectiveMode(s) {
    if (s.mode === 'laya' || s.mode === 'jev' || s.mode === 'off') return s.mode;
    return s.active === 'laya' ? 'laya' : 'off';
  }
  function savedForm(s) {
    var jev = s.jev || {};
    // "auto" (never chosen) is its own state in the form, not "Off": showing
    // it as Off next to "Not chosen yet" contradicted itself.
    return { mode: s.mode === 'auto' ? 'auto' : effectiveMode(s),
             provider: jev.provider || 'typesafe', key: '',
             consent: !!jev.consent, rerank: !!(jev.rerank && jev.rerank.enabled) };
  }
  function isDirty() {
    if (!status || !draft) return false;
    var saved = savedForm(status);
    return draft.mode !== saved.mode || draft.provider !== saved.provider ||
      draft.consent !== saved.consent || draft.rerank !== saved.rerank || !!draft.key.trim();
  }
  function layaReady() { return status.laya.state === 'ready'; }
  function jevKeySaved() {
    return !!status.jev.has_key && status.jev.provider === draft.provider;
  }
  var NAMES = { laya: 'On this Mac', jev: 'Online with Jev', off: 'Off', auto: 'Automatic' };
  function runningText(s) {
    if (s.active === 'laya') return 'On this Mac — your memories never leave it.';
    if (s.active === 'jev') return 'Online with Jev (' + providerLabel(s.jev.provider) + ').';
    return 'Nothing — answers are not checked.';
  }
  function providerLabel(p) { return p === 'openrouter' ? 'OpenRouter' : 'TypeSafe'; }

  /* ── readiness, in plain words, with the action that fixes it ── */
  function layaReadiness() {
    var laya = status.laya;
    if (!status.apple_silicon) return { ready: false, text: 'Needs a Mac with Apple Silicon. Choose Online with Jev instead.' };
    if (status.execution && status.execution.state !== 'ready' && status.active === 'laya') return { ready: false, text: status.execution.detail, action: 'check' };
    if (laya.state === 'ready') return { ready: true, text: 'Ready.' };
    if (laya.state === 'installing') return { ready: false, text: 'Being set up — you can Save once it says Ready.' };
    if (laya.action === 'check') return { ready: false, text: 'SLM found an install it hasn’t checked yet. Choose “Check this install” below — nothing is downloaded.', action: 'check' };
    if (laya.action === 'setup') return { ready: false, text: 'The setup didn’t finish. Choose “Repair” below, or “Use an install you already have”.', action: 'repair' };
    return { ready: false, text: 'Not set up yet. Choose “Set up” below (downloads about 1.1 GB), or “Use an install you already have”.', action: 'setup' };
  }
  function jevReadiness() {
    if (!jevKeySaved() && !draft.key.trim()) return { ready: false, text: 'Needs your Jev key — paste it under Online with Jev below.', action: 'key' };
    if (!draft.consent) return { ready: false, text: 'Needs you to tick the notice about what is sent — under Online with Jev below.', action: 'consent' };
    return { ready: true, text: draft.key.trim() ? 'Ready once you Save.' : 'Ready.' };
  }

  /* ── render ───────────────────────────────────────────────────── */
  function render() {
    if (!mountEl || !status) return;
    var focused = document.activeElement && mountEl.contains(document.activeElement)
      ? document.activeElement.getAttribute('data-ac') : null;
    mountEl.innerHTML = '';
    mountEl.appendChild(el('p', { text: 'Lets your memory say “I don’t have that” instead of guessing. One option runs at a time.' },
      { fontSize: '12.5px', color: 'var(--fg-2)', margin: '0 0 10px' }));
    var running = el('div', { class: 'ac-running', role: 'status' },
      { fontSize: '13px', marginBottom: '12px' });
    running.appendChild(el('strong', { text: 'Running now: ' }));
    running.appendChild(document.createTextNode(runningText(status)));
    mountEl.appendChild(running);
    if (!canManage) mountEl.appendChild(note('You can look, but only an admin can change this.', 'bad'));
    mountEl.appendChild(buildChoices());
    mountEl.appendChild(buildLaya());
    mountEl.appendChild(buildJev());
    mountEl.appendChild(buildSaveBar());
    if (focused) {
      var again = mountEl.querySelector('[data-ac="' + focused + '"]');
      if (again) again.focus();
    }
  }

  function buildChoices() {
    var wrap = el('fieldset', { class: 'ac-choices' },
      { border: '0', padding: '0', margin: '0 0 10px' });
    wrap.appendChild(el('legend', { text: 'Check answers' }, { fontSize: '12px', color: 'var(--fg-2)', marginBottom: '6px' }));
    if (draft.mode === 'auto') wrap.appendChild(buildAutomaticState());
    [['laya', 'On this Mac — private (recommended)'], ['jev', 'Online with Jev'], ['off', 'Off']]
      .forEach(function (opt) {
        var label = el('label', null, { display: 'flex', alignItems: 'baseline', gap: '8px',
          marginBottom: '6px', fontSize: '13.5px', cursor: canManage ? 'pointer' : 'not-allowed' });
        var radio = el('input', { type: 'radio', name: 'ac-mode', value: opt[0], 'data-ac': 'mode-' + opt[0] });
        radio.checked = draft.mode === opt[0];
        radio.disabled = !canManage;
        radio.addEventListener('change', function () { choose(opt[0]); });
        label.appendChild(radio);
        var words = el('span');
        words.appendChild(document.createTextNode(opt[1]));
        var state = choiceState(opt[0]);
        if (state) {
          words.appendChild(el('span', { text: ' ' + state.text, class: 'ac-choice-state' },
            { fontSize: '12px', color: state.ready ? 'var(--success, #1a7f37)' : 'var(--fg-3)' }));
        }
        label.appendChild(words);
        wrap.appendChild(label);
      });
    var chosen = draft.mode === 'laya' ? layaReadiness() : draft.mode === 'jev' ? jevReadiness() : null;
    if (chosen && !chosen.ready) wrap.appendChild(buildGuidance(chosen));
    return wrap;
  }

  /* Nothing chosen yet: say what Automatic is doing now, with no radio
     claiming a choice the person never made. */
  function buildAutomaticState() {
    var box = el('div', { class: 'ac-auto-state', role: 'status' },
      { fontSize: '13.5px', marginBottom: '8px' });
    box.appendChild(el('strong', { text: 'Automatic' }));
    box.appendChild(document.createTextNode(status.active === 'laya'
      ? ' — checking on this Mac. It turned on by itself once it was set up and checked.'
      : ' — not checking yet. On this Mac turns on by itself once it is set up and checked.'));
    box.appendChild(el('div', { text: 'Choose an option below to decide yourself.' },
      { fontSize: '12px', color: 'var(--fg-3)' }));
    return box;
  }

  /* The short state shown beside each choice: always what it is now. */
  function choiceState(mode) {
    if (mode === 'laya') {
      var laya = status.laya;
      if (!status.apple_silicon) return { text: '— needs a Mac with Apple Silicon' };
      if (laya.state === 'ready') return { ready: true, text: '✓ Ready' };
      if (laya.state === 'installing') return { text: '— being set up below' };
      if (laya.action === 'check') return { text: '— check it below' };
      if (laya.action === 'setup') return { text: '— repair it below' };
      return { text: '— set it up below' };
    }
    if (mode === 'jev') {
      if (!jevKeySaved() && !draft.key.trim()) return { text: '— add your key to use it' };
      if (!draft.consent) return { text: '— tick the notice below to use it' };
      if (!jevKeySaved()) return { text: '— Save to use it' };
      return { ready: true, text: '✓ Ready' };
    }
    return null;
  }
  /* Paths with the home folder shown as "~". */
  function homeShort(path) {
    return String(path || '').replace(/^\/(Users|home)\/[^/]+(?=\/|$)/, '~');
  }

  function buildGuidance(ready) {
    var box = el('div', { class: 'ac-guidance', role: 'note' }, { background: 'var(--violet-soft)',
      border: '1px solid var(--violet-line)', borderRadius: 'var(--r-md)', padding: '10px 12px',
      margin: '6px 0 4px', fontSize: '12.5px' });
    box.appendChild(el('div', { text: 'To use ' + NAMES[draft.mode] + ': ' + ready.text }));
    var actions = [];
    if (ready.action === 'check') actions.push(['Check this install', layaCheck]);
    if (ready.action === 'repair') actions.push(['Repair', layaSetup]);
    if (ready.action === 'setup') actions.push(['Set up', layaSetup]);
    if (ready.action === 'repair' || ready.action === 'setup') {
      actions.push(['Use an install you already have', function () { ui.existingOpen = true; render(); focusAc('existing-python'); }]);
    }
    if (ready.action === 'key') actions.push(['Go to the key box', function () { focusAc('jev-key'); }]);
    if (ready.action === 'consent') actions.push(['Show the notice', function () { focusAc('jev-consent'); }]);
    var r = row();
    actions.forEach(function (a, i) {
      var b = btn(a[0], i === 0, 'guide-' + i);
      b.addEventListener('click', a[1]);
      r.appendChild(b);
    });
    if (actions.length) box.appendChild(r);
    return box;
  }
  function focusAc(id) {
    var target = mountEl && mountEl.querySelector('[data-ac="' + id + '"]');
    if (!target) return;
    if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'center' });
    target.focus();
  }

  function choose(mode) {
    draft.mode = mode;
    savedNote = ''; formError = '';
    render();
  }

  /* ── On this Mac ─────────────────────────────────────────────── */
  function layaHeadline(laya) {
    if (status.active === 'laya' && status.execution && status.execution.state !== 'ready') return status.execution.detail;
    if (contactLost && laya.state === 'installing') return 'Lost contact with SLM, so progress is unknown. Reload the page.';
    switch (laya.state) {
      case 'ready': return 'Ready' + (laya.managed ? '.' : ' — using an install you already had.');
      case 'installing': return (laya.step || 'Setting up') + '… ' + Math.round((laya.progress || 0) * 100) + '%';
      case 'unsupported': return 'Needs a Mac with Apple Silicon.';
      case 'failed': return laya.error || 'It needs attention.';
      default: return 'Not set up yet.';
    }
  }
  function buildLaya() {
    var laya = status.laya;
    var box = card('On this Mac', 'ac-laya');
    // Found-but-unchecked is a next step, not an error: no red for it.
    var tone = laya.state === 'ready' && (!status.execution || status.execution.state === 'ready') ? 'ok'
      : laya.state === 'failed' && !/hasn.t checked it yet/.test(laya.error || '') ? 'bad' : '';
    box.appendChild(note(layaHeadline(laya), tone, 'ac-laya-state'));
    if (laya.state === 'installing' && !contactLost) {
      var track = el('div', { role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100',
        'aria-valuenow': String(Math.round((laya.progress || 0) * 100)) },
        { height: '8px', borderRadius: 'var(--r-pill)', background: 'var(--card-2)',
          border: '1px solid var(--border)', overflow: 'hidden', margin: '8px 0', maxWidth: '320px' });
      track.appendChild(el('div', null, { height: '100%', width: Math.round((laya.progress || 0) * 100) + '%',
        background: 'var(--violet)' }));
      box.appendChild(track);
    }
    if (status.adopt && !status.adopt.running && status.adopt.state === 'failed' && status.adopt.error &&
        laya.state !== 'ready') {
      box.appendChild(note('Last check of an install: ' + status.adopt.error, 'bad', 'ac-adopt-error'));
    }
    var buttons = row();
    if (laya.state === 'installing') {
      if (status.setup_running) {
        var cancel = btn('Cancel setup', false, 'laya-cancel');
        cancel.addEventListener('click', layaCancel);
        buttons.appendChild(cancel);
        buttons.appendChild(note('What is downloaded so far is kept.', 'dim'));
      }
    } else if (status.apple_silicon) {
      if (laya.state === 'not_installed') buttons.appendChild(onClick(btn('Set up', true, 'laya-setup'), layaSetup));
      if (laya.action === 'check') buttons.appendChild(onClick(btn('Check this install', true, 'laya-check'), layaCheck));
      if (laya.action === 'setup') buttons.appendChild(onClick(btn('Repair', true, 'laya-repair'), layaSetup));
      if (laya.state === 'ready') buttons.appendChild(onClick(btn(status.laya_test_running ? 'Testing…' : 'Test', false, 'laya-test'), layaTest));
      if (laya.state !== 'not_installed') {
        var external = !laya.managed;
        buttons.appendChild(onClick(btn(external ? 'Forget this install' : 'Remove', false, 'laya-remove'),
          function () { layaRemove(external); }));
      }
    }
    appendCalibration(box, buttons);
    box.appendChild(buttons);
    if (status.laya_test_running) box.appendChild(note('Testing — this loads the model once and asks it one question.', 'dim'));
    box.appendChild(testResult(status.tests && status.tests.laya, 'ac-laya-test'));
    if (status.apple_silicon && laya.state !== 'installing') box.appendChild(buildExisting(laya));
    box.appendChild(buildDetails(laya));
    return box;
  }
  function appendCalibration(box, buttons) {
    var calibration = status.calibration || { state: 'idle' };
    if (status.active === 'laya') {
      var calibrating = /^(running|warming|cancelling)$/.test(calibration.state);
      var calibrate = btn(calibration.state === 'running' ? 'Calibrating…' : 'Calibrate on my store', false, 'calibrate');
      calibrate.disabled = !canManage || calibrating || (status.execution && status.execution.state !== 'ready');
      calibrate.addEventListener('click', function () {
        send('/calibration', 'POST', { model: 'llama3.2:latest', samples: 32 }).then(function (result) {
          if (!result.ok) { toast(why(result)); return; }
          status.calibration = result.data; schedulePoll(); render();
        });
      });
      buttons.appendChild(calibrate);
      if (calibrating) {
        var cancelCalibration = btn('Cancel calibration', false, 'calibration-cancel');
        cancelCalibration.disabled = !canManage || calibration.state === 'cancelling';
        cancelCalibration.addEventListener('click', function () {
          send('/calibration/cancel', 'POST', {}).then(function (result) {
            if (!result.ok) { toast(why(result)); return; }
            status.calibration = result.data; schedulePoll(); render();
          });
        });
        buttons.appendChild(cancelCalibration);
      }
      box.appendChild(note('Calibration ' + calibration.state + ': ' + (calibration.done || 0) + '/' + (calibration.total || 0) + '. ' + (calibration.detail || 'Questions and labels stay on this machine; uses your installed GPU model.'), 'dim'));
    }
  }
  function onClick(b, fn) { b.addEventListener('click', fn); return b; }

  function testResult(result, cls) {
    if (!result) return el('div', { class: cls });
    var when = '';
    try { when = new Date(result.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); } catch (e) { when = ''; }
    var secs = result.seconds == null ? '' : ' — answered in ' + result.seconds + ' s';
    return note(result.ok ? 'Tested ✓ at ' + when + secs : 'Test failed at ' + when + ': ' + result.message,
      result.ok ? 'ok' : 'bad', cls);
  }

  function buildExisting(laya) {
    var d = el('details', { class: 'ac-existing' }, { marginTop: '10px' });
    d.open = ui.existingOpen;
    d.addEventListener('toggle', function () { ui.existingOpen = d.open; });
    d.appendChild(el('summary', { text: 'Use an install you already have' },
      { fontSize: '12.5px', color: 'var(--fg-2)', cursor: 'pointer' }));
    if (!ui.python && laya.python && !laya.managed) {
      ui.python = laya.python; ui.model = laya.model_path || ''; ui.cache = laya.hf_home || '';
    }
    d.appendChild(note('For networks that block the download. Point SLM at the Python program and model folder of an install you made earlier.', 'dim'));
    var py = input('existing-python', 'Python program, e.g. /Users/you/.local/share/laya-venv/bin/python');
    var model = input('existing-model', 'Model folder (optional if the cache folder is given)');
    var cache = input('existing-cache', 'Model cache folder (optional)');
    py.value = ui.python; model.value = ui.model; cache.value = ui.cache;
    py.addEventListener('input', function () { ui.python = py.value; });
    model.addEventListener('input', function () { ui.model = model.value; });
    cache.addEventListener('input', function () { ui.cache = cache.value; });
    d.appendChild(row(py)); d.appendChild(row(model)); d.appendChild(row(cache));
    var use = btn(status.adopt && status.adopt.running ? 'Checking…' : 'Check and use it', true, 'existing-use');
    use.disabled = !canManage || !!(status.adopt && status.adopt.running);
    use.addEventListener('click', function () {
      if (!ui.python.trim()) { formError = ''; toast('Enter the Python program first.'); focusAc('existing-python'); return; }
      adopt(ui.python.trim(), ui.cache.trim(), ui.model.trim());
    });
    d.appendChild(row(use));
    return d;
  }

  function buildDetails(laya) {
    var d = el('details', { class: 'ac-details' }, { marginTop: '8px' });
    d.open = ui.detailsOpen;
    d.addEventListener('toggle', function () { ui.detailsOpen = d.open; });
    d.appendChild(el('summary', { text: 'Details' }, { fontSize: '12px', color: 'var(--fg-3)', cursor: 'pointer' }));
    var lines = [
      'Status: ' + laya.state + (laya.step ? ' — ' + laya.step : ''),
      'Installed by SLM: ' + (laya.managed ? 'yes' : 'no'),
      'Python program: ' + (homeShort(laya.python) || '—'),
      'Model folder: ' + (homeShort(laya.model_path) || '—'),
      'Model cache folder: ' + (homeShort(laya.hf_home) || '—'),
    ];
    lines.forEach(function (t) {
      d.appendChild(el('div', { text: t, class: 'mono' }, { fontSize: '11.5px', color: 'var(--fg-2)', wordBreak: 'break-all' }));
    });
    return d;
  }

  function layaSetup() {
    send('/laya/setup', 'POST').then(function (r) {
      if (!r.ok) toast(why(r));
      refresh();
    });
  }
  function layaCheck() {
    var laya = status.laya;
    adopt(laya.python, laya.hf_home || '', laya.model_path || '');
  }
  function adopt(python, cache, model) {
    send('/laya/adopt', 'POST', { python: python, hf_home: cache, model_path: model }).then(function (r) {
      if (!r.ok) { ui.existingOpen = true; toast(why(r)); refresh(); return; }
      pendingAdopt = true;
      refresh();
    });
  }
  function layaCancel() {
    send('/laya/cancel', 'POST').then(function (r) { if (!r.ok) toast(why(r)); refresh(); });
  }
  function layaTest() {
    if (status.laya_test_running) return;
    send('/laya/test', 'POST').then(function (r) { if (!r.ok) toast(why(r)); refresh(); });
  }
  function layaRemove(external) {
    var question = external
      ? 'Stop using this install? Its files stay where they are.'
      : 'Remove the on-device check from this Mac? You can set it up again later.';
    if (!window.confirm(question)) return;
    send('/laya/remove', 'POST').then(function (r) {
      if (!r.ok) toast(why(r));
      ui.python = ''; ui.model = ''; ui.cache = '';
      refresh();
    });
  }

  /* ── Online with Jev ─────────────────────────────────────────── */
  // Approved wording — do not edit (tests/test_server/test_answer_check_rerank_api.py).
  var CONSENT_TEXT = 'When on, each recall sends your question and its top ' +
    '3 memories to {provider}. Don’t use this for client, confidential ' +
    'or personal material.';
  var RERANK_CONSENT_TEXT =
    "When on, each recall sends your question and its top {k} memories to {provider} to choose the best order. Don't use this for client, confidential or personal material.";

  function buildJev() {
    var jev = status.jev || {};
    var box = card('Online with Jev', 'ac-jev');
    var provider = el('select', { 'data-ac': 'jev-provider', 'aria-label': 'Provider' });
    Object.assign(provider.style, { height: '34px', borderRadius: 'var(--r-md)', border: '1px solid var(--border)',
      background: 'var(--card-2)', color: 'var(--fg)', padding: '0 10px', fontSize: '13px' });
    [['typesafe', 'TypeSafe'], ['openrouter', 'OpenRouter']].forEach(function (p) {
      var o = el('option', { value: p[0], text: p[1] });
      o.selected = draft.provider === p[0];
      provider.appendChild(o);
    });
    provider.disabled = !canManage;
    provider.addEventListener('change', function () { draft.provider = provider.value; savedNote = ''; render(); });
    var saved = jevKeySaved();
    var key = input('jev-key', saved ? 'Key saved (' + jev.key_hint + ') — paste a new one to replace it' : 'Paste your key', 'password');
    key.value = draft.key;
    key.setAttribute('autocomplete', 'off');
    key.addEventListener('input', function () {
      var had = !!draft.key.trim();
      draft.key = key.value; savedNote = '';
      if (had !== !!draft.key.trim()) render(); else markDirty();
    });
    box.appendChild(row(provider, key));
    if (jev.key_problem) box.appendChild(note(jev.key_problem, 'bad'));
    if (draft.key.trim()) box.appendChild(note('Not saved yet — choose Save below.', 'dim', 'ac-key-unsaved'));

    var test = btn('Test connection', false, 'jev-test');
    test.addEventListener('click', jevTest);
    var r = row(test);
    if (saved) r.appendChild(onClick(btn('Remove key', false, 'jev-remove'), jevRemoveKey));
    else if (jev.key_problem) r.appendChild(onClick(btn('Remove key', false, 'jev-remove'), jevRemoveKey));
    box.appendChild(r);
    if (!saved || draft.key.trim()) {
      box.appendChild(note(draft.key.trim() ? 'Test uses the saved key — Save first, then Test.'
        : 'Test needs a saved key — paste one and Save.', 'dim'));
    } else {
      box.appendChild(note('A test sends one small request to ' + providerLabel(draft.provider) + ' and uses a little of your credit.', 'dim'));
    }
    box.appendChild(testResult(jev.provider === draft.provider ? status.tests && status.tests.jev : null, 'ac-jev-test'));

    box.appendChild(checkbox('jev-consent', CONSENT_TEXT.replace('{provider}', providerLabel(draft.provider)),
      draft.consent, function (on) {
        draft.consent = on;
        if (!on) draft.rerank = false;
        savedNote = ''; render();
      }));
    var k = (jev.rerank && jev.rerank.k) || 20;
    var reorder = checkbox('jev-rerank', 'Also use Jev to reorder results. ' +
      RERANK_CONSENT_TEXT.replace('{k}', String(k)).replace('{provider}', providerLabel(draft.provider)),
      draft.rerank, function (on) {
        if (on && !draft.consent) { draft.rerank = false; render(); toast('Tick the notice above first.'); return; }
        // A Jev option must never look on while a different judge is chosen —
        // that was a silent state where it could never actually run. Ticking
        // it while e.g. On this Mac is selected offers the real fix: switch
        // the judge. Declining leaves the previous judge selected and the box
        // unticked; there is no middle state.
        if (on && draft.mode !== 'jev') {
          if (!window.confirm('Switch to Jev? ' + NAMES[draft.mode] +
              ' would stop checking answers; Jev would take over, with reordering on.')) {
            draft.rerank = false; render(); return;
          }
          draft.mode = 'jev';
        }
        draft.rerank = on; savedNote = ''; render();
      });
    box.appendChild(reorder);
    box.appendChild(note('Ticking a box above accepts its notice. Reordering works only while Jev is chosen.', 'dim'));
    if (!draft.consent) box.appendChild(note('Reordering needs the notice above ticked first.', 'dim'));
    else if (draft.rerank && draft.mode !== 'jev') {
      box.appendChild(note('Reordering with Jev is off while ' + NAMES[draft.mode] + ' is chosen. Your choice is kept for when you choose Jev again.', 'dim', 'ac-rerank-paused'));
    }
    box.appendChild(buildLadder(jev));
    return box;
  }

  /* Background judges (4.2.30 W2, D-20): their own consent, given here or with
     `slm consent give` at a terminal. Acts at once; no Save needed. */
  function buildLadder(jev) {
    var ladder = jev.ladder || {};
    var wrap = el('div', { class: 'ac-ladder' }, { marginTop: '14px', borderTop: '1px solid var(--border)', paddingTop: '10px' });
    wrap.appendChild(el('div', { text: 'Background judges' }, { fontWeight: '600', fontSize: '13px' }));
    var ready = status.mode === 'jev' && !!jev.consent;
    if (!ready) {
      wrap.appendChild(note('Turn on the online answer check with Jev and save first: the background judges use the same connection.', 'dim', 'ac-ladder-needs'));
      return wrap;
    }
    wrap.appendChild(note(ladder.notice || '', null, 'ac-ladder-notice'));
    if (ladder.consent) {
      wrap.appendChild(note('Allowed' + (ladder.since ? ' since ' + String(ladder.since).slice(0, 10) : '') + '.', 'ok', 'ac-ladder-state'));
      if (ladder.flag !== 'on') {
        wrap.appendChild(note('They only watch for now: they will not ask Jev until they are switched on (slm flags).', 'dim', 'ac-ladder-watch'));
      }
      wrap.appendChild(row(onClick(btn('Stop', false, 'ac-ladder-stop'), function () { ladderConsent(false); })));
    } else {
      wrap.appendChild(row(onClick(btn('I agree', false, 'ac-ladder-allow'), function () { ladderConsent(true); })));
    }
    return wrap;
  }
  function ladderConsent(on) {
    send('/jev/ladder', 'POST', { accepted: on, provider: (status.jev || {}).provider || draft.provider })
      .then(function (r) {
        if (!r.ok) toast(why(r));
        refresh(true);
      });
  }
  function checkbox(id, text, checked, onChange) {
    var label = el('label', null, { display: 'flex', gap: '8px', alignItems: 'flex-start', marginTop: '10px',
      fontSize: '12.5px', color: 'var(--fg)', cursor: canManage ? 'pointer' : 'not-allowed' });
    var box = el('input', { type: 'checkbox', 'data-ac': id }, { marginTop: '2px' });
    box.checked = checked;
    box.disabled = !canManage;
    box.addEventListener('change', function () { onChange(box.checked); });
    label.appendChild(box);
    label.appendChild(el('span', { text: text }));
    return label;
  }
  function jevTest() {
    if (!jevKeySaved() || draft.key.trim()) {
      toast(draft.key.trim() ? 'Save the key first, then Test.' : 'Paste your key and Save first.');
      focusAc(draft.key.trim() ? 'ac-save' : 'jev-key');
      return;
    }
    var b = mountEl.querySelector('[data-ac="jev-test"]');
    if (b) { b.disabled = true; b.textContent = 'Testing…'; }
    send('/jev/test', 'POST', { provider: draft.provider }).then(function (r) {
      if (!r.ok) toast(why(r));
      refresh();
    });
  }
  function jevRemoveKey() {
    if (!window.confirm('Remove the saved key for ' + providerLabel(draft.provider) + '? Jev turns off until you add one again.')) return;
    send('/jev/key', 'DELETE', { provider: draft.provider }).then(function (r) {
      if (!r.ok) toast(why(r));
      refresh(true);
    });
  }

  /* ── Save ─────────────────────────────────────────────────────── */
  function buildSaveBar() {
    var bar = el('div', { class: 'ac-savebar' }, { display: 'flex', gap: '10px', alignItems: 'center',
      flexWrap: 'wrap', paddingTop: '4px' });
    var save = btn('Save', true, 'ac-save');
    save.addEventListener('click', saveAll);
    bar.appendChild(save);
    var state = el('span', { class: 'ac-save-state', role: 'status', 'aria-live': 'polite' }, { fontSize: '12.5px' });
    if (formError) { state.textContent = formError; state.style.color = 'var(--danger)'; }
    else if (isDirty()) { state.textContent = '● Unsaved changes'; state.style.color = 'var(--warning, #b26b00)'; }
    else if (savedNote) { state.textContent = savedNote; state.style.color = 'var(--success, #1a7f37)'; }
    bar.appendChild(state);
    return bar;
  }
  function markDirty() {
    var state = mountEl && mountEl.querySelector('.ac-save-state');
    if (state && isDirty() && !formError) { state.textContent = '● Unsaved changes'; state.style.color = 'var(--warning, #b26b00)'; }
  }
  function saveAll() {
    var saved = savedForm(status);
    var leavingJev = saved.mode === 'jev' && draft.mode !== 'jev';
    if (leavingJev && saved.rerank && !window.confirm('Reordering with Jev turns off while ' +
        NAMES[draft.mode] + ' is selected. It comes back when you choose Jev again. Save?')) return;
    var running = effectiveMode(status);
    if (running !== draft.mode && running !== 'off' && draft.mode !== 'off' && draft.mode !== 'auto' &&
        !leavingJev && !window.confirm('Only one option runs at a time: this turns ' + NAMES[running] + ' off. Save?')) return;
    var body = { provider: draft.provider, key: draft.key.trim(), consent: draft.consent, rerank: draft.rerank };
    // The never-chosen start state stays as it is unless the person picked something.
    if (draft.mode !== 'auto') body.mode = draft.mode;
    var b = mountEl.querySelector('[data-ac="ac-save"]');
    if (b) { b.disabled = true; b.textContent = 'Saving…'; }
    send('/save', 'POST', body).then(function (r) {
      if (!r.ok) {
        formError = why(r); savedNote = '';
        render();
        return;
      }
      formError = '';
      savedNote = '✓ ' + (r.data.message || 'Saved.');
      draft.key = '';
      status = r.data;
      draft = savedForm(status);
      render();
    });
  }

  /* ── polling, refresh, bootstrap ──────────────────────────────── */
  function busy(s) {
    return !!(s.execution && s.execution.state === 'warming') || !!(s.calibration && /^(running|warming|cancelling)$/.test(s.calibration.state)) || s.laya.state === 'installing' || !!(s.adopt && s.adopt.running) || !!s.laya_test_running;
  }
  function stopPoll() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }
  function schedulePoll() {
    if (pollTimer) return;
    pollFailures = 0;
    pollTimer = setInterval(function () {
      getStatus().then(function (s) {
        if (!pollTimer) return;
        if (!s || !s.laya) {
          pollFailures += 1;
          if (pollFailures >= POLL_MAX_FAILURES) { stopPoll(); contactLost = true; render(); }
          return;
        }
        pollFailures = 0;
        accept(s, false);
        if (!busy(s)) stopPoll();
      });
    }, POLL_MS);
  }
  /* Take a fresh status. The person's unsaved edits are kept; only fields
     they have not touched follow the server. */
  function accept(s, resetDraft) {
    var before = status ? savedForm(status) : null;
    var dirtyBefore = isDirty();
    status = s;
    contactLost = false;
    var now = savedForm(s);
    if (!draft || resetDraft || !dirtyBefore) draft = Object.assign(now, { key: draft ? draft.key : '' });
    else if (before) Object.keys(now).forEach(function (k) {
      if (k !== 'key' && draft[k] === before[k]) draft[k] = now[k];
    });
    if (pendingAdopt && !(s.adopt && s.adopt.running)) {
      pendingAdopt = false;
      toast(s.laya.state === 'ready' ? 'That install works — it is ready to use.' : (s.adopt.error || 'That install could not be used.'));
      if (s.laya.state === 'ready') ui.existingOpen = false;
    }
    render();
  }
  function refresh(resetDraft) {
    if (!mountEl) return;
    getStatus().then(function (s) {
      if (!s || !s.laya) { if (status) render(); return; }
      accept(s, !!resetDraft);
      if (busy(s)) schedulePoll();
    });
  }
  function applyPermission(whoami) {
    var permissions = (whoami && whoami.permissions) || [];
    canManage = permissions.indexOf('manage') !== -1;
  }
  window.addEventListener('beforeunload', function (e) {
    if (!isDirty()) return undefined;
    e.preventDefault();
    e.returnValue = 'You have unsaved answer-check changes.';
    return e.returnValue;
  });

  function renderInto(container) {
    if (!container) return;
    mountEl = container;
    fetch('/api/rbac/whoami', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; })
      .then(applyPermission)
      .then(function () { refresh(true); });
  }

  // effectiveMode and isDirty are exposed for the UI tests.
  window.SLMAnswerCheck = { renderInto: renderInto, effectiveMode: effectiveMode,
                            isDirty: isDirty };
}());
