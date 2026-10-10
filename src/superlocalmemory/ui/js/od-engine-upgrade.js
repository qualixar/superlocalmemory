/* od-engine-upgrade.js — "Upgrade memory engine", in the Embeddings group of Settings (4.1.25).
 *
 * Moves an existing store to the newer memory engine (the model that also reads pictures
 * and documents). The work is the ordinary background switch: this card shows the plan
 * (what changes, memory, disk, time), starts it, and hands the 202 body to the shared
 * progress panel (od-reindex.js), which shows progress, Cancel, and polls. Afterwards the
 * card offers one click back (Roll back) and "Free the old data".
 *
 * Exposes window.odEngineUpgrade:
 *   panel(post)  element for the Embeddings group; post(url, body) is the caller's
 *                authenticated POST, resolving to a Response and rejecting (Error with
 *                .status and .body) on non-2xx
 *   refresh()    read the plan and the previous-engine state once
 * Every server string is set with textContent, never parsed as HTML.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var API = '/api/v3/embedding/reindex';
  var NEW_MODEL = 'google/embeddinggemma-2';
  var POLL_MS = 2000;

  var _post = null, _root = null, _body = null, _gen = 0, _busy = false;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = String(text);
    return e;
  }

  function button(label, cls, onClick) {
    var b = el('button', cls || 'btn sm', label);
    b.type = 'button';
    b.addEventListener('click', function () { onClick(b); });
    return b;
  }

  function size(mb) {
    mb = Number(mb) || 0;
    return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB';
  }

  function costText(p) {
    return p.memories + ' memories, ' + (p.minutes_label || 'a few minutes') + ', about ' + size(p.ram_mb)
      + ' of memory while it works, about ' + size(p.disk_mb) + ' of disk (the previous engine\'s data is kept until you free it).';
  }

  function confirmThen(opts, go) {
    if (typeof window.confirmDestructive !== 'function') { go(); return; }
    window.confirmDestructive(opts).then(function (yes) { if (yes) go(); });
  }

  function message(text, isError) {
    var m = el('p', isError ? 'od-engine-upgrade-error' : 'muted', text);
    m.setAttribute('role', 'status');
    if (isError) m.style.color = 'var(--danger)';
    return m;
  }

  function failText(e, fallback) {
    var d = e && e.body && (typeof e.body.detail === 'string' ? e.body.detail : e.body.error);
    return (e && e.message) || d || fallback;
  }

  function api(url) {
    return fetch(url).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: !!r.ok, status: r.status, data: data || {} };
      });
    }, function () { return { ok: false, status: 0, data: {} }; });
  }

  // -- actions ---------------------------------------------------------------

  function start(plan, note, goBtn) {
    confirmThen({
      title: 'Upgrade memory engine', target: 'Memory engine',
      consequence: plan.explain + ' ' + costText(plan), confirmLabel: 'Upgrade',
    }, function () {
      _busy = true;
      goBtn.disabled = true;
      note.textContent = 'Starting…';
      _post(API + '/upgrade', {}).then(function (r) { return r.json(); }).then(function (body) {
        note.textContent = 'Started: your memories are being re-read in the background.';
        if (window.odReindex) window.odReindex.track(body);
      }).catch(function (e) {
        _busy = false;
        goBtn.disabled = false;
        note.textContent = failText(e, 'Could not start the upgrade.');
        if (e && e.body && e.body.error === 'reindex_running' && window.odReindex) window.odReindex.conflict(e.body);
      });
    });
  }

  // The Documents & Images pane owns the turn-on flow (memory check, confirmation, progress).
  function openMediaPane() {
    if (typeof window.slmNavigate === 'function') window.slmNavigate('media-pane');
  }

  // The reason as a person reads it: the server appends the command for developers.
  function plainReason(plan) {
    var text = String(plan.reason || '');
    var cmd = plan.turn_on_command;
    if (!cmd || text.indexOf(cmd) < 0) return text;
    text = text.split(cmd).join('').replace(/\s*(Run)?\s*:\s*$/, '').trim();
    return /[.!?]$/.test(text) ? text : text + '.';
  }

  function rollBack(note) {
    confirmThen({
      title: 'Roll back the memory engine', target: 'Memory engine',
      consequence: 'Goes back to the previous engine. Recall keeps working while it does.', confirmLabel: 'Roll back',
    }, function () {
      note.textContent = 'Starting…';
      _post(API + '/rollback', {}).then(function (r) { return r.json(); }).then(function (body) {
        note.textContent = 'Rolling back in the background.';
        if (window.odReindex) window.odReindex.track(body);
      }).catch(function (e) { note.textContent = failText(e, 'Could not roll back.'); });
    });
  }

  function freeOld(note) {
    confirmThen({
      title: 'Free the old data', target: 'Previous engine data',
      consequence: 'Deletes the previous engine\'s data. You can no longer roll back after this.', confirmLabel: 'Free it',
    }, function () {
      _post(API + '/forget-previous', {}).then(function () { return refresh(); })
        .catch(function (e) { note.textContent = failText(e, 'Could not free the old data.'); });
    });
  }

  // -- rendering -------------------------------------------------------------

  function head() {
    var h = el('div');
    var badge = el('span', 'badge', 'Preview');
    badge.style.marginRight = '8px';
    h.appendChild(badge);
    var t = el('b', null, 'Upgrade memory engine');
    h.appendChild(t);
    return h;
  }

  function onNewEngine(status) {
    return !!(status && status.previous_vectors_kept && String(status.live || '').indexOf(NEW_MODEL) === 0);
  }

  function actions(plan, status, note) {
    var row = el('div');
    row.style.marginTop = '6px';
    if (!plan.already) {
      var go = button('Upgrade memory engine', 'btn sm primary', function (b) { start(plan, note, b); });
      go.disabled = !plan.available || _busy;
      row.appendChild(go);
    }
    if (plan.needs_media) {
      row.appendChild(button('Turn on images & documents first', 'btn sm', openMediaPane));
    }
    if (onNewEngine(status)) {
      row.appendChild(button('Roll back', 'btn sm', function () { rollBack(note); }));
      row.appendChild(button('Free the old data', 'btn sm ghost', function () { freeOld(note); }));
    }
    return row;
  }

  function render(plan, status) {
    _body.textContent = '';
    _root.style.display = '';
    _body.appendChild(head());
    _body.appendChild(el('p', 'muted', plan.explain));
    if (!plan.already) {
      _body.appendChild(el('p', 'muted', 'From ' + plan.from.model + ' to ' + plan.to.model + '. ' + costText(plan)));
    }
    if (plan.reason) _body.appendChild(message(plainReason(plan), false));
    var note = message('', false);
    _body.appendChild(actions(plan, status, note));
    _body.appendChild(note);
  }

  function renderProblem(res) {
    if (res.status === 404) { _root.style.display = 'none'; return; }
    _body.textContent = '';
    _root.style.display = '';
    _body.appendChild(head());
    var d = res.data || {};
    var text = typeof d.detail === 'string' ? d.detail : 'The upgrade is not available right now.';
    _body.appendChild(message(text, false));
  }

  // -- loading ---------------------------------------------------------------

  function schedule() {
    var gen = ++_gen;
    window.setTimeout(function () { if (gen === _gen) refresh(); }, POLL_MS);
  }

  function refresh() {
    if (!_root) return Promise.resolve();
    _gen += 1;
    return Promise.all([api(API + '/upgrade'), api(API)]).then(function (both) {
      var plan = both[0], status = both[1];
      if (!plan.ok || !plan.data.from) { renderProblem(plan); return; }
      _busy = false;
      render(plan.data, status.ok ? status.data : null);
      if (plan.data.media_enabled && plan.data.env_state === 'installing') schedule();
    });
  }

  function panel(post) {
    _post = post;
    _root = el('div');
    _root.id = 'od-engine-upgrade';
    _root.style.fontSize = '12px';
    _root.style.display = 'none';
    _body = el('div');
    _root.appendChild(_body);
    if (window.odReindex && typeof window.odReindex.onFinish === 'function') {
      window.odReindex.onFinish(function () { refresh(); });
    }
    return _root;
  }

  window.odEngineUpgrade = { panel: panel, refresh: refresh, POLL_MS: POLL_MS };
}());
