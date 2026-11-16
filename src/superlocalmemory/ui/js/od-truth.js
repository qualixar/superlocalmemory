// od-truth.js: the Dashboard home "Needs your answer" card (4.2.30, D-37..D-39).
//
// Two kinds of item, in plain words for people who never open a terminal:
//   - a dispute: two memories say different things about the same subject.
//     Buttons: "Keep this one" on each side, "Both are true",
//     "They are about different things", and "Not sure" (hides it here only,
//     nothing is changed).
//   - a changed file: a memory came from a file that changed or is gone.
//     Button: "Still true" (you looked at the file and the memory holds).
// Every answer goes to the same route agents and the CLI use
// (POST /api/disputes/{id}/settle, POST /api/changed-sources/{fact}/still-true).
// Memory text is only ever set with textContent. The card hides itself when
// nothing needs an answer.
(function () {
  'use strict';

  var LIMIT = 20;
  var skipped = {};

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function button(label, cls, onClick) {
    var b = el('button', 'btn sm ' + (cls || 'ghost'), label);
    b.type = 'button';
    b.addEventListener('click', function (e) { e.preventDefault(); onClick(b); });
    return b;
  }
  function day(iso) {
    if (!iso) return 'date unknown';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso).slice(0, 10) : d.toLocaleDateString();
  }

  function card() { return $('dash-attention-card'); }
  function body() { return $('dash-attention-body'); }

  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }

  function post(url, payload) {
    return fetch(url, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || !j.success) throw new Error(detailOf(j, 'That did not work (HTTP ' + r.status + ').'));
        return j;
      });
    });
  }

  function busy(row, on) {
    row.querySelectorAll('button').forEach(function (b) { b.disabled = on; });
  }

  function done(row, text) {
    row.textContent = '';
    row.appendChild(el('p', 'dash-card-meta tone-ok', text));
    setTimeout(load, 1500);
  }

  function failed(row, err) {
    busy(row, false);
    var note = row.querySelector('.truth-error') || row.appendChild(el('p', 'dash-card-meta tone-warn truth-error'));
    note.textContent = err && err.message ? err.message : 'That did not work.';
  }

  function settle(row, dispute, keep) {
    busy(row, true);
    post('/api/disputes/' + encodeURIComponent(dispute.dispute_id) + '/settle',
         { keep: keep, channel: 'dashboard' })
      .then(function (r) {
        var retired = (r.retired || []).length;
        done(row, retired ? 'Done. The other memory was retired; you can undo it under Corrections.'
                          : 'Done. Both memories stay.');
      }, function (e) { failed(row, e); });
  }

  function side(row, dispute, s) {
    var box = el('div', 'truth-side');
    var head = el('div', 'truth-side-head');
    head.appendChild(el('span', 'dash-card-meta', 'Said ' + day(s.said_at)));
    if (s.current) head.appendChild(el('span', 'chip-pill', 'Shown now'));
    box.appendChild(head);
    box.appendChild(el('p', 'truth-text', s.visible ? (s.content || '') : '(You cannot see this memory.)'));
    if (s.visible) box.appendChild(button('Keep this one', 'primary', function () { settle(row, dispute, s.fact_id); }));
    return box;
  }

  function disputeRow(d) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-dispute', d.dispute_id);
    row.appendChild(el('p', 'dash-card-lead',
      d.reason === 'rolled_back' ? 'A change was undone between these two. Which is true now?'
                                 : 'These two memories disagree. Which is true now?'));
    (d.sides || []).forEach(function (s) { row.appendChild(side(row, d, s)); });
    var more = el('div', 'truth-actions');
    more.appendChild(button('Both are true', 'ghost', function () { settle(row, d, 'both'); }));
    more.appendChild(button('They are about different things', 'ghost', function () { settle(row, d, 'unrelated'); }));
    more.appendChild(button('Not sure', 'ghost', function () {
      skipped[d.dispute_id] = true;
      row.remove();
      refreshCount();
    }));
    row.appendChild(more);
    return row;
  }

  function sourceRow(c) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-fact', c.fact_id);
    row.appendChild(el('p', 'dash-card-lead', c.state === 'missing'
      ? 'A memory came from a file that is gone.'
      : 'A memory came from a file that changed since it was saved.'));
    row.appendChild(el('p', 'truth-file', c.file || ''));
    row.appendChild(el('p', 'dash-card-meta', 'Look at the file. If the memory still holds, say so.'));
    var actions = el('div', 'truth-actions');
    if (c.state !== 'missing') {
      actions.appendChild(button('Still true', 'primary', function () {
        busy(row, true);
        post('/api/changed-sources/' + encodeURIComponent(c.fact_id) + '/still-true', {})
          .then(function () { done(row, 'Noted. The memory is marked as checked.'); },
                function (e) { failed(row, e); });
      }));
    }
    row.appendChild(actions);
    return row;
  }

  function refreshCount() {
    var b = body(), c = card();
    if (!b || !c) return;
    var n = b.querySelectorAll('.truth-item').length;
    var badge = $('dash-attention-count');
    if (badge) badge.textContent = n ? String(n) : '';
    c.hidden = n === 0;
  }

  function render(data) {
    var b = body(), c = card();
    if (!b || !c) return;
    b.textContent = '';
    (data.disputes || []).forEach(function (d) {
      if (!skipped[d.dispute_id]) b.appendChild(disputeRow(d));
    });
    (data.changed_sources || []).forEach(function (s) { b.appendChild(sourceRow(s)); });
    refreshCount();
  }

  function load() {
    if (!card()) return Promise.resolve();
    return fetch('/api/truth/attention?limit=' + LIMIT, { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(render, function () { var c = card(); if (c) c.hidden = true; });
  }

  window.SLMTruth = { reload: load, _render: render };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 300); });
}());
