// od-proposals.js: the Dashboard home "Proposed changes" card (4.2.30, WS-6, I-10).
//
// SuperLocalMemory may suggest changes to your memory: retire an older memory,
// merge copies, rewrite a summary. A suggestion is a set of changes decided
// together and never touches your memory until you answer:
//   - Apply   : all of the set, or none of it.
//   - Reject  : changes nothing.
//   - Undo    : on a set you applied, puts back exactly what it replaced.
// Every answer goes to the same routes `slm proposals` uses
// (POST /api/proposals/{id}/apply|reject|rollback); the daemon records who decided and the
// channel (the dashboard's own door), and "Recent decisions by your agents" lists it with an Undo. Memory text is only ever set with
// textContent. The card hides itself when there is nothing to show.
(function () {
  'use strict';

  var KIND = {
    retire: 'Retire a memory', merge: 'Merge copies', edit_derived: 'Rewrite a summary',
    create_page: 'Add a wiki page', wiki_edit: 'Edit from the Markdown wiki'
  };
  var pending = [];
  var applied = [];
  var count = 0;

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
  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }

  function post(id, action) {
    return fetch('/api/proposals/' + encodeURIComponent(id) + '/' + action, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
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
  function failed(row, err) {
    busy(row, false);
    var note = row.querySelector('.truth-error') || row.appendChild(el('p', 'dash-card-meta tone-warn truth-error'));
    note.textContent = err && err.message ? err.message : 'That did not work.';
  }
  function done(row, text) {
    row.textContent = '';
    row.appendChild(el('p', 'dash-card-meta tone-ok', text));
    setTimeout(load, 1500);
  }

  function decide(row, set, action, message) {
    busy(row, true);
    post(set.set_id, action).then(function () { done(row, message); },
                                  function (e) { failed(row, e); });
  }

  function members(row, set) {
    (set.proposals || []).forEach(function (p) {
      var box = el('div', 'truth-side');
      box.appendChild(el('span', 'dash-card-meta', KIND[p.kind] || 'Change'));
      box.appendChild(el('p', 'truth-text', p.summary || ''));
      row.appendChild(box);
    });
  }

  function pendingRow(set) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-set', set.set_id);
    row.appendChild(el('p', 'dash-card-lead', set.title ||
      'SuperLocalMemory suggests ' + ((set.proposals || []).length === 1 ? 'this change' : 'these changes') + '.'));
    row.appendChild(el('p', 'dash-card-meta', 'Suggested ' + day(set.created_at) +
      '. Nothing changes until you apply it, and you can undo it afterwards.'));
    members(row, set);
    var actions = el('div', 'truth-actions');
    actions.appendChild(button('Apply', 'primary', function () {
      decide(row, set, 'apply', 'Applied. You can undo it below.');
    }));
    actions.appendChild(button('Reject', 'ghost', function () {
      decide(row, set, 'reject', 'Rejected. Nothing was changed.');
    }));
    row.appendChild(actions);
    return row;
  }

  function appliedRow(set) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-set', set.set_id);
    row.appendChild(el('p', 'dash-card-meta', 'Applied ' + day(set.applied_at) + '. Undo puts back exactly what it replaced.'));
    members(row, set);
    var actions = el('div', 'truth-actions');
    actions.appendChild(button('Undo', 'ghost', function () {
      decide(row, set, 'rollback', 'Undone. Everything it replaced is back as it was.');
    }));
    row.appendChild(actions);
    return row;
  }

  function render() {
    var c = $('dash-proposals-card'), b = $('dash-proposals-body');
    if (!c || !b) return;
    b.textContent = '';
    c.hidden = !(pending.length || applied.length);
    var badge = $('dash-proposals-count');
    if (badge) badge.textContent = count ? String(count) : '';
    pending.forEach(function (s) { b.appendChild(pendingRow(s)); });
    if (applied.length) {
      b.appendChild(el('p', 'dash-card-meta', 'Recently applied'));
      applied.forEach(function (s) { b.appendChild(appliedRow(s)); });
    }
  }

  function get(state, limit) {
    return fetch('/api/proposals?state=' + state + '&limit=' + limit, { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
  }

  function load() {
    if (!$('dash-proposals-card')) return Promise.resolve();
    return Promise.all([get('proposed', 10), get('applied', 3)]).then(function (both) {
      pending = both[0].sets || [];
      count = both[0].count || pending.length;
      applied = both[1].sets || [];
      render();
    }, function () { var c = $('dash-proposals-card'); if (c) c.hidden = true; });
  }

  window.SLMProposals = { reload: load, _render: render };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 300); });
}());
