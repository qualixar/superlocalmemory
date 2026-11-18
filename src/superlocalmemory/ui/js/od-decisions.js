// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// od-decisions.js: the Dashboard home "Recent decisions by your agents" card (4.2.30, D-49).
//
// SuperLocalMemory is a memory system for agents, so an agent (or a script, or you) may release
// a memory the instruction firewall holds, settle a dispute, apply or roll back a set of proposed
// changes, and give a blind audit label. Nothing needs you to work. Every such decision is
// recorded: who decided (an agent id, a client name or "dashboard"), through which channel (mcp,
// cli, http, dashboard) and when. This card lists them, newest first, and each one that can be
// taken back has an Undo that uses the same door that made it:
//   POST /api/decisions/{id}/undo   (a promotion is withdrawn, a set rolled back, a dispute
//                                    reopened, a label retracted; the undo is recorded too).
// From GET /api/decisions. Memory text (the short preview of a promoted memory) and every other
// server string is set with textContent, never parsed as HTML. The card hides itself when
// nothing has been decided yet.
(function () {
  'use strict';

  var KIND = {
    promotion: 'Memory promoted', proposal: 'Proposed changes', dispute: 'Dispute',
    label: 'Audit label', signing: 'Memories signed again'
  };
  var CHANNEL = { mcp: 'an agent over MCP', cli: 'the command line', http: 'HTTP',
                  dashboard: 'the dashboard' };
  var UNDONE = { promote: 'Withdrawn', apply: 'Rolled back', settle: 'Reopened', label: 'Retracted' };
  var rows = [];
  var counts = { total: 0, undone: 0, open: 0 };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function when(iso) {
    if (!iso) return 'time unknown';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
  }
  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }

  function who(d) {
    return (d.actor === 'dashboard' ? 'You (on the dashboard)' : d.actor) +
      ' through ' + (CHANNEL[d.channel] || d.channel);
  }

  function undo(row, d, btn) {
    btn.disabled = true;
    fetch('/api/decisions/' + encodeURIComponent(d.decision_id) + '/undo', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || !j.success) throw new Error(detailOf(j, 'That did not work (HTTP ' + r.status + ').'));
        row.textContent = '';
        row.appendChild(el('p', 'dash-card-meta tone-ok', 'Undone. It is recorded below.'));
        setTimeout(load, 900);
      });
    }).catch(function (e) {
      btn.disabled = false;
      var note = row.querySelector('.truth-error') ||
        row.appendChild(el('p', 'dash-card-meta tone-warn truth-error'));
      note.textContent = e && e.message ? e.message : 'That did not work.';
    });
  }

  function decisionRow(d) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-decision', d.decision_id);
    row.appendChild(el('p', 'dash-card-lead', d.summary || KIND[d.kind] || 'A decision'));
    if (d.preview) row.appendChild(el('p', 'dash-card-meta truth-text', d.preview));
    row.appendChild(el('p', 'dash-card-meta', who(d) + ' · ' + when(d.decided_at)));
    if (d.undone) {
      row.appendChild(el('p', 'dash-card-meta tone-ok', (UNDONE[d.action] || 'Undone') + ' by ' +
        d.undone_by + ' through ' + (CHANNEL[d.undone_channel] || d.undone_channel) + ' · ' +
        when(d.undone_at)));
    } else if (d.can_undo) {
      var actions = el('div', 'truth-actions');
      var b = el('button', 'btn sm ghost', 'Undo');
      b.type = 'button';
      b.addEventListener('click', function (e) { e.preventDefault(); undo(row, d, b); });
      actions.appendChild(b);
      row.appendChild(actions);
    }
    return row;
  }

  function render() {
    var c = $('dash-decisions-card'), b = $('dash-decisions-body');
    if (!c || !b) return;
    b.textContent = '';
    c.hidden = !rows.length;
    var badge = $('dash-decisions-count');
    if (badge) badge.textContent = counts.open ? String(counts.open) : '';
    if (!rows.length) return;
    b.appendChild(el('p', 'dash-card-meta',
      'Your agents can decide these themselves; nothing here is waiting for you. ' +
      'Each one is recorded, and Undo takes it back.'));
    rows.forEach(function (d) { b.appendChild(decisionRow(d)); });
  }

  function load() {
    if (!$('dash-decisions-card')) return Promise.resolve();
    return fetch('/api/decisions?limit=10', { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (j) {
        rows = j.decisions || [];
        counts = j.counts || counts;
        render();
      }, function () { var c = $('dash-decisions-card'); if (c) c.hidden = true; });
  }

  window.SLMDecisions = { reload: load, _render: render };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 400); });
}());
