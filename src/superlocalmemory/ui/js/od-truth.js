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
//
// Also: the memory detail's "Where this came from" and "When it was true" sections
// (SLMTruth.detail) and the Health page's "Truth checks" card (odMountTruthChecks),
// all read-only.
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
      : d.reason === 'origin_below_head'
        ? 'One of these came from a source SuperLocalMemory trusts less, so it will not choose. Which is true now?'
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

  // ----- memory detail: "When it was true" (§8.10.3 b) -----
  var STATE_WORDS = {
    head: 'Shown now', superseded: 'Replaced', historical: 'Earlier value',
    disputed: 'In question', duplicate: 'Same as another memory', compatible: 'Also true',
    unbound: 'Not about this', gone: 'No longer kept'
  };

  function valueRow(v) {
    var row = el('div', 'truth-side');
    var head = el('div', 'truth-side-head');
    head.appendChild(el('span', 'dash-card-meta', 'From ' + day(v.began)));
    head.appendChild(el('span', 'chip-pill', v.current ? 'Shown now' : (STATE_WORDS[v.state] || v.state)));
    if (v.this_memory) head.appendChild(el('span', 'chip-pill violet', 'This memory'));
    row.appendChild(head);
    row.appendChild(el('p', 'truth-text', v.visible ? (v.content || '') : '(You cannot see this memory.)'));
    return row;
  }

  // Slot keys are stored normalised ("release_cadence"); people read words.
  function words(key) { return String(key || '').replace(/_/g, ' '); }

  function slotBlock(s) {
    var box = el('div', 'truth-item');
    var what = s.legacy ? 'Part of a correction: one memory replaced another.'
      : 'It states the value of: ' + [s.subject, s.relation].filter(Boolean).map(words).join(' · ') +
        (s.qualifier ? ' (' + words(s.qualifier) + ')' : '');
    box.appendChild(el('p', 'dash-card-lead', what));
    (s.values || []).forEach(function (v) { box.appendChild(valueRow(v)); });
    return box;
  }

  // ----- memory detail: "Where this came from" (WS-5, §8.10.3 h) -----
  // The class arrives with its plain-words label from the server; nothing is parsed.
  function sentence(text) {
    var s = String(text || '');
    return s ? s.charAt(0).toUpperCase() + s.slice(1) + '.' : '';
  }

  function originBlock(o) {
    var box = el('div', 'truth-item');
    box.setAttribute('data-origin', o['class'] || '');
    box.appendChild(el('p', 'dash-card-lead', sentence(o.label)));
    if (o.taint && o.taint !== o['class'] && o.taint !== 'unknown') {
      box.appendChild(el('p', 'dash-card-meta',
        'It was built from something less trusted: ' + o.taint_label + '.'));
    }
    if (o['class'] === 'unknown') {
      box.appendChild(el('p', 'dash-card-meta',
        'SuperLocalMemory did not record where this came from, so it cannot say how far to trust it.'));
    }
    return box;
  }

  function detail(factId, container) {
    if (!container || !factId) return Promise.resolve();
    container.textContent = '';
    return fetch('/api/truth/facts/' + encodeURIComponent(factId), { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (d) {
        var slots = d.slots || [], open = d.disputes || [], src = d.source;
        var stale = src && src.state !== 'fresh';
        container.textContent = '';
        if (d.origin && d.origin['class']) {
          container.appendChild(el('h3', 'truth-detail-title', 'Where this came from'));
          container.appendChild(originBlock(d.origin));
        }
        if (!slots.length && !open.length && !stale) return;
        container.appendChild(el('h3', 'truth-detail-title', 'When it was true'));
        open.forEach(function (dsp) {
          var row = disputeRow(dsp);
          container.appendChild(row);
        });
        if (stale) container.appendChild(sourceRow({ fact_id: factId, state: src.state, file: src.file }));
        slots.forEach(function (s) { container.appendChild(slotBlock(s)); });
      }, function () { container.textContent = ''; });
  }

  // ----- Health page: truth checks (§8.10.3 d) -----
  var CHECK_WORDS = {
    'I-01': 'Every value shown now comes from a memory that still exists.',
    'I-02': 'Nothing has two values shown at once.',
    'I-03': 'Every open question compares against the value shown now.',
    'I-04': 'Values in question have an open question; empty ones show nothing.',
    'I-05': 'Values follow each other in time without overlapping.',
    'I-06': 'Every replaced value was retired, or is waiting to be.',
    'I-07': 'Every change SuperLocalMemory made to a value is recorded with its reason.',
    'I-08': 'Undoing a change also brings the earlier value back.',
    'I-12': 'Every end date comes after its start and says where it came from.',
    'I-16': 'Values shown now have no open question.'
  };
  var STATUS_WORDS = {
    ok: ['OK', 'tone-ok'], violated: ['Problem found', 'tone-warn'],
    not_checked: ['Not checked yet', ''], not_applicable: ['Not used here', '']
  };

  function checkRows(list, data) {
    list.textContent = '';
    var bad = 0;
    (data.results || []).forEach(function (r) {
      var words = STATUS_WORDS[r.status] || [r.status, ''];
      if (r.status === 'violated') bad++;
      var row = el('div', 'list-row');
      row.appendChild(el('span', 'truth-check-text', CHECK_WORDS[r.invariant] || r.title));
      row.appendChild(el('span', 'truth-check-status ' + words[1],
        words[0] + (r.status === 'violated' && r.count ? ' (' + r.count + ')' : '')));
      list.appendChild(row);
    });
    var certs = data.certificates || [];
    var live = certs.filter(function (c) { return !c.voided_at; });
    list.appendChild(el('p', 'dash-card-lead truth-certs-title', 'Judges allowed to act on their own'));
    if (!live.length) {
      list.appendChild(el('p', 'dash-card-meta', 'None yet. A judge earns this only from your ' +
        'answers to its audit questions, and loses it when they stop supporting it.'));
    }
    certs.forEach(function (c) {
      var row = el('div', 'list-row');
      row.appendChild(el('span', 'truth-check-text',
        c.tier + ' may answer "' + String(c.verdict).replace(/_/g, ' ') + '" (' + c.n + ' audited, ' +
        c.errors + ' wrong)'));
      row.appendChild(el('span', 'truth-check-status ' + (c.voided_at ? '' : 'tone-ok'),
        c.voided_at ? (c.void_reason === 'drift' ? 'Stopped: its recent answers changed'
                                                 : 'Stopped: answers no longer support it')
                    : 'Since ' + day(c.issued_at)));
      list.appendChild(row);
    });
    var second = data.second_opinions || [];
    if (second.length) {
      list.appendChild(el('p', 'dash-card-lead truth-certs-title', 'Second opinions'));
      second.forEach(function (s) {
        var row = el('div', 'list-row');
        row.appendChild(el('span', 'truth-check-text', 'A stronger judge rechecked ' + s.pairs +
          ' of ' + s.lower_tier + "'s answers"));
        row.appendChild(el('span', 'truth-check-status ' + (s.disagreed ? 'tone-warn' : 'tone-ok'),
          s.disagreed ? 'Disagreed on ' + s.disagreed : 'Agreed'));
        list.appendChild(row);
      });
    }
    var lead = el('p', 'dash-card-lead ' + (bad ? 'tone-warn' : 'tone-ok'),
      bad ? bad + (bad === 1 ? ' check found a problem.' : ' checks found a problem.') +
            ' Run slm doctor --truth in a terminal for the details.'
          : 'Everything that was checked is consistent.');
    list.insertBefore(lead, list.firstChild);
  }

  function mountChecks(container) {
    if (!container || document.getElementById('od-truth-checks')) return;
    var card = el('div', 'card');
    card.id = 'od-truth-checks';
    card.style.marginTop = '16px';
    var head = el('div', 'card-head');
    head.appendChild(el('h3', null, 'Truth checks'));
    card.appendChild(head);
    var pad = el('div', 'card-pad dash-kv');
    pad.appendChild(el('p', 'dash-card-meta', 'Checks that the values SuperLocalMemory shows now, ' +
      'and their history, are consistent. It only reads; nothing is changed.'));
    var list = el('div', 'truth-checks');
    var run = button('Run checks', 'primary', function () {
      run.disabled = true;
      list.textContent = 'Checking…';
      fetch('/api/truth/checks', { credentials: 'same-origin' })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { checkRows(list, d); },
              function () { list.textContent = 'The checks could not run. Try again in a moment.'; })
        .then(function () { run.disabled = false; });
    });
    pad.appendChild(run);
    pad.appendChild(list);
    card.appendChild(pad);
    container.appendChild(card);
  }

  // ----- Dashboard home: blind audit questions (W2, R-16) -----
  // One pair at a time, never the judge's own answer, so the person's answer is
  // independent. Hidden when nothing is waiting.
  var audit = { items: [], count: 0 };

  function auditCard() { return $('dash-audit-card'); }

  function auditShow() {
    var c = auditCard(), b = $('dash-audit-body');
    if (!c || !b) return;
    b.textContent = '';
    var item = audit.items[0];
    c.hidden = !item;
    if (!item) return;
    var badge = $('dash-audit-count');
    if (badge) badge.textContent = String(audit.count);
    b.appendChild(el('p', 'dash-card-meta', 'SuperLocalMemory checks its own judgement against yours. ' +
      'Pick what is true; "I can\'t tell" is always fine. Your answers stay on this computer.'));
    b.appendChild(el('p', 'dash-card-lead', item.question));
    (item.memories || []).forEach(function (m) {
      var box = el('div', 'truth-side');
      box.appendChild(el('span', 'dash-card-meta', m.name + ' · said ' + day(m.said_at)));
      box.appendChild(el('p', 'truth-text', m.content || ''));
      b.appendChild(box);
    });
    var actions = el('div', 'truth-actions truth-audit-options');
    (item.options || []).forEach(function (o) {
      // Every option looks the same: no styling may hint at an answer (a blind label).
      actions.appendChild(button(o.text, 'ghost', function () {
        auditAnswer(item, o.label);
      }));
    });
    actions.appendChild(button('Skip', 'ghost', function () { audit.items.shift(); audit.count = Math.max(0, audit.count - 1); auditShow(); }));
    b.appendChild(actions);
  }

  function auditAnswer(item, label) {
    var b = $('dash-audit-body');
    if (b) busy(b, true);
    post('/api/audit/' + encodeURIComponent(item.item_id) + '/answer', { label: label, channel: 'dashboard' })
      .then(function () {
        audit.items.shift();
        audit.count = Math.max(0, audit.count - 1);
        if (!audit.items.length && audit.count) return auditLoad();
        auditShow();
      }, function (e) { if (b) failed(b, e); });
  }

  function auditLoad() {
    if (!auditCard()) return Promise.resolve();
    return fetch('/api/audit/pending?limit=10', { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (d) { audit = { items: d.items || [], count: d.count || 0 }; auditShow(); },
            function () { var c = auditCard(); if (c) c.hidden = true; });
  }

  window.odMountTruthChecks = mountChecks;
  function reloadAll() { return Promise.all([load(), auditLoad()]); }
  window.SLMTruth = { reload: reloadAll, detail: detail, audit: auditLoad, _render: render };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(reloadAll, 300); });
}());
