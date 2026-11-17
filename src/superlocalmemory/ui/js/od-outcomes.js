// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// 4.2.30 W3-5 (§8.10.3 g): the Health page's "What memory did for you" card.
// The same report as `slm health --memory` and MCP health(memory=true), from
// GET /api/outcomes/health: deliveries, how often memories are used (with a
// range), whether memory measurably helps (randomised, a claim only when the
// anytime-valid range excludes 0), and judge certificates in plain words.
// Read-only and on demand (a button): on a large store it reads every receipt.
// Nothing from the server is parsed as HTML.
(function () {
  'use strict';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function pct(x) { return x == null ? 'not known yet' : Math.round(100 * x) + '%'; }
  function points(x) { return (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' points'; }

  var CONTRAST = { memory: 'Giving one more memory at session start',
                   channel: 'Searching with every retrieval method' };
  var OUTCOME = { used: 'the agent acting on a memory', reask: 'the agent having to ask memory again' };

  function row(list, text, status, tone) {
    var r = el('div', 'list-row');
    r.style.flexWrap = 'wrap';  // a long label wraps as a whole on a phone, never word by word
    r.style.gap = '4px 12px';
    var label = el('span', 'truth-check-text', text);
    label.style.flex = '1 1 14em';
    r.appendChild(label);
    if (status != null) r.appendChild(el('span', 'truth-check-status ' + (tone || ''), status));
    list.appendChild(r);
    return r;
  }

  function memoryRow(list, f, status) {
    var r = row(list, 'Memory ' + String(f.fact_id).slice(0, 12) + ' (' + f.kind + '): used ' +
      f.uses + ' of ' + f.deliveries + ' times', status);
    if (typeof window.odOpenMemory === 'function') {
      var b = el('button', 'btn ghost', 'Open');
      b.style.whiteSpace = 'nowrap';
      b.style.flexShrink = '0';
      b.addEventListener('click', function (e) { e.preventDefault(); window.odOpenMemory(String(f.fact_id)); });
      r.appendChild(b);
    }
  }

  function effectText(e) {
    var what = CONTRAST[e.contrast] ||
      ('Searching with ' + String(e.contrast).split(':').pop().replace(/_/g, ' '));
    return what + ' (' + e.path.replace('_', ' ') + '), measured by ' + OUTCOME[e.outcome];
  }

  function effectStatus(e) {
    if (e.aipw == null) return ['Not enough data yet (' + e.n + ' of about ' + e.planned + ')', ''];
    if (e.claim === 'helps') return ['Helps: ' + points(e.aipw), 'tone-ok'];
    if (e.claim === 'hurts') return ['Hurts: ' + points(e.aipw), 'tone-warn'];
    return ['No clear effect yet (' + e.n + ' of about ' + e.planned + ')', ''];
  }

  function render(list, d) {
    list.textContent = '';
    var led = d.ledger || {};
    var recent = led.deliveries || {};
    var total = Object.keys(recent).reduce(function (s, k) { return s + recent[k]; }, 0);
    list.appendChild(el('p', 'dash-card-lead', 'SuperLocalMemory gave your agents memories ' + total +
      ' times in the last ' + (led.recent_days || 30) + ' days.'));
    list.appendChild(el('p', 'dash-card-meta', led.withholding === 'on'
      ? 'Measuring is on: one session start in ten leaves out one memory, or one way of searching, ' +
        'chosen at random and recorded, so SuperLocalMemory can tell whether memory helps.'
      : 'Measuring is off, so whether memory helps cannot be estimated yet. Turn it on with ' +
        'slm flags (outcomes.holdout).'));

    var use = d.use || {};
    list.appendChild(el('p', 'dash-card-lead truth-certs-title', 'How often memories are used'));
    row(list, use.rate == null ? 'Not enough deliveries yet to say.'
      : 'When a memory is given, the agent acts on it ' + pct(use.rate) + ' of the time. ' +
        use.never_used + ' of ' + use.facts_delivered + ' memories given were never used.');
    (use.most_used || []).slice(0, 5).forEach(function (f) {
      memoryRow(list, f, pct(f.low) + ' to ' + pct(f.high));
    });
    if ((use.rarely_used || []).length) {
      list.appendChild(el('p', 'dash-card-meta', 'Given often, rarely used (clearly below the usual rate):'));
      use.rarely_used.slice(0, 5).forEach(function (f) { memoryRow(list, f, 'under ' + pct(f.high)); });
    }

    list.appendChild(el('p', 'dash-card-lead truth-certs-title', 'Does memory help?'));
    list.appendChild(el('p', 'dash-card-meta', 'Compared at random, never by guess. "Helps" or ' +
      '"Hurts" appears only when the range that stays valid however often you look excludes zero.'));
    (d.effects || []).filter(function (e) {
      return e.contrast === 'memory' || e.contrast === 'channel' || e.claim;
    }).forEach(function (e) {
      var s = effectStatus(e);
      row(list, effectText(e), s[0], s[1]);
    });

    var certs = d.certificates || [];
    var live = certs.filter(function (c) { return !c.voided_at; });
    list.appendChild(el('p', 'dash-card-meta', 'Judges allowed to act on their own: ' + live.length +
      '. Second opinions that disagreed: ' + (d.second_opinions || []).length + '.'));
  }

  function mount(container) {
    if (!container || document.getElementById('od-memory-outcomes')) return;
    var card = el('div', 'card');
    card.id = 'od-memory-outcomes';
    card.style.marginTop = '16px';
    var head = el('div', 'card-head');
    head.appendChild(el('h3', null, 'What memory did for you'));
    card.appendChild(head);
    var pad = el('div', 'card-pad dash-kv');
    pad.appendChild(el('p', 'dash-card-meta', 'Which memories your agents used, and whether memory ' +
      'measurably helps. It only reads; nothing is changed.'));
    var list = el('div', 'truth-checks');
    var run = el('button', 'btn primary', 'Show');
    run.addEventListener('click', function (e) {
      e.preventDefault();
      run.disabled = true;
      list.textContent = 'Reading…';
      fetch('/api/outcomes/health', { credentials: 'same-origin' })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { render(list, d); },
              function () { list.textContent = 'The report could not be read. Try again in a moment.'; })
        .then(function () { run.disabled = false; });
    });
    pad.appendChild(run);
    pad.appendChild(list);
    card.appendChild(pad);
    container.appendChild(card);
  }

  window.odMountMemoryOutcomes = mount;
  window.SLMOutcomes = { _render: render };
}());
