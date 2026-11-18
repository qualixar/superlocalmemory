// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// 4.2.30 W3 (D-41, §8.10.3): the Health page's "Shown to your agents" card.
// What SuperLocalMemory recently put in front of your agents, grouped by conversation (the
// start of a conversation, a prompt, a cue, or when an agent asked), each memory with the
// reason it was chosen, in plain words. Beside it two numbers: the prompts that did not need
// memory and were skipped, and the memories that would be withheld if the harm check acted
// (it only watches in 4.2.30). From GET /api/delivered/recent; read-only and on demand (a
// button): the two numbers read the receipt ledger. Nothing from the server is parsed as HTML.
(function () {
  'use strict';

  var API = '/api/delivered/recent';
  var DEFAULT_LIMIT = 20;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }
  function percent(x) { return (Math.round(x * 1000) / 10) + '%'; }

  function openButton(factId) {
    if (typeof window.odOpenMemory !== 'function') return null;
    var b = el('button', 'btn sm ghost', 'Open');
    b.type = 'button';
    b.style.whiteSpace = 'nowrap';
    b.style.flexShrink = '0';
    b.addEventListener('click', function (e) { e.preventDefault(); window.odOpenMemory(String(factId)); });
    return b;
  }

  function memoryRow(m) {
    var row = el('div', 'delivered-memory');
    var body = el('div', 'delivered-memory-body');
    body.appendChild(el('p', 'delivered-text', m.preview || ''));
    var why = m.why || {};
    var line = el('p', 'dash-card-meta delivered-why', why.words || '');
    if (why.standing && why.standing !== 'current') line.className += ' tone-warn';
    body.appendChild(line);
    row.appendChild(body);
    var open = openButton(m.fact_id);
    if (open) row.appendChild(open);
    return row;
  }

  function deliveryBlock(d) {
    var box = el('div', 'delivered-delivery');
    box.appendChild(el('p', 'delivered-kind', d.kind_words + (d.ago ? ' · ' + d.ago : '')));
    (d.memories || []).forEach(function (m) { box.appendChild(memoryRow(m)); });
    if (d.more) box.appendChild(el('p', 'dash-card-meta', plural(d.more, 'more memory', 'more memories') + ' not listed.'));
    if (!(d.memories || []).length) {
      box.appendChild(el('p', 'dash-card-meta', 'No memory text to show for this one.'));
    }
    if (d.left_out) {
      box.appendChild(el('p', 'dash-card-meta', plural(d.left_out, 'memory was', 'memories were') +
        ' left out on purpose: a random check of whether memory helps. It can be switched off in Settings.'));
    }
    return box;
  }

  function groupBlock(g, open) {
    var det = el('details', 'delivered-group');
    if (open) det.open = true;
    var n = (g.deliveries || []).reduce(function (s, d) { return s + (d.memories || []).length; }, 0);
    var who = g.client ? String(g.client) : 'An agent';
    det.appendChild(el('summary', 'delivered-summary',
      who + ' · ' + (g.ago || '') + ' · ' + plural(n, 'memory', 'memories') + ' in ' +
      plural((g.deliveries || []).length, 'delivery', 'deliveries')));
    // oldest first inside a conversation reads as it happened
    (g.deliveries || []).slice().reverse().forEach(function (d) { det.appendChild(deliveryBlock(d)); });
    return det;
  }

  function needWords(n) {
    if (!n || n.turns == null || n.turns === 0) {
      return 'No prompts have been checked yet. The per-prompt check is off, or nothing has been asked.';
    }
    var text = plural(n.skipped || 0, 'prompt', 'prompts') + ' of ' + n.turns + ' in the last ' +
      (n.days || 30) + ' days did not need memory, so none was added.';
    if (n.rules_bound != null) {
      text += ' Of the prompts that did need it, the rules skipped at most ' + percent(n.rules_bound) +
        ' (measured, not assumed).';
    }
    return text;
  }

  function harmWords(h) {
    var count = (h && h.would_withhold_count) || 0;
    // Nothing testable (no memory was both shown and left out often enough, or the report could not
    // be read) is not the same as "tested and fine": say so instead of claiming a result.
    if (!count && !(h && h.tests > 0)) return 'Not testable yet: no memory has been left out often ' +
      'enough to compare. This check only watches; nothing is ever withheld by it in this version.';
    if (!count) return 'No memory has been measured as making an agent worse. This check only watches; ' +
      'nothing is ever withheld by it in this version.';
    return plural(count, 'memory', 'memories') + ' would be withheld if the harm check acted, because ' +
      'agents did measurably worse when they were shown. It only watches: nothing is withheld.';
  }

  function numbers(d) {
    var box = el('div', 'delivered-numbers');
    var a = el('div', 'delivered-number');
    a.appendChild(el('span', 'dash-mini-label muted', 'Prompts that skipped memory'));
    a.appendChild(el('span', 'dash-mini-value', d.need_gate && d.need_gate.turns ? String(d.need_gate.skipped || 0) : 'none yet'));
    a.appendChild(el('p', 'dash-card-meta', needWords(d.need_gate)));
    box.appendChild(a);
    var b = el('div', 'delivered-number');
    b.appendChild(el('span', 'dash-mini-label muted', 'Memories it would withhold'));
    b.appendChild(el('span', 'dash-mini-value', String((d.harm_gate && d.harm_gate.would_withhold_count) || 0)));
    b.appendChild(el('p', 'dash-card-meta', harmWords(d.harm_gate)));
    ((d.harm_gate && d.harm_gate.would_withhold) || []).forEach(function (m) {
      var row = el('div', 'delivered-memory');
      var body = el('div', 'delivered-memory-body');
      body.appendChild(el('p', 'delivered-text', m.preview || ''));
      row.appendChild(body);
      var open = openButton(m.fact_id);
      if (open) row.appendChild(open);
      b.appendChild(row);
    });
    box.appendChild(b);
    return box;
  }

  function render(list, d) {
    list.textContent = '';
    var groups = d.groups || [];
    if (!groups.length) {
      list.appendChild(el('p', 'dash-card-lead', 'Nothing has been shown to an agent yet.'));
      list.appendChild(el('p', 'dash-card-meta', 'When an agent starts a conversation, gets a prompt, hits ' +
        'a cue, or asks, what it was given appears here with the reason.'));
    } else {
      list.appendChild(el('p', 'dash-card-lead', 'In the last ' + plural(d.deliveries, 'delivery', 'deliveries') +
        ' your agents were shown ' + plural(d.memories, 'memory', 'memories') + ', in ' +
        plural(groups.length, 'conversation', 'conversations') + '.'));
      groups.forEach(function (g, i) { list.appendChild(groupBlock(g, i === 0)); });
    }
    list.appendChild(numbers(d));
  }

  function mount(container) {
    if (!container || document.getElementById('od-delivered')) return;
    var card = el('div', 'card');
    card.id = 'od-delivered';
    card.style.marginTop = '16px';
    var head = el('div', 'card-head');
    head.appendChild(el('h3', null, 'Shown to your agents'));
    card.appendChild(head);
    var pad = el('div', 'card-pad dash-kv');
    pad.appendChild(el('p', 'dash-card-meta', 'What your agents were given recently, and why each memory ' +
      'was chosen. It only reads; nothing is changed.'));
    var list = el('div', 'delivered-list');
    var run = el('button', 'btn primary', 'Show');
    run.type = 'button';
    run.addEventListener('click', function (e) {
      e.preventDefault();
      run.disabled = true;
      list.textContent = 'Reading…';
      fetch(API + '?limit=' + DEFAULT_LIMIT, { credentials: 'same-origin' })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { render(list, d); },
              function () { list.textContent = 'This could not be read. Try again in a moment.'; })
        .then(function () { run.disabled = false; });
    });
    pad.appendChild(run);
    pad.appendChild(list);
    card.appendChild(pad);
    container.appendChild(card);
  }

  window.odMountDelivered = mount;
  window.SLMDelivered = { _render: render };
}());
