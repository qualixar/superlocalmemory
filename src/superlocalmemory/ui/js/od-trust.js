// od-trust.js: the Dashboard home "Held back from instructions" card (4.2.30, WS-5, D-49).
//
// Nothing is hidden from your agents: they recall, and are shown at session start, every memory
// whatever its origin. The firewall does one thing: a memory that came from a web page, a tool's
// output, another agent, an import or a model's text (or whose source was not recorded) cannot by
// itself become an instruction: a pin, a standing rule or decision, or core memory. An agent's own
// words and everything above them are trusted for those roles directly.
// This card lists the held memories in plain words (only once the firewall is switched on or in
// shadow; off, nothing is held back and the card stays hidden). Nothing here needs you: an
// agent can promote a memory with one explicit call, and you can too:
//   - Promote : it may now be an instruction (recorded: who, when, through the dashboard;
//     "Recent decisions by your agents" lists it with an Undo).
//   - Promote all : everything listed, one call.
// Nothing is ever deleted here; a memory that is not promoted is still found by recall, as
// evidence. Every call goes to the routes `slm trust promote` uses
// (POST /api/trust/promote/{fact}, /api/trust/promote-all); the page's own door is recorded
// by the daemon (the X-SLM-Channel header the dashboard's fetch adds), not sent in the body.
// Memory text is only ever set with textContent. The card hides itself when nothing waits.
(function () {
  'use strict';

  var items = [];
  var count = 0;
  var acting = false;
  var state = 'off';

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
  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }

  function post(path) {
    return fetch(path, {
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
    setTimeout(load, 1200);
  }

  // "From " + the class label reads badly for these two ("From written by a model."), so say them as a phrase.
  var FROM_WORDS = {
    model_generated: 'text a model wrote',
    unknown: 'a memory whose source was not recorded'
  };

  function itemRow(item) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-fact', item.fact_id);
    row.appendChild(el('p', 'dash-card-lead', item.preview || ''));
    var words = (item.wanted_words || []).join('; ');
    row.appendChild(el('p', 'dash-card-meta',
      'From ' + (FROM_WORDS[item.origin_class] || item.origin || 'an unknown source') +
      (words ? '. This memory ' + words + '.' : '.') +
      ' Until it is promoted, agents are not told to follow it.'));
    var actions = el('div', 'truth-actions');
    actions.appendChild(button('Promote', 'primary', function () {
      busy(row, true);
      post('/api/trust/promote/' + encodeURIComponent(item.fact_id)).then(
        function () { done(row, 'Promoted. It may now be a pin, a rule or part of your core memory. You can undo it under Recent decisions.'); },
        function (e) { failed(row, e); });
    }));
    row.appendChild(actions);
    return row;
  }

  function render() {
    var c = $('dash-trust-card'), b = $('dash-trust-body');
    if (!c || !b) return;
    b.textContent = '';
    // Off: nothing is held back, so nothing waits and the card stays away.
    c.hidden = !items.length || state === 'off';
    var badge = $('dash-trust-count');
    if (badge) badge.textContent = count ? String(count) : '';
    if (!items.length) return;
    b.appendChild(el('p', 'dash-card-meta', acting
      ? 'These memories came from a web page, a tool, another agent, an import or a model, and are held back from being a pin, a rule or core memory. Your agents still find them and see them as evidence. Promote the ones you trust, on purpose.'
      : 'The instruction firewall is not switched on yet, so nothing is held back. This is what it would hold.'));
    if (items.length > 1) {
      var all = el('div', 'truth-actions');
      all.appendChild(button('Promote all ' + count, 'ghost', function (btn) {
        btn.disabled = true;
        post('/api/trust/promote-all').then(
          function (j) {
            b.textContent = '';
            b.appendChild(el('p', 'dash-card-meta tone-ok', 'Promoted ' + (j.approved || count) + '.'));
            setTimeout(load, 1200);
          },
          function (e) { btn.disabled = false; failed(b, e); });
      }));
      b.appendChild(all);
    }
    items.forEach(function (item) { b.appendChild(itemRow(item)); });
  }

  function load() {
    if (!$('dash-trust-card')) return Promise.resolve();
    return fetch('/api/trust/pending?limit=20', { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (j) {
        items = j.items || [];
        count = j.count || items.length;
        acting = !!j.acting;
        state = j.state || (acting ? 'on' : 'off');
        render();
      }, function () { var c = $('dash-trust-card'); if (c) c.hidden = true; });
  }

  window.SLMTrust = { reload: load, _render: render };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 350); });
}());
