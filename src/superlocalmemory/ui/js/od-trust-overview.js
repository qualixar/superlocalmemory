// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// 4.2.30 WS-5 (§8.10.3 h): the Health page's "Trust overview" card. Where your memories came from, how
// many wait for your OK (a link to that card), whether memories are signed and none was changed
// outside SuperLocalMemory, whether the secret screen is holding credentials back, and whether
// the audit log is intact. From GET /api/trust/overview: counts only, never a memory's text or
// a credential. Read-only and on demand (a button): the audit log is read end to end.
// Nothing from the server is parsed as HTML.
(function () {
  'use strict';

  var API = '/api/trust/overview';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function cap(s) { s = String(s || ''); return s.charAt(0).toUpperCase() + s.slice(1); }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  function row(list, text, status, tone) {
    var r = el('div', 'list-row');
    r.style.flexWrap = 'wrap';
    r.style.gap = '4px 12px';
    var label = el('span', 'truth-check-text', text);
    label.style.flex = '1 1 14em';
    r.appendChild(label);
    if (status != null) r.appendChild(el('span', 'truth-check-status ' + (tone || ''), status));
    list.appendChild(r);
    return r;
  }
  function title(list, text) { list.appendChild(el('p', 'dash-card-lead truth-certs-title', text)); }

  var FIREWALL = {
    off: 'The instruction firewall is off, so nothing is held back from being an instruction.',
    shadow: 'The instruction firewall is watching: it shows what it would hold back, and holds nothing yet.',
    on: 'The instruction firewall is on: a memory from a tool, a web page or another agent cannot become ' +
        'a rule or part of your core memory by itself; one explicit, recorded call promotes it.'
  };

  function goToWaiting() {
    if (typeof window.slmNavigate === 'function') window.slmNavigate('dashboard-pane');
    setTimeout(function () {
      var c = document.getElementById('dash-trust-card');
      if (c && c.scrollIntoView) c.scrollIntoView({ block: 'start' });
    }, 150);
  }

  function renderOrigins(list, origins) {
    title(list, 'Where your memories came from');
    if (!origins.length) {
      list.appendChild(el('p', 'dash-card-meta', 'Not worked out yet. SuperLocalMemory classifies memories ' +
        'in the background once the origin check is on (slm flags set trust.taint on).'));
      return;
    }
    origins.forEach(function (o) { row(list, cap(o.label), String(o.count)); });
  }

  function renderWaiting(list, fw) {
    title(list, 'Held back from instructions');
    list.appendChild(el('p', 'dash-card-meta', FIREWALL[fw.state] || FIREWALL.off));
    var r = row(list, plural(fw.waiting || 0, 'memory is', 'memories are') + ' held back' +
      (fw.approved ? '; ' + fw.approved + ' promoted' : '') + '.');
    if (fw.waiting && fw.state !== 'off') {
      var b = el('button', 'btn sm ghost', 'Review them');
      b.type = 'button';
      b.style.whiteSpace = 'nowrap';
      b.addEventListener('click', function (e) { e.preventDefault(); goToWaiting(); });
      r.appendChild(b);
    }
  }

  function renderSecrets(list, s) {
    title(list, 'Secrets');
    var kinds = Object.keys(s.by_type || {});
    if (s.state === 'on' || s.state === 'shadow') {
      list.appendChild(el('p', 'dash-card-meta', 'A password or key in a saved memory is stored as a placeholder ' +
        'such as [REDACTED:TYPE:1234]; the original is not kept anywhere.'));
    } else {
      list.appendChild(el('p', 'dash-card-meta', 'The secret screen is off, so a password or key you save is ' +
        'stored as written (slm flags set trust.secret_screen on turns it on).'));
    }
    row(list, kinds.length
      ? plural(s.memories || 0, 'memory holds', 'memories hold') + ' a credential as a placeholder: ' +
        kinds.map(function (k) { return k.replace(/_/g, ' ') + ' ' + s.by_type[k]; }).join(', ') + '.'
      : 'No memory holds a credential placeholder.');
  }

  function renderSignatures(list, sig) {
    title(list, 'Signed memories');
    list.appendChild(el('p', 'dash-card-meta', sig.key_words || ''));
    var tone = sig.modified ? 'tone-warn' : (sig.ok ? 'tone-ok' : '');
    row(list, sig.words || '', sig.modified ? 'Problem found' : (sig.ok ? 'OK' : 'Not checked'), tone);
    if (sig.facts) {
      var minis = el('div', 'dash-minis');
      [['Signed', sig.signed], ['Not signed yet', sig.unsigned],
       ['Changed since signing', sig.modified], ['Waiting to be signed again', sig.moved_by_a_fold]]
        .forEach(function (p) {
          var m = el('div');
          m.appendChild(el('span', 'dash-mini-label muted', p[0]));
          m.appendChild(el('span', 'dash-mini-value', String(p[1] || 0)));
          minis.appendChild(m);
        });
      list.appendChild(minis);
    }
  }

  function renderAudit(list, audit) {
    title(list, 'Audit log');
    var status = audit.ok === true ? 'OK' : (audit.ok === false ? 'Problem found' : 'No log yet');
    row(list, audit.words || '', status, audit.ok === true ? 'tone-ok' : (audit.ok === false ? 'tone-warn' : ''));
  }

  function render(list, d) {
    list.textContent = '';
    renderOrigins(list, d.origins || []);
    renderWaiting(list, d.firewall || {});
    renderSecrets(list, d.secret_screen || {});
    renderSignatures(list, d.signatures || {});
    renderAudit(list, d.audit_chain || {});
  }

  function mount(container) {
    if (!container || document.getElementById('od-trust-overview')) return;
    var card = el('div', 'card');
    card.id = 'od-trust-overview';
    card.style.marginTop = '16px';
    var head = el('div', 'card-head');
    head.appendChild(el('h3', null, 'Trust overview'));
    card.appendChild(head);
    var pad = el('div', 'card-pad dash-kv');
    pad.appendChild(el('p', 'dash-card-meta', 'Where memories came from, what waits for your OK, and whether ' +
      'anything was changed outside SuperLocalMemory. It only reads; nothing is changed.'));
    var list = el('div', 'truth-checks');
    var run = el('button', 'btn primary', 'Check now');
    run.type = 'button';
    run.addEventListener('click', function (e) {
      e.preventDefault();
      run.disabled = true;
      list.textContent = 'Checking…';
      fetch(API, { credentials: 'same-origin' })
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

  window.odMountTrustOverview = mount;
  window.SLMTrustOverview = { _render: render };
}());
