// od-dash-home.js: the Dashboard home tiles and summary cards.
//
// Read-only. Every tile and card loads from its own GET endpoint, shows a
// skeleton while it waits, then a value or an honest "Unavailable" with a
// Retry button. One slow or failing endpoint never blanks the rest of the page.
// The Memories / Graph / Facts tiles are filled by dashboard.js (GET /api/stats).
(function () {
  'use strict';

  var TIMEOUT_MS = 25000;
  var STALE_MS = 30000;
  var lastLoad = 0;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function getJson(url) {
    var ctl = (typeof AbortController === 'function') ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, TIMEOUT_MS) : null;
    return fetch(url, { credentials: 'same-origin', signal: ctl ? ctl.signal : undefined })
      .then(function (r) {
        if (timer) clearTimeout(timer);
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }, function (e) { if (timer) clearTimeout(timer); throw e; });
  }

  function skeleton(node, lines) {
    node.textContent = '';
    var wrap = el('div', 'skel-rows');
    wrap.setAttribute('aria-label', 'Loading');
    for (var i = 0; i < (lines || 2); i++) wrap.appendChild(el('span', 'skel skel-line'));
    node.appendChild(wrap);
  }

  function retryButton(fn) {
    var b = el('button', 'kpi-retry', 'Retry');
    b.type = 'button';
    b.addEventListener('click', function (e) { e.stopPropagation(); fn(); });
    return b;
  }

  // ----- KPI tiles -----
  function tile(valueId, deltaId) {
    var v = $(valueId), d = $(deltaId);
    return {
      loading: function () {
        if (v) { v.textContent = ''; var s = el('span', 'skel skel-num'); s.setAttribute('role', 'img'); s.setAttribute('aria-label', 'Loading'); v.appendChild(s); }
        if (d) d.textContent = ' ';
      },
      value: function (text, delta, tone) {
        if (v) { v.textContent = text; v.className = v.className.replace(/\s*tone-\w+/g, '') + (tone ? ' tone-' + tone : ''); }
        if (d) d.textContent = delta || ' ';
      },
      failed: function (retry) {
        if (v) { v.textContent = ''; v.appendChild(el('span', 'kpi-unavailable', 'Unavailable')); }
        if (d) { d.textContent = ''; d.appendChild(retryButton(retry)); }
      }
    };
  }

  function load(t, fn) {
    t.loading();
    return fn().catch(function () { t.failed(function () { load(t, fn); }); });
  }

  // ----- tiles -----
  function docsTile() {
    var t = tile('k-docs', 'k-docs-delta');
    return load(t, function () {
      return getJson('/api/v3/features').then(function (f) {
        var media = (f && f.media) || {};
        if (!media.enabled) {
          t.value('Off', 'Turn on in Documents & Images', 'muted');
          return;
        }
        return getJson('/api/v3/documents').then(function (d) {
          var n = (d.documents || []).length;
          t.value(String(n) + (d.next_cursor ? '+' : ''), n === 1 ? 'document read' : 'documents read');
        });
      });
    });
  }

  var appsData = { local: null, conns: null, peers: null };

  function appsTile() {
    var t = tile('k-apps', 'k-apps-delta');
    return load(t, function () {
      return Promise.all([
        getJson('/api/v3/local-apps').catch(function () { return null; }),
        getJson('/api/v3/connections/status').catch(function () { return null; }),
        getJson('/api/v3/mesh/peers').catch(function () { return null; })
      ]).then(function (r) {
        if (!r[0] && !r[1] && !r[2]) throw new Error('all failed');
        appsData = { local: r[0], conns: r[1], peers: r[2] };
        var apps = r[0] ? (r[0].confirmed_count != null ? r[0].confirmed_count : (r[0].apps || []).length) : 0;
        var web = r[1] ? (r[1].connections || []).length : 0;
        var peers = r[2] ? (r[2].peers || []).length : 0;
        var total = apps + web;
        t.value(String(total), peers + (peers === 1 ? ' peer' : ' peers') + ' on the mesh');
        renderAppsCard();
      });
    });
  }

  function answerCheckTile() {
    var t = tile('k-ac', 'k-ac-delta');
    return load(t, function () {
      return getJson('/api/v3/answer-check').then(function (s) {
        var state = s.laya && s.laya.state;
        if (state === 'failed') return t.value('Needs repair', 'Open to repair', 'warn');
        if (s.active === 'laya') return t.value('On this Mac', 'Checks answers on this computer');
        if (s.active === 'jev') return t.value('Online', 'Uses the provider you chose');
        if (state === 'installing') return t.value('Setting up', 'Working in the background');
        t.value('Off', 'Open to set up', 'muted');
      });
    });
  }

  function healthTile() {
    var t = tile('k-health', 'k-health-delta');
    var body = $('dash-health-body');
    if (body) skeleton(body, 2);
    return load(t, function () {
      return getJson('/api/memory-health').then(function (d) {
        var enr = d.enrichment || {};
        var open = (d.repairs || []).length + (d.suggestions || []).length;
        var label, tone;
        if ((enr.errors || 0) > 0 || enr.last_error) { label = 'Needs attention'; tone = 'warn'; }
        else if (open > 0) { label = 'Needs a look'; tone = 'warn'; }
        else if (enr.status && enr.status !== 'completed') { label = 'Working'; tone = ''; }
        else { label = 'Up to date'; tone = 'ok'; }
        t.value(label, d.state || d.summary || '', tone);
        renderHealthCard(d, open);
      });
    }).then(function () {
      if (body && body.querySelector('.skel-rows')) {
        body.textContent = '';
        body.appendChild(el('p', 'muted', 'Memory health could not be read right now.'));
        body.appendChild(retryButton(healthTile));
      }
    });
  }

  function renderHealthCard(d, open) {
    var body = $('dash-health-body');
    if (!body) return;
    body.textContent = '';
    body.appendChild(el('p', 'dash-card-lead', d.state || 'Checked in the background.'));
    if (d.summary) body.appendChild(el('p', 'muted', d.summary));
    var c = d.coverage || {};
    var bits = [];
    if (c.facts != null) bits.push(Number(c.facts).toLocaleString() + ' facts');
    if (c.entities != null) bits.push(Number(c.entities).toLocaleString() + ' entities');
    if (c.tags != null) bits.push(Number(c.tags).toLocaleString() + ' tags');
    if (bits.length) body.appendChild(el('p', 'dash-card-meta', bits.join(' · ')));
    if (open > 0) body.appendChild(el('p', 'dash-card-meta tone-warn', open + (open === 1 ? ' suggestion waiting for your say-so' : ' suggestions waiting for your say-so')));
  }

  function renderAppsCard() {
    var body = $('dash-apps-body');
    if (!body) return;
    body.textContent = '';
    var names = [];
    if (appsData.local && appsData.local.apps) {
      appsData.local.apps.forEach(function (a) { if (a.name && names.indexOf(a.name) < 0) names.push(a.name); });
    }
    if (appsData.conns && appsData.conns.connections) {
      appsData.conns.connections.forEach(function (c) { if (c.host && names.indexOf(c.host) < 0) names.push(c.host); });
    }
    var peers = appsData.peers && appsData.peers.peers ? appsData.peers.peers : [];
    if (!names.length && !peers.length) {
      var empty = el('div', 'slm-empty slm-empty-compact');
      var img = el('img', 'slm-empty-mascot');
      img.src = '/static/assets/mascot/peek-320.webp'; img.alt = ''; img.width = 72; img.height = 72; img.loading = 'lazy';
      empty.appendChild(img);
      var txt = el('div', 'slm-empty-body');
      txt.appendChild(el('p', 'dash-card-lead', 'No apps connected yet'));
      txt.appendChild(el('p', 'muted', 'Connect an AI app and it shares this memory. In a terminal, run slm connect followed by the app name, or use Connected apps.'));
      empty.appendChild(txt);
      body.appendChild(empty);
      return;
    }
    if (names.length) {
      var list = el('div', 'dash-tags');
      names.slice(0, 8).forEach(function (n) { list.appendChild(el('span', 'chip-pill', n)); });
      if (names.length > 8) list.appendChild(el('span', 'chip-pill', '+' + (names.length - 8) + ' more'));
      body.appendChild(list);
    }
    body.appendChild(el('p', 'dash-card-meta', peers.length + (peers.length === 1 ? ' peer' : ' peers') + ' in the mesh right now'));
  }

  function loadAll() {
    lastLoad = Date.now();
    docsTile(); appsTile(); answerCheckTile(); healthTile();
    if (window.SLMTruth) window.SLMTruth.reload();
    var body = $('dash-apps-body');
    if (body) skeleton(body, 2);
  }

  // Tiles and buttons with data-goto open the matching pane.
  document.addEventListener('click', function (e) {
    var g = e.target.closest ? e.target.closest('[data-goto]') : null;
    if (!g || e.target.closest('a,input,textarea,select') || (e.target.closest('button') && !g.matches('button'))) return;
    var link = document.querySelector('.nav-link[data-tab="' + g.getAttribute('data-goto') + '"]');
    if (link) link.click();
  });
  document.addEventListener('keydown', function (e) {
    if ((e.key !== 'Enter' && e.key !== ' ') || !e.target.matches || !e.target.matches('.kpi[data-goto]')) return;
    e.preventDefault();
    e.target.click();
  });

  function arm() {
    document.querySelectorAll('.kpi[data-goto]').forEach(function (k) {
      k.setAttribute('role', 'link'); k.setAttribute('tabindex', '0');
    });
  }

  window.SLMDashHome = { reload: loadAll };
  document.addEventListener('DOMContentLoaded', function () {
    arm();
    setTimeout(loadAll, 250);
  });
  window.addEventListener('focus', function () {
    if (Date.now() - lastLoad > STALE_MS && document.getElementById('dashboard-pane')) loadAll();
  });
}());
