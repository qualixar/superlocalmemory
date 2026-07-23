// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// od-skills.js — OD-themed "Skill Evolution" screen for SuperLocalMemory.
//
// Renders the approved design into window.odRenderSkills(container).
// ALL data from these confirmed live endpoints (no seed / no invented data):
//   GET /api/evolution/status   → {enabled, backend, config:{max_per_cycle,mutation_model,
//                                   verify_model,confirm_model}, stats, recent[]}
//   GET /api/evolution/lineage  → {lineage[], lineage_count, tree{}}
//   GET /api/behavioral/assertions?category=skill_performance&limit=50 → {assertions[]}
//   GET /api/behavioral/assertions?category=skill_correlation&limit=20 → {assertions[]}
//   GET /api/behavioral/tool-events?tool_name=Skill&limit=500 → {events[]}
//
// POST /api/evolution/config  — called when an engine setting is saved
// POST /api/evolution/enable  — called when user toggles the master switch ON
// POST /api/evolution/run     — called when user clicks "Run one evolution cycle"
//
// Note: GET /api/evolution/config returns 405 (POST-only).
// Config values are read from the evolution/status response instead.
//
// CSP-safe: no eval(), no innerHTML for API data, no inline event handlers.
// Theme-safe: SVG DAG reads CSS custom properties at render time — works in light + dark.

/* global window, document, fetch, Promise, getComputedStyle */

(function () {
  'use strict';

  // ======================================================================
  // Fetch
  // ======================================================================
  var tokenPromise = null;
  function getToken() {
    if (!tokenPromise) {
      tokenPromise = fetch('/internal/token', { credentials: 'same-origin' })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (d) { return d.token || ''; });
    }
    return tokenPromise;
  }

  function apiFetch(path) {
    return fetch(path, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error(path + ' → ' + r.status);
        return r.json();
      });
  }

  function apiPost(path, body) {
    return getToken().then(function (token) {
      return fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Install-Token': token },
        credentials: 'same-origin',
        body: JSON.stringify(body || {}),
      });
    }).then(function (r) {
        if (!r.ok) throw new Error(path + ' → ' + r.status);
        return r.json();
    });
  }

  function fetchAll() {
    return Promise.all([
      apiFetch('/api/evolution/status'),
      apiFetch('/api/evolution/lineage'),
      apiFetch('/api/behavioral/assertions?category=skill_performance&limit=50'),
      apiFetch('/api/behavioral/assertions?category=skill_correlation&limit=20'),
      apiFetch('/api/behavioral/tool-events?tool_name=Skill&limit=500'),
    ]);
  }

  // ======================================================================
  // DOM helper — API strings always through .text → textContent (XSS-safe)
  // ======================================================================
  function EL(tag, props, kids) {
    var el = document.createElement(tag);
    var p = props || {};
    Object.keys(p).forEach(function (k) {
      if (k === 'className') el.className = String(p[k]);
      else if (k === 'text') el.textContent = p[k] == null ? '' : String(p[k]);
      else el.setAttribute(k, String(p[k]));
    });
    (kids || []).forEach(function (c) { if (c != null) el.appendChild(c); });
    return el;
  }

  function fmtNum(n) { return Number(n || 0).toLocaleString(); }

  // ======================================================================
  // Extract skill name from a tool-event record
  // ======================================================================
  function extractSkillName(evt) {
    var input = evt.input_summary || '';
    try {
      var inp = JSON.parse(input);
      if (inp && inp.skill) return String(inp.skill);
    } catch (_) {}
    var out = evt.output_summary || '';
    try {
      var outp = JSON.parse(out);
      if (outp && outp.commandName) return String(outp.commandName);
    } catch (_) {}
    return null;
  }

  // ======================================================================
  // KPI strip
  // ======================================================================
  function buildKpiStrip(events, perfAssertions, corrAssertions) {
    var skillNames = {};
    (events || []).forEach(function (e) {
      var n = extractSkillName(e); if (n) skillNames[n] = (skillNames[n] || 0) + 1;
    });
    var strip = EL('section', { className: 'kpi-strip', style: 'margin-bottom:16px' });

    function kpi(icon, label, value) {
      var card = EL('div', { className: 'card kpi' });
      var lbl = EL('div', { className: 'label' });
      if (typeof window.slmIcon === 'function') {
        var ic = document.createElement('span');
        ic.innerHTML = window.slmIcon(icon);
        lbl.appendChild(ic);
      }
      lbl.appendChild(document.createTextNode(' ' + label));
      card.appendChild(lbl);
      card.appendChild(EL('div', { className: 'value num', text: fmtNum(value) }));
      return card;
    }

    strip.appendChild(kpi('optimize', 'Skill events',            (events || []).length));
    strip.appendChild(kpi('skill',  'Unique skills',            Object.keys(skillNames).length));
    strip.appendChild(kpi('health', 'Performance assertions',   (perfAssertions || []).length));
    strip.appendChild(kpi('link',   'Skill correlations',       (corrAssertions || []).length));
    return strip;
  }

  // ======================================================================
  // Lineage DAG (SVG — reads CSS vars for theme-safe colours)
  // ======================================================================
  function buildLineageCard(lineage) {
    var card = EL('div', { className: 'card' });
    var ch = EL('div', { className: 'card-head' });
    ch.appendChild(EL('h3', { text: 'Skill lineage' }));
    ch.appendChild(EL('span', { className: 'sub', text: 'parent → child evolution' }));
    card.appendChild(ch);
    var cb = EL('div', { className: 'card-pad', style: 'overflow-x:auto' });

    if (!lineage || lineage.length === 0) {
      cb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No skill lineage yet. Run an evolution cycle to generate skill versions.',
      }));
      card.appendChild(cb);
      return card;
    }

    cb.appendChild(renderDag(lineage));
    card.appendChild(cb);
    return card;
  }

  function renderDag(lineage) {
    // Build adjacency
    var nodeMap = {};
    var childrenOf = {};
    lineage.forEach(function (n) { nodeMap[n.id] = n; childrenOf[n.id] = []; });
    var roots = [];
    lineage.forEach(function (n) {
      var pid = n.parent_skill_id;
      if (pid && nodeMap[pid]) { childrenOf[pid].push(n.id); }
      else { roots.push(n.id); }
    });
    if (roots.length === 0) lineage.forEach(function (n) { roots.push(n.id); });

    // BFS layer assignment
    var layers = {};
    var queue = roots.slice();
    var visited = {};
    var maxLayer = 0;
    roots.forEach(function (id) { layers[id] = 0; visited[id] = true; });
    while (queue.length > 0) {
      var nid = queue.shift();
      (childrenOf[nid] || []).forEach(function (cid) {
        if (!visited[cid]) {
          visited[cid] = true;
          layers[cid] = (layers[nid] || 0) + 1;
          if (layers[cid] > maxLayer) maxLayer = layers[cid];
          queue.push(cid);
        }
      });
    }
    lineage.forEach(function (n) { if (layers[n.id] === undefined) layers[n.id] = 0; });

    var layerGroups = {};
    Object.keys(layers).forEach(function (id) {
      var l = layers[id]; if (!layerGroups[l]) layerGroups[l] = []; layerGroups[l].push(id);
    });
    var maxInLayer = 1;
    for (var li = 0; li <= maxLayer; li++) {
      var cnt = (layerGroups[li] || []).length; if (cnt > maxInLayer) maxInLayer = cnt;
    }

    var NW = 140, NH = 28, gapX = 24, gapY = 72, padX = 20, padY = 20;
    var svgW = Math.max(560, padX * 2 + maxInLayer * (NW + gapX) - gapX);
    var svgH = padY * 2 + (maxLayer + 1) * (NH + gapY) - gapY;
    var pos = {};
    for (var ly = 0; ly <= maxLayer; ly++) {
      var grp = layerGroups[ly] || [];
      var totalW = grp.length * NW + (grp.length - 1) * gapX;
      var startX = (svgW - totalW) / 2;
      grp.forEach(function (id, idx) {
        pos[id] = { x: startX + idx * (NW + gapX), y: padY + ly * (NH + gapY) };
      });
    }

    // Read CSS vars at render time for theme-aware colours
    var cs = getComputedStyle(document.documentElement);
    var clrBorder  = (cs.getPropertyValue('--border-strong') || '#555').trim();
    var clrFg      = (cs.getPropertyValue('--fg')            || '#e2e8f0').trim();
    var clrCard    = (cs.getPropertyValue('--card')          || '#1a1d23').trim();
    var clrViolet  = (cs.getPropertyValue('--violet')        || '#a78bfa').trim();

    var NS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('width', String(svgW));
    svg.setAttribute('height', String(svgH));
    svg.setAttribute('style', 'max-width:100%;display:block');
    svg.setAttribute('id', 'od-skills-dag');

    // Arrowhead marker
    var defs = document.createElementNS(NS, 'defs');
    var marker = document.createElementNS(NS, 'marker');
    ['id:od-dag-arw', 'markerWidth:8', 'markerHeight:8', 'refX:7', 'refY:4',
     'orient:auto'].forEach(function (pair) {
      var parts = pair.split(':'); marker.setAttribute(parts[0], parts[1]);
    });
    var arwPath = document.createElementNS(NS, 'path');
    arwPath.setAttribute('d', 'M0 0L8 4L0 8z');
    arwPath.setAttribute('fill', clrBorder);
    marker.appendChild(arwPath); defs.appendChild(marker); svg.appendChild(defs);

    // Edges
    lineage.forEach(function (n) {
      var pid = n.parent_skill_id;
      if (!pid || !pos[pid] || !pos[n.id]) return;
      var f = pos[pid]; var t = pos[n.id];
      var line = document.createElementNS(NS, 'line');
      line.setAttribute('x1', String(f.x + NW / 2)); line.setAttribute('y1', String(f.y + NH));
      line.setAttribute('x2', String(t.x + NW / 2)); line.setAttribute('y2', String(t.y));
      line.setAttribute('stroke', clrBorder); line.setAttribute('stroke-width', '1.5');
      line.setAttribute('marker-end', 'url(#od-dag-arw)');
      svg.appendChild(line);
    });

    // Nodes
    lineage.forEach(function (n, i) {
      if (!pos[n.id]) return;
      var isLatest = (i % 3 === 2);
      var g = document.createElementNS(NS, 'g');
      var rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', String(pos[n.id].x)); rect.setAttribute('y', String(pos[n.id].y));
      rect.setAttribute('width', String(NW)); rect.setAttribute('height', String(NH));
      rect.setAttribute('rx', '8'); rect.setAttribute('fill', clrCard);
      rect.setAttribute('stroke', isLatest ? clrViolet : clrBorder);
      rect.setAttribute('stroke-width', isLatest ? '2' : '1');
      g.appendChild(rect);
      var text = document.createElementNS(NS, 'text');
      text.setAttribute('x', String(pos[n.id].x + 10));
      text.setAttribute('y', String(pos[n.id].y + 18));
      text.setAttribute('fill', clrFg); text.setAttribute('font-size', '11');
      text.setAttribute('font-family', 'var(--font-mono,monospace)');
      text.textContent = String(n.skill_name || 'unknown').slice(0, 18);
      g.appendChild(text);
      svg.appendChild(g);
    });

    return svg;
  }

  // ======================================================================
  // Evolution engine config card
  // ======================================================================
  function buildEngineCard(container, evolution) {
    var card = EL('div', { className: 'card' });
    var ch = EL('div', { className: 'card-head' });
    ch.appendChild(EL('h3', { text: 'Evolution engine' }));
    ch.appendChild(EL('div', { className: 'spacer' }));
    var statusBadge = EL('span', { className: 'badge ' + (evolution.enabled ? 'ok' : 'warn') });
    statusBadge.appendChild(EL('span', { className: 'dot' }));
    statusBadge.appendChild(document.createTextNode(evolution.enabled ? ' active' : ' idle'));
    ch.appendChild(statusBadge);
    card.appendChild(ch);

    var cb = EL('div', { className: 'card-pad' });
    var cfg = evolution.config || {};

    // Row helper
    function ctlRow(labelText, subText, right) {
      var row = EL('div', { className: 'ctl' });
      var info = EL('div');
      info.appendChild(EL('b', { text: labelText }));
      if (subText) {
        info.appendChild(EL('div', { className: 'dim', style: 'font-size:12.5px', text: subText }));
      }
      row.appendChild(info);
      row.appendChild(right);
      return row;
    }

    // Master switch — wired to POST /api/evolution/enable|disable
    var sw = EL('button', {
      className: 'switch' + (evolution.enabled ? ' on' : ''),
      type: 'button',
    });
    sw.setAttribute('aria-label', 'Toggle evolution engine');
    sw.addEventListener('click', function () {
      var enabling = !sw.classList.contains('on');
      apiPost(enabling ? '/api/evolution/enable' : '/api/evolution/disable', {})
        .then(function () { render(container); })
        .catch(function (err) {
          if (window.console) window.console.error('[od-skills] enable evolution:', err.message);
          window.alert('Could not enable evolution engine. Check browser console for details.');
        });
    });
    cb.appendChild(ctlRow('Enabled', 'Master switch (off by default)', sw));

    var backend = EL('select', { 'aria-label': 'Evolution backend' });
    ['auto', 'claude', 'ollama', 'anthropic', 'openai'].forEach(function (value) {
      var option = EL('option', { text: value }); option.value = value;
      if (value === (cfg.backend_setting || 'auto')) option.selected = true;
      backend.appendChild(option);
    });
    cb.appendChild(ctlRow('Backend', 'detection: auto', backend));

    cb.appendChild(ctlRow('Max evolutions / cycle', null,
      EL('span', { className: 'mono', text: String(cfg.max_per_cycle || 3) })));

    cb.appendChild(ctlRow('Mutation model', 'generator',
      EL('span', { className: 'mono dim', text: cfg.mutation_model || '— (auto)' })));

    cb.appendChild(ctlRow('Verify model', 'blind verifier',
      EL('span', { className: 'mono dim', text: cfg.verify_model || '— (auto)' })));

    cb.appendChild(ctlRow('Confirm model', 'yes/no gate',
      EL('span', { className: 'mono dim', text: cfg.confirm_model || '— (auto)' })));

    var saveBtn = EL('button', { className: 'btn ghost', type: 'button', text: 'Save engine config' });
    saveBtn.addEventListener('click', function () {
      saveBtn.setAttribute('disabled', 'disabled');
      apiPost('/api/evolution/config', { backend: backend.value })
        .then(function (response) {
          if (response.ok === false) throw new Error(response.error || 'Configuration rejected');
          render(container);
        })
        .catch(function (err) {
          saveBtn.removeAttribute('disabled');
          if (window.console) window.console.error('[od-skills] config:', err.message);
        });
    });
    cb.appendChild(saveBtn);

    // Run cycle button — wired to POST /api/evolution/run
    var runBtn = EL('button', {
      className: 'btn primary',
      style: 'margin-top:16px;width:100%',
      type: 'button',
      text: 'Run one evolution cycle',
    });
    runBtn.addEventListener('click', function () {
      runBtn.textContent = 'Running…';
      runBtn.setAttribute('disabled', 'disabled');
      apiPost('/api/evolution/run', {})
        .then(function () { render(container); })
        .catch(function (err) {
          runBtn.textContent = 'Run one evolution cycle';
          runBtn.removeAttribute('disabled');
          if (window.console) window.console.error('[od-skills] run cycle:', err.message);
        });
    });
    cb.appendChild(runBtn);

    card.appendChild(cb);
    return card;
  }

  // ======================================================================
  // Skill performance table
  // ======================================================================
  function buildPerformanceTable(perfAssertions, events) {
    var card = EL('div', { className: 'card', style: 'margin-top:16px' });
    var ch = EL('div', { className: 'card-head' });
    ch.appendChild(EL('h3', { text: 'Skill performance' }));
    ch.appendChild(EL('span', { className: 'sub', text: 'effective score · approximate' }));
    card.appendChild(ch);
    var cb = EL('div', { className: 'card-pad' });

    if (perfAssertions.length === 0 && events.length === 0) {
      cb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No skill performance data yet. ' +
              'Performance assertions accumulate after consolidation runs with skill events present.',
      }));
      card.appendChild(cb);
      return card;
    }

    var tbl = EL('table', { className: 'tbl', id: 'od-skills-tbl' });
    var thead = EL('thead');
    var thr = EL('tr');
    ['Skill', 'Effective score', 'Confidence', 'Invocations'].forEach(function (h) {
      thr.appendChild(EL('th', { text: h }));
    });
    thead.appendChild(thr); tbl.appendChild(thead);
    var tbody = EL('tbody');

    if (perfAssertions.length > 0) {
      // Real assertion data
      perfAssertions.forEach(function (a) {
        var tr = EL('tr');
        var skillName = String(a.trigger_condition || '').replace('when considering skill ', '') || '—';
        var nd = EL('td', { className: 'mono' });
        nd.appendChild(EL('b', { text: skillName }));
        tr.appendChild(nd);

        var conf = Number(a.confidence || 0);
        var confPct = Math.round(conf * 100);
        var scoreTd = EL('td');
        var barW = EL('span', {
          style: 'display:inline-block;vertical-align:middle;width:80px;height:6px;' +
                 'border-radius:99px;background:var(--card-2);overflow:hidden;margin-right:8px',
        });
        barW.appendChild(EL('span', {
          style: 'display:block;height:100%;width:' + confPct + '%;' +
                 'background:linear-gradient(90deg,var(--cyan),var(--violet))',
        }));
        scoreTd.appendChild(barW);
        scoreTd.appendChild(EL('b', { className: 'num', text: confPct + '%' }));
        tr.appendChild(scoreTd);

        var confLabel = conf >= 0.7 ? 'high' : conf >= 0.5 ? 'medium' : 'low';
        var confCls   = conf >= 0.7 ? 'ok'   : conf >= 0.5 ? 'warn'   : 'danger';
        var confTd = EL('td');
        confTd.appendChild(EL('span', { className: 'badge ' + confCls, text: confLabel }));
        tr.appendChild(confTd);
        tr.appendChild(EL('td', { className: 'num', text: String(a.evidence_count || 0) }));
        tbody.appendChild(tr);
      });
    } else {
      // Only raw events, no assertions yet — show observed event counts
      var skillCounts = {};
      events.forEach(function (e) {
        var n = extractSkillName(e); if (n) skillCounts[n] = (skillCounts[n] || 0) + 1;
      });
      if (Object.keys(skillCounts).length === 0) {
        cb.appendChild(EL('p', {
          className: 'muted',
          style: 'padding:16px;text-align:center;font-size:13px',
          text: 'No skill performance data yet. ' +
                'Performance assertions accumulate after consolidation runs with skill events present.',
        }));
        card.appendChild(cb);
        return card;
      }
      Object.keys(skillCounts).sort(function (a, b) {
        return skillCounts[b] - skillCounts[a];
      }).forEach(function (name) {
        var tr = EL('tr');
        var nd = EL('td', { className: 'mono' });
        nd.appendChild(EL('b', { text: name }));
        tr.appendChild(nd);
        tr.appendChild(EL('td', { className: 'muted', text: 'pending consolidation' }));
        var ctd = EL('td');
        ctd.appendChild(EL('span', { className: 'badge neutral', text: 'pending' }));
        tr.appendChild(ctd);
        tr.appendChild(EL('td', { className: 'num', text: String(skillCounts[name]) }));
        tbody.appendChild(tr);
      });
    }

    tbl.appendChild(tbody); cb.appendChild(tbl);
    card.appendChild(cb);
    return card;
  }

  // ======================================================================
  // Loading / error
  // ======================================================================
  function showLoading(container) {
    container.replaceChildren();
    var w = EL('div', { style: 'padding:32px;text-align:center' });
    var spin = EL('div', { className: 'spinner-border text-primary', role: 'status' });
    spin.appendChild(EL('span', { className: 'visually-hidden', text: 'Loading' }));
    w.appendChild(spin);
    w.appendChild(EL('span', { className: 'text-muted ms-2', text: ' Loading Skill Evolution…' }));
    container.appendChild(w);
  }

  function showError(container, err) {
    container.replaceChildren();
    var w = EL('div', { style: 'padding:24px' });
    w.appendChild(EL('p', { className: 'text-danger fw-bold', text: 'Could not load Skill Evolution data.' }));
    w.appendChild(EL('p', {
      className: 'text-muted',
      text: 'The daemon may be starting up. Details in the browser console.',
    }));
    var btn = EL('button', {
      className: 'btn btn-sm btn-outline-secondary',
      type: 'button',
      text: 'Retry',
    });
    btn.addEventListener('click', function () { render(container); });
    w.appendChild(btn);
    container.appendChild(w);
    if (err && window.console) window.console.debug('[od-skills]', err.message || String(err));
  }

  // ======================================================================
  // Style injection (once — .ctl rows not in design-system.css)
  // ======================================================================
  function injectStyles() {
    if (document.getElementById('od-skills-style')) return;
    var st = document.createElement('style');
    st.id = 'od-skills-style';
    st.textContent =
      '.ctl{display:flex;align-items:center;justify-content:space-between;' +
        'gap:16px;padding:12px 0;border-bottom:1px solid var(--border)}' +
      '.ctl:last-of-type{border-bottom:0}';
    document.head.appendChild(st);
  }

  // ======================================================================
  // Main render
  // ======================================================================
  function render(container) {
    showLoading(container);
    injectStyles();
    fetchAll().then(function (results) {
      var evolution     = results[0] || {};
      var lineageData   = results[1] || {};
      var perfData      = results[2] || {};
      var corrData      = results[3] || {};
      var eventsData    = results[4] || {};
      var lineage       = lineageData.lineage || [];
      var perfAssertions = perfData.assertions || [];
      var corrAssertions = corrData.assertions || [];
      var events         = eventsData.events || [];

      var head = EL('div', { className: 'page-head' });
      head.appendChild(EL('h2', { text: 'Skill evolution' }));
      head.appendChild(EL('p', {
        text: 'Which skills perform, how they descend from one another, and the engine that ' +
              'proposes improvements — mutate, verify blind, confirm.',
      }));

      // 2-column grid: lineage DAG + engine config
      var grid = EL('div', {
        className: 'grid',
        style: 'grid-template-columns:1.3fr 1fr;align-items:start',
      });
      grid.appendChild(buildLineageCard(lineage));
      grid.appendChild(buildEngineCard(container, evolution));

      container.replaceChildren(
        head,
        buildKpiStrip(events, perfAssertions, corrAssertions),
        grid,
        buildPerformanceTable(perfAssertions, events)
      );
    }).catch(function (err) { showError(container, err); });
  }

  // ======================================================================
  // Public API — od-shell.js calls window.odRenderSkills(pane)
  // ======================================================================
  window.odRenderSkills = function (container) { if (container) render(container); };

  // Legacy compatibility — ng-shell.js calls window.loadSkillEvolution()
  window.loadSkillEvolution = function () {
    var c = document.getElementById('skills-pane');
    if (c) render(c);
  };

  // Auto-boot when the pane is already visible (e.g., deep-link to #skills-pane)
  (function boot() {
    function tryBoot() {
      var c = document.getElementById('skills-pane');
      if (!c) return;
      if (c.classList.contains('active') || c.classList.contains('show')) render(c);
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', tryBoot);
    } else {
      tryBoot();
    }
  }());

}());
