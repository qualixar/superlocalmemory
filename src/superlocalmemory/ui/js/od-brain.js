// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// od-brain.js — OD-themed "Living Brain" screen for SuperLocalMemory.
//
// Renders the approved design into window.odRenderBrain(container).
// ALL data from these confirmed live endpoints (no seed / no invented data):
//   GET /api/learning/status   → ranking_phase, stats, engagement, tech_preferences,
//                                 workflow_patterns, persisted source-quality posteriors
//   GET /api/behavioral/status → reward_telemetry, explicit outcome counts,
//                                 patterns[], transfers, recent_outcomes[]
//   GET /api/behavioral/assertions?category=skill_performance&limit=50 → assertions[]
//   GET /api/behavioral/tool-events?limit=500 → events[].{created_at,tool_name,…}
//
// Heatmap is built from tool-events aggregated by UTC date — real activity density,
// not the random slmHeatmap() fallback.
//
// CSP-safe: no eval(), no innerHTML for API data, no inline event handlers.
// Immutable style: EL() always creates new nodes; no mutation of existing DOM.

/* global window, document, fetch, Promise, setTimeout, getComputedStyle */

(function () {
  'use strict';

  // ======================================================================
  // Fetch
  // ======================================================================
  function apiFetch(path) {
    return fetch(path, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error(path + ' → ' + r.status);
        return r.json();
      });
  }

  function fetchAll() {
    return Promise.all([
      apiFetch('/api/learning/status'),
      apiFetch('/api/behavioral/status'),
      apiFetch('/api/behavioral/assertions?category=skill_performance&limit=50'),
      apiFetch('/api/behavioral/tool-events?limit=500'),
    ]);
  }

  // ======================================================================
  // DOM helper — API strings always go through .text → textContent (XSS-safe)
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

  // ======================================================================
  // Formatters
  // ======================================================================
  function fmtNum(n) { return Number(n || 0).toLocaleString(); }

  function fmtKB(kb) {
    var v = Number(kb || 0);
    if (v === 0) return '0 KB';
    return v < 1024 ? v.toFixed(1) + ' KB' : (v / 1024).toFixed(1) + ' MB';
  }

  function phaseLabel(raw) {
    return (raw || 'cold_start')
      .replace(/_/g, '-')
      .replace(/\b([a-z])/g, function (c) { return c.toUpperCase(); });
  }

  // ======================================================================
  // Heatmap — real activity density from tool-events aggregated by UTC date
  // ======================================================================
  function buildDateMap(events) {
    var map = {};
    (events || []).forEach(function (e) {
      if (!e.created_at) return;
      var d = String(e.created_at).slice(0, 10);
      map[d] = (map[d] || 0) + 1;
    });
    return map;
  }

  function signalLevel(count) {
    if (count <= 0) return 0;
    if (count <= 3) return 1;
    if (count <= 10) return 2;
    if (count <= 25) return 3;
    return 4;
  }

  function buildHeatmap(el, dateMap, weeks) {
    var totalDays = (weeks || 26) * 7;
    var today = new Date(); today.setHours(0, 0, 0, 0);
    el.replaceChildren();
    for (var i = 0; i < totalDays; i++) {
      var d = new Date(today);
      d.setDate(today.getDate() - (totalDays - 1 - i));
      var ds = d.toISOString().slice(0, 10);
      var cnt = (dateMap && dateMap[ds]) || 0;
      var cell = document.createElement('i');
      cell.setAttribute('data-l', String(signalLevel(cnt)));
      cell.setAttribute('title', ds + ': ' + cnt + ' event' + (cnt === 1 ? '' : 's'));
      el.appendChild(cell);
    }
  }

  // ======================================================================
  // UI primitives
  // ======================================================================
  function meter(pct, color) {
    var outer = EL('div', { className: 'meter' });
    var fill = document.createElement('i');
    fill.style.width = Math.min(100, Math.max(0, pct)) + '%';
    if (color) fill.style.background = color;
    outer.appendChild(fill);
    return outer;
  }

  function inlineBar(pct) {
    var wrap = EL('span', {
      style: 'display:inline-block;vertical-align:middle;width:56px;height:6px;' +
             'border-radius:99px;background:var(--card-2);overflow:hidden;margin-right:6px',
    });
    wrap.appendChild(EL('span', {
      style: 'display:block;height:100%;width:' + Math.min(100, pct) + '%;' +
             'background:linear-gradient(90deg,var(--cyan),var(--violet))',
    }));
    return wrap;
  }

  function heatLegend() {
    var w = EL('div', { className: 'heat-legend' });
    w.appendChild(document.createTextNode('less '));
    for (var i = 0; i <= 4; i++) w.appendChild(EL('i', { 'data-l': String(i) }));
    w.appendChild(document.createTextNode(' more'));
    return w;
  }

  // isNumeric=true → class="value num" (tabular digits, 30px default from CSS)
  // isNumeric=false → class="value" + font-size:24px (design uses smaller text for long labels)
  function kpiCard(icon, label, value, delta, dUp, vColor, isNumeric) {
    var card = EL('div', { className: 'card kpi' });
    var lbl = EL('div', { className: 'label' });
    if (typeof window.slmIcon === 'function') {
      var ic = document.createElement('span');
      ic.innerHTML = window.slmIcon(icon);
      lbl.appendChild(ic);
    }
    lbl.appendChild(document.createTextNode(' ' + label));
    var val = EL('div', { className: isNumeric ? 'value num' : 'value', text: value });
    if (!isNumeric) val.style.fontSize = '24px';
    if (vColor) val.style.color = vColor;
    card.appendChild(lbl);
    card.appendChild(val);
    card.appendChild(EL('div', { className: 'delta ' + (dUp ? 'up' : ''), text: delta }));
    return card;
  }

  // ======================================================================
  // Phase style injection (once — not in design-system.css yet)
  // ======================================================================
  function injectPhaseStyle() {
    if (document.getElementById('od-brain-phase-style')) return;
    var st = document.createElement('style');
    st.id = 'od-brain-phase-style';
    st.textContent =
      '.phase{flex:1;text-align:center;padding:10px 6px;border-radius:var(--r-md);' +
        'background:var(--card-2);border:1px solid var(--border)}' +
      '.phase b{display:block;font-size:13px}' +
      '.phase span{font-size:11px;color:var(--fg-3)}' +
      '.phase.done{border-color:color-mix(in srgb,var(--ok) 40%,transparent)}' +
      '.phase.done b{color:var(--ok)}' +
      '.phase.now{border-color:var(--violet-line);background:var(--violet-soft)}' +
      '.phase.now b{color:var(--violet)}';
    document.head.appendChild(st);
  }

  // ======================================================================
  // Tab: OVERVIEW
  // ======================================================================
  function buildOverview(learning, behavioral, dateMap) {
    var sec = EL('section', { className: 'tabpane active', 'data-p': 'overview' });
    var stats = (learning && learning.stats) || {};
    var eng = (learning && learning.engagement) || {};
    var beh = behavioral || {};
    var ranker = (learning && learning.ranker_phase) || {};
    var gates = ranker.gates || {};
    var ruleGate = Math.max(1, Number(gates.rule_based_min_signals || 1));
    var mlGate = Math.max(ruleGate, Number(gates.ml_model_min_signals || ruleGate));
    var signals = Number(ranker.signals != null
      ? ranker.signals : stats.ranker_signal_count || 0);
    var pct = Math.min(100, Math.round(signals / mlGate * 100));
    var phase = ranker.key || (learning && learning.ranking_phase) || 'baseline';
    var phaseNumber = Number(ranker.phase || 1);
    var modelActive = Boolean(ranker.model_active);
    var phaseDelta = modelActive
      ? 'Verified active model'
      : signals < mlGate
        ? fmtNum(mlGate - signals) + ' to ML data gate'
        : 'ML data gate met · verified model required';
    var healthStatus = (eng.health_status || 'INACTIVE').toUpperCase();
    var healthColor = healthStatus === 'HEALTHY' ? 'var(--ok)'
      : healthStatus === 'ACTIVE' ? 'var(--cyan)' : undefined;
    var pCount = ((beh.patterns) || []).length;

    // KPI strip
    var strip = EL('div', { className: 'kpi-strip', style: 'margin-bottom:16px' });
    // Ranking phase: text label → isNumeric=false (font-size:24px to match design)
    strip.appendChild(kpiCard('skill', 'Ranking phase', phaseLabel(phase),
      phaseDelta, modelActive, undefined, false));
    // Feedback signals: numeric → isNumeric=true
    strip.appendChild(kpiCard('optimize', 'Feedback signals', fmtNum(signals),
      '▲ ' + fmtNum(stats.unique_queries || 0) + ' unique queries', true, undefined, true));
    // Engagement health: text label → isNumeric=false
    strip.appendChild(kpiCard('health', 'Engagement health',
      healthStatus.charAt(0) + healthStatus.slice(1).toLowerCase(),
      (eng.days_active || 0) + ' days active · ' + Number(eng.memories_per_day || 0).toFixed(1) + ' mem/day',
      healthStatus === 'HEALTHY', healthColor, false));
    // Patterns: numeric → isNumeric=true
    strip.appendChild(kpiCard('brain', 'Patterns learned', String(pCount),
      '▲ ' + (beh.cross_project_transfers || 0) + ' transferable', pCount > 0, undefined, true));
    sec.appendChild(strip);

    // 2-column grid
    var grid = EL('div', { className: 'grid', style: 'grid-template-columns:1fr 1fr;align-items:start' });

    // Progress card
    var pcrd = EL('div', { className: 'card' });
    var ph = EL('div', { className: 'card-head' });
    ph.appendChild(EL('h3', { text: 'Adaptive ranking progress' }));
    ph.appendChild(EL('span', { className: 'sub', text: 'baseline → rule-based → ML model' }));
    pcrd.appendChild(ph);
    var pb = EL('div', { className: 'card-pad' });
    var pmeta = EL('div', {
      style: 'display:flex;justify-content:space-between;font-size:12px;color:var(--fg-2);margin-bottom:8px',
    });
    pmeta.appendChild(EL('span', { text: fmtNum(signals) + ' / ' + fmtNum(mlGate) + ' signals' }));
    pmeta.appendChild(EL('span', { className: 'num', text: pct + '%' }));
    pb.appendChild(pmeta);
    pb.appendChild(meter(pct));
    var phasesRow = EL('div', {
      style: 'display:flex;justify-content:space-between;margin-top:18px;gap:12px',
    });
    [
      {
        t: 'Baseline',
        d: '0–' + Math.max(0, ruleGate - 1) + ' signals',
        done: phaseNumber > 1,
        now: phaseNumber === 1,
      },
      {
        t: 'Rule-Based',
        d: fmtNum(ruleGate) + '–' + fmtNum(Math.max(ruleGate, mlGate - 1)) + ' signals',
        done: phaseNumber > 2,
        now: phaseNumber === 2,
      },
      {
        t: 'ML Model',
        d: fmtNum(mlGate) + '+ · ' + (modelActive ? 'verified active' : 'verified model required'),
        done: false,
        now: phaseNumber === 3 && modelActive,
      },
    ].forEach(function (phI) {
      var el = EL('div', {
        className: 'phase' + (phI.done ? ' done' : '') + (phI.now ? ' now' : ''),
      });
      el.appendChild(EL('b', { text: phI.t }));
      el.appendChild(EL('span', { text: phI.d }));
      phasesRow.appendChild(el);
    });
    pb.appendChild(phasesRow);
    pcrd.appendChild(pb);
    grid.appendChild(pcrd);

    // Privacy card
    var priv = EL('div', { className: 'card' });
    var prvh = EL('div', { className: 'card-head' });
    prvh.appendChild(EL('h3', { text: 'Privacy & data' }));
    prvh.appendChild(EL('span', { className: 'sub', text: 'learning.db — separate from memory.db' }));
    priv.appendChild(prvh);
    var prvb = EL('div', { className: 'card-pad', style: 'display:flex;flex-direction:column;gap:2px' });
    [
      ['Learning DB size',  fmtKB(stats.db_size_kb)],
      ['Patterns learned',  String(pCount)],
      ['Models trained',    String(stats.models_trained || 0)],
      ['Verified active models', String(stats.models_active_verified || 0)],
      ['Sources tracked',   String(stats.tracked_sources || 0)],
    ].forEach(function (row) {
      var r = EL('div', { className: 'list-row' });
      r.appendChild(EL('span', { className: 'muted', style: 'flex:1', text: row[0] }));
      r.appendChild(EL('b', { className: 'num', text: row[1] }));
      prvb.appendChild(r);
    });
    var lr = EL('div', { className: 'list-row' });
    lr.appendChild(EL('span', { className: 'muted', style: 'flex:1', text: 'Never leaves device' }));
    var badge = EL('span', { className: 'badge ok' });
    badge.appendChild(EL('span', { className: 'dot' }));
    badge.appendChild(document.createTextNode(' local'));
    lr.appendChild(badge);
    prvb.appendChild(lr);
    priv.appendChild(prvb);
    grid.appendChild(priv);
    sec.appendChild(grid);

    // Activity heatmap (tool events are activity, never reward labels)
    var hmc = EL('div', { className: 'card', style: 'margin-top:16px' });
    var hmh = EL('div', { className: 'card-head' });
    hmh.appendChild(EL('h3', { text: 'Memory activity' }));
    hmh.appendChild(EL('span', { className: 'sub', text: 'tool events · last 26 weeks' }));
    hmh.appendChild(EL('div', { className: 'spacer' }));
    hmh.appendChild(heatLegend());
    hmc.appendChild(hmh);
    var hmb = EL('div', { className: 'card-pad', style: 'overflow-x:auto' });
    var hmEl = EL('div', { className: 'heatmap', id: 'od-brain-heat1' });
    buildHeatmap(hmEl, dateMap, 26);
    hmb.appendChild(hmEl);
    hmc.appendChild(hmb);
    sec.appendChild(hmc);
    return sec;
  }

  // ======================================================================
  // Tab: REWARD SIGNAL
  // ======================================================================
  function buildReward(behavioral) {
    var sec = EL('section', { className: 'tabpane', 'data-p': 'reward' });
    var beh = behavioral || {};
    var reward = beh.reward_telemetry || {};
    var timeline = reward.timeline || [];
    var rewardDateMap = {};
    timeline.forEach(function (point) {
      if (point && point.date) rewardDateMap[String(point.date)] = Number(point.count || 0);
    });

    // Density heatmap
    var hmcr = EL('div', { className: 'card', style: 'margin-bottom:16px' });
    var hmhr = EL('div', { className: 'card-head' });
    hmhr.appendChild(EL('h3', { text: 'Reward signal density' }));
    hmhr.appendChild(EL('span', {
      className: 'sub',
      text: 'settled numeric labels per day · last ' + Number(reward.window_days || 182) + ' days',
    }));
    hmhr.appendChild(EL('div', { className: 'spacer' }));
    hmhr.appendChild(heatLegend());
    hmcr.appendChild(hmhr);
    var hmbr = EL('div', { className: 'card-pad', style: 'overflow-x:auto' });
    var hmElR = EL('div', { className: 'heatmap', id: 'od-brain-heat2' });
    buildHeatmap(hmElR, rewardDateMap, 26);
    hmbr.appendChild(hmElR);
    hmcr.appendChild(hmbr);
    sec.appendChild(hmcr);

    // 2-column: sparkline + outcome mix
    var grid = EL('div', { className: 'grid', style: 'grid-template-columns:1fr 1fr;align-items:start' });

    // Average settled reward and real daily series
    var fbCard = EL('div', { className: 'card' });
    var fbH = EL('div', { className: 'card-head' });
    fbH.appendChild(EL('h3', { text: 'Average settled reward' }));
    fbH.appendChild(EL('span', {
      className: 'sub',
      text: fmtNum(reward.count || 0) + ' finalized labels',
    }));
    fbCard.appendChild(fbH);
    var fbB = EL('div', { className: 'card-pad' });
    if (reward.average != null) {
      fbB.appendChild(EL('div', {
        className: 'value num',
        style: 'font-size:30px;margin-bottom:12px',
        text: Number(reward.average).toFixed(3),
      }));
    }
    var fbSp = EL('div', { id: 'od-brain-sp-fb' });
    var sparkVals = timeline.slice(-30).map(function (point) {
      return Number(point.average || 0);
    });
    var hasSparkData = sparkVals.length > 0;
    if (hasSparkData && typeof window.slmSpark === 'function') {
      fbSp.innerHTML = window.slmSpark(sparkVals, { w: 600, h: 150, color: 'var(--violet)' });
      var sv1 = fbSp.querySelector('svg'); if (sv1) sv1.style.height = '150px';
    } else {
      fbSp.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:32px;text-align:center;font-size:13px',
        text: 'No settled reward history is available yet.',
      }));
    }
    fbB.appendChild(fbSp);
    fbCard.appendChild(fbB);
    grid.appendChild(fbCard);

    // Settled reward distribution
    var outCard = EL('div', { className: 'card' });
    var outH = EL('div', { className: 'card-head' });
    outH.appendChild(EL('h3', { text: 'Reward distribution' }));
    outH.appendChild(EL('span', { className: 'sub', text: 'engagement-derived settled labels' }));
    outCard.appendChild(outH);
    var outB = EL('div', { className: 'card-pad', id: 'od-brain-outcomes' });
    var total = Number(reward.count || 0);
    var bd = reward.distribution || {};
    if (total === 0) {
      outB.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No settled reward labels yet. Recall engagement will populate this view.',
      }));
    } else {
      [
        ['Positive (> 0.6)', bd.positive || 0, 'var(--ok)'],
        ['Neutral (0.4–0.6)', bd.neutral || 0, 'var(--cyan)'],
        ['Negative (< 0.4)', bd.negative || 0, 'var(--danger)'],
      ].forEach(function (r) {
        var p2 = Math.round(r[1] / total * 100);
        var rw = EL('div', { style: 'margin-bottom:13px' });
        var rm = EL('div', {
          style: 'display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px',
        });
        rm.appendChild(EL('span', { text: r[0] }));
        rm.appendChild(EL('b', { className: 'num', text: p2 + '%' }));
        rw.appendChild(rm);
        rw.appendChild(meter(p2, r[2]));
        outB.appendChild(rw);
      });
    }
    outCard.appendChild(outB);
    grid.appendChild(outCard);
    sec.appendChild(grid);
    return sec;
  }

  // ======================================================================
  // Tab: BEHAVIOUR
  // ======================================================================
  function buildBehaviour(learning, behavioral) {
    var sec = EL('section', { className: 'tabpane', 'data-p': 'behaviour' });
    var l = learning || {};
    var beh = behavioral || {};
    var grid1 = EL('div', { className: 'grid', style: 'grid-template-columns:1fr 1fr;align-items:start' });

    // Tech preferences table
    var tc = EL('div', { className: 'card' });
    var tch = EL('div', { className: 'card-head' });
    tch.appendChild(EL('h3', { text: 'Tech preferences' }));
    tch.appendChild(EL('span', { className: 'sub', text: 'Layer 1 · confidence-weighted' }));
    tc.appendChild(tch);
    var tcb = EL('div', { className: 'card-pad' });
    var techItems = l.tech_preferences || [];
    if (techItems.length === 0) {
      tcb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No tech preferences learned yet. These emerge after consistent cross-session usage.',
      }));
    } else {
      var tbl = EL('table', { className: 'tbl' });
      var thead = EL('thead');
      var thr = EL('tr');
      ['Preference', 'Value', 'Confidence', 'Evidence'].forEach(function (h) {
        thr.appendChild(EL('th', { text: h }));
      });
      thead.appendChild(thr); tbl.appendChild(thead);
      var tbody = EL('tbody');
      techItems.slice(0, 10).forEach(function (t) {
        var tr = EL('tr');
        tr.appendChild(EL('td', { className: 'muted', text: String(t.name || t.preference || '—') }));
        var vd = EL('td');
        vd.appendChild(EL('b', { text: String(t.value || '—') }));
        tr.appendChild(vd);
        var cp = Math.round(Number(t.confidence || t.strength || 0) * 100);
        var cd = EL('td');
        cd.appendChild(inlineBar(cp));
        cd.appendChild(document.createTextNode(cp + '%'));
        tr.appendChild(cd);
        tr.appendChild(EL('td', { className: 'num', text: String(t.evidence_count || t.count || 0) }));
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody); tcb.appendChild(tbl);
    }
    tc.appendChild(tcb); grid1.appendChild(tc);

    // Workflow patterns
    var wc = EL('div', { className: 'card' });
    var wch = EL('div', { className: 'card-head' });
    wch.appendChild(EL('h3', { text: 'Workflow patterns' }));
    wch.appendChild(EL('span', { className: 'sub', text: 'Layer 3 · sequence & temporal' }));
    wc.appendChild(wch);
    var wcb = EL('div', { className: 'card-pad' });
    var wfPats = l.workflow_patterns || [];
    if (wfPats.length === 0) {
      wcb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No sequence or temporal patterns recorded yet. Interest signals are not shown as workflows.',
      }));
    } else {
      wfPats.slice(0, 3).forEach(function (wf) {
        var wd = EL('div', { style: 'margin-top:12px' });
        wd.appendChild(EL('span', { className: 'badge violet', text: String(wf.type || 'pattern') }));
        wd.appendChild(EL('div', {
          style: 'margin-top:6px;font-size:13px',
          text: String(wf.description || wf.pattern || '—'),
        }));
        wcb.appendChild(wd);
      });
    }
    wc.appendChild(wcb); grid1.appendChild(wc);
    sec.appendChild(grid1);

    var grid2 = EL('div', {
      className: 'grid',
      style: 'grid-template-columns:1fr 1fr;align-items:start;margin-top:16px',
    });

    // Cross-project transfers
    var xc = EL('div', { className: 'card' });
    var xch = EL('div', { className: 'card-head' });
    xch.appendChild(EL('h3', { text: 'Cross-project transfers' }));
    xch.appendChild(EL('span', { className: 'sub', text: 'patterns reused across projects' }));
    xc.appendChild(xch);
    var xcb = EL('div', { className: 'card-pad' });
    var xPats = beh.cross_project_patterns || [];
    if (xPats.length === 0) {
      xcb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No cross-project patterns yet. These appear when the same behaviour is detected in 2+ projects.',
      }));
    } else {
      xPats.slice(0, 5).forEach(function (x) {
        var item = EL('div', { className: 'feed-item' });
        if (typeof window.slmIcon === 'function') {
          var ic = EL('span', { className: 'ic', style: 'background:var(--violet-soft);color:var(--violet)' });
          ic.innerHTML = window.slmIcon('link');
          item.appendChild(ic);
        }
        var info = EL('div');
        info.appendChild(EL('b', { text: String(x.pattern_key || x.name || '—') }));
        info.appendChild(EL('div', {
          className: 'dim',
          style: 'font-size:12px;margin-top:2px',
          text: String(x.description || x.pattern_type || '—'),
        }));
        item.appendChild(info); xcb.appendChild(item);
      });
    }
    xc.appendChild(xcb); grid2.appendChild(xc);

    // Recent outcomes
    var rc = EL('div', { className: 'card' });
    var rch = EL('div', { className: 'card-head' });
    rch.appendChild(EL('h3', { text: 'Recent outcomes' }));
    rc.appendChild(rch);
    var rcb = EL('div', { className: 'card-pad' });
    var recOuts = beh.recent_outcomes || [];
    if (recOuts.length === 0) {
      rcb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No outcome reports yet. Use /report-outcome to teach the ranker.',
      }));
    } else {
      recOuts.slice(0, 8).forEach(function (r) {
        var item = EL('div', { className: 'feed-item' });
        var out = String(r.outcome || 'unknown');
        var bcls = out === 'success' ? 'ok' : out === 'failure' ? 'danger' : 'warn';
        var bdg = EL('span', { className: 'badge ' + bcls });
        bdg.appendChild(EL('span', { className: 'dot' }));
        bdg.appendChild(document.createTextNode(' ' + out));
        item.appendChild(bdg);
        item.appendChild(EL('span', {
          style: 'flex:1;font-size:13px',
          text: String(r.action_type || 'other'),
        }));
        item.appendChild(EL('time', { text: String(r.timestamp || '') }));
        rcb.appendChild(item);
      });
    }
    rc.appendChild(rcb); grid2.appendChild(rc);
    sec.appendChild(grid2);
    return sec;
  }

  // ======================================================================
  // Tab: CONNECTED CLIENTS
  // ======================================================================
  function buildClients(dateMap) {
    var sec = EL('section', { className: 'tabpane', 'data-p': 'clients' });

    // Sparkline from tool-event activity (last 22 data-points matching design)
    var evc = EL('div', { className: 'card', style: 'margin-bottom:16px' });
    var evh = EL('div', { className: 'card-head' });
    evh.appendChild(EL('h3', { text: 'Connected-client evolution' }));
    evh.appendChild(EL('span', { className: 'sub', text: 'tool-event activity over time (proxy metric)' }));
    evc.appendChild(evh);
    var evb = EL('div', { className: 'card-pad' });
    var today = new Date(); today.setHours(0, 0, 0, 0);
    var cVals = [];
    for (var ci = 21; ci >= 0; ci--) {
      var cd = new Date(today); cd.setDate(today.getDate() - ci);
      cVals.push((dateMap && dateMap[cd.toISOString().slice(0, 10)]) || 0);
    }
    var evSp = EL('div', { id: 'od-brain-sp-clients' });
    var hasData = cVals.some(function (v) { return v > 0; });
    if (hasData && typeof window.slmSpark === 'function') {
      evSp.innerHTML = window.slmSpark(cVals, { w: 600, h: 150, color: 'var(--cyan)' });
      var sv2 = evSp.querySelector('svg'); if (sv2) sv2.style.height = '150px';
    } else {
      evSp.appendChild(EL('p', { className: 'muted', style: 'padding:32px;text-align:center',
        text: 'No event history in this period.' }));
    }
    evb.appendChild(evSp);
    // TODO: GET /api/clients — no live connected-client session data available via these endpoints.
    // tool-events captures tool invocations but not distinct client identities or session counts.
    evb.appendChild(EL('p', {
      className: 'muted',
      style: 'margin-top:12px;font-size:13px',
      text: 'Bars above represent all tool invocations logged to this daemon. ' +
            'A dedicated client-session endpoint is not yet exposed via the public API.',
    }));
    evc.appendChild(evb);
    sec.appendChild(evc);

    // Empty-state clients table
    var tc = EL('div', { className: 'card' });
    var tch = EL('div', { className: 'card-head' });
    tch.appendChild(EL('h3', { text: 'Clients' }));
    tc.appendChild(tch);
    var tcb = EL('div', { className: 'card-pad' });
    tcb.appendChild(EL('p', {
      className: 'muted',
      style: 'padding:16px;text-align:center;font-size:13px',
      text: 'Client details are not yet available via API. Coming in a future daemon release.',
    }));
    tc.appendChild(tcb);
    sec.appendChild(tc);
    return sec;
  }

  // ======================================================================
  // Tab: SOURCE QUALITY
  // ======================================================================
  function buildSourceQuality(learning) {
    var sec = EL('section', { className: 'tabpane', 'data-p': 'sources' });
    var scores = (learning && learning.source_scores) || {};
    var entries = Object.keys(scores).sort(function (a, b) {
      return Number(scores[b]) - Number(scores[a]);
    });
    var card = EL('div', { className: 'card' });
    var ch = EL('div', { className: 'card-head' });
    ch.appendChild(EL('h3', { text: 'Source quality' }));
    ch.appendChild(EL('span', { className: 'sub', text: 'persisted source-outcome posterior · 0.0–1.0' }));
    card.appendChild(ch);
    var cb = EL('div', { className: 'card-pad' });
    if (entries.length === 0) {
      cb.appendChild(EL('p', {
        className: 'muted',
        style: 'padding:16px;text-align:center;font-size:13px',
        text: 'No source-quality observations yet. Recall hits alone do not establish source quality.',
      }));
    } else {
      entries.forEach(function (k) {
        var v = Number(scores[k]);
        var row = EL('div', { style: 'margin-bottom:14px' });
        var meta = EL('div', {
          style: 'display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px',
        });
        meta.appendChild(EL('span', { className: 'mono', text: k }));
        meta.appendChild(EL('b', { className: 'num', text: v.toFixed(2) }));
        row.appendChild(meta);
        row.appendChild(meter(v * 100));
        cb.appendChild(row);
      });
    }
    card.appendChild(cb); sec.appendChild(card);
    return sec;
  }

  // ======================================================================
  // Tab row
  // ======================================================================
  function buildTabRow(pCount) {
    var tabDiv = EL('div', { className: 'tabs', id: 'od-brain-tabs' });
    [
      ['overview',   'Overview',           null],
      ['reward',     'Reward signal',       null],
      ['behaviour',  'Behaviour',           pCount],
      ['clients',    'Connected clients',   null],
      ['sources',    'Source quality',      null],
    ].forEach(function (td, i) {
      var btn = EL('button', { className: 'tab' + (i === 0 ? ' active' : ''), 'data-t': td[0] });
      btn.appendChild(document.createTextNode(td[1]));
      if (td[2] !== null) {
        btn.appendChild(document.createTextNode(' '));
        btn.appendChild(EL('span', { className: 'cnt', text: String(td[2]) }));
      }
      tabDiv.appendChild(btn);
    });
    return tabDiv;
  }

  // ======================================================================
  // Wire tab switching — addEventListener only (CSP-safe, no onclick)
  // ======================================================================
  function wireTabs(container) {
    var tabRow = container.querySelector('#od-brain-tabs');
    if (!tabRow) return;
    var btns = tabRow.querySelectorAll('.tab');
    btns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        btns.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var t = btn.getAttribute('data-t');
        container.querySelectorAll('.tabpane').forEach(function (s) {
          s.classList.toggle('active', s.getAttribute('data-p') === t);
        });
      });
    });
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
    w.appendChild(EL('span', { className: 'text-muted ms-2', text: ' Loading Brain…' }));
    container.appendChild(w);
  }

  function showError(container, err) {
    container.replaceChildren();
    var w = EL('div', { style: 'padding:24px' });
    w.appendChild(EL('p', { className: 'text-danger fw-bold', text: 'Could not load Brain data.' }));
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
    if (err && window.console) window.console.debug('[od-brain]', err.message || String(err));
  }

  // ======================================================================
  // Main render
  // ======================================================================
  function render(container) {
    showLoading(container);
    injectPhaseStyle();
    fetchAll().then(function (results) {
      var learning = results[0] || {};
      var behavioral = results[1] || {};
      var events = ((results[3] && results[3].events) || []);
      var dateMap = buildDateMap(events);
      var pCount = ((behavioral.patterns) || []).length;

      var head = EL('div', { className: 'page-head' });
      head.appendChild(EL('h2', { text: 'The living brain' }));
      head.appendChild(EL('p', {
        text: 'How your memory is getting smarter — ranking phase, the reward signal it learns from, ' +
              'and the behavioural patterns it has extracted. Everything trained on-device from your own usage.',
      }));

      container.replaceChildren(
        head,
        buildTabRow(pCount),
        buildOverview(learning, behavioral, dateMap),
        buildReward(behavioral),
        buildBehaviour(learning, behavioral),
        buildClients(dateMap),
        buildSourceQuality(learning)
      );
      wireTabs(container);
    }).catch(function (err) { showError(container, err); });
  }

  // ======================================================================
  // Public API — od-shell.js calls window.odRenderBrain(pane)
  // ======================================================================
  window.odRenderBrain = function (container) { if (container) render(container); };

  // Legacy compatibility — ng-shell.js calls window.loadBrain()
  window.loadBrain = function () {
    var c = document.getElementById('brain-pane');
    if (c) render(c);
  };

  // Auto-boot when the pane is already visible (e.g., deep-link to #brain-pane)
  (function boot() {
    function tryBoot() {
      var c = document.getElementById('brain-pane');
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
