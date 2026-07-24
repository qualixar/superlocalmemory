// od-health.js — OD Health pane renderer v2 (design-faithful)
// Renders into window.odRenderHealth(container).
//
// Endpoints (confirmed with curl against daemon :8765):
//   GET /health               → {status, version, pid, state, readiness}
//   GET /api/stats            → {overview: {db_size_mb, total_memories, ...}}
//   GET /api/v3/math/health   → {health: {fisher, sheaf, langevin}, overall, note}
//   GET /api/agents           → {agents: [{agent_id, profile_id, registered_at}], count, stats}
//   GET /api/trust/stats      → {avg_trust_score, total_signals, enforcement, by_signal_type}
//   GET /api/events/stats     → {total_events, events_last_24h, "memory.stored", "memory.recalled", ...}
//
// JSON key mapping for events/stats: top-level dot-notation keys e.g. data["memory.stored"]
// JSON key mapping for agents: data.stats.total_agents, data.agents[].{agent_id, profile_id, registered_at}
//
// CRIT fixes applied:
//   (a) No hardcoded numbers — all values from real API or '—' with TODO comment.
//   (b) Agent table 6 columns: 2 real (Agent, Last seen), 4 with '—' (Protocol, Trust,
//       Writes, Recalls — no per-agent endpoint exists).
//   (c) KPI 4th card "Mesh broker" shows 'Unknown' — no /api/mesh/* endpoint confirmed.
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

/* global window, document, fetch, Promise */
(function () {
  'use strict';

  // ─── Helpers ──────────────────────────────────────────────────────────────

  /** HTML-escape all API strings before inserting via innerHTML. */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Format a Unix-seconds float or ISO string as "Xm ago".
   * Returns '—' for falsy input.
   */
  function timeAgo(ts) {
    if (!ts) return '—';
    var ms = (typeof ts === 'number') ? ts * 1000 : new Date(ts).getTime();
    var d = (Date.now() - ms) / 1000;
    if (d < 0) d = 0;
    if (d < 60)    return Math.floor(d) + 's ago';
    if (d < 3600)  return Math.floor(d / 60) + 'm ago';
    if (d < 86400) return Math.floor(d / 3600) + 'h ago';
    return Math.floor(d / 86400) + 'd ago';
  }

  /** Format a number with locale separators; safe for null/undefined. */
  function fmtNum(n) { return Number(n || 0).toLocaleString(); }

  /**
   * Map a status string to an OD badge/color class.
   * Returns: 'ok' | 'warn' | 'danger' | 'neutral'
   */
  function statusClass(st) {
    var s = String(st || '').toLowerCase();
    if (s === 'ok' || s === 'healthy' || s === 'running' || s === 'active' ||
        s === 'configured' || s === 'ready') return 'ok';
    if (s === 'degraded' || s === 'stale' || s === 'warning' || s === 'warn') return 'warn';
    if (s === 'error' || s === 'dead' || s === 'critical') return 'danger';
    return 'neutral';
  }

  /** Return the CSS var color for a status class. */
  function statusColor(cls) {
    if (cls === 'ok')     return 'var(--ok)';
    if (cls === 'warn')   return 'var(--warn)';
    if (cls === 'danger') return 'var(--danger)';
    return 'var(--fg-2)';
  }

  /** Get an inline SVG from the shell icon registry; safe if shell not loaded. */
  function safeIcon(name) {
    return (typeof window.slmIcon === 'function') ? window.slmIcon(name) : '';
  }

  // ─── Skeleton HTML ────────────────────────────────────────────────────────

  /**
   * Static structural HTML for the health pane — zero API data inside.
   * All dynamic content is written by the populate* functions.
   * IDs use od-h-* prefix to avoid colliding with legacy Bootstrap IDs.
   */
  function buildSkeleton() {
    return (
      // Page header
      '<div class="page-head">' +
        '<h2>System health</h2>' +
        '<p>Math layers, the connected-agent stream and daemon status — ' +
        'click any card to drill down. All green means your memory is learning correctly.</p>' +
      '</div>' +

      // KPI strip: 4 status cards (Daemon, Memory DB, Math layers, Mesh broker)
      '<section class="kpi-strip" style="margin-bottom:16px" id="od-h-status">' +
        '<div class="card kpi"><div class="dim" style="font-size:12px;padding:20px">Loading status…</div></div>' +
      '</section>' +

      // Two-column: Math health + Trust signals
      '<div class="grid" style="grid-template-columns:1fr 1fr;align-items:start">' +

        '<div class="card">' +
          '<div class="card-head">' +
            '<h3>Math-layer health</h3>' +
            '<span class="sub">Scoring layers · Consistency · Lifecycle</span>' +
          '</div>' +
          '<div class="card-pad" id="od-h-math">' +
            '<div class="dim" style="text-align:center;padding:20px">Loading…</div>' +
          '</div>' +
        '</div>' +

        '<div class="card">' +
          '<div class="card-head">' +
            '<h3>Trust signals</h3>' +
            '<span class="sub" id="od-h-trust-mode">loading…</span>' +
          '</div>' +
          '<div class="card-pad">' +
            '<div style="display:flex;gap:24px;margin-bottom:16px">' +
              '<div>' +
                '<div class="dim" style="font-size:11px">Avg trust score</div>' +
                '<div class="num" style="font-size:24px;font-weight:700" id="od-h-trust-avg">—</div>' +
              '</div>' +
              '<div>' +
                '<div class="dim" style="font-size:11px">Total signals</div>' +
                '<div class="num" style="font-size:24px;font-weight:700" id="od-h-trust-total">—</div>' +
              '</div>' +
            '</div>' +
            '<div id="od-h-signals"></div>' +
          '</div>' +
        '</div>' +

      '</div>' +

      // Connected agents card (full-width)
      '<div class="card" style="margin-top:16px">' +
        '<div class="card-head">' +
          '<h3>Connected agents</h3>' +
          '<span class="sub">live write / recall stream</span>' +
          '<div class="spacer"></div>' +
          // pulse class + explicit dot color as in design
          '<span class="badge neutral pulse" style="border-radius:99px">' +
            '<span class="dot" style="background:var(--ok)"></span> live' +
          '</span>' +
        '</div>' +
        '<div class="card-pad">' +
          // Summary stats row: 4 metrics
          '<div style="display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap" id="od-h-agent-stats">' +
            '<div>' +
              '<div class="dim" style="font-size:11px">Total agents</div>' +
              '<div class="num" style="font-size:22px;font-weight:700" id="od-h-agent-total">—</div>' +
            '</div>' +
            '<div>' +
              '<div class="dim" style="font-size:11px">Active 24h</div>' +
              '<div class="num" style="font-size:22px;font-weight:700" id="od-h-agent-active">—</div>' +
            '</div>' +
            '<div>' +
              '<div class="dim" style="font-size:11px">Total writes</div>' +
              '<div class="num" style="font-size:22px;font-weight:700" id="od-h-agent-writes">—</div>' +
            '</div>' +
            '<div>' +
              '<div class="dim" style="font-size:11px">Total recalls</div>' +
              '<div class="num" style="font-size:22px;font-weight:700" id="od-h-agent-recalls">—</div>' +
            '</div>' +
          '</div>' +
          // Agents table: 6 columns matching design
          // Protocol, Trust, Writes, Recalls — no per-agent endpoint; display '—'
          '<table class="tbl" id="od-h-agents-tbl">' +
            '<thead><tr>' +
              '<th>Agent</th>' +
              '<th>Protocol</th>' +
              '<th>Trust</th>' +
              '<th>Writes</th>' +
              '<th>Recalls</th>' +
              '<th>Last seen</th>' +
            '</tr></thead>' +
            '<tbody id="od-h-agents-body">' +
              '<tr><td colspan="6" class="dim" style="text-align:center;padding:24px">Loading…</td></tr>' +
            '</tbody>' +
          '</table>' +
        '</div>' +
      '</div>' +

      // System Health — Components (v3.8.2 UX-6): what's installed vs missing,
      // with copy-paste fix commands + a Retry-now button. The daemon
      // auto-heals fixable items on start; this is transparency + fallback.
      // Rendered by od-components.js → window.odRenderComponents.
      '<div class="card" style="margin-top:16px">' +
        '<div class="card-head">' +
          '<h3>System Health — Components</h3>' +
          '<span class="sub">models &amp; dependencies · self-healing</span>' +
        '</div>' +
        '<div class="card-pad" id="od-h-components">' +
          '<div class="dim" style="text-align:center;padding:20px">Loading…</div>' +
        '</div>' +
      '</div>'
    );
  }

  // ─── KPI strip ────────────────────────────────────────────────────────────

  /**
   * Populate the 4 KPI status cards from real API data.
   * CRIT(a): no hardcoded values — all from fetch results or '—' with explicit note.
   * CRIT(c): 4th card (Mesh broker) has no API endpoint → shows 'Unknown'.
   *
   * @param {Object|null} healthData  /health response
   * @param {Object|null} statsData   /api/stats response
   * @param {Object|null} mathData    /api/v3/math/health response
   */
  function populateKpiStrip(healthData, statsData, mathData) {
    var strip = document.getElementById('od-h-status');
    if (!strip) return;

    // Card 1: Daemon — from /health
    var daemonCls = (healthData && healthData.status === 'ok') ? 'ok' : 'warn';
    var daemonVal = daemonCls === 'ok' ? 'Healthy' : 'Degraded';
    var daemonDetail = healthData
      ? 'port 8765 · v' + esc(String(healthData.version || '?'))
      : 'daemon unreachable';

    // Card 2: Memory DB — from /api/stats
    var dbMb = (statsData && statsData.overview)
      ? Number(statsData.overview.db_size_mb || 0).toFixed(1)
      : null;
    var dbCls = dbMb !== null ? 'ok' : 'neutral';
    var dbVal = dbMb !== null ? 'Healthy' : 'Unknown';
    var dbDetail = dbMb !== null ? 'memory.db · ' + esc(dbMb) + ' MB' : 'stats unavailable';

    // Card 3: Math layers — from /api/v3/math/health
    var mathLayerCount = (mathData && mathData.health)
      ? Object.keys(mathData.health).length : 0;
    var mathOverall = mathData ? String(mathData.overall || '') : '';
    var mathCls = statusClass(mathOverall);
    var mathVal = mathCls === 'ok' ? 'Active' : (mathOverall ? esc(mathOverall) : 'Unknown');
    var mathDetail = esc(String(mathLayerCount)) + ' / 3 layers online';

    // Card 4: Mesh broker — TODO: no /api/mesh/* endpoint confirmed
    var meshCls = 'neutral';
    var meshVal = 'Unknown';
    var meshDetail = 'no endpoint — configure mesh to enable'; // TODO: wire to /api/mesh/status

    var CARDS = [
      { label: 'Daemon',       cls: daemonCls, value: daemonVal, detail: daemonDetail },
      { label: 'Memory DB',    cls: dbCls,     value: dbVal,     detail: dbDetail },
      { label: 'Math layers',  cls: mathCls,   value: mathVal,   detail: mathDetail },
      { label: 'Mesh broker',  cls: meshCls,   value: meshVal,   detail: meshDetail },
    ];

    strip.innerHTML = CARDS.map(function (c) {
      return (
        '<div class="card kpi" style="cursor:pointer">' +
          '<div class="label">' + safeIcon('health') + ' ' + esc(c.label) + '</div>' +
          '<div class="value" style="font-size:20px;color:' + statusColor(c.cls) + '">' + c.value + '</div>' +
          '<div class="dim" style="font-size:12px;margin-top:4px">' + c.detail + '</div>' +
        '</div>'
      );
    }).join('');
  }

  // ─── Math health ──────────────────────────────────────────────────────────

  /**
   * Render the 3 math-layer cards from /api/v3/math/health.
   * Display names for known keys; falls through to raw key for unknown ones.
   * CRIT(a): all values from API — description, mode, threshold, temperature.
   */
  function populateMathHealth(mathData) {
    var el = document.getElementById('od-h-math');
    if (!el) return;

    if (!mathData || !mathData.health || Object.keys(mathData.health).length === 0) {
      el.innerHTML = '<p class="muted" style="padding:8px 0">Math health data unavailable.</p>';
      return;
    }

    var NAMES = {
      fisher:   'Scoring layer',
      sheaf:    'Consistency layer',
      langevin: 'Lifecycle layer',
    };

    var html = '';
    var layers = mathData.health;

    // Map API status values to design-canonical display labels.
    // 'configured' from the live API maps to 'active' (same semantic class).
    var STATUS_LABEL = {
      ok:         'active',
      active:     'active',
      configured: 'active',
      running:    'active',
      healthy:    'active',
      warn:       'degraded',
      degraded:   'degraded',
      error:      'error',
      critical:   'error',
    };

    Object.keys(layers).forEach(function (key) {
      var layer = layers[key];
      var bc = statusClass(layer.status);
      var rawStatus = String(layer.status || '').toLowerCase();
      var displayStatus = STATUS_LABEL[rawStatus] || STATUS_LABEL[bc] || esc(layer.status || 'unknown');
      var displayName = NAMES[key] || esc(key);

      // Key-value metadata pairs — only from real API fields
      var kvs = [];
      if (layer.mode      != null) kvs.push(['mode',        String(layer.mode)]);
      if (layer.threshold != null) kvs.push(['threshold',   String(layer.threshold)]);
      if (layer.temperature != null) kvs.push(['temperature', String(layer.temperature)]);

      var kvsHtml = kvs.map(function (kv) {
        return '<span class="mono" style="font-size:12px">' +
          '<span class="dim">' + esc(kv[0]) + '</span> ' + esc(kv[1]) +
          '</span>';
      }).join('');

      html += (
        '<div style="padding:14px 0;border-bottom:1px solid var(--border)">' +
          '<div style="display:flex;align-items:center;gap:10px">' +
            '<b>' + esc(displayName) + '</b>' +
            '<span class="badge ' + bc + '">' +
              '<span class="dot"></span> ' + displayStatus +
            '</span>' +
          '</div>' +
          '<p class="muted" style="font-size:12.5px;margin-top:5px">' +
            esc(layer.description || '') +
          '</p>' +
          (kvs.length
            ? '<div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap">' + kvsHtml + '</div>'
            : '') +
        '</div>'
      );
    });

    // API's own note (clarifies config-derived vs live probe)
    if (mathData.note) {
      html += '<p class="dim" style="font-size:11px;margin-top:10px;font-style:italic">' +
        esc(mathData.note) + '</p>';
    }

    el.innerHTML = html;
  }

  // ─── Trust signals ────────────────────────────────────────────────────────

  /** Color map for known signal types — OD palette tokens. */
  var SIGNAL_COLORS = {
    recalled:          'var(--ok)',
    high_importance:   'var(--cyan)',
    high_volume:       'var(--warn)',
    quick_delete:      'var(--danger)',
  };

  /**
   * Render trust signal bars from /api/trust/stats.
   * Shows percentages relative to total signals (design-faithful).
   * Honest empty state when by_signal_type is empty (current live API).
   */
  function populateTrustSignals(trustStats) {
    var modeEl  = document.getElementById('od-h-trust-mode');
    var avgEl   = document.getElementById('od-h-trust-avg');
    var totalEl = document.getElementById('od-h-trust-total');
    var barsEl  = document.getElementById('od-h-signals');

    if (!trustStats) {
      if (modeEl)  modeEl.textContent  = 'trust data unavailable';
      if (barsEl)  barsEl.innerHTML    = '<p class="muted">Could not load trust data.</p>';
      return;
    }

    if (modeEl) modeEl.textContent = 'enforcement: ' + (trustStats.enforcement || 'unknown');

    if (avgEl) {
      avgEl.textContent = trustStats.avg_trust_score != null
        ? Number(trustStats.avg_trust_score).toFixed(2)
        : '—';
    }

    var totalSig = Number(trustStats.total_signals || 0);
    if (totalEl) totalEl.textContent = fmtNum(totalSig);

    if (!barsEl) return;

    var sigMap = trustStats.by_signal_type || {};
    var keys = Object.keys(sigMap);

    if (keys.length === 0) {
      barsEl.innerHTML =
        '<p class="muted" style="font-size:13px">No trust signals recorded yet. ' +
        'Signals accumulate as agents write and recall memories.</p>';
      return;
    }

    var total = keys.reduce(function (acc, k) { return acc + Number(sigMap[k] || 0); }, 0) || 1;

    barsEl.innerHTML = keys.map(function (k) {
      var val = Number(sigMap[k] || 0);
      var pct = Math.round((val / total) * 100);
      var barColor = SIGNAL_COLORS[k] || 'var(--violet)';
      return (
        '<div style="margin-bottom:12px">' +
          '<div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px">' +
            '<span class="mono">' + esc(k) + '</span>' +
            '<b class="num">' + pct + '%</b>' +
          '</div>' +
          '<div class="meter"><i style="width:' + pct + '%;background:' + barColor + '"></i></div>' +
        '</div>'
      );
    }).join('');
  }

  // ─── Agent summary stats ──────────────────────────────────────────────────

  /**
   * Populate the 4-metric summary row above the agent table.
   * Total agents  → /api/agents stats.total_agents (real)
   * Active 24h    → TODO: no per-agent last_seen endpoint; displays '—'
   * Total writes  → /api/events/stats["memory.stored"] (real)
   * Total recalls → /api/events/stats["memory.recalled"] (real)
   */
  function populateAgentSummary(agentsData, eventsStats) {
    var totalEl   = document.getElementById('od-h-agent-total');
    var activeEl  = document.getElementById('od-h-agent-active');
    var writesEl  = document.getElementById('od-h-agent-writes');
    var recallsEl = document.getElementById('od-h-agent-recalls');

    var stats = (agentsData && agentsData.stats) || {};
    var agents = (agentsData && agentsData.agents) || [];
    var total  = stats.total_agents != null ? stats.total_agents : agents.length;

    if (totalEl)   totalEl.textContent   = fmtNum(total);
    // TODO: no per-agent last_seen endpoint in current API; active_24h cannot be determined
    if (activeEl)  activeEl.textContent  = '—';

    var writes  = eventsStats ? (eventsStats['memory.stored']   || 0) : 0;
    var recalls = eventsStats ? (eventsStats['memory.recalled'] || 0) : 0;

    if (writesEl)  writesEl.textContent  = fmtNum(writes);
    if (recallsEl) recallsEl.textContent = fmtNum(recalls);
  }

  // ─── Agent table ──────────────────────────────────────────────────────────

  /**
   * Populate the connected-agents table from /api/agents.
   * CRIT(b): 6-column design layout preserved; columns with no API data show '—'.
   *   Real data: Agent (agent_id), Last seen (derived from registered_at)
   *   TODO data: Protocol, Trust, Writes, Recalls — no per-agent endpoint
   * CRIT(c): agent_id goes through esc() — XSS-safe.
   */
  function populateAgentTable(agentsData) {
    var tbody = document.getElementById('od-h-agents-body');
    if (!tbody) return;

    var agents = (agentsData && agentsData.agents) || [];

    if (agents.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="6" class="dim" style="text-align:center;padding:24px">' +
        'No agents registered. Agents appear automatically when they connect via MCP, CLI, or REST.' +
        '</td></tr>';
      return;
    }

    var rows = '';
    for (var i = 0; i < agents.length; i++) {
      var a = agents[i];
      // CRIT(c): agent_id through esc()
      rows += (
        '<tr>' +
          '<td><b>' + esc(a.agent_id || '') + '</b></td>' +
          // TODO: Protocol — no protocol field in /api/agents response
          '<td><span class="badge neutral">—</span></td>' +
          // TODO: Trust score — only aggregate available via /api/trust/stats, not per-agent
          '<td><span class="badge neutral">—</span></td>' +
          // TODO: Writes per-agent — no per-agent write count in current API
          '<td class="num dim">—</td>' +
          // TODO: Recalls per-agent — no per-agent recall count in current API
          '<td class="num dim">—</td>' +
          '<td class="dim">' + esc(timeAgo(a.registered_at)) + '</td>' +
        '</tr>'
      );
    }
    tbody.innerHTML = rows;
  }

  // ─── Main entry point ─────────────────────────────────────────────────────

  /**
   * Fetch all health endpoints, inject OD markup, and populate with real data.
   * Idempotent: each call clears and re-renders the container.
   *
   * @param {HTMLElement} container  Usually document.getElementById("health-pane").
   */
  function odRenderHealth(container) {
    if (!container) return;

    container.innerHTML =
      '<div style="color:var(--fg-2);text-align:center;padding:48px 0">' +
      'Loading health data…</div>';

    Promise.all([
      fetch('/health').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/stats').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/v3/math/health').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/agents').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/trust/stats').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/events/stats').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
    ]).then(function (results) {
      var healthData  = results[0];
      var statsData   = results[1];
      var mathData    = results[2];
      var agentsData  = results[3];
      var trustStats  = results[4];
      var eventsStats = results[5];

      // Inject static skeleton — no API data inside
      container.innerHTML = buildSkeleton();

      // Populate each section with real API data
      populateKpiStrip(healthData, statsData, mathData);
      populateMathHealth(mathData);
      populateTrustSignals(trustStats);
      populateAgentSummary(agentsData, eventsStats);
      populateAgentTable(agentsData);
      // v3.8.2 UX-6: component "what's missing" report (od-components.js).
      if (typeof window.odRenderComponents === 'function') {
        window.odRenderComponents(document.getElementById('od-h-components'));
      }

    }).catch(function (err) {
      container.innerHTML =
        '<div style="color:var(--danger);text-align:center;padding:48px 0">' +
        'Failed to load health data. Check the daemon is running on port 8765.</div>';
      if (typeof console !== 'undefined') {
        console.error('[od-health] render error:', err);
      }
    });
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  window.odRenderHealth = odRenderHealth;

  // Auto-run: if the health pane is already in the DOM at script load, render it.
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('health-pane');
    if (pane) odRenderHealth(pane);
  });

}());
