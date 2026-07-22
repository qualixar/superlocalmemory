// SuperLocalMemory — od-agents.js  v1.0
// OD-styled Multi-Agent Memory pane. Self-contained, CSP-safe.
// Renders into the "agents-pane" tab container via the shell's triggerTabLoad.
// Exposes window.odRenderAgents(container).
//
// Endpoint:
//   GET /api/agents/memory-activity?limit=20 →
//   { ok, profile_id, total_memories, agent_count,
//     agents: [{agent_id, count, last_active, source_types:[..]}],  // desc by count
//     recent: [{agent_id, content, created_at, source_type, session_id}] }
//
// CRIT fixes baked in:
//   C1 — all agent_id / content / source_type values through escapeHtml() before innerHTML
//   C2 — empty state for zero agents / zero memories (adapter guide shown)
//   C3 — fetch failure shows error card with Retry; never throws, never silent
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

(function () {
  'use strict';

  var API_URL    = '/api/agents/memory-activity?limit=20';
  var REFRESH_MS = 15000;

  // ── Module state ──────────────────────────────────────────────────────────
  var _container = null;
  var _timer     = null;
  var _observer  = null;
  var _allRecent = [];   // full recent list — client-side filter only

  // ── Helpers ───────────────────────────────────────────────────────────────
  // CRIT C1: every API-derived string must pass through this before innerHTML.
  // Same implementation as od-optimize.js and od-mesh.js.
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function timeAgo(iso) {
    if (!iso) return 'never';
    var d = (Date.now() - new Date(iso).getTime()) / 1000;
    if (d < 5)     return 'just now';
    if (d < 60)    return Math.floor(d) + 's ago';
    if (d < 3600)  return Math.floor(d / 60) + 'm ago';
    if (d < 86400) return Math.floor(d / 3600) + 'h ago';
    return Math.floor(d / 86400) + 'd ago';
  }

  function fmtDate(iso) {
    if (!iso) return '—';
    try {
      var d = new Date(iso);
      return d.toLocaleDateString() + ' ' + d.toTimeString().slice(0, 5);
    } catch (e) { return String(iso); }
  }

  // Render source_type chips (array of strings)
  function fmtChips(types) {
    if (!Array.isArray(types) || types.length === 0) {
      return '<span style="color:var(--fg-3)">—</span>';
    }
    return types.map(function (t) {
      return (
        '<span style="display:inline-block;padding:1px 7px;border-radius:99px;' +
        'font-size:11px;background:var(--card-2);color:var(--fg-2);margin-right:4px">' +
        escapeHtml(t) + '</span>'
      );
    }).join('');
  }

  // Deterministic avatar color from agent_id string
  function avatarColor(id) {
    var palette = ['var(--violet)', 'var(--cyan)', 'var(--ok)', 'var(--warn)'];
    var s = String(id || '?');
    var h = 0;
    for (var i = 0; i < s.length; i++) {
      h = (h * 31 + s.charCodeAt(i)) & 0x7fffffff;
    }
    return palette[h % palette.length];
  }

  // ── Build layout HTML ─────────────────────────────────────────────────────
  function buildHTML() {
    return (
      '<div id="od-agents-root">' +

        // Page head
        '<div class="page-head">' +
          '<h2>Multi-Agent Memory</h2>' +
          '<p>Memory written by agents across CrewAI, LangChain, LangGraph, ' +
            'Semantic Kernel, and other agentic frameworks. Each agent stamps ' +
            '<code>SLM_AGENT_ID</code> on every write &mdash; this page visualises ' +
            'that activity across all agents sharing this SLM instance.</p>' +
        '</div>' +

        // Error band — CRIT C3
        '<div id="od-agents-err"' +
          ' style="display:none;background:var(--warn-soft);border:1px solid var(--warn);' +
                  'border-radius:var(--r-md);padding:12px 16px;margin-bottom:16px;' +
                  'align-items:center;justify-content:space-between;gap:12px">' +
          '<span style="color:var(--warn);font-size:13.5px" id="od-agents-err-msg">' +
            'Could not load agent memory data' +
          '</span>' +
          '<button id="od-agents-retry"' +
            ' style="padding:5px 12px;font-size:12.5px;border-radius:var(--r-md);' +
                    'border:1px solid var(--warn);background:transparent;' +
                    'color:var(--warn);cursor:pointer;font-weight:600">' +
            'Retry' +
          '</button>' +
        '</div>' +

        // KPI strip
        '<section class="kpi-strip" style="margin-bottom:16px">' +
          '<div class="card kpi">' +
            '<div class="label"><span data-ic="memories"></span> Total memories</div>' +
            '<div class="value num" id="od-agents-kpi-total">—</div>' +
            '<div class="delta" id="od-agents-kpi-profile">&nbsp;</div>' +
          '</div>' +
          '<div class="card kpi">' +
            '<div class="label"><span data-ic="mesh"></span> Active agents</div>' +
            '<div class="value num" id="od-agents-kpi-count">—</div>' +
            '<div class="delta">stamped SLM_AGENT_ID</div>' +
          '</div>' +
        '</section>' +

        // Per-agent cards
        '<div class="card" style="margin-bottom:16px">' +
          '<div class="card-head">' +
            '<h3>Agent activity</h3>' +
            '<span class="sub">memories written per agent, ranked by volume</span>' +
          '</div>' +
          '<div class="launch-grid" id="od-agents-grid"' +
            ' style="grid-template-columns:repeat(auto-fill,minmax(220px,1fr));' +
                    'padding:16px 20px;gap:14px">' +
            '<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--fg-2);' +
              'font-size:14px">Loading agent data…</div>' +
          '</div>' +
        '</div>' +

        // Recent memories table with client-side agent filter
        '<div class="card">' +
          '<div class="card-head">' +
            '<h3>Recent memories</h3>' +
            '<span class="sub">newest first</span>' +
            '<div class="spacer"></div>' +
            '<select id="od-agents-filter"' +
              ' style="font-size:12.5px;padding:3px 8px;border-radius:var(--r-sm);' +
                      'background:var(--card-2);border:1px solid var(--border);' +
                      'color:var(--fg-1)">' +
              '<option value="">All agents</option>' +
            '</select>' +
          '</div>' +
          '<div class="card-pad" style="overflow-x:auto">' +
            '<div id="od-agents-table">' +
              '<div style="text-align:center;padding:32px;color:var(--fg-3);font-size:13px">' +
                'Loading recent memories…' +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div>' +

      '</div>' // od-agents-root
    );
  }

  // ── Render: KPI strip ─────────────────────────────────────────────────────
  function renderKpi(data) {
    var totEl  = document.getElementById('od-agents-kpi-total');
    var cntEl  = document.getElementById('od-agents-kpi-count');
    var profEl = document.getElementById('od-agents-kpi-profile');
    if (totEl)  totEl.textContent  = Number(data.total_memories || 0).toLocaleString();
    if (cntEl)  cntEl.textContent  = Number(data.agent_count   || 0).toLocaleString();
    if (profEl) profEl.textContent = 'profile: ' + escapeHtml(data.profile_id || 'default');
  }

  // ── Render: per-agent cards ───────────────────────────────────────────────
  function renderAgentCards(agents) {
    var grid = document.getElementById('od-agents-grid');
    if (!grid) return;

    // CRIT C2: empty state when no agent has written memory yet
    if (!agents || agents.length === 0) {
      grid.innerHTML =
        '<div class="card launch-card"' +
          ' style="grid-column:1/-1;display:flex;flex-direction:column;' +
                  'align-items:center;justify-content:center;' +
                  'padding:48px 24px;text-align:center;cursor:default">' +
          '<div style="width:44px;height:44px;border-radius:12px;' +
            'background:var(--card-2);display:flex;align-items:center;' +
            'justify-content:center;margin-bottom:16px;color:var(--fg-3)">' +
            (typeof window.slmIcon === 'function' ? window.slmIcon('mesh') : '') +
          '</div>' +
          '<h3 style="font-size:15px;margin-bottom:6px">No multi-agent memory yet</h3>' +
          '<p style="font-size:12.5px;color:var(--fg-2);max-width:44ch;line-height:1.55">' +
            'Connect agents via the CrewAI, LangChain, LangGraph, Semantic Kernel, ' +
            'or Agent Framework adapters. Each stamps <code>SLM_AGENT_ID</code> on ' +
            'every write so their activity appears here.' +
          '</p>' +
        '</div>';
      return;
    }

    var html = '';
    agents.forEach(function (a) {
      var rawId  = String(a.agent_id || 'unknown');
      var id     = escapeHtml(rawId);
      var count  = Number(a.count || 0);
      var letter = escapeHtml(rawId.charAt(0).toUpperCase());  // SEC: escape before innerHTML
      var bg     = avatarColor(rawId);
      var ago    = escapeHtml(timeAgo(a.last_active));

      html +=
        '<div class="card launch-card" style="cursor:default">' +
          '<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px">' +
            '<span class="avatar" style="background:' + bg + '">' + letter + '</span>' +
            '<div style="flex:1;min-width:0">' +
              '<h3 style="font-size:15px;white-space:nowrap;overflow:hidden;' +
                'text-overflow:ellipsis" title="' + id + '">' + id + '</h3>' +
              '<span class="mono dim" style="font-size:11px">active ' + ago + '</span>' +
            '</div>' +
            '<span class="badge ok">' + count.toLocaleString() + '</span>' +
          '</div>' +
          '<div style="font-size:12px;color:var(--fg-2);margin-bottom:6px">' +
            '<span class="dim">source types</span>' +
          '</div>' +
          '<div>' + fmtChips(a.source_types) + '</div>' +
        '</div>';
    });
    grid.innerHTML = html;
  }

  // ── Populate agent filter dropdown ────────────────────────────────────────
  function populateFilter(agents) {
    var sel = document.getElementById('od-agents-filter');
    if (!sel) return;
    // Drop all options except the first "All agents" placeholder
    while (sel.options.length > 1) sel.remove(1);
    if (!agents) return;
    agents.forEach(function (a) {
      var opt = document.createElement('option');
      opt.value       = String(a.agent_id || '');
      opt.textContent = String(a.agent_id || 'unknown');
      sel.appendChild(opt);
    });
  }

  // ── Render: recent memories table ────────────────────────────────────────
  function renderRecentTable(records) {
    var wrap = document.getElementById('od-agents-table');
    if (!wrap) return;

    if (!records || records.length === 0) {
      wrap.innerHTML =
        '<div style="text-align:center;padding:32px;color:var(--fg-3);font-size:13px">' +
          'No memories match the current filter.' +
        '</div>';
      return;
    }

    var rows = records.map(function (r) {
      // CRIT C1: all three user-derived fields escaped
      var rawId      = String(r.agent_id    || '—');
      var rawContent = String(r.content     || '—');
      var rawSrc     = String(r.source_type || '—');
      var rawSess    = String(r.session_id  || '—');

      var agId    = escapeHtml(rawId);
      var srcType = escapeHtml(rawSrc);
      var dt      = escapeHtml(fmtDate(r.created_at));

      // Truncate content for display; keep full version in title attribute
      var dispContent = rawContent.length > 120 ? rawContent.slice(0, 120) + '…' : rawContent;
      var content     = escapeHtml(dispContent);
      var contentFull = escapeHtml(rawContent);

      var dispSess    = rawSess.length > 12 ? rawSess.slice(0, 12) + '…' : rawSess;
      var sess        = escapeHtml(dispSess);
      var sessFull    = escapeHtml(rawSess);

      var avatarBg = avatarColor(rawId);
      var letter   = escapeHtml(rawId.charAt(0).toUpperCase());  // SEC: escape before innerHTML

      return (
        '<tr>' +
          '<td style="white-space:nowrap;padding:7px 12px 7px 0">' +
            '<span style="display:inline-flex;align-items:center;gap:6px">' +
              '<span class="avatar"' +
                ' style="width:22px;height:22px;font-size:10px;background:' + avatarBg + '">' +
                letter +
              '</span>' +
              '<span class="mono" style="font-size:12px">' + agId + '</span>' +
            '</span>' +
          '</td>' +
          '<td style="max-width:340px;font-size:13px;padding:7px 12px 7px 0"' +
            ' title="' + contentFull + '">' + content + '</td>' +
          '<td style="white-space:nowrap;padding:7px 12px 7px 0">' +
            '<span style="padding:1px 7px;border-radius:99px;font-size:11px;' +
              'background:var(--card-2);color:var(--fg-2)">' + srcType + '</span>' +
          '</td>' +
          '<td class="mono dim" style="font-size:11px;white-space:nowrap;padding:7px 12px 7px 0">' +
            dt +
          '</td>' +
          '<td class="mono dim" style="font-size:11px;white-space:nowrap;padding:7px 0"' +
            ' title="' + sessFull + '">' + sess + '</td>' +
        '</tr>'
      );
    }).join('');

    wrap.innerHTML =
      '<table style="width:100%;border-collapse:collapse;font-size:13px">' +
        '<thead>' +
          '<tr style="border-bottom:1px solid var(--border)">' +
            '<th style="text-align:left;padding:4px 12px 8px 0;font-weight:600;' +
              'font-size:12px;color:var(--fg-2)">Agent</th>' +
            '<th style="text-align:left;padding:4px 12px 8px 0;font-weight:600;' +
              'font-size:12px;color:var(--fg-2)">Memory</th>' +
            '<th style="text-align:left;padding:4px 12px 8px 0;font-weight:600;' +
              'font-size:12px;color:var(--fg-2)">Source</th>' +
            '<th style="text-align:left;padding:4px 12px 8px 0;font-weight:600;' +
              'font-size:12px;color:var(--fg-2)">Written</th>' +
            '<th style="text-align:left;padding:4px 0 8px 0;font-weight:600;' +
              'font-size:12px;color:var(--fg-2)">Session</th>' +
          '</tr>' +
        '</thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>';
  }

  // ── Error/retry UI ─────────────────────────────────────────────────────────
  function showError(msg) {
    var band  = document.getElementById('od-agents-err');
    var msgEl = document.getElementById('od-agents-err-msg');
    if (band)  band.style.display  = 'flex';
    if (msgEl) msgEl.textContent   = msg || 'Could not load agent memory data';

    // Degrade gracefully — show errors in both sub-panels
    var grid = document.getElementById('od-agents-grid');
    if (grid) {
      grid.innerHTML =
        '<div style="grid-column:1/-1;text-align:center;padding:32px;' +
          'color:var(--fg-3);font-size:13px">Could not load agent data.</div>';
    }
    var table = document.getElementById('od-agents-table');
    if (table) {
      table.innerHTML =
        '<div style="text-align:center;padding:32px;color:var(--fg-3);font-size:13px">' +
          'Could not load recent memories.' +
        '</div>';
    }
  }

  function hideError() {
    var band = document.getElementById('od-agents-err');
    if (band) band.style.display = 'none';
  }

  // ── Data loader ───────────────────────────────────────────────────────────
  function loadAll() {
    fetch(API_URL, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        if (!data || !data.ok) throw new Error('API returned error');
        hideError();
        _allRecent = Array.isArray(data.recent) ? data.recent : [];

        renderKpi(data);
        renderAgentCards(Array.isArray(data.agents) ? data.agents : []);
        populateFilter(Array.isArray(data.agents) ? data.agents : []);
        renderRecentTable(_allRecent);
      })
      .catch(function (err) {
        // CRIT C3: never throw, always show a visible error card
        var msg = 'Could not load agent memory data' +
          (err && err.message ? ' (' + err.message + ')' : '');
        showError(msg);
      });
  }

  // ── Client-side filter: re-render recent table by selected agent ──────────
  function applyFilter() {
    var sel  = document.getElementById('od-agents-filter');
    var filt = sel ? sel.value : '';
    var subset = filt
      ? _allRecent.filter(function (r) { return r.agent_id === filt; })
      : _allRecent;
    renderRecentTable(subset);
  }

  // ── Event wiring (CSP-safe, no onXxx= attributes) ─────────────────────────
  function wireEvents(container) {
    var sel = container.querySelector('#od-agents-filter');
    if (sel) sel.addEventListener('change', applyFilter);

    var retryBtn = container.querySelector('#od-agents-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', function () {
        hideError();
        loadAll();
      });
    }
  }

  // ── Refresh management: MutationObserver re-activates on tab show ─────────
  function startRefreshObserver() {
    var pane = document.getElementById('agents-pane');
    if (!pane || _observer) return;
    _observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        if (m.type !== 'attributes' || m.attributeName !== 'class') return;
        var t = m.target;
        if (t.classList.contains('active') && t.classList.contains('show')) {
          loadAll();
          if (_timer) clearInterval(_timer);
          _timer = setInterval(function () {
            if (t.classList.contains('active')) {
              loadAll();
            } else {
              clearInterval(_timer);
              _timer = null;
            }
          }, REFRESH_MS);
        }
      });
    });
    _observer.observe(pane, { attributes: true });
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.odRenderAgents = function (container) {
    if (!container) return;
    _container = container;

    // Idempotent: reload data if already rendered
    if (container.querySelector('#od-agents-root')) {
      loadAll();
      return;
    }

    container.innerHTML = buildHTML();

    // Fill data-ic SVG placeholders (icons in KPI strip)
    if (typeof window.slmIcon === 'function') {
      container.querySelectorAll('[data-ic]').forEach(function (el) {
        el.innerHTML = window.slmIcon(el.getAttribute('data-ic'));
      });
    }

    wireEvents(container);
    loadAll();
    startRefreshObserver();

    // Polling timer — self-stops when pane becomes inactive
    if (_timer) clearInterval(_timer);
    _timer = setInterval(function () {
      var pane = document.getElementById('agents-pane');
      if (pane && pane.classList.contains('active')) {
        loadAll();
      } else {
        clearInterval(_timer);
        _timer = null;
      }
    }, REFRESH_MS);
  };

  // Auto-run at DOMContentLoaded if the pane is already in the DOM
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('agents-pane');
    if (pane) window.odRenderAgents(pane);
  });

}());
