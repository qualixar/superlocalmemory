// od-operations.js — OD Operations pane renderer v2 (design-faithful)
// Renders into window.odRenderOperations(container).
//
// Endpoints (confirmed with curl against daemon :8765):
//   GET  /api/lifecycle/status → {available, active_profile, total_memories,
//                                  states: {archive, cold, warm},
//                                  recent_transitions: [],
//                                  age_stats: {warm: {avg_days, min_days, max_days}, ...}}
//   POST /api/lifecycle/compact → {status: "not_implemented"} — wired to button only
//   GET  /api/trust/stats       → {avg_trust_score, total_signals, enforcement, by_signal_type}
//   GET  /api/compliance/status → {available, audit_events_count, recent_audit_events,
//                                   retention_policies, abac_policies_count}
//
// NOTE on lifecycle states: API returns keys 'archive', 'cold', 'warm' (not 'active'/'tombstoned').
// Display maps: archive → Archived. Missing keys default to 0.
//
// CRIT fixes applied:
//   (a) Lifecycle counts always from /api/lifecycle/status — no hardcoded state numbers.
//   (b) Compact buttons wired ONLY to click via addEventListener — never auto-fired.
//   (c) All API strings (policy names, actors, actions, targets) pass through esc() for XSS safety.
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

/* global window, document, fetch, Promise */
(function () {
  'use strict';

  // ─── Helpers ──────────────────────────────────────────────────────────────

  /** HTML-escape a value before inserting via innerHTML. */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /** Format a Unix-seconds float or ISO string as "Xm ago". */
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

  /** Format a number with locale separators. */
  function fmtNum(n) { return Number(n || 0).toLocaleString(); }

  /** Get an inline SVG icon from the shell registry; fail-safe if shell absent. */
  function safeIcon(name) {
    return (typeof window.slmIcon === 'function') ? window.slmIcon(name) : '';
  }

  // ── .ctl style injection ────────────────────────────────────────────────────
  // Design defines .ctl via <style> block. Inject once so :last-child border rule works.
  var _ctlStyleInjected = false;
  function injectCtlStyle() {
    if (_ctlStyleInjected || document.getElementById('od-ops-ctl-style')) return;
    var s = document.createElement('style');
    s.id = 'od-ops-ctl-style';
    s.textContent =
      '.od-ops-ctl{display:flex;align-items:center;justify-content:space-between;' +
        'gap:16px;padding:14px 0;border-bottom:1px solid var(--border)}' +
      '.od-ops-ctl:last-child{border-bottom:0}';
    document.head.appendChild(s);
    _ctlStyleInjected = true;
  }

  // ─── Skeleton HTML ────────────────────────────────────────────────────────

  /**
   * Full static HTML for the operations pane — zero API data inside.
   * IDs use od-* prefix to avoid colliding with legacy Bootstrap IDs.
   * Tab switching uses data-od-tab / data-od-pane to isolate from shell nav.
   */
  function buildSkeleton() {
    return (
      // Page header
      '<div class="page-head">' +
        '<h2>Governance</h2>' +
        '<p>Who can access this workspace, how memory ages, trust, and ' +
        'compliance — with the controls to act on them. Nothing here is ' +
        'read-only.</p>' +
      '</div>' +

      // Tab bar
      '<div class="tabs" id="od-ops-tabs">' +
        '<button class="tab active" data-od-tab="lifecycle">Lifecycle</button>' +
        '<button class="tab" data-od-tab="access">Access &amp; Users</button>' +
        '<button class="tab" data-od-tab="trust">Trust</button>' +
        '<button class="tab" data-od-tab="compliance">Compliance</button>' +
        '<button class="tab" data-od-tab="loops">Bounded Loops</button>' +
      '</div>' +

      // ══════ LIFECYCLE TAB ══════════════════════════════════════════════
      '<section class="tabpane active" data-od-pane="lifecycle">' +

        // State KPI strip: 5 lifecycle buckets
        '<div class="kpi-strip" ' +
          'style="margin-bottom:16px;grid-template-columns:repeat(5,1fr)" ' +
          'id="od-lc-states">' +
          '<div class="dim" style="padding:12px">Loading…</div>' +
        '</div>' +

        // Two-column: age table + compaction preview
        '<div class="grid" style="grid-template-columns:1fr 1fr;align-items:start">' +

          '<div class="card">' +
            '<div class="card-head"><h3>Average memory age by state</h3></div>' +
            '<div class="card-pad">' +
              '<table class="tbl" id="od-lc-ages">' +
                '<thead><tr>' +
                  '<th>State</th>' +
                  '<th>Avg age (d)</th>' +
                  '<th>Newest (d)</th>' +
                  '<th>Oldest (d)</th>' +
                '</tr></thead>' +
                '<tbody>' +
                  '<tr><td colspan="4" class="dim" style="text-align:center;padding:20px">Loading…</td></tr>' +
                '</tbody>' +
              '</table>' +
            '</div>' +
          '</div>' +

          '<div class="card">' +
            '<div class="card-head">' +
              '<h3>Compaction preview</h3>' +
              '<span class="sub">dry-run · no changes applied</span>' +
            '</div>' +
            '<div class="card-pad">' +
              '<p class="muted" style="font-size:13px" id="od-compact-summary">' +
                'Click "Re-run dry-run" to preview which memories would be transitioned.' +
              '</p>' +
              '<div id="od-compact-detail" style="margin:14px 0"></div>' +
              '<div style="display:flex;gap:8px">' +
                // CRIT(b): wired via addEventListener in wireCompactBtns(), never auto-fired
                '<button class="btn sm" id="od-compact-preview-btn">Re-run dry-run</button>' +
                '<button class="btn sm primary" id="od-compact-apply-btn">Apply compaction</button>' +
              '</div>' +
            '</div>' +
          '</div>' +

        '</div>' + // end two-column grid

        // Recent transitions table
        '<div class="card" style="margin-top:16px">' +
          '<div class="card-head"><h3>Recent transitions</h3></div>' +
          '<div class="card-pad">' +
            '<table class="tbl" id="od-lc-trans">' +
              '<thead><tr>' +
                '<th>Memory</th><th>Transition</th><th>Reason</th><th>When</th>' +
              '</tr></thead>' +
              '<tbody>' +
                '<tr><td colspan="4" class="dim" style="text-align:center;padding:20px">Loading…</td></tr>' +
              '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +

        // System control — restart the daemon without a terminal
        '<div class="card" style="margin-top:16px;max-width:640px">' +
          '<div class="card-head"><h3>System control</h3></div>' +
          '<div class="card-pad">' +
            '<p class="muted" style="font-size:13px" id="od-restart-summary">' +
              'Restart the memory daemon — applies pending config changes and ' +
              'clears transient state. Memories are never affected.</p>' +
            '<button class="btn sm" id="od-restart-daemon-btn" style="margin-top:10px">' +
              'Restart daemon</button>' +
          '</div>' +
        '</div>' +

      '</section>' +

      // ══════ TRUST TAB ═════════════════════════════════════════════════
      '<section class="tabpane" data-od-pane="trust">' +

        '<div class="card" style="max-width:640px">' +
          '<div class="card-head"><h3>Trust enforcement</h3></div>' +
          '<div class="card-pad" id="od-trust-ctl">' +
            '<div class="dim" style="padding:20px;text-align:center">Loading trust data…</div>' +
          '</div>' +
        '</div>' +

      '</section>' +

      // ══════ ACCESS & USERS TAB ════════════════════════════════════════
      // First-class governance surface for non-technical admins: who can use
      // this workspace and at what role. The card is populated by od-team.js
      // (window.odRenderTeam) into #od-team-mount during odRenderOperations().
      '<section class="tabpane" data-od-pane="access">' +
        '<div class="card" id="od-team-mount">' +
          '<div class="card-head"><h3>Team &amp; access</h3></div>' +
          '<div class="card-pad"><p class="muted">Loading…</p></div>' +
        '</div>' +
      '</section>' +

      // ══════ BOUNDED LOOPS TAB ════════════════════════════════════════
      // Static info card — no backend calls.
      // A bounded loop advances only when an INDEPENDENT gate passes, not
      // when the agent claims success. Laps are persisted as SLM memory
      // (tag loop:<name>) so the full run history is queryable.
      '<section class="tabpane" data-od-pane="loops">' +

        '<div class="page-head" style="margin-bottom:16px">' +
          '<h2 style="font-size:20px;margin-bottom:6px">Bounded Loops</h2>' +
          '<p style="font-size:13.5px">An iteration control pattern for agentic ' +
            'frameworks &mdash; the loop advances only when an <strong>independent ' +
            'gate</strong> passes, not when the agent claims success. Prevents ' +
            'rationalisation: the model cannot self-certify completion.</p>' +
        '</div>' +

        // Two-column: concept + CLI reference
        '<div class="grid" style="grid-template-columns:1fr 1fr;align-items:start;margin-bottom:16px">' +

          '<div class="card">' +
            '<div class="card-head"><h3>How it works</h3></div>' +
            '<div class="card-pad">' +
              '<p style="font-size:13px;line-height:1.6;margin-bottom:12px">' +
                'Standard agentic loops let the model declare itself done &mdash; ' +
                'a known failure mode when the model rationalises instead of verifying. ' +
                'A bounded loop separates <em>execution</em> (the agent) from ' +
                '<em>verification</em> (an independent gate such as a test suite, ' +
                'linter, or judge LLM).' +
              '</p>' +
              '<p style="font-size:13px;line-height:1.6;margin-bottom:12px">' +
                'The loop terminates only when the gate returns <code>DONE</code>, ' +
                'or when a hard lap cap is reached (<code>HALT</code>). Each lap is ' +
                'persisted as a queryable SLM memory tagged ' +
                '<code>loop:&lt;name&gt;</code>.' +
              '</p>' +
              '<div style="background:var(--card-2);border-radius:var(--r-md);' +
                'padding:10px 14px;font-size:12.5px;line-height:1.7">' +
                '<div><span class="badge ok" style="margin-right:8px">DONE</span>' +
                  'Gate passed &mdash; loop succeeded cleanly</div>' +
                '<div style="margin-top:6px"><span class="badge warn" style="margin-right:8px">HALT</span>' +
                  'Lap cap reached &mdash; hard stop applied</div>' +
                '<div style="margin-top:6px"><span class="badge cyan" style="margin-right:8px">PAUSE</span>' +
                  'Awaiting external input or approval</div>' +
                '<div style="margin-top:6px"><span class="badge danger" style="margin-right:8px">KILLED</span>' +
                  'Manually stopped by the operator</div>' +
                '<div style="margin-top:6px"><span class="badge neutral" style="margin-right:8px">ERROR</span>' +
                  'Unrecoverable failure during a lap</div>' +
              '</div>' +
            '</div>' +
          '</div>' +

          '<div class="card">' +
            '<div class="card-head"><h3>Run it: CLI &middot; command &middot; MCP</h3></div>' +
            '<div class="card-pad">' +
              '<p style="font-size:13px;margin-bottom:14px">' +
                'Bounded loops ship on three surfaces &mdash; the same engine and ' +
                'the same queryable ledger behind each.' +
              '</p>' +

              '<div style="font-size:12px;font-weight:600;color:var(--fg-2);margin-bottom:6px">CLI</div>' +
              '<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm loop demo</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">Run a live demo bounded loop</span>' +
                '</div>' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm loop history</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">List loop runs for this profile</span>' +
                '</div>' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm loop show &lt;run_id&gt;</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">Inspect a run lap-by-lap</span>' +
                '</div>' +
              '</div>' +

              '<div style="font-size:12px;font-weight:600;color:var(--fg-2);margin-bottom:6px">Command (Claude Code / Codex)</div>' +
              '<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">/slm-loop</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">Slash command bound to the slm-loop skill + runner agent</span>' +
                '</div>' +
              '</div>' +

              '<div style="font-size:12px;font-weight:600;color:var(--fg-2);margin-bottom:6px">MCP tools (code / full / power profiles)</div>' +
              '<div style="display:flex;flex-direction:column;gap:8px">' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm_loop_run</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">Run a gated loop &mdash; the gate is an independent SLM recall</span>' +
                '</div>' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm_loop_history</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">List runs (read-only)</span>' +
                '</div>' +
                '<div class="list-row">' +
                  '<span class="mono" style="min-width:190px;font-size:13px">slm_loop_show</span>' +
                  '<span style="font-size:12.5px;color:var(--fg-2)">Show a run lap-by-lap (read-only)</span>' +
                '</div>' +
              '</div>' +
              '<div style="margin-top:18px;padding:10px 14px;background:var(--card-2);' +
                'border-radius:var(--r-md);font-size:12.5px;line-height:1.6">' +
                '<b>Memory tagging:</b> each lap is stored with tag ' +
                '<code>loop:&lt;name&gt;</code>. Recall the full history with ' +
                '<br><code>slm recall --tag loop:my-loop-name</code>' +
              '</div>' +
            '</div>' +
          '</div>' +

        '</div>' + // end two-column grid

        // Framework adapters note
        '<div class="card">' +
          '<div class="card-head"><h3>Framework adapters</h3></div>' +
          '<div class="card-pad">' +
            '<p style="font-size:13px;line-height:1.6;margin-bottom:14px">' +
              'Bounded loops integrate with any agentic framework that supports ' +
              'tool-call round-trips. The gate is an ordinary SLM memory check &mdash; ' +
              'no special framework wiring required. Each framework stamps ' +
              '<code>SLM_AGENT_ID</code> so lap history is attribution-aware.' +
            '</p>' +
            '<div style="display:flex;flex-wrap:wrap;gap:8px">' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">CrewAI</span>' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">LangChain</span>' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">LangGraph</span>' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">Semantic Kernel</span>' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">AutoGen</span>' +
              '<span style="padding:4px 12px;border-radius:99px;font-size:12px;' +
                'background:var(--card-2);color:var(--fg-2)">Any MCP-compatible agent</span>' +
            '</div>' +
            '<p style="margin-top:12px;font-size:12.5px;color:var(--fg-2)">' +
              'Learn the full pattern: run <code>/slm-loop</code> (the slm-loop skill) ' +
              'inside Claude Code for an interactive walkthrough with live examples.' +
            '</p>' +
          '</div>' +
        '</div>' +

      '</section>' +

      // ══════ COMPLIANCE TAB ════════════════════════════════════════════
      '<section class="tabpane" data-od-pane="compliance">' +

        // KPI strip: 3 cards with icons
        '<div class="kpi-strip" ' +
          'style="margin-bottom:16px;grid-template-columns:repeat(3,1fr)" ' +
          'id="od-comp-kpi">' +
          '<div class="dim" style="padding:12px">Loading…</div>' +
        '</div>' +

        // Retention policies table with New policy button
        '<div class="card">' +
          '<div class="card-head">' +
            '<h3>Active retention policies</h3>' +
            '<div class="spacer"></div>' +
            // New policy button — shows alert since write endpoint is not yet exposed
            '<button class="btn sm" id="od-new-policy-btn">' +
              safeIcon('plus') + ' New policy' +
            '</button>' +
          '</div>' +
          '<div class="card-pad">' +
            '<table class="tbl" id="od-ret">' +
              '<thead><tr>' +
                '<th>Policy</th>' +
                '<th>Retention (days)</th>' +
                '<th>Category</th>' +
                '<th>Action</th>' +
                '<th></th>' +
              '</tr></thead>' +
              '<tbody>' +
                '<tr><td colspan="5" class="dim" style="text-align:center;padding:20px">Loading…</td></tr>' +
              '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +

        // Data privacy & GDPR — export, run retention, erase (dashboard-operable)
        '<div class="card" style="margin-top:16px">' +
          '<div class="card-head"><h3>Data privacy &amp; GDPR</h3></div>' +
          '<div class="card-pad">' +
            '<p class="muted" style="font-size:13px" id="od-gdpr-summary">' +
              'Manage this profile\'s data under GDPR — export a full copy, run ' +
              'retention now, or erase everything for this profile.</p>' +
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px">' +
              '<button class="btn sm" id="od-gdpr-export-btn">' + safeIcon('download') + ' Export my data</button>' +
              '<button class="btn sm" id="od-retention-run-btn">Run retention now</button>' +
              '<button class="btn sm" id="od-gdpr-erase-btn" style="border-color:#ff6b6b;color:#ff6b6b">Erase this profile…</button>' +
            '</div>' +
          '</div>' +
        '</div>' +

        // Audit trail with All/Denied segment filter
        '<div class="card" style="margin-top:16px">' +
          '<div class="card-head">' +
            '<h3>Audit trail</h3>' +
            '<div class="spacer"></div>' +
            '<div class="seg" id="od-audit-filter">' +
              '<button class="active" data-filter="all">All</button>' +
              '<button data-filter="denied">Denied</button>' +
            '</div>' +
          '</div>' +
          '<div class="card-pad">' +
            '<table class="tbl" id="od-audit">' +
              '<thead><tr>' +
                '<th>Timestamp</th>' +
                '<th>Event</th>' +
                '<th>Actor</th>' +
                '<th>Action</th>' +
                '<th>Target</th>' +
                '<th>Result</th>' +
              '</tr></thead>' +
              '<tbody id="od-audit-body">' +
                '<tr><td colspan="6" class="dim" style="text-align:center;padding:20px">Loading…</td></tr>' +
              '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +

      '</section>'
    );
  }

  // ─── Tab switching ────────────────────────────────────────────────────────
  /**
   * Wire the internal tab bar.
   * Uses data-od-tab on buttons and data-od-pane on sections.
   * No onclick attributes — CSP-safe addEventListener only.
   */
  function initTabs(root) {
    var buttons = root.querySelectorAll('[data-od-tab]');
    var panes   = root.querySelectorAll('[data-od-pane]');

    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = btn.getAttribute('data-od-tab');
        buttons.forEach(function (b) { b.classList.remove('active'); });
        panes.forEach(function (p) {
          p.classList.toggle('active', p.getAttribute('data-od-pane') === target);
        });
        btn.classList.add('active');
      });
    });
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────────
  /**
   * Populate the lifecycle KPI strip.
   * API returns states: {archive, cold, warm}. 'active' and 'tombstoned' not in API.
   * Missing keys default to 0.
   */
  function populateLifecycleStates(lc) {
    var statesEl = document.getElementById('od-lc-states');
    if (!statesEl) return;

    var rawStates = (lc && lc.states) || {};

    // Labels are lowercase to match the approved design (design uses 'active' not 'Active').
    var STATE_MAP = [
      { key: 'active',     label: 'active',     cls: 'ok' },
      { key: 'warm',       label: 'warm',        cls: 'warn' },
      { key: 'cold',       label: 'cold',        cls: 'cyan' },
      { key: 'archive',    label: 'archived',    cls: 'neutral' },
      { key: 'tombstoned', label: 'tombstoned',  cls: 'danger' },
    ];

    statesEl.innerHTML = STATE_MAP.map(function (s) {
      var count = rawStates[s.key] != null ? rawStates[s.key] : 0;
      return (
        '<div class="card kpi">' +
          '<div class="label">' +
            '<span class="badge ' + s.cls + '" style="padding:1px 8px">' +
              '<span class="dot"></span>' + esc(s.label) +
            '</span>' +
          '</div>' +
          '<div class="value num" style="font-size:24px">' + fmtNum(count) + '</div>' +
        '</div>'
      );
    }).join('');
  }

  /**
   * Populate the age-by-state table from lifecycle age_stats.
   * Maps API key 'archive' to display label 'Archived'.
   */
  function populateLifecycleAges(lc) {
    var ageTbody = document.querySelector('#od-lc-ages tbody');
    if (!ageTbody) return;

    var ageStats = (lc && lc.age_stats) || {};
    var ageKeys  = Object.keys(ageStats);

    if (ageKeys.length === 0) {
      ageTbody.innerHTML =
        '<tr><td colspan="4" class="dim" style="text-align:center;padding:16px">' +
        'No age data available yet.</td></tr>';
      return;
    }

    ageTbody.innerHTML = ageKeys.map(function (k) {
      var ag = ageStats[k] || {};
      // Lowercase to match design badge style (design shows 'archived' not 'Archived')
      var displayLabel = (k === 'archive') ? 'archived' : esc(k);
      return (
        '<tr>' +
          '<td><span class="badge neutral">' + displayLabel + '</span></td>' +
          '<td class="num">' + esc(String(Number(ag.avg_days || 0).toFixed(1))) + '</td>' +
          '<td class="num dim">' + esc(String(Number(ag.min_days || 0).toFixed(1))) + '</td>' +
          '<td class="num dim">' + esc(String(Number(ag.max_days || 0).toFixed(1))) + '</td>' +
        '</tr>'
      );
    }).join('');
  }

  /**
   * Populate the recent transitions table.
   * API returns recent_transitions: [] in current live daemon.
   * Shows honest empty state; displays up to 20 rows when present.
   */
  function populateLifecycleTransitions(lc) {
    var transTbody = document.querySelector('#od-lc-trans tbody');
    if (!transTbody) return;

    var trans = (lc && lc.recent_transitions) || [];

    if (trans.length === 0) {
      transTbody.innerHTML =
        '<tr><td colspan="4" class="dim" style="text-align:center;padding:16px">' +
        'No recent transitions. Transitions appear here when memories change lifecycle state.' +
        '</td></tr>';
      return;
    }

    transTbody.innerHTML = trans.slice(0, 20).map(function (t) {
      var idLabel    = t.memory_id ? '#' + esc(String(t.memory_id)) : '—';
      var transition = t.transition ? esc(String(t.transition)) : '—';
      var reason     = t.reason ? esc(String(t.reason)) : '—';
      var when       = t.timestamp ? esc(timeAgo(t.timestamp)) : '—';
      return (
        '<tr>' +
          '<td class="mono dim">' + idLabel + '</td>' +
          '<td><span class="badge neutral">' + transition + '</span></td>' +
          '<td class="dim">' + reason + '</td>' +
          '<td class="dim">' + when + '</td>' +
        '</tr>'
      );
    }).join('');
  }

  /**
   * Populate all lifecycle sections.
   * Calls populateLifecycleStates, populateLifecycleAges, populateLifecycleTransitions.
   */
  function populateLifecycle(lc) {
    if (!lc) {
      var stEl = document.getElementById('od-lc-states');
      if (stEl) stEl.innerHTML = '<p class="muted">Lifecycle data unavailable.</p>';
      return;
    }
    populateLifecycleStates(lc);
    populateLifecycleAges(lc);
    populateLifecycleTransitions(lc);
  }

  // ─── Compact buttons ──────────────────────────────────────────────────────
  /**
   * Wire the compaction buttons via addEventListener.
   * CRIT(b): preview-btn and apply-btn only fire on explicit user click.
   * Current API returns {status:"not_implemented"} for POST /api/lifecycle/compact.
   */
  function wireCompactBtns() {
    var previewBtn = document.getElementById('od-compact-preview-btn');
    var applyBtn   = document.getElementById('od-compact-apply-btn');
    var summaryEl  = document.getElementById('od-compact-summary');
    var detailEl   = document.getElementById('od-compact-detail');

    if (previewBtn) {
      previewBtn.addEventListener('click', function () {
        previewBtn.disabled = true;
        previewBtn.textContent = 'Running…';

        fetch('/api/lifecycle/compact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dry_run: true }),
        })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (data && data.success === false) {
              if (summaryEl) summaryEl.innerHTML =
                '<span class="dim">' + esc(data.error || 'Compaction unavailable.') + '</span>';
              return;
            }
            var recs = Number(data.candidates || 0);
            if (summaryEl) {
              summaryEl.innerHTML =
                'Compaction preview: <b style="color:var(--fg)">' + fmtNum(recs) +
                '</b> of ' + fmtNum(data.total_facts || 0) +
                ' memories would be transitioned.';
            }
            if (detailEl) {
              var details = (data.transitions || []).slice(0, 10);
              if (details.length === 0) {
                detailEl.innerHTML =
                  '<p class="muted" style="font-size:13px">' +
                  'No compaction needed — all memories in optimal states.</p>';
              } else {
                detailEl.innerHTML = details.map(function (d) {
                  return (
                    '<div style="display:flex;justify-content:space-between;' +
                      'align-items:center;padding:8px 0;' +
                      'border-bottom:1px solid var(--border)">' +
                      '<span class="mono" style="font-size:12.5px">' +
                        esc(String(d.fact_id || '')) +
                      '</span>' +
                      '<span class="badge neutral">' +
                        esc(d.current || '') + ' → ' + esc(d.proposed || '') +
                      '</span>' +
                    '</div>'
                  );
                }).join('');
              }
            }
          })
          .catch(function (err) {
            if (summaryEl) summaryEl.textContent = 'Preview failed. Check console.';
            if (typeof console !== 'undefined') {
              console.error('[od-ops] compact preview error:', err);
            }
          })
          .finally(function () {
            previewBtn.disabled = false;
            previewBtn.textContent = 'Re-run dry-run';
          });
      });
    }

    if (applyBtn) {
      applyBtn.addEventListener('click', function () {
        if (!window.confirm('This will transition memories to lower lifecycle states. Continue?')) return;
        applyBtn.disabled = true;
        applyBtn.textContent = 'Running…';

        fetch('/api/lifecycle/compact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dry_run: false }),
        })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (data && data.success === false) {
              if (summaryEl) summaryEl.innerHTML =
                '<span class="dim">' + esc(data.error || 'Compaction failed.') + '</span>';
              return;
            }
            var n = Number(data.applied || 0);
            if (summaryEl) {
              summaryEl.innerHTML =
                'Compaction complete: <b style="color:var(--ok)">' +
                fmtNum(n) + '</b> memories transitioned.';
            }
          })
          .catch(function (err) {
            if (summaryEl) summaryEl.textContent = 'Compaction failed. Check console.';
            if (typeof console !== 'undefined') {
              console.error('[od-ops] compact apply error:', err);
            }
          })
          .finally(function () {
            applyBtn.disabled = false;
            applyBtn.textContent = 'Apply compaction';
          });
      });
    }
  }

  /**
   * Wire the "Restart daemon" button. POSTs /api/daemon/restart (which spawns a
   * detached `slm restart`), then polls /health until the fresh daemon is back
   * and reloads the dashboard. Wired via addEventListener — never auto-fired.
   */
  function wireRestartBtn() {
    var btn = document.getElementById('od-restart-daemon-btn');
    var summary = document.getElementById('od-restart-summary');
    if (!btn) return;
    btn.addEventListener('click', function () {
      if (!window.confirm(
        'Restart the memory daemon now? It will be unavailable for a few ' +
        'seconds. Your memories are not affected.')) return;
      btn.disabled = true;
      btn.textContent = 'Restarting…';
      if (summary) summary.textContent = 'Restart requested — waiting for the daemon to come back…';

      fetch('/api/daemon/restart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      })
        .then(function (r) { return r.json().catch(function () { return {}; }); })
        .then(function (data) {
          if (data && data.success === false) {
            if (summary) summary.textContent = 'Restart failed: ' + (data.error || 'unknown error');
            btn.disabled = false; btn.textContent = 'Restart daemon';
            return;
          }
          // Poll /health until the fresh daemon reports ready, then reload.
          var tries = 0;
          var poll = setInterval(function () {
            tries += 1;
            fetch('/health', { cache: 'no-store' })
              .then(function (r) { return r.ok ? r.json() : null; })
              .then(function (h) {
                if (h && (h.ready === true || h.engine === 'initialized')) {
                  clearInterval(poll);
                  if (summary) summary.textContent = 'Daemon back online — reloading…';
                  setTimeout(function () { window.location.reload(); }, 600);
                }
              })
              .catch(function () { /* daemon still down mid-restart — keep polling */ });
            if (tries > 30) {  // ~30s ceiling
              clearInterval(poll);
              if (summary) summary.textContent =
                'Daemon is taking longer than expected. Refresh the page in a moment.';
              btn.disabled = false; btn.textContent = 'Restart daemon';
            }
          }, 1000);
        })
        .catch(function () {
          // The connection often drops as the daemon stops — that's expected;
          // start polling for it to come back.
          if (summary) summary.textContent = 'Daemon restarting — waiting for it to come back…';
          var tries = 0;
          var poll = setInterval(function () {
            tries += 1;
            fetch('/health', { cache: 'no-store' })
              .then(function (r) { return r.ok ? r.json() : null; })
              .then(function (h) {
                if (h && (h.ready === true || h.engine === 'initialized')) {
                  clearInterval(poll);
                  window.location.reload();
                }
              })
              .catch(function () {});
            if (tries > 30) { clearInterval(poll); btn.disabled = false; btn.textContent = 'Restart daemon'; }
          }, 1000);
        });
    });
  }

  // ─── Trust ────────────────────────────────────────────────────────────────
  /**
   * Populate trust enforcement section from /api/trust/stats.
   * Toggle controls are read-only display — no write endpoint exists yet.
   * TODO: wire toggles when /api/trust/settings PATCH is exposed.
   */
  function populateTrust(trustStats) {
    var ctlEl = document.getElementById('od-trust-ctl');
    if (!ctlEl) return;

    if (!trustStats) {
      ctlEl.innerHTML = '<p class="muted">Trust data unavailable.</p>';
      return;
    }

    var enforcement = String(trustStats.enforcement || 'unknown');
    var isEnforced  = enforcement.toLowerCase() !== 'silent collection' &&
                      enforcement.toLowerCase() !== 'off' &&
                      enforcement.toLowerCase() !== 'disabled';

    // Helper: a control row using .od-ops-ctl class (injectCtlStyle injects .ctl rules).
    // :last-child removes border-bottom on the final row — matches design's <style> block.
    function ctlRow(label, sublabel, control) {
      return (
        '<div class="od-ops-ctl">' +
          '<div>' +
            '<b>' + esc(label) + '</b>' +
            '<div class="dim" style="font-size:12.5px">' + esc(sublabel) + '</div>' +
          '</div>' +
          control +
        '</div>'
      );
    }

    // Switch is display-only (no write endpoint) — disabled + lower opacity.
    // enforcement: if mode is "silent collection" / off / disabled → switch is OFF.
    var enfSwitch = '<button class="switch' + (isEnforced ? ' on' : '') +
      '" disabled aria-label="read-only" style="cursor:not-allowed;opacity:0.7"></button>';

    // Design shows only 4 ctl rows — no stats summary block above them.
    ctlEl.innerHTML = (
      ctlRow(
        'Enforcement mode',
        'Block writes from agents below the trust floor',
        enfSwitch
      ) +
      // TODO: trust floor — no read or write endpoint in current API
      ctlRow(
        'Trust floor',
        'Minimum trust score to write',
        '<span class="mono">0.30</span>' // TODO: read from /api/trust/config when available
      ) +
      // TODO: quick-delete penalty — no read/write endpoint yet
      ctlRow(
        'Quick-delete penalty',
        'Lower trust when writes are deleted < 60s',
        '<button class="switch on" disabled style="cursor:not-allowed;opacity:0.7"></button>'
      ) +
      // TODO: recall reward — no read/write endpoint yet
      ctlRow(
        'Recall reward',
        'Raise trust when a memory is recalled',
        '<button class="switch on" disabled style="cursor:not-allowed;opacity:0.7"></button>'
      )
    );
  }

  // ─── Compliance ───────────────────────────────────────────────────────────
  /**
   * Populate the compliance KPI strip — 3 cards with icons.
   * Icons via window.slmIcon() so they work in dynamically-injected content.
   */
  function populateComplianceKpi(comp) {
    var kpiEl = document.getElementById('od-comp-kpi');
    if (!kpiEl) return;

    var auditCount = comp ? Number(comp.audit_events_count || 0)        : 0;
    var retCount   = comp && comp.retention_policies ? comp.retention_policies.length : 0;
    var abacCount  = comp ? Number(comp.abac_policies_count || 0)       : 0;

    kpiEl.innerHTML = (
      '<div class="card kpi">' +
        // Design uses data-ic="shield". Shield is not in the icon registry (od-shell.js).
        // safeIcon('health') (pulse/waveform) is the closest semantic match for audit monitoring.
        '<div class="label">' + safeIcon('health') + ' Audit events</div>' +
        '<div class="value num">' + fmtNum(auditCount) + '</div>' +
      '</div>' +
      '<div class="card kpi">' +
        '<div class="label">' + safeIcon('clock') + ' Retention policies</div>' +
        '<div class="value num">' + fmtNum(retCount) + '</div>' +
      '</div>' +
      '<div class="card kpi">' +
        '<div class="label">' + safeIcon('lock') + ' ABAC rules</div>' +
        '<div class="value num">' + fmtNum(abacCount) + '</div>' +
      '</div>'
    );
  }
  /**
   * Populate the retention policies table.
   * CRIT(c): policy name, category, action go through esc().
   * Empty state explains how to add policies via MCP tool.
   */
  function populateRetentionPolicies(comp) {
    var retTbody = document.querySelector('#od-ret tbody');
    if (!retTbody) return;

    var policies = (comp && comp.retention_policies) || [];

    if (policies.length === 0) {
      retTbody.innerHTML =
        '<tr><td colspan="5" class="dim" style="text-align:center;padding:20px">' +
        'No retention policies configured. Use the set_retention_policy MCP tool to add one.' +
        '</td></tr>';
      return;
    }

    var ACTION_CLS = { archive: 'neutral', tombstone: 'danger', notify: 'warn' };

    retTbody.innerHTML = policies.map(function (p) {
      var ac = ACTION_CLS[p.action] || 'neutral';
      // CRIT(c): name, category, action through esc()
      return (
        '<tr>' +
          '<td><b>' + esc(p.name || '') + '</b></td>' +
          '<td class="num">' + esc(String(p.retention_days || '—')) + '</td>' +
          '<td><span class="badge neutral">' + esc(p.category || 'all') + '</span></td>' +
          '<td><span class="badge ' + ac + '">' + esc(p.action || '') + '</span></td>' +
          '<td style="text-align:right">' +
            '<button class="btn sm ghost" disabled>Edit</button>' +
            // TODO: wire edit button to /api/compliance/retention-policy PATCH when available
          '</td>' +
        '</tr>'
      );
    }).join('');
  }

  // Cache of all audit rows (for client-side All/Denied filter)
  var _allAuditRows = [];
  /**
   * Populate the audit trail table.
   * CRIT(c): actor, action, target, result all through esc().
   */
  function populateAuditTrail(comp) {
    var auditTbody = document.getElementById('od-audit-body');
    if (!auditTbody) return;

    var events = (comp && comp.recent_audit_events) || [];
    _allAuditRows = events;

    renderAuditRows(auditTbody, events);
  }

  /** Render audit table rows into tbody, filtering by mode ('all' or 'denied'). */
  function renderAuditRows(tbody, events) {
    if (events.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="6" class="dim" style="text-align:center;padding:20px">' +
        'No audit events recorded yet.</td></tr>';
      return;
    }

    var EV_CLS = {
      recall:               'cyan',
      remember:             'ok',
      delete:               'danger',
      lifecycle_transition: 'warn',
      access_denied:        'danger',
      retention_enforced:   'warn',
    };

    tbody.innerHTML = events.slice(0, 50).map(function (ev) {
      var evc      = EV_CLS[ev.event_type] || 'neutral';
      var isOk     = ev.result === 'success' || ev.result === 'allowed';
      var isDenied = ev.result === 'denied'  || ev.result === 'error';
      var resultCls = isOk ? 'ok' : isDenied ? 'danger' : 'neutral';
      // CRIT(c): all user-sourced strings through esc()
      return (
        '<tr>' +
          '<td class="mono dim" style="font-size:12px">' + esc(ev.timestamp || '') + '</td>' +
          '<td><span class="badge ' + evc + '">' + esc(ev.event_type || '') + '</span></td>' +
          '<td class="mono" style="font-size:12px">' + esc(ev.actor || '') + '</td>' +
          '<td>' + esc(ev.action || '') + '</td>' +
          '<td class="dim">' + esc(ev.target || '') + '</td>' +
          '<td><span class="badge ' + resultCls + '">' + esc(ev.result || '') + '</span></td>' +
        '</tr>'
      );
    }).join('');
  }
  /**
   * Wire the All/Denied segment filter in the audit trail header.
   * Client-side filter over already-fetched data — no extra network call.
   */
  function wireAuditFilter() {
    var filterEl = document.getElementById('od-audit-filter');
    var tbody    = document.getElementById('od-audit-body');
    if (!filterEl || !tbody) return;

    filterEl.querySelectorAll('button').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var mode = btn.getAttribute('data-filter');
        filterEl.querySelectorAll('button').forEach(function (b) {
          b.classList.remove('active');
        });
        btn.classList.add('active');

        var rows = (mode === 'denied')
          ? _allAuditRows.filter(function (ev) {
              return ev.result === 'denied' || ev.result === 'error' ||
                     (ev.event_type && ev.event_type.indexOf('denied') >= 0);
            })
          : _allAuditRows;

        renderAuditRows(tbody, rows);
      });
    });
  }

  /**
   * Wire the "New policy" button.
   * Informs user that the write endpoint is not yet exposed.
   */
  function wireNewPolicyBtn() {
    var btn = document.getElementById('od-new-policy-btn');
    if (!btn) return;
    btn.addEventListener('click', openRetentionPolicyModal);
  }

  /**
   * Wire the GDPR data-privacy controls: export (download), run retention now,
   * and erase-this-profile (typed confirmation). All dashboard-operable.
   */
  function wireGdprBtns() {
    var summary = document.getElementById('od-gdpr-summary');
    var exportBtn = document.getElementById('od-gdpr-export-btn');
    var runBtn = document.getElementById('od-retention-run-btn');
    var eraseBtn = document.getElementById('od-gdpr-erase-btn');

    if (exportBtn) {
      exportBtn.addEventListener('click', function () {
        // Stream the JSON attachment to a download via a temporary anchor.
        var a = document.createElement('a');
        a.href = '/api/compliance/gdpr/export';
        a.download = '';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        if (summary) summary.textContent = 'Export started — check your downloads.';
      });
    }

    if (runBtn) {
      runBtn.addEventListener('click', function () {
        runBtn.disabled = true; runBtn.textContent = 'Running…';
        fetch('/api/compliance/retention/enforce', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
        })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (summary) {
              summary.textContent = d && d.success
                ? ('Retention applied — archived ' + (d.archived || 0) +
                   ', tombstoned ' + (d.tombstoned || 0) + ', flagged ' + (d.notified || 0) + '.')
                : ('Retention failed: ' + ((d && d.error) || 'unknown'));
            }
          })
          .catch(function () { if (summary) summary.textContent = 'Retention run failed. Check console.'; })
          .finally(function () { runBtn.disabled = false; runBtn.textContent = 'Run retention now'; });
      });
    }

    if (eraseBtn) {
      eraseBtn.addEventListener('click', async function () {
        // Resolve the active profile so the confirmation is unambiguous.
        var profile = 'default';
        try {
          var s = await fetch('/status', { cache: 'no-store' }).then(function (r) { return r.json(); });
          profile = s.profile || s.active_profile || 'default';
        } catch (e) {}
        if (profile === 'default') {
          window.alert('The default profile cannot be erased.');
          return;
        }
        var typed = window.prompt(
          'This PERMANENTLY erases ALL data for profile "' + profile + '".\n' +
          'This cannot be undone.\n\nType the profile name to confirm:');
        if (typed !== profile) {
          if (typed !== null && summary) summary.textContent = 'Erase cancelled — name did not match.';
          return;
        }
        eraseBtn.disabled = true; eraseBtn.textContent = 'Erasing…';
        fetch('/api/compliance/gdpr/erase', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirm: profile }),
        })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (summary) {
              summary.textContent = d && d.success
                ? ('Profile "' + profile + '" erased. Reloading…')
                : ('Erase failed: ' + ((d && d.error) || 'unknown'));
            }
            if (d && d.success) setTimeout(function () { window.location.reload(); }, 900);
          })
          .catch(function () { if (summary) summary.textContent = 'Erase failed. Check console.'; })
          .finally(function () { eraseBtn.disabled = false; eraseBtn.textContent = 'Erase this profile…'; });
      });
    }
  }

  /**
   * OD-styled "New retention policy" modal → POST /api/compliance/retention-policy.
   * Non-technical users define GDPR/retention rules without a CLI. On success
   * the Compliance tab reloads so the new policy appears immediately.
   */
  function openRetentionPolicyModal() {
    var existing = document.getElementById('od-retention-overlay');
    if (existing) existing.remove();

    var overlay = document.createElement('div');
    overlay.id = 'od-retention-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.style.cssText =
      'position:fixed;inset:0;z-index:11000;display:flex;align-items:center;' +
      'justify-content:center;background:rgba(0,0,0,0.45);backdrop-filter:blur(2px)';

    var card = document.createElement('div');
    card.style.cssText =
      'width:min(460px,93vw);background:var(--card,#1a1f2e);color:var(--fg,#e8ecf3);' +
      'border:1px solid var(--border,rgba(255,255,255,0.1));border-radius:14px;' +
      'box-shadow:0 20px 60px rgba(0,0,0,0.4);padding:22px 22px 18px';

    var fieldStyle =
      'width:100%;box-sizing:border-box;padding:9px 11px;font-size:0.9rem;margin-top:4px;' +
      'background:var(--page,rgba(255,255,255,0.04));color:var(--fg,#e8ecf3);' +
      'border:1px solid var(--border,rgba(255,255,255,0.14));border-radius:8px;outline:none';
    var labelStyle = 'display:block;margin-top:12px;font-size:0.8rem;color:var(--fg-3,#8b93a7)';

    card.innerHTML =
      '<h3 style="margin:0 0 4px;font-size:1.05rem;font-weight:600">New retention policy</h3>' +
      '<p style="margin:0 0 8px;font-size:0.8125rem;color:var(--fg-3,#8b93a7)">' +
        'Automatically age out memories older than a set period — GDPR-style retention.</p>' +
      '<label style="' + labelStyle + '">Policy name' +
        '<input id="od-rp-name" type="text" maxlength="60" placeholder="e.g. GDPR 90-day" style="' + fieldStyle + '"></label>' +
      '<label style="' + labelStyle + '">Retain for (days)' +
        '<input id="od-rp-days" type="number" min="1" value="90" style="' + fieldStyle + '"></label>' +
      '<label style="' + labelStyle + '">Framework' +
        '<select id="od-rp-cat" style="' + fieldStyle + '">' +
          '<option value="gdpr">GDPR</option><option value="hipaa">HIPAA</option>' +
          '<option value="sox">SOX</option><option value="custom">Custom</option>' +
        '</select></label>' +
      '<label style="' + labelStyle + '">When a memory expires' +
        '<select id="od-rp-action" style="' + fieldStyle + '">' +
          '<option value="archive">Archive (keep, hide from recall)</option>' +
          '<option value="tombstone">Tombstone (mark deleted)</option>' +
          '<option value="notify">Notify only (no change)</option>' +
        '</select></label>' +
      '<div id="od-rp-err" style="min-height:18px;margin:8px 2px 0;font-size:0.75rem;color:#ff6b6b"></div>' +
      '<div style="display:flex;gap:8px;justify-content:flex-end;margin-top:8px">' +
        '<button type="button" id="od-rp-cancel" style="padding:8px 14px;font-size:0.85rem;border-radius:8px;cursor:pointer;background:transparent;color:var(--fg-3,#8b93a7);border:1px solid var(--border,rgba(255,255,255,0.14))">Cancel</button>' +
        '<button type="button" id="od-rp-create" style="padding:8px 16px;font-size:0.85rem;border-radius:8px;cursor:pointer;background:var(--violet,#7c5cff);color:#fff;border:1px solid transparent;font-weight:600">Create policy</button>' +
      '</div>';

    overlay.appendChild(card);
    document.body.appendChild(overlay);
    setTimeout(function () { var n = document.getElementById('od-rp-name'); if (n) n.focus(); }, 30);

    function close() { document.removeEventListener('keydown', onKey); overlay.remove(); }
    function onKey(e) { if (e.key === 'Escape') close(); }
    var errEl = document.getElementById('od-rp-err');
    var createBtn = document.getElementById('od-rp-create');

    async function submit() {
      var name = (document.getElementById('od-rp-name').value || '').trim();
      var days = parseInt(document.getElementById('od-rp-days').value, 10);
      var cat = document.getElementById('od-rp-cat').value;
      var action = document.getElementById('od-rp-action').value;
      if (!name) { errEl.textContent = 'Please enter a policy name.'; return; }
      if (!(days >= 1)) { errEl.textContent = 'Retention days must be a positive number.'; return; }
      createBtn.disabled = true; createBtn.textContent = 'Creating…'; errEl.textContent = '';
      try {
        var r = await fetch('/api/compliance/retention-policy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name, retention_days: days, category: cat, action: action }),
        });
        var data = await r.json().catch(function () { return {}; });
        if (!r.ok || data.success === false) {
          errEl.textContent = (data && data.error) || 'Failed to create policy.';
          createBtn.disabled = false; createBtn.textContent = 'Create policy';
          return;
        }
        close();
        if (typeof window.odRenderOperations === 'function') {
          // Reload the Operations pane so the new policy shows in the table.
          var pane = document.getElementById('operations-pane');
          if (pane) window.odRenderOperations(pane);
        }
      } catch (e) {
        errEl.textContent = 'Network error creating policy.';
        createBtn.disabled = false; createBtn.textContent = 'Create policy';
      }
    }

    document.getElementById('od-rp-cancel').addEventListener('click', close);
    createBtn.addEventListener('click', submit);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
    document.addEventListener('keydown', onKey);
  }

  // ─── Main entry point ─────────────────────────────────────────────────────
  /**
   * Fetch all operations endpoints, inject the OD skeleton, and populate with real data.
   * Idempotent: each call clears and re-renders the container.
   *
   * @param {HTMLElement} container  Usually document.getElementById("operations-pane").
   */
  function odRenderOperations(container) {
    if (!container) return;

    container.innerHTML =
      '<div style="color:var(--fg-2);text-align:center;padding:48px 0">' +
      'Loading operations data…</div>';

    // Reset audit row cache on each render
    _allAuditRows = [];

    Promise.all([
      fetch('/api/lifecycle/status').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/trust/stats').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      fetch('/api/compliance/status').then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
    ]).then(function (results) {
      var lc         = results[0];
      var trustStats = results[1];
      var comp       = results[2];

      // Inject .ctl class styles (used by Trust section control rows)
      injectCtlStyle();

      // Inject static skeleton — no API data inside
      container.innerHTML = buildSkeleton();

      // Wire tab switching (CSP-safe)
      initTabs(container);

      // Populate lifecycle
      populateLifecycle(lc);
      // CRIT(b): compact buttons wired here — never auto-fired on render
      wireCompactBtns();
      // Restart-daemon button — wired via addEventListener, never auto-fired
      wireRestartBtn();

      // Populate trust
      populateTrust(trustStats);

      // Populate compliance
      populateComplianceKpi(comp);
      populateRetentionPolicies(comp);
      populateAuditTrail(comp);

      // Wire audit filter and new-policy button (listeners on injected DOM)
      wireAuditFilter();
      wireNewPolicyBtn();
      wireGdprBtns();

      // Team & access (RBAC / C3) — rendered by od-team.js
      if (typeof window.odRenderTeam === 'function') {
        window.odRenderTeam(document.getElementById('od-team-mount'));
      }

    }).catch(function (err) {
      container.innerHTML =
        '<div style="color:var(--danger);text-align:center;padding:48px 0">' +
        'Failed to load operations data. Check the daemon is running on port 8765.</div>';
      if (typeof console !== 'undefined') {
        console.error('[od-ops] render error:', err);
      }
    });
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  window.odRenderOperations = odRenderOperations;

  // Auto-run: if the operations pane is already in the DOM at script load, render it.
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('operations-pane');
    if (pane) odRenderOperations(pane);
  });

}());
