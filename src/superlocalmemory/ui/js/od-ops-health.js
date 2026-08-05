/* od-ops-health.js — Wave-3 Admin Operations & Health panel
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0-or-later
 *
 * Injects an "Admin: Ops Health" tab into the Governance pane tab bar
 * rendered by od-operations.js. Uses a MutationObserver so it fires after
 * od-operations.js's innerHTML replacement completes. Additive only — never
 * modifies or removes od-operations.js tabs or sections.
 *
 * Surfaces:
 *   - Dead-letter DLQ entries (ingestion exhausted)
 *   - Degraded completion manifests
 *   - Exhausted projection obligations
 *   - Writer stall warning
 *   - Per-entry action buttons: retry / force_reconcile / cancel
 */

(function () {
  'use strict';

  /* -----------------------------------------------------------------------
   * Constants
   * --------------------------------------------------------------------- */
  var POLL_MS = 30000;      // auto-refresh interval when tab is active
  var TAB_KEY = 'admin-ops-health';
  var TAB_LABEL = 'Admin: Ops Health';
  var PANE_ID = 'od-pane-admin-ops-health';
  var _pollTimer = null;
  var _injected = false;

  /* -----------------------------------------------------------------------
   * DOM helpers
   * --------------------------------------------------------------------- */
  function $id(id) { return document.getElementById(id); }

  function _esc(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* -----------------------------------------------------------------------
   * Fetch helpers — never throws
   * --------------------------------------------------------------------- */
  function _get(path) {
    return fetch(path, { credentials: 'same-origin' })
      .then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; });
      })
      .catch(function (err) { return { ok: false, status: 0, data: null, err: String(err) }; });
  }

  function _post(path, body) {
    return fetch(path, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; });
      })
      .catch(function (err) { return { ok: false, status: 0, data: null, err: String(err) }; });
  }

  /* -----------------------------------------------------------------------
   * Tab injection — called once od-operations.js has rendered #od-ops-tabs
   * --------------------------------------------------------------------- */
  function _injectTab() {
    var tabBar = $id('od-ops-tabs');
    if (!tabBar) return;

    /* Don't inject twice (idempotent) */
    if (_injected || tabBar.querySelector('[data-od-tab="' + TAB_KEY + '"]')) {
      _injected = true;
      return;
    }

    /* Add tab button */
    var tabBtn = document.createElement('button');
    tabBtn.className = 'tab';
    tabBtn.setAttribute('data-od-tab', TAB_KEY);
    tabBtn.style.cssText = 'border-color: var(--bs-warning, #ffc107); color: var(--bs-warning, #ffc107);';
    tabBtn.textContent = TAB_LABEL;
    tabBar.appendChild(tabBtn);

    /* Add tab pane section */
    var pane = document.createElement('section');
    pane.className = 'tabpane';
    pane.setAttribute('data-od-pane', TAB_KEY);
    pane.id = PANE_ID;
    pane.innerHTML = _buildPaneHtml();
    tabBar.parentNode.appendChild(pane);

    /* Wire tab click into od-operations.js's existing tab-switching logic
     * by dispatching a click on the same element od-operations uses */
    tabBtn.addEventListener('click', function () { _onTabActivated(); });

    /* Also patch od-operations.js's tab system if it uses delegated events */
    _patchTabSystem(tabBar);

    _injected = true;
    _attachActionListeners(pane);
  }

  function _buildPaneHtml() {
    return (
      '<div style="padding:16px">' +

      /* Header */
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">' +
        '<div>' +
          '<h2 style="font-size:20px;margin-bottom:4px">Operations Health</h2>' +
          '<p style="font-size:13px;opacity:0.7;margin:0">Stuck ingestion, failed projections, writer stalls — visible here first. OWNER or ADMIN access required.</p>' +
        '</div>' +
        '<button id="oh-refresh-btn" style="padding:6px 14px;border-radius:6px;border:1px solid var(--bs-secondary,#6c757d);background:transparent;color:inherit;cursor:pointer">&#x21bb; Refresh</button>' +
      '</div>' +

      /* Health banner (error state) */
      '<div id="oh-banner-error" style="display:none;padding:10px 14px;border-radius:8px;background:rgba(220,53,69,0.15);border:1px solid rgba(220,53,69,0.4);margin-bottom:14px;font-size:13px">' +
        '&#x26a0; <span id="oh-banner-text">Failures detected.</span>' +
        '<span style="opacity:0.6;margin-left:8px">Resolve below or run <code>slm ops list</code>.</span>' +
      '</div>' +

      /* Health banner (ok state) */
      '<div id="oh-banner-ok" style="display:none;padding:10px 14px;border-radius:8px;background:rgba(25,135,84,0.15);border:1px solid rgba(25,135,84,0.4);margin-bottom:14px;font-size:13px">' +
        '&#x2713; All operations healthy &mdash; no failures detected.' +
      '</div>' +

      /* Writer stall warning */
      '<div id="oh-stall-banner" style="display:none;padding:10px 14px;border-radius:8px;background:rgba(255,193,7,0.12);border:1px solid rgba(255,193,7,0.5);margin-bottom:14px;font-size:13px">' +
        '&#x23f3; <strong>Writer stalled</strong> &mdash; operation <code id="oh-stall-op">?</code> inflight for <span id="oh-stall-age">?</span>s. Other team members may experience slow writes.' +
      '</div>' +

      /* KPI strip */
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px">' +
        '<div style="border:1px solid var(--bs-border-color,rgba(255,255,255,0.12));border-radius:8px;padding:12px;text-align:center">' +
          '<div id="oh-kpi-dlq" style="font-size:22px;font-weight:600">-</div>' +
          '<div style="font-size:12px;opacity:0.6">Dead-letter</div>' +
        '</div>' +
        '<div style="border:1px solid var(--bs-border-color,rgba(255,255,255,0.12));border-radius:8px;padding:12px;text-align:center">' +
          '<div id="oh-kpi-degraded" style="font-size:22px;font-weight:600">-</div>' +
          '<div style="font-size:12px;opacity:0.6">Degraded</div>' +
        '</div>' +
        '<div style="border:1px solid var(--bs-border-color,rgba(255,255,255,0.12));border-radius:8px;padding:12px;text-align:center">' +
          '<div id="oh-kpi-exhausted" style="font-size:22px;font-weight:600">-</div>' +
          '<div style="font-size:12px;opacity:0.6">Exhausted</div>' +
        '</div>' +
      '</div>' +

      /* Table wrap */
      '<div id="oh-table-wrap"><p style="opacity:0.5;font-size:13px">Click Refresh or navigate to this tab to load.</p></div>' +

      '</div>'
    );
  }

  /* Patches od-operations.js's internal tab router so our pane hides/shows
   * correctly when other tabs are clicked.  Intercepts the tab bar's click
   * events and extends the existing handler. */
  function _patchTabSystem(tabBar) {
    tabBar.addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('[data-od-tab]') : e.target;
      if (!btn || !btn.getAttribute) return;
      var key = btn.getAttribute('data-od-tab');
      if (!key) return;

      /* Show our pane only when our tab is clicked */
      var ourPane = $id(PANE_ID);
      if (!ourPane) return;

      if (key === TAB_KEY) {
        /* Deactivate all od-operations.js panes first */
        var sibs = tabBar.parentNode.querySelectorAll('[data-od-pane]');
        sibs.forEach(function (s) { s.classList.remove('active'); });
        ourPane.classList.add('active');
        _onTabActivated();
      } else {
        ourPane.classList.remove('active');
      }
    }, true /* capture — fires before od-operations.js's handler */);
  }

  /* -----------------------------------------------------------------------
   * Tab activation — load + auto-refresh
   * --------------------------------------------------------------------- */
  function _onTabActivated() {
    _load();
    _startAutoRefresh();
  }

  function _startAutoRefresh() {
    _stopAutoRefresh();
    _pollTimer = setInterval(_load, POLL_MS);
  }

  function _stopAutoRefresh() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  }

  /* -----------------------------------------------------------------------
   * Data loading
   * --------------------------------------------------------------------- */
  function _load() {
    var wrap = $id('oh-table-wrap');
    if (wrap) wrap.innerHTML = '<p style="opacity:0.5;font-size:13px">Loading…</p>';

    _get('/operations/failed').then(function (res) {
      if (res.status === 403) { _renderPermissionDenied(); return; }
      if (!res.ok || !res.data) { _renderError(res.err || ('HTTP ' + res.status)); return; }
      _renderData(res.data);
    });

    _get('/status').then(function (res) {
      if (!res.ok || !res.data) return;
      _renderStall(res.data);
    });
  }

  /* -----------------------------------------------------------------------
   * Rendering
   * --------------------------------------------------------------------- */
  function _set(id, val) { var el = $id(id); if (el) el.textContent = val; }
  function _show(id) { var el = $id(id); if (el) el.style.display = ''; }
  function _hide(id) { var el = $id(id); if (el) el.style.display = 'none'; }

  function _renderPermissionDenied() {
    _hide('oh-banner-error'); _hide('oh-banner-ok');
    var w = $id('oh-table-wrap');
    if (w) w.innerHTML = '<p style="color:#ffc107;font-size:13px">&#x1f512; Permission denied. This panel requires <strong>OWNER</strong> or <strong>ADMIN</strong> role.</p>';
    _set('oh-kpi-dlq', '-'); _set('oh-kpi-degraded', '-'); _set('oh-kpi-exhausted', '-');
  }

  function _renderError(msg) {
    _hide('oh-banner-error'); _hide('oh-banner-ok');
    var w = $id('oh-table-wrap');
    if (w) w.innerHTML = '<p style="opacity:0.5;font-size:13px">Could not load operations: ' + _esc(msg) + '</p>';
  }

  function _renderData(data) {
    var dlq = data.dead_letter || [];
    var degraded = data.degraded_manifests || [];
    var exhausted = data.exhausted_obligations || [];
    var total = (data.total !== undefined) ? data.total : (dlq.length + degraded.length + exhausted.length);

    _set('oh-kpi-dlq', dlq.length);
    _set('oh-kpi-degraded', degraded.length);
    _set('oh-kpi-exhausted', exhausted.length);

    if (total > 0) {
      var parts = [];
      if (dlq.length) parts.push(dlq.length + ' dead-letter');
      if (degraded.length) parts.push(degraded.length + ' degraded');
      if (exhausted.length) parts.push(exhausted.length + ' exhausted');
      _set('oh-banner-text', parts.join(', ') + ' — action required.');
      _show('oh-banner-error'); _hide('oh-banner-ok');
    } else {
      _hide('oh-banner-error'); _show('oh-banner-ok');
    }

    var all = dlq.concat(degraded).concat(exhausted);
    var w = $id('oh-table-wrap');
    if (w) {
      w.innerHTML = _renderTable(all);
      _attachActionListeners(w);
    }
  }

  function _renderStall(status) {
    if (status.writer_stalled) {
      _set('oh-stall-op', status.writer_stalled_op_id || '?');
      _set('oh-stall-age', status.writer_stalled_age_s != null ? Math.round(status.writer_stalled_age_s) : '?');
      _show('oh-stall-banner');
    } else {
      _hide('oh-stall-banner');
    }
  }

  function _catLabel(cat) {
    return { dead_letter: 'Dead-letter', degraded_manifest: 'Degraded', exhausted_obligation: 'Exhausted' }[cat] || cat;
  }

  function _renderTable(entries) {
    if (!entries || entries.length === 0) {
      return '<p style="opacity:0.5;font-size:13px">No failed operations.</p>';
    }
    var style = 'width:100%;border-collapse:collapse;font-size:13px';
    var thStyle = 'padding:8px 10px;text-align:left;opacity:0.6;border-bottom:1px solid rgba(255,255,255,0.1)';
    var tdStyle = 'padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.06)';
    var rows = entries.map(function (e) {
      var opId = _esc(e.operation_id || '?');
      var cat = _esc(_catLabel(e.category));
      var profile = _esc(e.profile_id || '-');
      var attempts = e.attempts !== undefined ? e.attempts : '-';
      var err = e.error ? ('<span style="color:#f66;font-size:12px">' + _esc(e.error.slice(0, 60)) + '</span>') : '';
      var btns =
        '<button class="oh-action" data-op="' + opId + '" data-act="retry" style="margin-right:4px;padding:3px 8px;border-radius:4px;border:1px solid;cursor:pointer;font-size:12px">Retry</button>' +
        '<button class="oh-action" data-op="' + opId + '" data-act="force_reconcile" style="margin-right:4px;padding:3px 8px;border-radius:4px;border:1px solid;cursor:pointer;font-size:12px">Reconcile</button>' +
        '<button class="oh-action" data-op="' + opId + '" data-act="cancel" style="padding:3px 8px;border-radius:4px;border:1px solid rgba(220,53,69,0.5);color:#f66;cursor:pointer;font-size:12px">Cancel</button>';
      return '<tr><td style="' + tdStyle + '"><code style="font-size:11px">' + opId + '</code></td>' +
        '<td style="' + tdStyle + '">' + cat + '</td>' +
        '<td style="' + tdStyle + '">' + profile + '</td>' +
        '<td style="' + tdStyle + '">' + attempts + '</td>' +
        '<td style="' + tdStyle + '">' + err + '</td>' +
        '<td style="' + tdStyle + '">' + btns + '</td></tr>';
    }).join('');

    return '<table style="' + style + '"><thead><tr>' +
      ['Operation ID', 'Category', 'Profile', 'Attempts', 'Error', 'Actions'].map(function (h) {
        return '<th style="' + thStyle + '">' + h + '</th>';
      }).join('') +
      '</tr></thead><tbody>' + rows + '</tbody></table>';
  }

  /* -----------------------------------------------------------------------
   * Action buttons
   * --------------------------------------------------------------------- */
  function _attachActionListeners(container) {
    var btns = container.querySelectorAll('.oh-action');
    for (var i = 0; i < btns.length; i++) {
      (function (btn) {
        btn.addEventListener('click', function () {
          _resolveOp(btn.getAttribute('data-op'), btn.getAttribute('data-act'));
        });
      })(btns[i]);
    }

    var refreshBtn = $id('oh-refresh-btn');
    if (refreshBtn && !refreshBtn._wired) {
      refreshBtn._wired = true;
      refreshBtn.addEventListener('click', _load);
    }
  }

  function _resolveOp(opId, action) {
    if (!window.confirm('Action: ' + action + '\nOperation: ' + opId + '\n\nProceed?')) return;
    _post('/operations/' + encodeURIComponent(opId) + '/resolve', { action: action })
      .then(function (res) {
        if (res.status === 403) { _toast('error', 'Permission denied.'); return; }
        var d = res.data || {};
        if (d.success) { _toast('ok', 'Resolved: ' + opId); setTimeout(_load, 800); }
        else { _toast('error', 'Not resolved: ' + (d.reason || d.error || 'unknown')); }
      });
  }

  function _toast(kind, msg) {
    var w = $id('oh-table-wrap');
    if (!w) return;
    var el = document.createElement('div');
    el.style.cssText = 'padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:13px;' +
      (kind === 'ok' ? 'background:rgba(25,135,84,0.2);' : 'background:rgba(220,53,69,0.2);');
    el.textContent = msg;
    w.insertBefore(el, w.firstChild);
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 5000);
  }

  /* -----------------------------------------------------------------------
   * Injection observer — waits for od-operations.js to render #od-ops-tabs
   * --------------------------------------------------------------------- */
  function _waitForTabBar() {
    /* Fast path: already rendered */
    if ($id('od-ops-tabs')) { _injectTab(); return; }

    if (typeof MutationObserver === 'undefined') {
      /* Fallback: polling (od-operations.js renders lazily on tab click) */
      var t = setInterval(function () {
        if ($id('od-ops-tabs')) { clearInterval(t); _injectTab(); }
      }, 200);
      /* Stop after 5 min — well past any realistic page session */
      setTimeout(function () { clearInterval(t); }, 300000);
      return;
    }

    /* Observe the whole document body so we catch od-operations.js's
     * deferred render regardless of which element it targets. */
    var obs = new MutationObserver(function () {
      if ($id('od-ops-tabs') && !_injected) {
        _injectTab();
        if (_injected) obs.disconnect();
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });

    /* Safety: stop observing after 5 min */
    setTimeout(function () { obs.disconnect(); }, 300000);
  }

  /* -----------------------------------------------------------------------
   * Init
   * --------------------------------------------------------------------- */
  function init() {
    _waitForTabBar();
  }

  window.OpsHealth = { load: _load, init: init };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
