// od-components.js — System Health "What's Missing" report (v3.8.2 UX-6)
//
// window.odRenderComponents(el) renders the component-registry snapshot from
//   GET  /api/v3/components          → {components:[...], summary:{...}}
// into `el`, showing per-component status, a copy-paste fix command for the
// items SLM can't repair on its own, and a "Retry now" button that triggers
//   POST /api/v3/components/heal     (install token auto-attached by core.js)
// The daemon auto-heals on start; this panel is transparency + a manual
// fallback for the non-technical user.
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

/* global window, document, fetch, navigator, setTimeout */
(function () {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Severity/glyph/colour for one component, honouring its category so an
  // absent optional backend is not shown as an error.
  function view(c) {
    var st = String(c.status || '').toLowerCase();
    if (st === 'ok') return { g: '✓', col: 'var(--ok)', tip: 'ready' };
    if (st === 'retrying') return { g: '⟳', col: 'var(--accent, #4a9)', tip: 'repairing' };
    if (st === 'degraded') return { g: '⚠', col: 'var(--warn)', tip: 'degraded' };
    // missing — severity depends on how much SLM needs it.
    if (c.category === 'required') return { g: '✗', col: 'var(--danger)', tip: 'missing' };
    if (c.category === 'recommended') return { g: '⚠', col: 'var(--warn)', tip: 'missing' };
    return { g: '○', col: 'var(--fg-2)', tip: 'optional' };
  }

  function rowHtml(c) {
    var v = view(c);
    var fix = '';
    if (c.status !== 'ok' && c.fix_cmd) {
      if (c.auto_fixable) {
        fix = '<div class="muted" style="margin-top:2px;font-size:12px">'
            + 'SLM repairs this automatically.</div>';
      } else {
        fix = '<div style="margin-top:4px;display:flex;gap:6px;align-items:center">'
            + '<code style="font-size:12px;background:var(--bg-2,#0002);'
            + 'padding:2px 6px;border-radius:4px">' + esc(c.fix_cmd) + '</code>'
            + '<button class="od-copy-fix" data-cmd="' + esc(c.fix_cmd) + '" '
            + 'style="font-size:11px;padding:1px 6px;cursor:pointer">Copy</button>'
            + '</div>';
      }
    }
    return '<div style="padding:8px 0;border-bottom:1px solid var(--border,#8881)">'
      + '<div style="display:flex;gap:8px;align-items:baseline">'
      + '<span style="color:' + v.col + ';font-weight:700;width:16px;text-align:center">'
      + v.g + '</span>'
      + '<div style="flex:1">'
      + '<span style="font-weight:600">' + esc(c.label) + '</span>'
      + ' <span class="muted" style="font-size:12px">' + esc(c.detail || v.tip) + '</span>'
      + fix
      + '</div></div></div>';
  }

  function render(el, data) {
    var comps = (data && data.components) || [];
    var s = (data && data.summary) || {};
    var healthy = !!s.healthy;
    var hasAutoFix = (s.auto_fixable_missing || 0) > 0;

    var head = '<div style="display:flex;justify-content:space-between;'
      + 'align-items:center;margin-bottom:6px">'
      + '<h3 style="margin:0">System Health — Components</h3>'
      + '<span style="padding:2px 10px;border-radius:12px;font-size:12px;'
      + 'background:' + (healthy ? 'var(--ok)' : 'var(--warn)') + ';color:#000">'
      + (healthy ? 'All good' : 'Needs attention') + '</span></div>';

    var summary = '<div class="muted" style="font-size:12px;margin-bottom:8px">'
      + (s.ok || 0) + ' ready · ' + (s.missing || 0) + ' missing · '
      + (s.degraded || 0) + ' degraded'
      + (s.retrying ? ' · ' + s.retrying + ' repairing' : '') + '</div>';

    var actions = '<div style="margin-top:10px;display:flex;gap:8px">'
      + '<button id="od-comp-recheck" style="padding:4px 12px;cursor:pointer">Recheck</button>'
      + (hasAutoFix
          ? '<button id="od-comp-retry" style="padding:4px 12px;cursor:pointer;'
            + 'background:var(--accent,#4a9);color:#000;border:none;border-radius:4px">'
            + 'Retry now</button>'
          : '')
      + '</div>'
      + '<div class="muted" style="font-size:11px;margin-top:6px">'
      + 'SLM auto-repairs fixable items on start. Copy a command for anything '
      + 'it can’t install for you.</div>';

    el.innerHTML = head + summary
      + '<div>' + comps.map(rowHtml).join('') + '</div>' + actions;

    // Copy buttons.
    el.querySelectorAll('.od-copy-fix').forEach(function (b) {
      b.addEventListener('click', function () {
        var cmd = b.getAttribute('data-cmd') || '';
        if (navigator.clipboard) navigator.clipboard.writeText(cmd);
        var old = b.textContent; b.textContent = 'Copied';
        setTimeout(function () { b.textContent = old; }, 1200);
      });
    });
    var recheck = el.querySelector('#od-comp-recheck');
    if (recheck) recheck.addEventListener('click', function () { load(el); });
    var retry = el.querySelector('#od-comp-retry');
    if (retry) retry.addEventListener('click', function () { doRetry(el, retry); });
  }

  function load(el) {
    if (!el) return;
    el.innerHTML = '<p class="muted" style="padding:8px 0">Checking components…</p>';
    fetch('/api/v3/components')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data) {
          el.innerHTML = '<p class="muted" style="padding:8px 0">'
            + 'Component health unavailable (is the daemon running?).</p>';
          return;
        }
        render(el, data);
      })
      .catch(function () {
        el.innerHTML = '<p class="muted" style="padding:8px 0">'
          + 'Could not load component health.</p>';
      });
  }

  // Trigger a background heal, then poll a few times so the panel reflects
  // the retrying → ok transition without the user refreshing.
  function doRetry(el, btn) {
    btn.disabled = true; btn.textContent = 'Repairing…';
    fetch('/api/v3/components/heal', { method: 'POST', credentials: 'same-origin' })
      .then(function () {
        var tries = 0;
        (function poll() {
          tries += 1;
          load(el);
          if (tries < 4) setTimeout(poll, 2500);
        })();
      })
      .catch(function () { load(el); });
  }

  window.odRenderComponents = load;
})();
