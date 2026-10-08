// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// od-apps-ui.js — small DOM toolkit shared by the Connected apps pane.
//
// Exposes: window.odAppsUi
//
// CSP-safe by construction: every node is built with createElement /
// createElementNS and filled with textContent or text nodes. Nothing here uses
// innerHTML, inline handlers or inline <style>. App names and hosts come from
// third-party OAuth registrations, so they must only ever travel through
// textContent / setAttribute (never markup).
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var uid = 0;

  // ---------------------------------------------------------------------------
  // Element factory
  //   h('div', { className: 'x', text: 'hi', 'aria-label': 'y', hidden: true }, [child, 'text'])
  // true -> empty attribute, false/null/undefined -> attribute omitted.
  // ---------------------------------------------------------------------------
  function h(tag, props, kids) {
    var el = document.createElement(tag);
    var p = props || {};
    Object.keys(p).forEach(function (key) {
      var value = p[key];
      if (value === false || value === null || value === undefined) return;
      if (key === 'className') el.className = String(value);
      else if (key === 'text') el.textContent = String(value);
      else el.setAttribute(key, value === true ? '' : String(value));
    });
    (kids || []).forEach(function (kid) {
      if (kid === null || kid === undefined || kid === false) return;
      el.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    });
    return el;
  }

  function nextId(prefix) { uid += 1; return prefix + '-' + uid; }

  // ---------------------------------------------------------------------------
  // Icons (static path data only; lucide-style 24x24 strokes)
  // ---------------------------------------------------------------------------
  var ICONS = {
    monitor: [['rect', { x: 2, y: 3, width: 20, height: 14, rx: 2 }], ['path', { d: 'M8 21h8M12 17v4' }]],
    lock: [['rect', { x: 4, y: 10, width: 16, height: 10, rx: 2 }], ['path', { d: 'M8 10V7a4 4 0 0 1 8 0v3' }]],
    eye: [['path', { d: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z' }], ['circle', { cx: 12, cy: 12, r: 3 }]],
    approve: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M8 12.5l3 3 5-6' }]],
    check: [['path', { d: 'M5 12.5l4.5 4.5L19 7' }]],
    copy: [['rect', { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ['path', { d: 'M5 15V6a2 2 0 0 1 2-2h9' }]],
    refresh: [['path', { d: 'M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5' }]],
    close: [['path', { d: 'M6 6l12 12M18 6L6 18' }]],
    chevron: [['path', { d: 'M9 6l6 6-6 6' }]],
    alert: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M12 7.5v5M12 16h.01' }]],
    link: [['path', { d: 'M9 15l6-6M10 6l1-1a4 4 0 0 1 6 6l-1 1M14 18l-1 1a4 4 0 0 1-6-6l1-1' }]],
    arrow: [['path', { d: 'M5 12h14M13 6l6 6-6 6' }]]
  };

  function icon(name, size) {
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', String(size || 18));
    svg.setAttribute('height', String(size || 18));
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.8');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    (ICONS[name] || []).forEach(function (part) {
      var shape = document.createElementNS(SVG_NS, part[0]);
      Object.keys(part[1]).forEach(function (attr) { shape.setAttribute(attr, String(part[1][attr])); });
      svg.appendChild(shape);
    });
    return svg;
  }

  // ---------------------------------------------------------------------------
  // Text helpers
  // ---------------------------------------------------------------------------
  // Monogram from a name: the initials of its first two words ("Claude Code" ->
  // "CC"), or its first two letters for a single word ("Composio" -> "Co").
  // Letters and digits only; a name made only of symbols falls back to its
  // first character so the avatar is never empty.
  function monogram(name) {
    var raw = String(name || '').trim();
    if (!raw) return '?';
    var words = raw.match(/[\p{L}\p{N}]+/gu) || [];
    if (words.length >= 2) return (Array.from(words[0])[0] + Array.from(words[1])[0]).toUpperCase();
    if (words.length === 1) {
      var letters = Array.from(words[0]);
      return letters[0].toUpperCase() + (letters[1] || '').toLowerCase();
    }
    return Array.from(raw)[0];
  }

  // Plain-language relative time. Returns null for a missing or invalid stamp
  // so callers can omit the phrase instead of printing "Invalid Date".
  function relativeTime(ms, now) {
    if (typeof ms !== 'number' || !isFinite(ms) || ms <= 0) return null;
    var diff = Math.max(0, (now === undefined || now === null ? Date.now() : now) - ms);
    var seconds = Math.floor(diff / 1000);
    if (seconds < 45) return 'just now';
    var minutes = Math.floor(seconds / 60);
    if (minutes < 1) return '1 minute ago';
    if (minutes < 60) return minutes === 1 ? '1 minute ago' : minutes + ' minutes ago';
    var hours = Math.floor(minutes / 60);
    if (hours < 24) return hours === 1 ? '1 hour ago' : hours + ' hours ago';
    var days = Math.floor(hours / 24);
    if (days === 1) return 'yesterday';
    if (days < 7) return days + ' days ago';
    if (days < 30) { var weeks = Math.floor(days / 7); return weeks === 1 ? '1 week ago' : weeks + ' weeks ago'; }
    try {
      return 'on ' + new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    } catch (_) { return null; }
  }

  function fullDate(ms) {
    if (typeof ms !== 'number' || !isFinite(ms) || ms <= 0) return '';
    try { return new Date(ms).toLocaleString(); } catch (_) { return ''; }
  }

  // ---------------------------------------------------------------------------
  // Network: same helper chain as od-connections.js. window.slmFetch is the
  // dashboard's timeout wrapper; core.js also attaches the install-token header
  // to every same-origin POST, so mutations need no extra credentials here.
  // Resolves { ok, status, data } and only rejects on a transport failure.
  // ---------------------------------------------------------------------------
  function request(path, init) {
    var fetcher = typeof window.slmFetch === 'function' ? window.slmFetch : window.fetch;
    var options = Object.assign({ credentials: 'same-origin' }, init || {});
    return fetcher(path, options).then(function (response) {
      var parsed = typeof response.json === 'function'
        ? Promise.resolve().then(function () { return response.json(); }).catch(function () { return null; })
        : Promise.resolve(null);
      return parsed.then(function (data) { return { ok: !!response.ok, status: response.status, data: data }; });
    });
  }

  // ---------------------------------------------------------------------------
  // Toast: one polite live region, one short-lived message per call.
  // ---------------------------------------------------------------------------
  function toast(message) {
    var region = document.getElementById('apps-toast-region');
    if (!region) {
      region = h('div', { id: 'apps-toast-region', className: 'apps-toast-region', role: 'status', 'aria-live': 'polite' });
      document.body.appendChild(region);
    }
    var item = h('div', { className: 'apps-toast' }, [icon('check', 16), h('span', { text: message })]);
    region.appendChild(item);
    window.setTimeout(function () { if (item.parentNode) item.parentNode.removeChild(item); }, 3600);
    return item;
  }

  // ---------------------------------------------------------------------------
  // Copy button: clipboard when available, text selection as the fallback.
  // ---------------------------------------------------------------------------
  function copyButton(label, getValue, input) {
    var text = h('span', { text: label });
    var button = h('button', { type: 'button', className: 'btn secondary sm' }, [icon('copy', 15), text]);
    var timer = null;
    button.addEventListener('click', function () {
      var value = getValue();
      function done() {
        text.textContent = 'Copied';
        if (timer !== null) window.clearTimeout(timer);
        timer = window.setTimeout(function () { text.textContent = label; timer = null; }, 2000);
      }
      var clipboard = window.navigator && window.navigator.clipboard;
      if (clipboard && clipboard.writeText) {
        clipboard.writeText(value).then(done).catch(function () { if (input) input.select(); });
      } else if (input) input.select();
    });
    return button;
  }

  // ---------------------------------------------------------------------------
  // Confirm dialog (accessible modal): role=dialog, aria-modal, labelled and
  // described, focus moves in, Tab is trapped, Escape and backdrop cancel,
  // focus returns to the opener. While onConfirm is pending the dialog cannot
  // be dismissed so a half-sent request is never abandoned silently.
  //   onConfirm(controller) -> Promise; the caller closes or reports an error.
  // ---------------------------------------------------------------------------
  function confirmDialog(options) {
    var opener = options.returnFocus || document.activeElement;
    var titleId = nextId('apps-dialog-title');
    var bodyId = nextId('apps-dialog-body');
    var busy = false;
    var closed = false;

    var title = h('h2', { id: titleId, className: 'apps-dialog-title', text: options.title });
    var body = h('p', { id: bodyId, className: 'apps-dialog-body', text: options.body });
    var error = h('p', { className: 'apps-dialog-error', role: 'alert', hidden: true });
    var cancel = h('button', { type: 'button', className: 'btn secondary', text: options.cancelLabel || 'Cancel' });
    var confirm = h('button', { type: 'button', className: 'btn danger solid', text: options.confirmLabel || 'Confirm' });
    var dialog = h('div', {
      className: 'apps-dialog', role: 'dialog', 'aria-modal': 'true',
      'aria-labelledby': titleId, 'aria-describedby': bodyId, tabindex: '-1'
    }, [title, body, error, h('div', { className: 'apps-dialog-actions' }, [cancel, confirm])]);
    var backdrop = h('div', { className: 'apps-dialog-backdrop' }, [dialog]);

    var controller = {
      element: dialog,
      setBusy: function (value, label) {
        busy = !!value;
        cancel.disabled = busy; confirm.disabled = busy;
        confirm.textContent = busy && label ? label : (options.confirmLabel || 'Confirm');
        dialog.setAttribute('aria-busy', busy ? 'true' : 'false');
      },
      setError: function (message) {
        error.textContent = message || '';
        error.hidden = !message;
      },
      close: function () {
        if (closed) return;
        closed = true;
        document.removeEventListener('keydown', onKey, true);
        if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
        var target = opener && opener.isConnected ? opener : (typeof options.fallbackFocus === 'function' ? options.fallbackFocus() : null);
        if (target && typeof target.focus === 'function') target.focus();
      }
    };

    function focusable() {
      return Array.from(dialog.querySelectorAll('button:not([disabled])'));
    }
    function onKey(event) {
      if (event.key === 'Escape') {
        if (busy) { event.preventDefault(); return; }
        event.preventDefault(); controller.close();
      } else if (event.key === 'Tab') {
        var items = focusable();
        if (!items.length) { event.preventDefault(); dialog.focus(); return; }
        var first = items[0]; var last = items[items.length - 1];
        var active = document.activeElement;
        if (!dialog.contains(active)) { event.preventDefault(); first.focus(); }
        else if (event.shiftKey && active === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && active === last) { event.preventDefault(); first.focus(); }
      }
    }

    cancel.addEventListener('click', function () { if (!busy) controller.close(); });
    backdrop.addEventListener('mousedown', function (event) { if (event.target === backdrop && !busy) controller.close(); });
    confirm.addEventListener('click', function () {
      if (busy) return;
      controller.setError('');
      options.onConfirm(controller);
    });

    document.body.appendChild(backdrop);
    document.addEventListener('keydown', onKey, true);
    // Safe default for a destructive prompt: start on Cancel.
    cancel.focus();
    return controller;
  }

  window.odAppsUi = {
    h: h,
    icon: icon,
    monogram: monogram,
    relativeTime: relativeTime,
    fullDate: fullDate,
    request: request,
    toast: toast,
    copyButton: copyButton,
    confirmDialog: confirmDialog,
    nextId: nextId
  };
}());
