// od-auth-gate.js — product-wide login gate (v3.8.0)
//
// Personal mode (require_login = false): no gate — the loopback machine owner
// is trusted, exactly as before. Backward compatible: single-operator users
// never see a login screen.
//
// Company/Enterprise mode (require_login = true):
//   * First run (user_count == 0): a "Create admin" screen. The operator picks
//     the username AND password — SLM ships NO default credentials and never
//     writes a plaintext password to disk (the backend stores an scrypt hash).
//   * Thereafter: a "Sign in" screen unless the caller already holds a valid
//     HttpOnly session cookie (set by POST /api/rbac/login).
//
// The gate is UX + attribution only. The daemon independently enforces auth on
// every sensitive read/mutation server-side, so this screen fails OPEN on a
// transient status error (never locks the operator out of a working daemon).
//
// CSP-safe: no inline handlers, no on*= attributes — every control is wired via
// addEventListener. All server-supplied text is inserted via textContent.
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

/* global window, document, fetch */
(function () {
  'use strict';

  var OVERLAY_ID = 'slm-auth-gate';

  // ─── DOM helpers (CSP-safe, XSS-safe) ───────────────────────────────────────
  function elmt(tag, opts) {
    opts = opts || {};
    var e = document.createElement(tag);
    if (opts.cls) e.className = opts.cls;
    if (opts.id) e.id = opts.id;
    if (opts.text != null) e.textContent = String(opts.text);
    if (opts.type) e.type = opts.type;
    if (opts.placeholder) e.placeholder = opts.placeholder;
    if (opts.autocomplete) e.setAttribute('autocomplete', opts.autocomplete);
    if (opts.style) e.style.cssText = opts.style;
    if (opts.attrs) {
      Object.keys(opts.attrs).forEach(function (k) { e.setAttribute(k, opts.attrs[k]); });
    }
    return e;
  }

  function overlay() {
    var d = document.getElementById(OVERLAY_ID);
    if (d) return d;
    d = elmt('div', {
      id: OVERLAY_ID,
      attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Sign in' },
      style: [
        'position:fixed', 'inset:0', 'z-index:100000',
        'display:flex', 'align-items:center', 'justify-content:center',
        'background:#0b0d10', 'background:var(--bg,#0b0d10)',
        'padding:24px',
      ].join(';'),
    });
    document.body.appendChild(d);
    return d;
  }

  function removeOverlay() {
    var d = document.getElementById(OVERLAY_ID);
    if (d && d.parentNode) d.parentNode.removeChild(d);
  }

  function card() {
    var c = elmt('div', {
      style: [
        'width:100%', 'max-width:380px',
        'background:#16181d', 'background:var(--card,#16181d)',
        'border:1px solid rgba(255,255,255,0.08)', 'border-radius:14px',
        'padding:28px', 'box-shadow:0 12px 40px rgba(0,0,0,0.45)',
        'color:#e8eaed', 'color:var(--text,#e8eaed)',
        'font-family:Inter,system-ui,sans-serif',
      ].join(';'),
    });
    var brand = elmt('div', { style: 'display:flex;align-items:center;gap:10px;margin-bottom:18px' });
    var logo = elmt('img', { attrs: { src: 'static/assets/slm-icon.svg', alt: '', width: '28', height: '28' } });
    var name = elmt('div');
    name.appendChild(elmt('div', { text: 'SuperLocalMemory', style: 'font-weight:650;font-size:15px' }));
    name.appendChild(elmt('div', { text: 'by Qualixar', style: 'font-size:12px;opacity:0.6' }));
    brand.appendChild(logo);
    brand.appendChild(name);
    c.appendChild(brand);
    return c;
  }

  function field(labelText, inputOpts) {
    var wrap = elmt('div', { style: 'margin-bottom:12px' });
    var lbl = elmt('label', { text: labelText, style: 'display:block;font-size:12px;opacity:0.75;margin-bottom:4px' });
    var input = elmt('input', inputOpts);
    input.style.cssText = [
      'width:100%', 'box-sizing:border-box', 'padding:9px 11px',
      'background:rgba(255,255,255,0.05)', 'border:1px solid rgba(255,255,255,0.12)',
      'border-radius:8px', 'color:inherit', 'font-size:14px',
    ].join(';');
    if (inputOpts && inputOpts.id) lbl.setAttribute('for', inputOpts.id);
    wrap.appendChild(lbl);
    wrap.appendChild(input);
    return wrap;
  }

  function primaryBtn(text) {
    var b = elmt('button', { text: text, type: 'button' });
    b.style.cssText = [
      'width:100%', 'padding:10px', 'margin-top:6px', 'border:none',
      'border-radius:8px', 'background:#6c5ce7', 'background:var(--violet,#6c5ce7)',
      'color:#fff', 'font-size:14px', 'font-weight:600', 'cursor:pointer',
    ].join(';');
    return b;
  }

  function msgLine() {
    return elmt('p', { style: 'font-size:12.5px;margin:10px 0 0;min-height:16px;color:#ff7b7b' });
  }

  // ─── API (uses the globally patched fetch: X-Install-Token on POST) ──────────
  function api(url, method, body) {
    var init = { method: method || 'GET', credentials: 'same-origin' };
    if (body) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (d) {
        return { ok: r.ok, status: r.status, data: d };
      });
    });
  }

  // ─── Screens ────────────────────────────────────────────────────────────────
  function showSpinner() {
    var o = overlay();
    o.textContent = '';
    var c = card();
    c.appendChild(elmt('p', { text: 'Checking access…', style: 'opacity:0.7;font-size:13px;margin:0' }));
    o.appendChild(c);
  }

  function showLogin(prefillMsg) {
    var o = overlay();
    o.textContent = '';
    var c = card();
    c.appendChild(elmt('h2', { text: 'Sign in', style: 'font-size:18px;margin:0 0 16px' }));
    var user = { id: 'slm-gate-user', placeholder: 'Username', autocomplete: 'username' };
    var pass = { id: 'slm-gate-pass', type: 'password', placeholder: 'Password', autocomplete: 'current-password' };
    c.appendChild(field('Username', user));
    c.appendChild(field('Password', pass));
    var btn = primaryBtn('Sign in');
    var msg = msgLine();
    if (prefillMsg) { msg.style.color = '#7bd88f'; msg.textContent = prefillMsg; }
    c.appendChild(btn);
    c.appendChild(msg);
    o.appendChild(c);

    function submit() {
      var u = (document.getElementById('slm-gate-user') || {}).value || '';
      var p = (document.getElementById('slm-gate-pass') || {}).value || '';
      if (!u || !p) { msg.style.color = '#ff7b7b'; msg.textContent = 'Enter your username and password.'; return; }
      btn.disabled = true;
      api('/api/rbac/login', 'POST', { username: u, password: p }).then(function (r) {
        btn.disabled = false;
        if (r.ok) { window.location.reload(); return; }
        msg.style.color = '#ff7b7b';
        msg.textContent = (r.data && r.data.detail) || 'Sign in failed.';
      }).catch(function () {
        btn.disabled = false;
        msg.style.color = '#ff7b7b';
        msg.textContent = 'Could not reach the daemon.';
      });
    }
    btn.addEventListener('click', submit);
    c.addEventListener('keydown', function (e) { if (e.key === 'Enter') submit(); });
  }

  function showCreateAdmin() {
    var o = overlay();
    o.textContent = '';
    var c = card();
    c.appendChild(elmt('h2', { text: 'Create the first admin', style: 'font-size:18px;margin:0 0 6px' }));
    c.appendChild(elmt('p', {
      text: 'This workspace requires sign-in. Choose the admin username and a '
        + 'password you control — SuperLocalMemory ships no default login.',
      style: 'font-size:12.5px;opacity:0.7;margin:0 0 16px',
    }));
    c.appendChild(field('Username', { id: 'slm-gate-nu', placeholder: 'e.g. admin', autocomplete: 'username' }));
    c.appendChild(field('Display name (optional)', { id: 'slm-gate-nn', placeholder: 'Full name', autocomplete: 'name' }));
    c.appendChild(field('Password (min 8 characters)', { id: 'slm-gate-np', type: 'password', placeholder: 'Password', autocomplete: 'new-password' }));
    c.appendChild(field('Confirm password', { id: 'slm-gate-nc', type: 'password', placeholder: 'Re-enter password', autocomplete: 'new-password' }));
    var btn = primaryBtn('Create admin & sign in');
    var msg = msgLine();
    c.appendChild(btn);
    c.appendChild(msg);
    o.appendChild(c);

    function submit() {
      var u = (document.getElementById('slm-gate-nu') || {}).value || '';
      var n = (document.getElementById('slm-gate-nn') || {}).value || '';
      var p = (document.getElementById('slm-gate-np') || {}).value || '';
      var cc = (document.getElementById('slm-gate-nc') || {}).value || '';
      if (!u) { msg.textContent = 'Choose a username.'; return; }
      if (p.length < 8) { msg.textContent = 'Password must be at least 8 characters.'; return; }
      if (p !== cc) { msg.textContent = 'Passwords do not match.'; return; }
      btn.disabled = true;
      api('/api/rbac/users', 'POST', { username: u, password: p, display_name: n, role: 'admin' })
        .then(function (r) {
          if (!r.ok) {
            btn.disabled = false;
            msg.style.color = '#ff7b7b';
            msg.textContent = (r.data && r.data.detail) || 'Could not create the admin account.';
            return;
          }
          // Auto sign-in so the operator lands straight in the dashboard.
          return api('/api/rbac/login', 'POST', { username: u, password: p }).then(function () {
            window.location.reload();
          });
        }).catch(function () {
          btn.disabled = false;
          msg.style.color = '#ff7b7b';
          msg.textContent = 'Could not reach the daemon.';
        });
    }
    btn.addEventListener('click', submit);
    c.addEventListener('keydown', function (e) { if (e.key === 'Enter') submit(); });
  }

  // ─── Decision flow ──────────────────────────────────────────────────────────
  function decide() {
    api('/api/rbac/status', 'GET').then(function (r) {
      var s = (r && r.data) || {};
      if (!r.ok || !s.require_login) { removeOverlay(); return; }   // Personal mode
      if ((s.user_count || 0) === 0) { showCreateAdmin(); return; } // First run
      api('/api/rbac/whoami', 'GET').then(function (w) {
        var who = (w && w.data) || {};
        if (w.ok && who.kind === 'user') { removeOverlay(); return; } // Already signed in
        showLogin();
      }).catch(function () { removeOverlay(); });                    // Fail open
    }).catch(function () { removeOverlay(); });                      // Fail open
  }

  // Paint the overlay immediately (this script is at end of <body>, so body
  // exists) to avoid flashing dashboard content to unauthenticated users in
  // company mode, then resolve the real state.
  showSpinner();
  decide();
}());
