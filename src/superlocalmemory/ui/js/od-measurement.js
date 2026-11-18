/* od-measurement.js - "Help SLM measure what works" (4.2.30 D-28).
 *
 * Two surfaces over GET/POST /api/outcomes/measurement:
 *   - a Settings card: the notice in plain words, whether measuring is on, and one
 *     switch. Acts at once, no Save. Off stops the random withholding and swapping;
 *     receipts (ids and numbers, this computer only) stay unless turned off with
 *     `slm measurement off --receipts`.
 *   - a banner on the dashboard home, shown while the notice is pending: Learn more
 *     (opens the Settings card), Turn off, OK (read it; keep measuring).
 * Nothing from the server is parsed as HTML. A change that is refused says why in
 * words; the banner stays until a change or OK actually succeeds.
 *
 * Exposes window.SLMMeasurement { renderInto(mount), mountBanner(host), reload() }.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar - AGPL-3.0-or-later
 */
(function () {
  'use strict';

  var API = '/api/outcomes/measurement';
  var CARD_ID = 'settings-measurement';
  var state = null;
  var cardMount = null;
  var bannerHost = null;
  var message = '';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function btn(label, cls, onClick) {
    var b = el('button', 'btn sm ' + (cls || 'ghost'), label);
    b.type = 'button';
    b.addEventListener('click', function (e) { e.preventDefault(); onClick(b); });
    return b;
  }
  function toast(m) { if (typeof window.showToast === 'function') window.showToast(m); }

  function read() {
    return fetch(API, { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function post(path, body) {
    return fetch(API + path, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; })
        .then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
    }).catch(function () {
      return { ok: false, status: 0, data: {} };
    });
  }
  function why(r) {
    if (r.status === 403) return 'Only the owner or an admin can change this. Ask them, or run slm measurement off.';
    var d = r.data && r.data.detail;
    return typeof d === 'string' && d ? d : 'That did not work. Try again in a moment.';
  }

  /* ── Settings card ────────────────────────────────────────────── */
  function paragraphs(text) {
    var box = el('div', 'od-measure-notice');
    String(text || '').split('\n\n').forEach(function (p) {
      if (!p.trim()) return;
      var para = el('p', 'dash-card-meta', p.trim());
      para.style.whiteSpace = 'pre-line';  // the last paragraph is a few short lines
      box.appendChild(para);
    });
    return box;
  }

  function buildSwitch(measuring) {
    var wrap = el('div', 'od-measure-switch');
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '10px';
    wrap.style.margin = '10px 0';
    var sw = el('button', 'btn sm ' + (measuring ? 'primary' : 'ghost'), measuring ? 'On' : 'Off');
    sw.type = 'button';
    sw.id = 'od-measure-switch';
    sw.setAttribute('role', 'switch');
    sw.setAttribute('aria-checked', measuring ? 'true' : 'false');
    sw.setAttribute('aria-label', 'Help SLM measure what works');
    sw.addEventListener('click', function (e) { e.preventDefault(); change(!measuring, sw); });
    wrap.appendChild(sw);
    wrap.appendChild(el('span', null, measuring
      ? 'SLM is measuring whether memory helps.'
      : 'SLM is not withholding or swapping anything.'));
    return wrap;
  }

  function paintCard() {
    if (!cardMount) return;
    cardMount.textContent = '';
    if (!state) {
      cardMount.appendChild(el('p', 'dash-card-meta',
        'Could not read the measurement settings. Is SLM running? Try again in a moment, or run slm measurement status.'));
      return;
    }
    cardMount.appendChild(paragraphs(state.notice));
    if (state.kill_switch) {
      cardMount.appendChild(el('p', 'dash-card-meta',
        'The kill switch is on, so every 4.2.30 measure is off whatever is set here (slm flags kill off releases it).'));
    }
    cardMount.appendChild(buildSwitch(!!state.measuring));
    var note = el('p', 'dash-card-meta', 'Turning this off stops the withholding and swapping. ' +
      'Receipts (ids and numbers, kept on this computer) are still written. To stop them too, run slm measurement off --receipts.');
    cardMount.appendChild(note);
    var list = el('div', 'truth-checks');
    (state.flags || []).forEach(function (f) {
      var r = el('div', 'list-row');
      r.appendChild(el('span', 'truth-check-text', f.label + ': ' + f.what));
      r.appendChild(el('span', 'truth-check-status ' + (f.state === 'on' ? 'tone-ok' : ''), f.state === 'on' ? 'On' : 'Off'));
      list.appendChild(r);
    });
    cardMount.appendChild(list);
    var live = el('p', 'dash-card-meta', message);
    live.id = 'od-measure-message';
    live.setAttribute('role', 'status');
    live.setAttribute('aria-live', 'polite');
    cardMount.appendChild(live);
  }

  function change(on, button) {
    if (button) button.disabled = true;
    return post('', { on: on }).then(function (r) {
      if (r.ok && r.data && r.data.flags) {
        state = r.data;
        message = on ? 'Measuring is on.' : 'Measuring is off. It takes effect at the next session start.';
        toast(message);
      } else {
        message = why(r);
        toast(message);
      }
      paintCard();
      paintBanner();
      return r.ok;
    });
  }

  function renderInto(mount) {
    cardMount = mount;
    mount.setAttribute('data-od-measure', 'card');
    mount.textContent = '';
    mount.appendChild(el('p', 'dash-card-meta', 'Loading…'));
    return read().then(function (s) { state = s; paintCard(); });
  }

  /* ── Dashboard banner ─────────────────────────────────────────── */
  function goToCard() {
    var link = document.querySelector('.nav-link[data-tab="settings-pane"]');
    if (link) link.click();
    var started = Date.now();
    (function look() {
      var target = document.getElementById(CARD_ID);
      if (target) {
        if (typeof target.scrollIntoView === 'function') target.scrollIntoView({ block: 'start' });
        var head = target.querySelector('h3') || target;
        if (!head.hasAttribute('tabindex')) head.setAttribute('tabindex', '-1');
        head.focus();
        return;
      }
      if (Date.now() - started < 2000) setTimeout(look, 100);
    }());
  }

  function dismiss(button, alsoTurnOff) {
    button.disabled = true;
    var step = alsoTurnOff ? change(false, null) : Promise.resolve(true);
    return step.then(function (ok) {
      if (!ok) { button.disabled = false; paintBannerNote(message); return null; }
      return post('/acknowledge').then(function (r) {
        if (r.ok && r.data && r.data.flags) { state = r.data; paintBanner(); paintCard(); }
        else { message = why(r); button.disabled = false; paintBannerNote(message); }
      });
    });
  }

  function paintBannerNote(text) {
    var n = bannerHost && bannerHost.querySelector('.od-measure-note');
    if (n) n.textContent = text;
  }

  function paintBanner() {
    if (!bannerHost) return;
    bannerHost.textContent = '';
    if (!state || !state.pending) return;
    var card = el('div', 'card card-pad od-measure-banner');
    card.setAttribute('role', 'region');
    card.setAttribute('aria-label', 'What SuperLocalMemory measures');
    card.appendChild(el('p', null, state.measuring
      ? 'SLM now measures whether memory helps your agents. In a small share of session starts it leaves one memory out at random, ' +
        'and swaps two neighbouring results in a few recalls. Everything is logged as ids and numbers on this computer only.'
      : 'SLM can measure whether memory helps your agents, but this is turned off here, so nothing is withheld or swapped.'));
    var actions = el('div', 'od-star-actions');
    actions.appendChild(btn('Learn more', 'ghost', goToCard));
    if (state.measuring) actions.appendChild(btn('Turn off', 'ghost', function (b) { dismiss(b, true); }));
    actions.appendChild(btn('OK', 'primary', function (b) { dismiss(b, false); }));
    card.appendChild(actions);
    var note = el('p', 'dash-card-meta od-measure-note', '');
    note.setAttribute('role', 'status');
    card.appendChild(note);
    bannerHost.appendChild(card);
  }

  function mountBanner(host) {
    bannerHost = host || null;
    if (!bannerHost) return Promise.resolve();
    return read().then(function (s) { state = s; paintBanner(); });
  }

  function reload() {
    return read().then(function (s) { state = s; paintCard(); paintBanner(); });
  }

  window.SLMMeasurement = { renderInto: renderInto, mountBanner: mountBanner, reload: reload, CARD_ID: CARD_ID };

  function init() { mountBanner(document.getElementById('od-measurement-banner')); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
