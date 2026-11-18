/* od-killswitch.js - "Turn off everything new in 4.2.30" (D-23, blueprint 8.10.3 (k)).
 *
 * A Settings card over GET/POST /api/flags/kill and POST /api/flags/rollback-retirements:
 *   - what the switch does and what it does not undo (retirements), in plain words from the server;
 *   - the state, and one button that acts at once (no Save);
 *   - a second button, "Bring back memories that 4.2.30 retired": it counts first (a dry run),
 *     asks, then does it as one recorded decision that Recent decisions can undo.
 * Every server string is set with textContent, never parsed as HTML. A refused change says why in
 * words. The layout wraps, so it holds at 375 px.
 *
 * Exposes window.SLMKillSwitch { renderInto(mount) }.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar - AGPL-3.0-or-later
 */
(function () {
  'use strict';

  var KILL = '/api/flags/kill';
  var BACK = '/api/flags/rollback-retirements';
  var state = null;
  var mount = null;
  var message = '';
  var asking = null;   // a number while we wait for "Yes, bring them back"

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
  function memories(n) { return n === 1 ? '1 memory' : n + ' memories'; }

  function read() {
    return fetch(KILL, { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function post(path, body) {
    return fetch(path, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; })
        .then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
    }).catch(function () { return { ok: false, status: 0, data: {} }; });
  }
  function why(r) {
    if (r.status === 403) return 'Only the owner or an admin can change this. Ask them, or run slm flags kill on.';
    var d = r.data && r.data.detail;
    if (d && typeof d === 'object') d = d.message || d.code;
    return typeof d === 'string' && d ? d : 'That did not work. Try again in a moment.';
  }

  function row() {
    var r = el('div', 'od-kill-actions');
    r.style.display = 'flex';
    r.style.flexWrap = 'wrap';
    r.style.alignItems = 'center';
    r.style.gap = '10px';
    r.style.margin = '10px 0';
    return r;
  }

  function paint() {
    if (!mount) return;
    mount.textContent = '';
    if (!state || !state.kill_switch) {
      mount.appendChild(el('p', 'dash-card-meta',
        'Could not read the switch. Is SLM running? Try again in a moment, or run slm flags.'));
      return;
    }
    var on = !!state.kill_switch.on;
    mount.appendChild(el('p', 'dash-card-meta', state.does));
    mount.appendChild(el('p', 'dash-card-meta', state.does_not));
    var status = el('p', 'dash-card-lead', state.kill_switch.words);
    status.id = 'od-kill-state';
    status.setAttribute('role', 'status');
    mount.appendChild(status);
    var actions = row();
    var sw = btn(on ? 'Turn everything new back on' : 'Turn off everything new in 4.2.30',
      on ? 'ghost' : 'primary', function (b) { change(!on, b); });
    sw.id = 'od-kill-switch';
    sw.setAttribute('aria-pressed', on ? 'true' : 'false');
    actions.appendChild(sw);
    mount.appendChild(actions);
    paintBack();
    var live = el('p', 'dash-card-meta', message);
    live.id = 'od-kill-message';
    live.setAttribute('role', 'status');
    live.setAttribute('aria-live', 'polite');
    mount.appendChild(live);
  }

  function paintBack() {
    var n = Number(state.would_come_back) || 0;
    mount.appendChild(el('h4', null, 'Bring back memories that 4.2.30 retired'));
    mount.appendChild(el('p', 'dash-card-meta', n
      ? memories(n) + ' that 4.2.30 retired would come back. It is recorded, and Recent decisions can undo it.'
      : 'Nothing that 4.2.30 retired is waiting to come back.'));
    var actions = row();
    if (asking != null) {
      actions.appendChild(el('span', null, memories(asking) + ' would come back. Bring them back?'));
      actions.appendChild(btn('Yes, bring them back', 'primary', function (b) { bringBack(b); }));
      actions.appendChild(btn('Cancel', 'ghost', function () { asking = null; paint(); }));
    } else {
      var go = btn('Bring back memories that 4.2.30 retired', 'ghost', function (b) { count(b); });
      go.id = 'od-kill-bring-back';
      go.disabled = !n;
      actions.appendChild(go);
    }
    mount.appendChild(actions);
  }

  function change(on, button) {
    if (button) button.disabled = true;
    return post(KILL, { on: on }).then(function (r) {
      if (r.ok && r.data && r.data.kill_switch) {
        state = r.data;
        message = r.data.kill_switch.on
          ? 'Everything new is off. Recalls now read the way they did before 4.2.30.'
          : 'Everything new is back as you had set it.';
      } else {
        message = why(r);
      }
      toast(message);
      paint();
      return r.ok;
    });
  }

  function count(button) {
    button.disabled = true;
    return post(BACK, { dry_run: true }).then(function (r) {
      if (r.ok && r.data && r.data.success) {
        var n = Number(r.data.would_come_back) || 0;
        asking = n || null;
        message = n ? '' : 'Nothing that 4.2.30 retired is waiting to come back.';
      } else {
        message = why(r);
      }
      paint();
    });
  }

  function bringBack(button) {
    button.disabled = true;
    return post(BACK, {}).then(function (r) {
      asking = null;
      if (r.ok && r.data && r.data.success) {
        var n = Number(r.data.came_back) || 0;
        message = 'Brought back ' + memories(n) + '. It is under Recent decisions, where Undo retires them again.';
        if (r.data.refused) message += ' ' + memories(Number(r.data.refused)) + ' could not be brought back and were left as they are.';
        return read().then(function (s) { if (s) state = s; toast(message); paint(); });
      }
      message = why(r);
      toast(message);
      paint();
      return null;
    });
  }

  function renderInto(host) {
    mount = host;
    host.setAttribute('data-od-kill', 'card');
    host.textContent = '';
    host.appendChild(el('p', 'dash-card-meta', 'Loading...'));
    return read().then(function (s) { state = s; paint(); });
  }

  window.SLMKillSwitch = { renderInto: renderInto };
}());
