// od-features.js — the turn-on card for images and documents, shared helpers
// for the new panes. Public: window.odFeatures.
// XSS-safe: every server-provided string goes in with textContent, never as HTML.
// Writes use the page's fetch, which core.js wraps with the local write credential.
(function () {
  'use strict';

  var POLL_MS = 2000;
  var SIZE_NOTE = 'about 1.5 GB';

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = String(text);
    return e;
  }

  function button(label, cls, onClick) {
    var b = el('button', cls || 'btn sm', label);
    b.type = 'button';
    b.addEventListener('click', function () { onClick(b); });
    return b;
  }

  // Resolves {ok, status, data}; a dropped connection is status 0, never a throw.
  function api(method, url, body) {
    var init = { method: method };
    if (body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: !!r.ok, status: r.status, data: data || {} };
      });
    }, function () { return { ok: false, status: 0, data: {} }; });
  }

  function failText(res, fallback) {
    var d = res.data && res.data.detail;
    if (typeof d === 'string' && d) return d;
    if (d && typeof d.message === 'string') return d.message;
    return res.status === 0 ? 'Could not reach the local service.' : fallback;
  }

  function notAvailable(host, what) {
    host.textContent = '';
    var p = el('p', 'muted', (what || 'This') + ' is not available in this build.');
    p.setAttribute('role', 'status');
    host.appendChild(p);
  }

  // ---------------------------------------------------------------- media card
  function isOn(m) {
    return !!(m.enabled && m.env_state === 'ready' && !m.restart_required);
  }

  function cardShell(title, sub) {
    var card = el('div', 'card card-pad');
    var head = el('div', 'card-head');
    head.appendChild(el('h3', null, title));
    card.appendChild(head);
    if (sub) card.appendChild(el('p', 'muted', sub));
    return card;
  }

  function messageLine(card) {
    var m = el('p', 'muted');
    m.setAttribute('role', 'status');
    card.appendChild(m);
    return m;
  }

  function confirmThen(opts, go) {
    if (typeof window.confirmDestructive !== 'function') { go(); return; }
    window.confirmDestructive(opts).then(function (yes) { if (yes) go(); });
  }

  // The welcome card asks for the same turn-on flow as the pane's own button: it opens
  // the pane, and the pane's next "off" card starts that flow (with its confirmation).
  var enableRequested = false;

  function startEnable(host, opts, msg, m) {
    if (lowRam(m)) return;
    confirmThen({
      title: 'Turn on images and documents', target: 'Images and documents',
      consequence: 'Downloads ' + SIZE_NOTE + ' of models. Everything stays on this computer.',
      confirmLabel: 'Turn on',
    }, function () {
      msg.textContent = 'Starting…';
      api('POST', '/api/v3/features/media/enable', { yes: true }).then(function (res) {
        if (res.data && res.data.media) return render(host, res.data.media, opts);
        msg.textContent = failText(res, 'Could not turn on images and documents.');
      });
    });
  }

  // The daemon decides (one function, 16 GB) and says so in ram_ok / ram_message; an
  // older daemon says nothing, which counts as fine.
  function lowRam(m) {
    return !!m && m.ram_ok === false;
  }

  function offCard(host, m, opts) {
    var card = cardShell('Turn on images and documents',
      'Remember pictures and read PDFs: ' + SIZE_NOTE + ' of models, and it stays on this computer.');
    var msg = messageLine(card);
    if (m.env_state === 'unsupported' && m.step) msg.textContent = m.step;
    if (lowRam(m) && m.ram_message) card.insertBefore(el('p', null, m.ram_message), msg);
    var go = button('Turn on', 'btn sm primary', function () { startEnable(host, opts, msg, m); });
    go.disabled = m.env_state === 'unsupported' || lowRam(m);
    card.appendChild(go);
    if (enableRequested && !go.disabled) {
      enableRequested = false;
      window.setTimeout(function () { startEnable(host, opts, msg, m); }, 0);
    }
    return card;
  }

  function progressBar(m) {
    var pct = Math.max(0, Math.min(100, Math.round((Number(m.progress) || 0) * 100)));
    var bar = el('div', 'od-media-progress');
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.setAttribute('aria-valuenow', String(pct));
    var fill = el('div', 'od-media-progress-fill');
    fill.style.width = pct + '%';
    bar.appendChild(fill);
    return bar;
  }

  function installingCard(m) {
    var card = cardShell('Setting up images and documents', 'This can take several minutes.');
    card.appendChild(progressBar(m));
    card.appendChild(el('p', 'muted', m.step || 'Working…'));
    return card;
  }

  function failedCard(host, m, opts) {
    var card = cardShell('Setup did not finish');
    card.appendChild(el('p', null, m.step || m.error || 'Setup did not finish.'));
    var msg = messageLine(card);
    card.appendChild(button('Try again', 'btn sm primary', function () { startEnable(host, opts, msg, m); }));
    return card;
  }

  function restartCard() {
    var card = cardShell('One more step', 'Restart so images and documents can start.');
    var status = messageLine(card);
    card.appendChild(button('Restart SuperLocalMemory', 'btn sm primary', function (b) {
      if (typeof window.odRestartDaemon === 'function') window.odRestartDaemon(b, status);
      else status.textContent = 'Restart from the Governance page.';
    }));
    return card;
  }

  function startDisable(host, opts, box, msg) {
    var remove = !!box.checked;
    confirmThen({
      title: 'Turn off images and documents', target: 'Images and documents',
      consequence: remove ? 'Turns off and deletes the downloaded files. Saved memories stay.'
        : 'Turns off. The downloaded files and saved memories stay.',
      confirmLabel: 'Turn off',
    }, function () {
      api('POST', '/api/v3/features/media/disable', { remove_files: remove }).then(function (res) {
        if (res.data && res.data.media) return render(host, res.data.media, opts);
        msg.textContent = failText(res, 'Could not turn off images and documents.');
      });
    });
  }

  function onCard(host, opts) {
    var card = cardShell('Images and documents are on', 'Everything stays on this computer.');
    var label = el('label', 'muted');
    var box = el('input');
    box.type = 'checkbox';
    box.checked = false;
    label.appendChild(box);
    label.appendChild(el('span', null, ' Also remove the downloaded files'));
    card.appendChild(label);
    var msg = messageLine(card);
    card.appendChild(button('Turn off', 'btn sm', function () { startDisable(host, opts, box, msg); }));
    return card;
  }

  function pickCard(host, m, opts) {
    if (isOn(m)) return onCard(host, opts);
    if (m.enabled && m.env_state === 'ready') return restartCard();
    if (m.enabled && m.env_state === 'installing') return installingCard(m);
    if (m.enabled && m.env_state === 'failed') return failedCard(host, m, opts);
    return offCard(host, m, opts);
  }

  function schedulePoll(host, opts, token) {
    window.setTimeout(function () {
      if (host.__odFeatToken !== token || !host.isConnected) return;
      api('GET', '/api/v3/features').then(function (res) {
        if (host.__odFeatToken !== token) return;
        if (res.ok && res.data.media) render(host, res.data.media, opts);
        else schedulePoll(host, opts, token);
      });
    }, POLL_MS);
  }

  function render(host, m, opts) {
    var token = (host.__odFeatToken || 0) + 1;
    host.__odFeatToken = token;
    host.textContent = '';
    host.appendChild(pickCard(host, m, opts));
    enableRequested = false;
    if (isOn(m)) { if (opts.onReady) opts.onReady(m); }
    else if (opts.onNotReady) opts.onNotReady(m);
    if (m.enabled && m.env_state === 'installing') schedulePoll(host, opts, token);
  }

  function mountMediaCard(host, opts) {
    opts = opts || {};
    return api('GET', '/api/v3/features').then(function (res) {
      if (res.status === 404) return notAvailable(host, 'Images and documents');
      if (!res.ok || !res.data.media) {
        host.textContent = '';
        host.appendChild(el('p', 'muted', failText(res, 'Could not read the feature status.')));
        return;
      }
      render(host, res.data.media, opts);
    });
  }

  // ------------------------------------------------------------- what's new
  // Two equal tiles that say what is on and what is not, so the card never offers
  // something that is already done. The state comes from the same features read the
  // Documents & Images pane makes.
  var WHATSNEW_KEY = 'slm.whatsnew.4.1.25';
  var OPEN_MEDIA = 'Open Documents & Images';

  // Storage can be missing or throw (private windows, blocked site data).
  function seenWhatsNew() {
    try { return window.localStorage.getItem(WHATSNEW_KEY) === '1'; } catch (e) { return false; }
  }

  function rememberWhatsNew() {
    try { window.localStorage.setItem(WHATSNEW_KEY, '1'); } catch (e) { /* the card is still hidden this visit */ }
  }

  function goTo(pane) {
    if (typeof window.slmNavigate === 'function') window.slmNavigate(pane);
  }

  function requestEnable() { enableRequested = true; goTo('media-pane'); }

  function tile(title, chip) {
    var t = el('div', 'od-whatsnew-tile');
    var head = el('div', 'od-whatsnew-tile-head');
    head.appendChild(el('h4', null, title));
    if (chip) head.appendChild(el('span', 'badge ok', chip));
    t.appendChild(head);
    return t;
  }

  function percentText(m) {
    var pct = Math.max(0, Math.min(100, Math.round((Number(m.progress) || 0) * 100)));
    return pct + '%';
  }

  // Which of the five situations the images tile is in.
  function mediaTile(m) {
    var on = isOn(m);
    var t = tile('Images & documents', on ? 'On' : '');
    if (on) {
      t.appendChild(el('p', 'muted', 'Pictures and PDFs are remembered, and they stay on this computer.'));
      t.appendChild(button(OPEN_MEDIA, 'btn sm primary', function () { goTo('media-pane'); }));
    } else if (m.enabled && m.env_state === 'installing') {
      t.appendChild(el('p', 'muted', 'Setting up: ' + (m.step || 'working') + ' (' + percentText(m) + ').'));
      t.appendChild(button(OPEN_MEDIA, 'btn sm', function () { goTo('media-pane'); }));
    } else if (m.enabled && m.env_state === 'ready') {
      t.appendChild(el('p', 'muted', 'Almost done. Restart SuperLocalMemory to finish; the button is on the next page.'));
      t.appendChild(button(OPEN_MEDIA, 'btn sm primary', function () { goTo('media-pane'); }));
    } else {
      t.appendChild(el('p', 'muted', 'Remember pictures and read PDFs. It downloads ' + SIZE_NOTE +
        ' of models once, and nothing leaves this computer.'));
      if (lowRam(m) && m.ram_message) t.appendChild(el('p', null, m.ram_message));
      else if (m.env_state === 'unsupported' && m.step) t.appendChild(el('p', null, m.step));
      var go = button('Turn on images & documents', 'btn sm primary', requestEnable);
      go.disabled = m.env_state === 'unsupported' || lowRam(m);
      t.appendChild(go);
    }
    return t;
  }

  function botsTile(apps) {
    var t = tile('Bot messages');
    if (apps > 0) {
      var line = apps === 1 ? '1 app is ready to message your other bots.' : apps + ' apps can message each other.';
      t.appendChild(el('p', 'muted', line));
      t.appendChild(button('Open Bot messages', 'btn sm primary', function () { goTo('botmsg-pane'); }));
    } else {
      t.appendChild(el('p', 'muted', 'Let the AI apps you connect leave each other messages. You choose which apps may.'));
      t.appendChild(button('Set up bot messages', 'btn sm primary', function () { goTo('apps-pane'); }));
    }
    return t;
  }

  function developerLine(m) {
    return !m.enabled || m.env_state === 'failed';
  }

  function drawWhatsNew(host, data, token) {
    var m = (data && data.media) || {};
    var apps = Number(data && data.mesh && data.mesh.apps_with_mesh) || 0;
    host.textContent = '';
    var card = el('div', 'card card-pad od-whatsnew');
    var close = button('×', 'btn sm ghost od-whatsnew-close', function () {
      host.__odWnToken = token + 1;           // a poll still in flight must not redraw it
      rememberWhatsNew();
      host.textContent = '';
    });
    close.setAttribute('aria-label', 'Dismiss');
    card.appendChild(close);
    card.appendChild(el('h3', null, 'New in SuperLocalMemory'));
    var tiles = el('div', 'od-whatsnew-tiles');
    tiles.appendChild(mediaTile(m));
    tiles.appendChild(botsTile(apps));
    card.appendChild(tiles);
    if (developerLine(m)) card.appendChild(el('p', 'muted od-whatsnew-dev', 'For developers: slm media enable'));
    host.appendChild(card);
    if (m.enabled && m.env_state === 'installing') pollWhatsNew(host, token);
  }

  function pollWhatsNew(host, token) {
    window.setTimeout(function () {
      if (host.__odWnToken !== token || !host.isConnected) return;
      api('GET', '/api/v3/features').then(function (res) {
        if (host.__odWnToken !== token) return;
        if (res.ok && res.data.media) drawWhatsNew(host, res.data, token);
        else pollWhatsNew(host, token);
      });
    }, POLL_MS);
  }

  // Resolves once drawn. A card the person dismissed costs no request. If the status
  // cannot be read, the card still shows its two starting offers.
  function mountWhatsNew(host) {
    host.textContent = '';
    var token = (host.__odWnToken || 0) + 1;
    host.__odWnToken = token;
    if (seenWhatsNew()) return Promise.resolve();
    return api('GET', '/api/v3/features').then(function (res) {
      if (host.__odWnToken !== token) return;
      drawWhatsNew(host, res.ok ? res.data : {}, token);
    });
  }

  function autoMountWhatsNew() {
    var host = document.getElementById('od-whatsnew');
    if (host) mountWhatsNew(host);
  }

  window.odFeatures = {
    el: el, button: button, api: api, failText: failText, notAvailable: notAvailable,
    confirmThen: confirmThen, mountMediaCard: mountMediaCard, isOn: isOn,
    mountWhatsNew: mountWhatsNew,
  };
  autoMountWhatsNew();
})();
