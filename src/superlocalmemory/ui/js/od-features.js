// od-features.js — the turn-on card for images and documents, shared helpers
// for the new panes. Public: window.odFeatures.
// XSS-safe: every server-provided string goes in with textContent, never as HTML.
// Writes use the page's fetch, which core.js wraps with the local write credential.
(function () {
  'use strict';

  var POLL_MS = 2000;
  var SIZE_NOTE = 'about 2.8 GB of disk space for the model and runtime';
  function mediaSize(m) {
    return m && m.installed_bytes ? (m.installed_bytes / 1024 ** 3).toFixed(1) + ' GB installed (measured)' : SIZE_NOTE + ' (estimate)';
  }

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
    // Pictures and documents work as soon as the environment is ready (checked on a real store,
    // 19 Nov 2026): saved, read, indexed and described with no restart. A restart only lets text and
    // pictures share one model process, so it is a note on the card, never a step that blocks.
    return !!(m.enabled && m.env_state === 'ready');
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
    confirmThen({
      title: 'Turn on images and documents', target: 'Images and documents',
      consequence: 'Installs ' + mediaSize(m) + '. Everything stays on this computer.',
      confirmLabel: 'Turn on',
    }, function () {
      msg.textContent = 'Starting…';
      api('POST', '/api/v3/features/media/enable', { yes: true }).then(function (res) {
        if (res.data && res.data.media) return render(host, res.data.media, opts);
        msg.textContent = failText(res, 'Could not turn on images and documents.');
      });
    });
  }

  // The daemon decides (one function: automatic from 16 GB, one click below) and says so in `mode`,
  // `words` and, on a smaller computer, `ram_warning`. Nothing is refused for memory. An older daemon
  // says nothing, which counts as the one-click path.
  function offCard(host, m, opts) {
    var auto = m.mode === 'auto' && !m.opted_out;
    var title = m.opted_out ? 'Images and documents are turned off by you'
      : (auto ? 'Images and documents set up by themselves' : 'Turn on images and documents');
    var detail = auto ? (m.words || 'SuperLocalMemory downloads the picture model by itself when your computer is idle.')
      : 'Remember pictures and read PDFs: ' + mediaSize(m) + ', and it stays on this computer.';
    var card = cardShell(title, detail);
    var msg = messageLine(card);
    if (m.env_state === 'unsupported' && m.step) msg.textContent = m.step;
    if (m.ram_warning) card.insertBefore(el('p', null, m.ram_warning), msg);
    var go = button(auto ? 'Turn on now' : 'Turn on', 'btn sm primary', function () { startEnable(host, opts, msg, m); });
    go.disabled = m.env_state === 'unsupported';
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
    var card = cardShell('Setting up images and documents', m.words || 'This can take several minutes.');
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

  function onCard(host, opts, m) {
    var sub = (m.words || 'Ready') + '. Everything stays on this computer.';
    if (m.restart_required) sub += ' It uses less memory after SuperLocalMemory next restarts.';
    var card = cardShell('Images and documents are on', sub);
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
    if (isOn(m)) return onCard(host, opts, m);
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
  var WHATSNEW_KEY = 'slm.whatsnew.4.2.0';
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
      t.appendChild(el('p', 'muted', m.words || ('Setting up: ' + (m.step || 'working') + ' (' + percentText(m) + ').')));
      t.appendChild(button(OPEN_MEDIA, 'btn sm', function () { goTo('media-pane'); }));
    } else if (m.enabled && m.env_state === 'ready') {
      t.appendChild(el('p', 'muted', 'Almost done. Restart SuperLocalMemory to finish; the button is on the next page.'));
      t.appendChild(button(OPEN_MEDIA, 'btn sm primary', function () { goTo('media-pane'); }));
    } else {
      var auto = m.mode === 'auto' && !m.opted_out;
      t.appendChild(el('p', 'muted', m.opted_out ? 'Turned off by you.' : (auto && m.words ? m.words
        : 'Remember pictures and read PDFs. Installs ' + mediaSize(m) + ' once, and nothing leaves this computer.')));
      if (m.ram_warning) t.appendChild(el('p', null, m.ram_warning));
      else if (m.env_state === 'unsupported' && m.step) t.appendChild(el('p', null, m.step));
      var go = button(auto ? 'Turn on now' : 'Turn on images & documents', 'btn sm primary', requestEnable);
      go.disabled = m.env_state === 'unsupported';
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
