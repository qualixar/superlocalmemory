// od-media-packs.js: "Better pictures and documents", the optional models on the Documents & Images pane
// (4.2.30, WS-8, D-19).
//
// For people who are not technical. Three plain switches, each off until chosen:
//   Describe pictures                  a small model writes a few words about each picture you save
//   Read scanned documents better      a larger model reads scanned pages (the standard reader is the fallback)
//   Remember where text was found      keeps "from page 3, top left" for text read from a picture or page
// Turning a model on downloads it: the size is shown on the button and in the confirmation before anything
// is downloaded, you watch the progress, and Turn off keeps your memories (and, unless you tick the box, the
// downloaded files). It reads and changes through the routes `slm media pack` uses:
//   GET  /api/v3/features/media/packs
//   POST /api/v3/features/media/packs/enable   {pack, yes: true, source: 'dashboard'}
//   POST /api/v3/features/media/packs/disable  {pack, remove_files}
// No raw ids are shown, and every server string is set with textContent, never as HTML.
// Public: window.odMediaPacks.mount(host).
(function () {
  'use strict';

  var POLL_MS = 2000;
  var PATH = '/api/v3/features/media/packs';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
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
    return res.status === 0 ? 'Could not reach the local service.' : fallback;
  }

  function confirmThen(opts, go) {
    if (typeof window.confirmDestructive !== 'function') { go(); return; }
    window.confirmDestructive(opts).then(function (yes) { if (yes) go(); });
  }

  function progressBar(p) {
    var pct = Math.max(0, Math.min(100, Math.round((Number(p.progress) || 0) * 100)));
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

  // The state in a word a person understands.
  function stateWord(p) {
    if (p.state === 'on') return 'On';
    if (p.state === 'downloading') return 'Downloading';
    if (p.state === 'ready') return 'Downloaded, off';
    if (p.state === 'failed') return 'Did not finish';
    return 'Off';
  }

  function row(host, p, redraw) {
    var box = el('div', 'od-pack-row');
    box.setAttribute('data-pack', p.pack);
    var head = el('div', 'card-head');
    head.appendChild(el('h4', null, p.label));
    head.appendChild(el('span', 'badge' + (p.state === 'on' ? ' ok' : ''), stateWord(p)));
    box.appendChild(head);
    box.appendChild(el('p', 'muted', p.what));
    var msg = el('p', 'muted');
    msg.setAttribute('role', 'status');

    if (p.killed) {
      msg.textContent = 'These features are switched off for now, so this cannot be turned on.';
    } else if (p.state === 'downloading') {
      box.appendChild(progressBar(p));
      msg.textContent = p.step || 'Downloading…';
    } else if (p.state === 'failed') {
      msg.textContent = p.error || 'The download did not finish. Check your connection and try again.';
    } else if (p.state === 'off' && p.size_bytes) {
      msg.textContent = p.download_words + '. It stays on this computer.';
    }
    box.appendChild(msg);

    if (!p.killed && p.state !== 'downloading') {
      if (p.state === 'on') {
        var remove = el('input');
        remove.type = 'checkbox';
        var label = el('label', 'muted');
        label.appendChild(remove);
        label.appendChild(el('span', null, p.size_bytes ? ' Also remove the downloaded files' : ' Also forget saved places'));
        if (p.size_bytes) box.appendChild(label);
        box.appendChild(button('Turn off', 'btn sm', function () {
          api('POST', PATH + '/disable', { pack: p.pack, remove_files: !!remove.checked }).then(function (res) {
            if (res.ok) redraw(); else msg.textContent = failText(res, 'Could not turn this off.');
          });
        }));
      } else {
        var verb = p.size_bytes ? 'Turn on (' + p.size_words + ' download)' : 'Turn on';
        var go = button(verb, 'btn sm primary', function () {
          var doIt = function () {
            msg.textContent = 'Starting…';
            api('POST', PATH + '/enable', { pack: p.pack, yes: true, source: 'dashboard' }).then(function (res) {
              if (res.ok) redraw(); else msg.textContent = failText(res, 'Could not turn this on.');
            });
          };
          if (!p.size_bytes) { doIt(); return; }
          confirmThen({
            title: p.label, target: p.label,
            consequence: p.download_words + '. It stays on this computer and you can turn it off any time.',
            confirmLabel: 'Download and turn on',
          }, doIt);
        });
        go.disabled = !p.media_ready;
        if (!p.media_ready) msg.textContent = 'Turn on Images & documents first; this adds to it.';
        box.appendChild(go);
      }
    }
    host.appendChild(box);
  }

  function mount(host) {
    var token = (host.__odPackToken || 0) + 1;
    host.__odPackToken = token;

    function redraw() { return mount(host); }

    return api('GET', PATH).then(function (res) {
      if (host.__odPackToken !== token) return;
      host.textContent = '';
      if (res.status === 404) return;            // an older daemon: the card simply is not there
      var card = el('div', 'card card-pad od-media-packs');
      card.appendChild(el('h3', null, 'Better pictures and documents'));
      card.appendChild(el('p', 'muted', 'Optional. Each part is off until you turn it on, and nothing is ' +
        'downloaded without your yes.'));
      if (!res.ok || !Array.isArray(res.data.packs)) {
        card.appendChild(el('p', 'muted', failText(res, 'Could not read these settings.')));
        host.appendChild(card);
        return;
      }
      var downloading = false;
      res.data.packs.forEach(function (p) {
        if (p.state === 'downloading') downloading = true;
        row(card, p, redraw);
      });
      host.appendChild(card);
      if (downloading) {
        window.setTimeout(function () {
          if (host.__odPackToken === token && host.isConnected) redraw();
        }, POLL_MS);
      }
    });
  }

  window.odMediaPacks = { mount: mount };
})();
