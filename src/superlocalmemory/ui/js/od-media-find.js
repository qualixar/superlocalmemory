// od-media-find.js — "Find a picture" box at the top of the Documents & Images pane.
// Public: window.odMediaFind.mount(host)
// Runs the dashboard's own search (POST /api/search, the one the Memories pane uses)
// and keeps only the results that carry a picture: a thumbnail, the first line, and
// how well it matched. Pictures are found by what they show and the text inside them.
// XSS-safe: every result string is set with textContent; a media id is used only if it
// is 32 hex characters.
(function () {
  'use strict';

  var ID_RE = /^[0-9a-f]{32}$/;
  var LIMIT = 50;
  var LINE_MAX = 140;
  var EMPTY = 'Nothing matched. Pictures are found by what they show and the text inside them.';
  var FAILED = 'Search did not work. Try again in a moment.';

  function F() { return window.odFeatures; }
  function el(tag, cls, text) { return F().el(tag, cls, text); }

  function firstLine(text) {
    var lines = String(text == null ? '' : text).split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      var t = lines[i].trim();
      if (t) return t.length > LINE_MAX ? t.slice(0, LINE_MAX - 1) + '…' : t;
    }
    return 'Picture';
  }

  // A read-only search: no write credential, and it must not clear the dashboard cache.
  function search(query) {
    return fetch('/api/search', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      slmInvalidatesCache: false, slmRequiresWriteAuth: false,
      body: JSON.stringify({ query: query, limit: LIMIT }),
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }

  function withPicture(results) {
    return (results || []).filter(function (r) {
      return r && r.media && ID_RE.test(String(r.media.media_id || ''));
    });
  }

  function card(r) {
    var line = firstLine(r.content);
    var li = el('li');
    var img = el('img');
    img.setAttribute('src', '/api/v3/media/' + r.media.media_id + '/thumb');
    img.setAttribute('alt', line);
    img.setAttribute('loading', 'lazy');
    img.addEventListener('error', function () { img.style.visibility = 'hidden'; });
    li.appendChild(img);
    li.appendChild(el('span', 'od-find-line', line));
    var score = Number(r.score);
    if (isFinite(score)) li.appendChild(el('span', 'muted', Math.round(score * 100) + '% match'));
    return li;
  }

  function show(ui, results) {
    ui.list.textContent = '';
    var found = withPicture(results);
    ui.note.textContent = found.length ? '' : EMPTY;
    found.forEach(function (r) { ui.list.appendChild(card(r)); });
  }

  function run(ui) {
    var q = ui.input.value.trim();
    if (!q) return Promise.resolve();
    var seq = ++ui.seq;
    ui.note.textContent = 'Searching…';
    ui.list.textContent = '';
    return search(q).then(function (data) {
      if (seq === ui.seq) show(ui, data && data.results);
    }, function () {
      if (seq !== ui.seq) return;
      ui.note.textContent = FAILED;
    });
  }

  function mount(host) {
    var box = el('div', 'card card-pad od-media-section');
    box.appendChild(el('h3', null, 'Find a picture'));
    box.appendChild(el('p', 'muted', 'Type what is in it, or words written on it.'));
    var form = el('form', 'od-find');
    form.setAttribute('role', 'search');
    var input = el('input');
    input.type = 'search';
    input.setAttribute('aria-label', 'Find a picture');
    input.setAttribute('placeholder', 'the slide with the quarterly numbers');
    input.setAttribute('autocomplete', 'off');
    var go = el('button', 'btn sm primary', 'Find');
    go.type = 'submit';
    form.appendChild(input);
    form.appendChild(go);
    var note = el('p', 'muted');
    note.setAttribute('role', 'status');
    var list = el('ul', 'od-find-results');
    var ui = { input: input, note: note, list: list, seq: 0 };
    form.addEventListener('submit', function (e) { e.preventDefault(); run(ui); });
    [form, note, list].forEach(function (n) { box.appendChild(n); });
    host.appendChild(box);
    return ui;
  }

  window.odMediaFind = { mount: mount };
})();
