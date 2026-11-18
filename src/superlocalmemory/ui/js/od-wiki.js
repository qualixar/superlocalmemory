// od-wiki.js: the Dashboard home "Living wiki" card (4.2.30, WS-6, D-17, D-18; blueprint 8.10.3 i).
//
// SuperLocalMemory compiles pages from your memories: what is current about a subject,
// how a family of decisions went, what is known about a topic. Every sentence of a page is
// a word-for-word piece of a memory it cites, or a fixed template over its values; no model
// wrote any of it, and a page says how many sentences it withheld. A page is never a memory.
//   - Read          : a page, each sentence with the memory it comes from ("Open memory").
//   - Rebuild pages : you compile the pages again and approve them (POST /api/wiki/rebuild,
//                     channel "dashboard": the same route `slm wiki rebuild` uses).
//   - Read my edits : your edits to the Markdown files become proposals, never writes
//                     (POST /api/wiki/sync); only while the Markdown wiki is on.
// Memory text is only ever set with textContent. The card hides itself when there is nothing
// to show and the Markdown wiki is off.
(function () {
  'use strict';

  var KIND = { current_state: 'Current state', story: 'Story of a decision', topic: 'Topic' };
  var data = { pages: [], status: null, flag: 'off' };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }
  function button(label, cls, onClick) {
    var b = el('button', 'btn sm ' + (cls || 'ghost'), label);
    b.type = 'button';
    b.addEventListener('click', function (e) { e.preventDefault(); onClick(b); });
    return b;
  }
  function detailOf(json, fallback) {
    var d = json && (json.detail || json.error);
    if (d && typeof d === 'object') d = d.message || d.code;
    return d ? String(d) : fallback;
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  function request(method, url, body) {
    var init = { method: method, credentials: 'same-origin' };
    if (body) { init.headers = { 'Content-Type': 'application/json' }; init.body = JSON.stringify(body); }
    return fetch(url, init).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok || !j.success) throw new Error(detailOf(j, 'That did not work (HTTP ' + r.status + ').'));
        return j;
      });
    });
  }

  // Follow a sentence to the memory it comes from: open the Memories pane and search for it.
  function openMemory(text) {
    var tab = document.querySelector('.nav-link[data-tab="memories-pane"]');
    if (tab) tab.click();
    var box = $('recall-lab-query'), go = $('recall-lab-search');
    if (box) box.value = text;
    if (go) go.click();
  }

  function note(host, text, tone) {
    var n = host.querySelector('.wiki-note') || host.appendChild(el('p', 'dash-card-meta wiki-note'));
    n.className = 'dash-card-meta wiki-note ' + (tone || '');
    n.textContent = text;
  }

  function withheldLine(w) {
    if (!w || !w.count) return null;
    var why = Object.keys(w.reasons || {}).map(function (k) {
      return w.reasons[k].count + ' ' + w.reasons[k].why;
    }).join('; ');
    return plural(w.count, 'sentence', 'sentences') + ' withheld' + (why ? ' (' + why + ')' : '') + '.';
  }

  function sentenceRow(s) {
    var row = el('div', 'truth-side');
    row.appendChild(el('p', 'truth-text', s.text));
    var meta = el('span', 'dash-card-meta', s.type === 'span' ? 'word for word from a memory'
                                                              : 'from slot values');
    row.appendChild(meta);
    (s.citations || []).slice(0, 2).forEach(function (c) {
      var label = 'Open memory' + (c.origin ? ' (' + c.origin + ')' : '');
      row.appendChild(button(label, 'ghost', function () { openMemory(s.text); }));
    });
    return row;
  }

  function showPage(host, p) {
    host.textContent = '';
    host.appendChild(el('p', 'dash-card-lead', p.title));
    host.appendChild(el('p', 'dash-card-meta', p.notice));
    if (p.stale) {
      host.appendChild(el('p', 'dash-card-meta tone-warn',
        'Some sentences no longer match your memories and are hidden. Rebuild the pages to refresh them.'));
    }
    (p.sentences || []).forEach(function (s) { host.appendChild(sentenceRow(s)); });
    var line = withheldLine(p.withheld);
    if (line) host.appendChild(el('p', 'dash-card-meta', line));
  }

  function toggle(row, page) {
    var host = row.querySelector('.wiki-page');
    if (host) { row.removeChild(host); return; }
    host = row.appendChild(el('div', 'wiki-page'));
    host.appendChild(el('p', 'dash-card-meta', 'Checking every sentence against your memories…'));
    request('GET', '/api/wiki/pages/' + encodeURIComponent(page.page_id)).then(
      function (j) { showPage(host, j.page); },
      function (e) { host.textContent = ''; host.appendChild(el('p', 'dash-card-meta tone-warn', e.message)); });
  }

  function pageRow(page) {
    var row = el('div', 'truth-item');
    row.setAttribute('data-page', page.page_id);
    row.appendChild(el('p', 'dash-card-lead', page.title));
    var counts = plural(page.shown_count, 'sentence', 'sentences') +
      (page.withheld_count ? ', ' + page.withheld_count + ' withheld' : '');
    row.appendChild(el('p', 'dash-card-meta', (KIND[page.kind] || 'Page') + '. ' + counts + '.'));
    var actions = el('div', 'truth-actions');
    actions.appendChild(button('Read', 'primary', function () { toggle(row, page); }));
    row.appendChild(actions);
    return row;
  }

  function act(btn, host, method, url, done) {
    btn.disabled = true;
    request(method, url, { channel: 'dashboard' }).then(function (j) {
      note(host, done(j), 'tone-ok');
      setTimeout(load, 1200);
    }, function (e) { btn.disabled = false; note(host, e.message, 'tone-warn'); });
  }

  function toolbar(body) {
    var bar = el('div', 'truth-actions');
    bar.appendChild(button('Rebuild pages', 'primary', function (b) {
      act(b, body, 'POST', '/api/wiki/rebuild', function (j) {
        return j.pages ? 'Compiled and approved ' + plural(j.pages, 'page', 'pages') + '.'
                       : 'The pages are up to date.';
      });
    }));
    if (data.flag === 'on') {
      bar.appendChild(button('Read my edits', 'ghost', function (b) {
        act(b, body, 'POST', '/api/wiki/sync', function (j) {
          return j.found ? plural(j.queued.length, 'edit', 'edits') + ' proposed; review them under Proposed changes.'
                         : 'No edits to read. (' + plural(j.deleted_lines || 0, 'deleted line', 'deleted lines') +
                           ' ignored: deleting a line forgets nothing.)';
        });
      }));
    }
    return bar;
  }

  function render() {
    var c = $('dash-wiki-card'), b = $('dash-wiki-body');
    if (!c || !b) return;
    b.textContent = '';
    var pages = data.pages, st = data.status || {};
    c.hidden = !(pages.length || data.flag !== 'off');
    var badge = $('dash-wiki-count');
    if (badge) badge.textContent = pages.length ? String(pages.length) : '';
    if (c.hidden) return;
    var where = data.flag === 'off' ? 'The Markdown folder is off (slm flags set compiler.wiki on).'
      : (data.flag === 'shadow' ? 'Markdown folder in shadow: it shows what it would write.'
                                : 'Markdown folder: ' + (st.dir || ''));
    b.appendChild(el('p', 'dash-card-meta',
      pages.length ? 'Pages built only from your memories: every sentence cites the memory it comes from. ' + where
                   : 'No pages yet. ' + where));
    b.appendChild(toolbar(b));
    if (st.pending_sets) {
      b.appendChild(el('p', 'dash-card-meta', plural(st.pending_sets, 'page change is', 'page changes are') +
        ' waiting under Proposed changes.'));
    }
    pages.forEach(function (p) { b.appendChild(pageRow(p)); });
  }

  function load() {
    if (!$('dash-wiki-card')) return Promise.resolve();
    return request('GET', '/api/wiki/pages').then(function (j) {
      data = { pages: j.pages || [], status: j.status || null, flag: j.flag || 'off' };
      render();
    }, function () { var c = $('dash-wiki-card'); if (c) c.hidden = true; });
  }

  window.SLMWiki = { reload: load, _render: render, _open: openMemory };
  document.addEventListener('DOMContentLoaded', function () { setTimeout(load, 300); });
}());
