// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
// Licensed under AGPL-3.0-or-later - see LICENSE file
//
// 4.2.30 (D-41, §8.10.3 l): the "What memory can do now" card on the Dashboard home, next to the
// What's new tiles. One plain line for each new thing, each with a link to its card or its
// guide, and shown only while that feature is not yet turned on: which ones are on comes from
// GET /api/whatsnew/state (the feature flags, read-only). The card hides itself when there is
// nothing left to offer, and a person can dismiss it (remembered in this browser).
// Nothing from the server is parsed as HTML.
(function () {
  'use strict';

  var API = '/api/whatsnew/state';
  var KEY = 'slm.whatsnew.4.2.30';
  var DOCS = 'https://github.com/qualixar/superlocalmemory/blob/main/docs/';

  // One line each, in the order a person meets them. `pane`: a page of this dashboard;
  // `doc`: a guide. `how`: the plain switch (only for features with a flag).
  var ENTRIES = [
    { id: 'need_gate', title: 'Memory on every prompt',
      line: 'Your agent can be given the right memory as you type, only when a prompt needs it.',
      doc: 'configuration.md' },
    { id: 'cues', title: 'Reminders on cues',
      line: 'A memory can come up when you open a file, run a command or hit an error it belongs to.',
      doc: 'configuration.md' },
    { id: 'measurement', title: 'What SuperLocalMemory measures',
      line: 'It now and then leaves one memory out, at random, to learn whether memory helps. Ids and numbers only, on this computer, and you can turn it off.',
      pane: 'settings-pane', paneLabel: 'Open Settings' },
    { id: 'origin', title: 'Where a memory came from',
      line: 'See whether a memory is your own words, a tool\'s output or a web page.',
      doc: 'configuration.md' },
    { id: 'firewall', title: 'Held back from instructions',
      line: 'A memory from a tool or a web page cannot become a rule by itself; one explicit, recorded call promotes it, and you can undo it.',
      doc: 'configuration.md' },
    { id: 'secrets', title: 'Secrets stored as placeholders',
      line: 'A password or key you save is kept as a placeholder, never in clear.',
      doc: 'configuration.md' },
    { id: 'signing', title: 'Signed memories',
      line: 'SuperLocalMemory can tell when a memory was changed outside it.',
      doc: 'signed-memories.md' },
    { id: 'why', title: 'Why was this shown?',
      line: 'Open any memory to see where it came from and which conversations it was shown in.',
      pane: 'memories-pane', paneLabel: 'Open Memories' },
    { id: 'proposals', title: 'Proposed changes',
      line: 'SuperLocalMemory can suggest tidy-ups, such as merging duplicates, that your agents or you apply, with an undo.',
      doc: 'cli-reference.md' },
    { id: 'wiki', title: 'Living wiki',
      line: 'Pages built only from your memories, with every sentence linked to the memory it came from.',
      doc: 'living-wiki.md' },
    { id: 'import', title: 'Bring memories from other apps',
      line: 'Import what ChatGPT, Claude, Mem0, Letta or your Markdown notes remember. You see a preview first, and one Undo removes the whole import.',
      doc: 'importing.md' },
    { id: 'decisions', title: 'Your agents decide, and you can undo',
      line: 'Agents can settle disputes and apply changes themselves. Each decision is listed on the Dashboard with an Undo.',
      pane: 'dashboard-pane', paneLabel: 'Open the Dashboard' }
  ];

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function dismissed() {
    try { return window.localStorage.getItem(KEY) === '1'; } catch (e) { return false; }
  }
  function remember() {
    try { window.localStorage.setItem(KEY, '1'); } catch (e) { /* hidden for this visit anyway */ }
  }

  function go(pane) {
    if (typeof window.slmNavigate === 'function') { window.slmNavigate(pane); return; }
    var link = document.querySelector('.nav-link[data-tab="' + pane + '"]');
    if (link) link.click();
  }

  function link(entry) {
    if (entry.pane) {
      var b = el('button', 'btn sm ghost', entry.paneLabel || 'Open');
      b.type = 'button';
      b.addEventListener('click', function (e) { e.preventDefault(); go(entry.pane); });
      return b;
    }
    var a = el('a', 'od-whatsnew-link', 'Read more');
    a.href = DOCS + entry.doc;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    return a;
  }

  function item(entry, flag) {
    var li = el('li', 'od-whatsnew-item');
    li.setAttribute('data-entry', entry.id);
    var text = el('span', 'od-whatsnew-item-text');
    text.appendChild(el('strong', null, entry.title + (/[.?!]$/.test(entry.title) ? ' ' : '. ')));
    text.appendChild(document.createTextNode(entry.line));
    if (flag) text.appendChild(el('code', 'od-whatsnew-how', 'slm flags set ' + flag + ' on'));
    li.appendChild(text);
    li.appendChild(link(entry));
    return li;
  }

  function draw(host, state) {
    host.textContent = '';
    var features = (state && state.features) || {};
    var shown = ENTRIES.filter(function (e) { return !(features[e.id] && features[e.id].on); });
    if (!shown.length || dismissed()) return;
    var card = el('div', 'card card-pad od-whatsnew od-whatsnew-4230');
    var close = el('button', 'btn sm ghost od-whatsnew-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss');
    close.addEventListener('click', function (e) { e.preventDefault(); remember(); host.textContent = ''; });
    card.appendChild(close);
    card.appendChild(el('h3', null, 'What memory can do now'));
    if (state.kill_switch) {
      card.appendChild(el('p', 'dash-card-meta', 'The kill switch is on, so every new feature is off. ' +
        'Turn it off with slm flags kill off to use them.'));
    }
    var list = el('ul', 'od-whatsnew-list');
    shown.forEach(function (e) { list.appendChild(item(e, features[e.id] && features[e.id].flag)); });
    card.appendChild(list);
    host.appendChild(card);
  }

  // Resolves when drawn. Nothing is asked of the server for a card the person dismissed, and a
  // state that cannot be read draws nothing (never a list that might offer what is already on).
  function mount(host) {
    if (!host) return Promise.resolve();
    host.textContent = '';
    if (dismissed()) return Promise.resolve();
    return fetch(API, { credentials: 'same-origin' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (s) { draw(host, s); }, function () { host.textContent = ''; });
  }

  window.odWhatsNew4230 = { mount: mount, ENTRIES: ENTRIES };
  document.addEventListener('DOMContentLoaded', function () {
    mount(document.getElementById('od-whatsnew-4230'));
  });
}());
