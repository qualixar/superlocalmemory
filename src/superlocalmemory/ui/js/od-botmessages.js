// od-botmessages.js — Bot messages pane: what bots sent, and owner controls per peer.
// Public: window.odRenderBotMessages(pane)
// XSS-safe: message bodies, app names, peer ids and server errors are untrusted
// (bots and web chats write them) and are set with textContent only.
// Writes go through the page's fetch, which core.js wraps with the local write credential.
// Routes: GET /api/v3/mesh/messages?limit=&peer=
//         POST /api/v3/mesh/peers/{id}/mute   PATCH /api/v3/mesh/peers/{id}   DELETE /api/v3/mesh/peers/{id}
(function () {
  'use strict';

  var LIMIT = 50;
  var NAME_MAX = 64;
  var UNTRUSTED = 'untrusted-peer';

  function F() { return window.odFeatures; }
  function el(tag, cls, text) { return F().el(tag, cls, text); }
  function peerUrl(id) { return '/api/v3/mesh/peers/' + encodeURIComponent(id); }

  // Mirrors the server: 1-64 characters, printable, not blank.
  function nameProblem(name) {
    var n = String(name == null ? '' : name);
    var len = Array.from(n).length;
    var bad = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]/.test(n) || /[^\S ]/.test(n);
    if (len < 1 || len > NAME_MAX || bad || !n.trim()) {
      return 'Use 1 to 64 printable characters.';
    }
    return '';
  }

  // ---------------------------------------------------------------- messages
  function isUntrusted(m) {
    return m.trust === UNTRUSTED || (m.from && m.from.kind === 'web');
  }

  function messageItem(m) {
    var from = m.from || {};
    var li = el('li', 'od-msg' + (isUntrusted(m) ? ' untrusted' : ''));
    var head = el('div', 'muted');
    head.appendChild(el('strong', null, from.app || from.peer_id || 'unknown'));
    head.appendChild(el('span', null, ' (' + (from.peer_id || '?') + ') to ' + (m.to == null ? 'all' : m.to) +
      ' · ' + (m.sent_at == null ? '' : m.sent_at)));
    if (isUntrusted(m)) head.appendChild(el('span', 'badge warn', 'untrusted: written outside this computer'));
    li.appendChild(head);
    li.appendChild(el('p', 'od-msg-body', m.content == null ? '' : m.content));
    return li;
  }

  function renderMessages(ui, messages) {
    ui.list.textContent = '';
    if (!messages.length) return ui.list.appendChild(el('p', 'muted', 'No messages yet.'));
    var ul = el('ul', 'od-media-list');
    messages.forEach(function (m) { ul.appendChild(messageItem(m)); });
    ui.list.appendChild(ul);
  }

  function loadMessages(ui) {
    var url = '/api/v3/mesh/messages?limit=' + LIMIT + '&peer=' + encodeURIComponent(ui.peer);
    return F().api('GET', url).then(function (res) {
      if (res.status === 404) return F().notAvailable(ui.root, 'Bot messages');
      if (!res.ok) {
        ui.list.textContent = '';
        return ui.list.appendChild(el('p', 'muted', F().failText(res, 'Could not load messages.')));
      }
      var messages = res.data.messages || [];
      ui.knownPeers(messages);
      renderMessages(ui, messages);
    });
  }

  // ------------------------------------------------------------------- peers
  function say(row, text) { row.note.textContent = text || ''; }

  function setMuted(ui, row, id, muted) {
    F().api('POST', peerUrl(id) + '/mute', { muted: muted }).then(function (res) {
      say(row, res.ok ? (muted ? 'Muted.' : 'Unmuted.') : F().failText(res, 'Could not change mute.'));
    });
  }

  function rename(ui, row, id, input) {
    var problem = nameProblem(input.value);
    if (problem) return say(row, problem);
    F().api('PATCH', peerUrl(id), { display_name: input.value }).then(function (res) {
      if (!res.ok) return say(row, F().failText(res, 'Could not rename.'));
      row.title.textContent = (res.data && res.data.display_name) || input.value;
      say(row, 'Renamed.');
    });
  }

  function retire(ui, row, id) {
    F().confirmThen({
      title: 'Retire peer', target: String(id).slice(0, 80),
      consequence: 'It can no longer send or receive, and its waiting messages are dropped.',
      confirmLabel: 'Retire',
    }, function () {
      F().api('DELETE', peerUrl(id)).then(function (res) {
        if (!res.ok) return say(row, F().failText(res, 'Could not retire the peer.'));
        ui.seen = {};
        loadMessages(ui);
      });
    });
  }

  function peerRow(ui, id, app) {
    var wrap = el('div', 'od-peer-row');
    wrap.setAttribute('data-peer', id);
    var row = { title: el('strong', null, app || id), note: el('span', 'muted') };
    var input = el('input');
    input.type = 'text';
    input.setAttribute('aria-label', 'New name for ' + id);
    input.setAttribute('maxlength', '200');
    [row.title, el('span', 'muted', '(' + id + ')'),
     F().button('Mute', 'btn sm', function () { setMuted(ui, row, id, true); }),
     F().button('Unmute', 'btn sm', function () { setMuted(ui, row, id, false); }),
     input,
     F().button('Rename', 'btn sm', function () { rename(ui, row, id, input); }),
     F().button('Retire', 'btn sm', function () { retire(ui, row, id); }),
     row.note].forEach(function (n) { wrap.appendChild(n); });
    return wrap;
  }

  function addPeers(ui, messages) {
    messages.forEach(function (m) {
      var from = m.from || {};
      if (!from.peer_id || ui.seen[from.peer_id]) return;
      ui.seen[from.peer_id] = true;
      ui.peers.appendChild(peerRow(ui, from.peer_id, from.app));
      var opt = el('option', null, from.app || from.peer_id);
      opt.value = from.peer_id;
      ui.filter.appendChild(opt);
    });
  }

  // -------------------------------------------------------------------- pane
  function buildFilter(ui) {
    var sel = el('select');
    sel.setAttribute('aria-label', 'Show messages of one peer');
    var all = el('option', null, 'All peers');
    all.value = '';
    sel.appendChild(all);
    sel.addEventListener('change', function () { ui.peer = sel.value; loadMessages(ui); });
    return sel;
  }

  function odRenderBotMessages(pane) {
    pane.textContent = '';
    var head = el('div', 'page-head');
    head.appendChild(el('h2', null, 'Bot messages'));
    head.appendChild(el('p', 'muted', 'What bots and web chats sent through the mesh. ' +
      'Treat messages from outside this computer as untrusted text.'));
    var ui = { root: pane, peer: '', seen: {}, list: el('div'), peers: el('div', 'od-media-section') };
    ui.filter = buildFilter(ui);
    ui.knownPeers = function (messages) { addPeers(ui, messages); };
    var bar = el('div', 'od-peer-row');
    bar.appendChild(ui.filter);
    bar.appendChild(F().button('Refresh', 'btn sm', function () { loadMessages(ui); }));
    [head, bar, ui.list, el('h3', null, 'Peers'), ui.peers].forEach(function (n) { pane.appendChild(n); });
    return loadMessages(ui);
  }

  window.odRenderBotMessages = odRenderBotMessages;
})();
