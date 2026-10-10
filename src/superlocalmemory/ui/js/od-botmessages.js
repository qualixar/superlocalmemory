// od-botmessages.js — Bot messages pane: what bots sent, and owner controls per peer.
// Public: window.odRenderBotMessages(pane)
// XSS-safe: message bodies, app names, peer ids and server errors are untrusted
// (bots and web chats write them) and are set with textContent only.
// Writes go through the page's fetch, which core.js wraps with the local write credential.
// Routes: GET /api/v3/mesh/messages?limit=&peer=   GET /api/v3/mesh/peers
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

  // ----------------------------------------------------------------- naming
  function shortId(id) { id = String(id == null ? '' : id); return id.length > 8 ? id.slice(0, 8) : id; }

  // Name shown for a peer: its display name, then its app, then a short id.
  function peerName(from) {
    return from.display_name || from.app || shortId(from.peer_id);
  }

  function nameOf(ui, id) {
    var info = ui.info[id];
    return info ? peerName(info) : shortId(id);
  }

  function kindLabel(kind) { return kind === 'web' ? 'web app' : 'this computer'; }

  function kindChip(kind) { return el('span', 'badge neutral od-kind', kindLabel(kind)); }

  // ---------------------------------------------------------------- time
  // Server times carry microseconds ("...51.148946+00:00"); trim to the milliseconds
  // every browser parses.
  function parseTime(iso) {
    if (iso == null || iso === '') return null;
    var t = Date.parse(String(iso).replace(/(\.\d{3})\d+/, '$1'));
    return isNaN(t) ? null : t;
  }

  function relative(ms) {
    var sec = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (sec < 45) return 'just now';
    var min = Math.round(sec / 60);
    if (min < 60) return min + ' min ago';
    var hrs = Math.round(min / 60);
    if (hrs < 24) return hrs + (hrs === 1 ? ' hour ago' : ' hours ago');
    var days = Math.round(hrs / 24);
    if (days < 30) return days + (days === 1 ? ' day ago' : ' days ago');
    return new Date(ms).toLocaleDateString();
  }

  // A <time> with the relative text and the full local time as its title.
  function timeEl(iso, cls) {
    var ms = parseTime(iso);
    if (ms == null) return el('time', cls, iso == null ? '' : String(iso));
    var t = el('time', cls, relative(ms));
    t.setAttribute('title', new Date(ms).toLocaleString());
    t.setAttribute('datetime', new Date(ms).toISOString());
    return t;
  }

  // -------------------------------------------------------------- messages
  function messageItem(ui, m) {
    var from = m.from || {};
    var li = el('li', 'od-msg' + (isUntrusted(m) ? ' untrusted' : ''));
    var head = el('div', 'od-msg-head');
    head.appendChild(el('strong', 'od-msg-from', from.peer_id ? nameOf(ui, from.peer_id) : (from.app || 'unknown')));
    head.appendChild(kindChip(from.kind || (ui.info[from.peer_id] || {}).kind));
    head.appendChild(el('span', 'od-msg-arrow muted', '→'));
    head.appendChild(el('span', 'od-msg-to', m.to == null ? 'everyone' : nameOf(ui, m.to)));
    if (isUntrusted(m)) head.appendChild(el('span', 'badge warn', 'untrusted: written outside this computer'));
    head.appendChild(timeEl(m.sent_at, 'od-msg-time muted'));
    li.appendChild(head);
    li.appendChild(el('p', 'od-msg-body', m.content == null ? '' : m.content));
    return li;
  }

  function openConnectedApps() {
    if (typeof window.slmNavigate === 'function') window.slmNavigate('apps-pane');
  }

  // The same three steps as docs/remote-access/hosts.md: the tick on the approval page
  // is not enough, the key on this computer must allow it too.
  function howItWorks() {
    var wrap = el('div');
    wrap.appendChild(el('h4', null, 'How it works'));
    var ol = el('ol', 'od-botmsg-steps');
    ol.appendChild(el('li', null, 'Connect an app in Connected apps.'));
    ol.appendChild(el('li', null, 'Tick "Allow talking to your other bots" when you approve it.'));
    var third = el('li', null, 'On this computer, switch on "Let these apps message your other bots" in Connected apps, or run: ');
    third.appendChild(el('code', null, 'slm remote keys allow web-<connection id> mesh'));
    third.appendChild(el('span', null, ' (find the name with '));
    third.appendChild(el('code', null, 'slm remote keys list'));
    third.appendChild(el('span', null, ').'));
    ol.appendChild(third);
    wrap.appendChild(ol);
    return wrap;
  }

  function emptyState(ui) {
    var box = el('div', 'od-botmsg-empty');
    if (ui.peer) { box.appendChild(el('p', 'muted', 'No messages from this peer yet.')); return box; }
    box.appendChild(el('p', 'muted', 'Bots you connect can leave each other messages here. ' +
      'Allow it per app in Connected apps.'));
    box.appendChild(howItWorks());
    box.appendChild(F().button('Open Connected apps', 'btn sm primary', openConnectedApps));
    return box;
  }

  function renderMessages(ui, messages) {
    ui.messages = messages;
    ui.list.textContent = '';
    if (!messages.length) return ui.list.appendChild(emptyState(ui));
    var ul = el('ul', 'od-msg-list');
    messages.forEach(function (m) { ul.appendChild(messageItem(ui, m)); });
    ui.list.appendChild(ul);
  }

  // Every connected peer, including ones that have not sent anything. A failure
  // here only means fewer rows; messages still load.
  function loadPeers(ui) {
    return F().api('GET', '/api/v3/mesh/peers').then(function (res) {
      if (res.ok && res.data) addPeers(ui, (res.data.peers || []).map(function (p) { return { from: p }; }));
    }, function () {});
  }

  // Start from a clean peer list so a refresh shows current names and mute state.
  function resetPeers(ui) {
    ui.seen = {};
    ui.info = {};
    ui.peers.textContent = '';
    if (ui.peersHead.parentNode) ui.peersHead.parentNode.removeChild(ui.peersHead);
    while (ui.filter.options.length > 1) ui.filter.remove(1);
  }

  function loadAll(ui) {
    resetPeers(ui);
    return loadPeers(ui).then(function () { return loadMessages(ui); }).then(function () {
      ui.filter.value = ui.peer;
      // The filtered peer is gone (retired): fall back to everyone.
      if (ui.filter.value !== ui.peer) { ui.peer = ''; ui.filter.value = ''; return loadMessages(ui); }
    });
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

  function showMuted(row, muted) {
    row.muted = muted;
    row.toggle.textContent = muted ? 'Unmute' : 'Mute';
    if (muted && !row.tag.parentNode) row.head.appendChild(row.tag);
    if (!muted && row.tag.parentNode) row.head.removeChild(row.tag);
  }

  function toggleMute(ui, row, id) {
    var want = !row.muted;
    F().api('POST', peerUrl(id) + '/mute', { muted: want }).then(function (res) {
      if (!res.ok) return say(row, F().failText(res, 'Could not change mute.'));
      if (ui.info[id]) ui.info[id].muted = want;
      showMuted(row, want);
      say(row, want ? 'Muted.' : 'Unmuted.');
    });
  }

  function renamed(ui, row, id, name) {
    if (ui.info[id]) ui.info[id].display_name = name;
    row.title.textContent = name;
    [].slice.call(ui.filter.options).forEach(function (o) { if (o.value === id) o.textContent = name; });
    if (ui.messages) renderMessages(ui, ui.messages);
  }

  function stopEditing(row) {
    row.edit.parentNode.replaceChild(row.title, row.edit);
    row.actions.insertBefore(row.renameBtn, row.actions.firstChild);
    row.actions.removeChild(row.saveBtn);
    row.actions.removeChild(row.cancelBtn);
  }

  function save(ui, row, id) {
    var problem = nameProblem(row.input.value);
    if (problem) return say(row, problem);
    F().api('PATCH', peerUrl(id), { display_name: row.input.value }).then(function (res) {
      if (!res.ok) return say(row, F().failText(res, 'Could not rename.'));
      var name = (res.data && res.data.display_name) || row.input.value;
      stopEditing(row);
      renamed(ui, row, id, name);
      say(row, 'Renamed.');
    });
  }

  // The name becomes a text field with Save and Cancel; Enter saves, Escape cancels.
  function startEditing(row) {
    row.input.value = row.title.textContent;
    row.title.parentNode.replaceChild(row.edit, row.title);
    row.actions.removeChild(row.renameBtn);
    row.actions.insertBefore(row.cancelBtn, row.actions.firstChild);
    row.actions.insertBefore(row.saveBtn, row.cancelBtn);
    say(row, '');
    row.input.focus();
  }

  function cancelEditing(row) { stopEditing(row); say(row, ''); }

  function retire(ui, row, id) {
    F().confirmThen({
      title: 'Retire peer', target: row.title.textContent.slice(0, 80),
      consequence: 'It can no longer send or receive, and its waiting messages are dropped.',
      confirmLabel: 'Retire',
    }, function () {
      F().api('DELETE', peerUrl(id)).then(function (res) {
        if (!res.ok) return say(row, F().failText(res, 'Could not retire the peer.'));
        loadAll(ui);
      });
    });
  }

  function editField(ui, row, id) {
    var wrap = el('span', 'od-peer-edit');
    row.input = el('input', 'od-peer-input');
    row.input.type = 'text';
    row.input.setAttribute('aria-label', 'New name for ' + nameOf(ui, id));
    row.input.setAttribute('maxlength', '200');
    row.input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); save(ui, row, id); }
      else if (e.key === 'Escape') { e.preventDefault(); cancelEditing(row); }
    });
    wrap.appendChild(row.input);
    return wrap;
  }

  function peerHead(from, row) {
    row.head = el('div', 'od-peer-head');
    row.title = el('strong', 'od-peer-name', peerName(from));
    row.tag = el('span', 'badge warn od-peer-muted', 'muted');
    [row.title, kindChip(from.kind)].forEach(function (n) { row.head.appendChild(n); });
    return row.head;
  }

  function peerMeta(from) {
    var meta = el('div', 'od-peer-meta muted');
    var id = String(from.peer_id == null ? '' : from.peer_id);
    var idEl = el('span', 'od-peer-id', id.length > 12 ? id.slice(0, 12) : id);
    idEl.setAttribute('title', id);
    meta.appendChild(idEl);
    if (parseTime(from.last_seen) != null) {
      meta.appendChild(el('span', null, ' · Last seen '));
      meta.appendChild(timeEl(from.last_seen));
    }
    return meta;
  }

  function peerActions(ui, row, id) {
    row.toggle = F().button('Mute', 'btn sm', function () { toggleMute(ui, row, id); });
    row.renameBtn = F().button('Rename', 'btn sm', function () { startEditing(row); });
    row.saveBtn = F().button('Save', 'btn sm primary', function () { save(ui, row, id); });
    row.cancelBtn = F().button('Cancel', 'btn sm', function () { cancelEditing(row); });
    row.actions = el('div', 'od-peer-actions');
    [row.renameBtn, row.toggle,
     F().button('Retire', 'btn sm od-peer-retire', function () { retire(ui, row, id); })]
      .forEach(function (b) { row.actions.appendChild(b); });
    return row.actions;
  }

  function peerRow(ui, from) {
    var id = from.peer_id;
    var card = el('div', 'od-peer-card');
    card.setAttribute('data-peer', id);
    var row = { note: el('span', 'od-peer-note muted'), muted: false };
    row.note.setAttribute('role', 'status');
    card.appendChild(peerHead(from, row));
    card.appendChild(peerMeta(from));
    row.edit = editField(ui, row, id);
    card.appendChild(peerActions(ui, row, id));
    card.appendChild(row.note);
    showMuted(row, !!from.muted);
    return card;
  }

  // Registers every sender, so names resolve in message headers; the first record
  // for a peer wins, and the peer list (richer than a message's sender block) loads first.
  function addPeers(ui, messages) {
    messages.forEach(function (m) {
      var from = m.from || {};
      if (!from.peer_id || ui.seen[from.peer_id]) return;
      ui.seen[from.peer_id] = true;
      ui.info[from.peer_id] = from;
      if (!ui.peersHead.parentNode) ui.peersBox.insertBefore(ui.peersHead, ui.peers);
      ui.peers.appendChild(peerRow(ui, from));
      var opt = el('option', null, peerName(from));
      opt.value = from.peer_id;
      ui.filter.appendChild(opt);
    });
  }

  // -------------------------------------------------------------------- pane
  var filterSeq = 0;

  function buildFilter(ui) {
    var sel = el('select', 'od-select');
    sel.id = 'od-botmsg-filter-' + (++filterSeq);
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
    var ui = { root: pane, peer: '', seen: {}, info: {}, messages: null, list: el('div'),
      peers: el('div', 'od-peer-grid'), peersHead: el('h3', null, 'Peers'),
      peersBox: el('div', 'od-media-section') };
    ui.peersBox.appendChild(ui.peers);
    ui.filter = buildFilter(ui);
    ui.knownPeers = function (messages) { addPeers(ui, messages); };
    var bar = el('div', 'od-botmsg-bar');
    var label = el('label', 'od-botmsg-label muted', 'Show messages from');
    label.setAttribute('for', ui.filter.id);
    bar.appendChild(label);
    bar.appendChild(ui.filter);
    bar.appendChild(F().button('Refresh', 'btn sm', function () { loadAll(ui); }));
    [head, bar, ui.list, ui.peersBox].forEach(function (n) { pane.appendChild(n); });
    return loadAll(ui);
  }

  window.odRenderBotMessages = odRenderBotMessages;
})();
