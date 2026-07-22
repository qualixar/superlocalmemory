// SuperLocalMemory — od-mesh.js  v2.0
// OD-styled Mesh Peers pane. Self-contained, CSP-safe.
// Renders into the "mesh-pane" tab container via the shell's triggerTabLoad.
// Exposes window.odRenderMesh(container).
//
// HTML structure mirrors approved design (mesh-peers.html <main class="content">).
// Design-system CSS classes used throughout — no invented inline colours.
//
// Endpoints (all via meshFetch → X-Install-Token):
//   GET  /mesh/status   → {broker_up, peer_count, uptime_s}
//   GET  /mesh/peers    → {peers:[{peer_id, session_id, status, last_heartbeat,
//                                   summary, host, port, agent_type, project_path}]}
//   POST /mesh/send     → body:{from_peer, to_peer, content, type} → {ok:true}
//
// JSON keys mapped:
//   status  : broker_up(bool), peer_count(int), uptime_s(int)
//   peer    : peer_id, session_id, summary, status, host, port,
//             agent_type, last_heartbeat, project_path
//   send OK : ok(bool) | send ERR: detail(str)
//
// CRIT fixes baked in:
//   C1 — send guarded: button disabled when no peer selected or peer is not active
//   C2 — fetch errors surface in conversation log; no silent broken state
//   C3 — all API strings through escapeHtml() before innerHTML assignment
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

(function () {
  'use strict';

  var REFRESH_MS = 5000;
  var SEND_FROM  = 'dashboard';

  // ── Token helpers (mirrors ng-mesh.js; never stored in sessionStorage) ────
  var _tokenCache = null;

  function fetchTokenFromServer() {
    return fetch('/internal/token', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var tok = d && typeof d.token === 'string' ? d.token.trim() : '';
        if (tok) _tokenCache = tok;
        return tok || null;
      })
      .catch(function () { return null; });
  }

  function ensureToken() {
    return _tokenCache ? Promise.resolve(_tokenCache) : fetchTokenFromServer();
  }

  // Authenticated fetch; retries once on 401 (stale token).
  function meshFetch(url, opts) {
    return ensureToken().then(function (token) {
      var base = Object.assign({ credentials: 'same-origin' }, opts || {});
      if (token) {
        base.headers = Object.assign({ 'X-Install-Token': token }, base.headers || {});
      }
      return fetch(url, base).then(function (r) {
        if (r.status !== 401) return r;
        return fetchTokenFromServer().then(function (fresh) {
          var retry = Object.assign({ credentials: 'same-origin' }, opts || {});
          if (fresh) {
            retry.headers = Object.assign({ 'X-Install-Token': fresh }, retry.headers || {});
          }
          return fetch(url, retry);
        });
      });
    });
  }

  // ── Module state ──────────────────────────────────────────────────────────
  var _peers      = [];       // last-fetched peer list
  var _selectedId = null;     // peer_id currently shown in conversation panel
  var _convLogs   = {};       // peer_id → [{role:'q'|'a', text}]
  var _container  = null;
  var _timer      = null;
  var _observer   = null;     // MutationObserver — re-starts refresh on tab activate

  // ── HTML helpers ──────────────────────────────────────────────────────────
  // CRIT C3: every API-derived string MUST pass through this.
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function badgeCls(st) {
    return st === 'active' ? 'ok' : st === 'stale' ? 'warn' : 'danger';
  }

  function timeAgo(iso) {
    if (!iso) return 'unknown';
    var d = (Date.now() - new Date(iso).getTime()) / 1000;
    if (d < 5)     return 'now';
    if (d < 60)    return Math.floor(d) + 's';
    if (d < 3600)  return Math.floor(d / 60) + 'm';
    if (d < 86400) return Math.floor(d / 3600) + 'h';
    return Math.floor(d / 86400) + 'd';
  }

  function fmtUptime(sec) {
    if (!sec) return '—';
    var h = Math.floor(sec / 3600);
    if (h > 24) return Math.floor(h / 24) + 'd';
    var m = Math.floor((sec % 3600) / 60);
    return h > 0 ? h + 'h ' + m + 'm' : m + 'm';
  }

  function sendSvg() {
    // Use shell's slmIcon if available; inline fallback for safety.
    if (typeof window.slmIcon === 'function') return window.slmIcon('send');
    return '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor"' +
      ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<line x1="22" y1="2" x2="11" y2="13"/>' +
      '<polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
  }

  // ── Layout HTML ───────────────────────────────────────────────────────────
  // Mirrors <main class="content"> from approved design (mesh-peers.html).
  function buildHTML() {
    return (
      '<div id="od-mesh-root">' +

        // Page head — description updated live from /mesh/status
        '<div class="page-head">' +
          '<h2>Mesh peers</h2>' +
          '<p id="od-mesh-page-desc">Loading mesh status…</p>' +
        '</div>' +

        // Two-column: peer cards (left) + conversation panel (right)
        '<div style="display:grid;grid-template-columns:1.15fr 1fr;gap:20px;align-items:start">' +

          // LEFT — peer card grid
          '<div>' +
            '<div class="launch-grid" id="od-mesh-peers-grid"' +
              ' style="grid-template-columns:1fr 1fr">' +
              '<div id="od-mesh-peers-ph"' +
                ' style="grid-column:1/-1;text-align:center;padding:40px;color:var(--fg-2)">' +
                '<div style="font-size:15px;margin-bottom:6px">Loading peers…</div>' +
              '</div>' +
            '</div>' +
          '</div>' +

          // RIGHT — conversation panel (sticky, full-height card)
          '<div class="card"' +
            ' style="position:sticky;top:86px;display:flex;flex-direction:column;' +
                    'height:calc(100vh - 120px)">' +

            // Card header: selected peer info
            '<div class="card-head">' +
              '<span class="avatar" id="od-mesh-cv-avatar"' +
                ' style="width:30px;height:30px">?</span>' +
              '<div>' +
                '<h3 id="od-mesh-cv-name">No peer selected</h3>' +
                '<span class="sub" id="od-mesh-cv-meta">' +
                  'Click a peer card to start messaging' +
                '</span>' +
              '</div>' +
              '<div class="spacer"></div>' +
              '<span class="badge neutral" id="od-mesh-cv-badge">—</span>' +
            '</div>' +

            // Conversation log
            '<div class="ask-log" id="od-mesh-cv-log"' +
              ' style="flex:1;max-height:none;padding:16px;display:flex;flex-direction:column;gap:10px">' +
              '<div style="text-align:center;color:var(--fg-3);font-size:13px;margin-top:30px">' +
                'Select a peer to start messaging' +
              '</div>' +
            '</div>' +

            // Input row — C1: disabled until active peer selected
            '<div class="ask-input" style="border-top:1px solid var(--border)">' +
              '<input id="od-mesh-cv-input" disabled autocomplete="off"' +
                ' placeholder="Select a peer to start messaging">' +
              '<button id="od-mesh-cv-send" disabled aria-label="Send message">' +
                sendSvg() +
              '</button>' +
            '</div>' +

          '</div>' + // end conversation panel

        '</div>' + // end two-column
      '</div>'    // od-mesh-root
    );
  }

  // ── Render: page-head description from /mesh/status ───────────────────────
  function renderPageDesc(data) {
    var el = document.getElementById('od-mesh-page-desc');
    if (!el) return;
    if (!data) {
      el.textContent = 'Agents sharing this memory mesh on your machine and LAN.' +
        ' Peers go stale after 5 min without a heartbeat, dead after 30.';
      return;
    }
    var up  = data.broker_up === true ? 'Broker up' : 'Broker offline';
    var cnt = data.peer_count != null
      ? data.peer_count + ' peer' + (data.peer_count === 1 ? '' : 's')
      : '';
    var upt = data.uptime_s != null ? 'uptime ' + fmtUptime(data.uptime_s) : '';
    var parts = [up];
    if (cnt) parts.push(cnt);
    if (upt) parts.push(upt);
    el.textContent = 'Agents sharing this memory mesh on your machine and LAN. ' +
      parts.join(' · ') + '. Peers go stale after 5 min without a heartbeat, dead after 30.';
  }

  // ── Render: peer cards ────────────────────────────────────────────────────
  function meshEmptyIcon() {
    if (typeof window.slmIcon === 'function') return window.slmIcon('mesh');
    return '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor"' +
      ' stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
      '<circle cx="12" cy="12" r="2.5"/><circle cx="5" cy="5" r="2"/>' +
      '<circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/>' +
      '<circle cx="19" cy="19" r="2"/>' +
      '<path d="M10 10L6.5 6.5M14 10l3.5-3.5M10 14l-3.5 3.5M14 14l3.5 3.5"/></svg>';
  }

  function renderPeers(peers) {
    var grid = document.getElementById('od-mesh-peers-grid');
    if (!grid) return;

    if (!peers || peers.length === 0) {
      // Design pattern: proper launch-card in the grid, not bare centered text
      grid.innerHTML =
        '<div class="card launch-card"' +
          ' style="grid-column:1/-1;display:flex;flex-direction:column;' +
          'align-items:center;justify-content:center;' +
          'padding:48px 24px;text-align:center;cursor:default">' +
          '<div style="width:44px;height:44px;border-radius:12px;' +
            'background:var(--card-2);display:flex;align-items:center;' +
            'justify-content:center;margin-bottom:16px;color:var(--fg-3)">' +
            meshEmptyIcon() +
          '</div>' +
          '<h3 style="font-size:15px;margin-bottom:6px">No peers connected</h3>' +
          '<p style="font-size:12.5px;color:var(--fg-2);' +
            'max-width:30ch;line-height:1.5">' +
            'Start another agent session. Peers register via the' +
            ' <code>mesh_summary</code> MCP tool.' +
          '</p>' +
        '</div>';
      return;
    }

    var html = '';
    peers.forEach(function (p) {
      var st       = p.status || 'unknown';
      var name     = escapeHtml(p.session_id || p.peer_id || 'Unknown');
      var pidFrag  = escapeHtml((p.peer_id || '').slice(0, 12));
      var summary  = escapeHtml(p.summary || 'No summary');
      var atype    = escapeHtml(p.agent_type || 'unknown');
      var ago      = escapeHtml(timeAgo(p.last_heartbeat));
      var proj     = escapeHtml(p.project_path || '~');
      var port     = p.port ? ' · :' + p.port : '';
      var letter   = (p.session_id || p.peer_id || '?').charAt(0).toUpperCase();
      var avatarBg = p.agent_type === 'mcp' ? 'var(--violet)' : 'var(--cyan)';
      var pid      = escapeHtml(p.peer_id || '');
      var toolCnt  = Number(p.tool_count || 0);
      var memCnt   = Number(p.memory_count || 0);
      // Latency — API may return latency_ms; dead peers always show '—'
      var latencyStr = p.latency_ms != null
        ? escapeHtml(String(p.latency_ms)) + ' ms'
        : (st === 'dead' ? '—' : '—');

      html +=
        '<div class="card launch-card" data-peer-id="' + pid + '"' +
          ' role="button" tabindex="0" aria-label="Select peer ' + name + '">' +
          '<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px">' +
            '<span class="avatar" style="background:' + avatarBg + '">' + letter + '</span>' +
            '<div style="flex:1;min-width:0">' +
              '<h3 style="font-size:15px;white-space:nowrap;overflow:hidden;' +
                'text-overflow:ellipsis">' + name + '</h3>' +
              '<span class="mono dim" style="font-size:11px">' + pidFrag + '</span>' +
            '</div>' +
            '<span class="badge ' + badgeCls(st) + '">' +
              '<span class="dot"></span>' + escapeHtml(st) +
            '</span>' +
          '</div>' +
          '<p style="font-size:12.5px;color:var(--fg-2);margin-bottom:12px;' +
            'overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;' +
            '-webkit-box-orient:vertical">' + summary + '</p>' +
          // Design: proto / latency / last  (three items)
          '<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px" class="mono">' +
            '<span><span class="dim">proto</span> ' + atype + '</span>' +
            '<span><span class="dim">memories</span> ' + memCnt + '</span>' +
            '<span><span class="dim">tools</span> ' + toolCnt + '</span>' +
            '<span><span class="dim">last</span> ' + ago + '</span>' +
          '</div>' +
          '<div class="mono dim" style="font-size:11px;margin-top:8px">' +
            proj + port +
          '</div>' +
        '</div>';
    });
    grid.innerHTML = html;
  }

  // ── Peer selection: update conversation panel header + input state ─────────
  function selectPeerById(pid) {
    _selectedId = pid || null;
    var peer = pid ? _peers.find(function (p) { return p.peer_id === pid; }) : null;

    var avatarEl = document.getElementById('od-mesh-cv-avatar');
    var nameEl   = document.getElementById('od-mesh-cv-name');
    var metaEl   = document.getElementById('od-mesh-cv-meta');
    var badgeEl  = document.getElementById('od-mesh-cv-badge');
    var inputEl  = document.getElementById('od-mesh-cv-input');
    var sendEl   = document.getElementById('od-mesh-cv-send');

    if (!peer) {
      if (avatarEl) { avatarEl.textContent = '?'; avatarEl.style.background = ''; }
      if (nameEl)   nameEl.textContent  = 'No peer selected';
      if (metaEl)   metaEl.textContent  = 'Click a peer card to start messaging';
      if (badgeEl)  { badgeEl.className = 'badge neutral'; badgeEl.textContent = '—'; }
      // CRIT C1: disable when no peer
      if (inputEl)  { inputEl.disabled = true; inputEl.placeholder = 'Select a peer to start messaging'; }
      if (sendEl)   { sendEl.disabled = true; sendEl.style.opacity = '0.4'; }
      renderConv();
      return;
    }

    var st      = peer.status || 'unknown';
    // CRIT C1: dead peers are gone from the broker — disable send.
    // Stale peers are still registered and can queue messages (matches design behaviour).
    var canSend = (st !== 'dead');
    var atype   = peer.agent_type || 'unknown';
    var letter  = (peer.session_id || peer.peer_id || '?').charAt(0).toUpperCase();
    var avatarBg = atype === 'mcp' ? 'var(--violet)' : 'var(--cyan)';

    if (avatarEl) { avatarEl.textContent = letter; avatarEl.style.background = avatarBg; }
    if (nameEl)   nameEl.textContent = peer.session_id || peer.peer_id || 'Unknown';
    if (metaEl)   metaEl.textContent =
      atype + ' · ' + st +
      (peer.host && peer.port ? ' · ' + peer.host + ':' + peer.port : '');
    if (badgeEl) {
      badgeEl.className = 'badge ' + badgeCls(st);
      badgeEl.innerHTML = '<span class="dot"></span> ' + escapeHtml(st);
    }
    if (inputEl) {
      inputEl.disabled    = !canSend;
      inputEl.placeholder = st === 'dead'
        ? 'Peer is unreachable'
        : 'Message ' + escapeHtml(peer.session_id || 'peer') + '…';
    }
    if (sendEl) {
      sendEl.disabled      = !canSend;
      sendEl.style.opacity = canSend ? '1' : '0.4';
    }
    renderConv();
  }

  // ── Render: conversation log ──────────────────────────────────────────────
  function renderConv() {
    var log = document.getElementById('od-mesh-cv-log');
    if (!log) return;
    var msgs = _selectedId ? (_convLogs[_selectedId] || []) : [];
    if (msgs.length === 0) {
      log.innerHTML =
        '<div style="text-align:center;color:var(--fg-3);font-size:13px;margin-top:30px">' +
          (_selectedId
            ? 'No messages yet — type below to send'
            : 'Select a peer to start messaging') +
        '</div>';
      return;
    }
    var html = '';
    msgs.forEach(function (m) {
      // .msg.q and .msg.a are defined in design-system.css
      html += '<div class="msg ' + m.role + '">' + escapeHtml(m.text) + '</div>';
    });
    log.innerHTML = html;
    log.scrollTop = log.scrollHeight;
  }

  // ── Send message to peer via POST /mesh/send ──────────────────────────────
  function sendMessage() {
    // CRIT C1: bail immediately if no peer selected
    if (!_selectedId) return;
    var inputEl = document.getElementById('od-mesh-cv-input');
    if (!inputEl) return;
    var text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';

    // Optimistic echo
    if (!_convLogs[_selectedId]) _convLogs[_selectedId] = [];
    _convLogs[_selectedId].push({ role: 'q', text: text });
    renderConv();

    var peerId = _selectedId; // capture before any async re-assign
    meshFetch('/mesh/send', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        from_peer: SEND_FROM,
        to_peer:   peerId,
        content:   text,
        type:      'text'
      })
    })
    .then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, data: d }; });
    })
    .then(function (res) {
      // CRIT C2: always surface errors — never silent
      var ack = (res.ok && res.data && res.data.ok)
        ? 'Delivered — queued in mesh (TTL 48h).'
        : 'Error: ' + escapeHtml(
            (res.data && res.data.detail) ? String(res.data.detail) : 'unknown error'
          );
      _convLogs[peerId].push({ role: 'a', text: ack });
      renderConv();
    })
    .catch(function () {
      // CRIT C2: network failure visible in log
      _convLogs[peerId].push({ role: 'a', text: 'Network error — could not reach broker.' });
      renderConv();
    });
  }

  // ── Data loaders ──────────────────────────────────────────────────────────
  function loadAll() {
    meshFetch('/mesh/status')
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; })
      .then(renderPageDesc);

    meshFetch('/mesh/peers')
      .then(function (r) { return r.ok ? r.json() : { peers: [] }; })
      .catch(function () { return { peers: [] }; })
      .then(function (d) {
        _peers = Array.isArray(d && d.peers) ? d.peers : [];
        renderPeers(_peers);
        // If the currently selected peer has expired, deselect
        if (_selectedId && !_peers.find(function (p) { return p.peer_id === _selectedId; })) {
          selectPeerById(null);
        }
      });
  }

  // ── Event wiring (CSP-safe delegation — no onXxx= attributes) ─────────────
  function wireEvents(container) {
    // Peer card click → select for conversation
    var grid = container.querySelector('#od-mesh-peers-grid');
    if (grid) {
      grid.addEventListener('click', function (e) {
        var card = e.target.closest('[data-peer-id]');
        if (!card) return;
        var pid = card.getAttribute('data-peer-id');
        if (pid) selectPeerById(pid);
      });
      grid.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        var card = e.target.closest('[data-peer-id]');
        if (!card) return;
        e.preventDefault();
        var pid = card.getAttribute('data-peer-id');
        if (pid) selectPeerById(pid);
      });
    }

    // Send button
    var sendBtn = container.querySelector('#od-mesh-cv-send');
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    // Enter key in input
    var inputEl = container.querySelector('#od-mesh-cv-input');
    if (inputEl) {
      inputEl.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
      });
    }
  }

  // ── Refresh management: MutationObserver re-starts on tab activate ────────
  function startRefreshObserver() {
    var pane = document.getElementById('mesh-pane');
    if (!pane || _observer) return;
    _observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        if (m.type !== 'attributes' || m.attributeName !== 'class') return;
        var t = m.target;
        if (t.classList.contains('active') && t.classList.contains('show')) {
          loadAll();
          if (_timer) clearInterval(_timer);
          _timer = setInterval(function () {
            if (t.classList.contains('active')) {
              loadAll();
            } else {
              clearInterval(_timer);
              _timer = null;
            }
          }, REFRESH_MS);
        }
      });
    });
    _observer.observe(pane, { attributes: true });
  }

  // ── Public API ────────────────────────────────────────────────────────────
  window.odRenderMesh = function (container) {
    if (!container) return;
    _container = container;

    // Idempotent: if already rendered, only reload data
    if (container.querySelector('#od-mesh-root')) {
      loadAll();
      return;
    }

    container.innerHTML = buildHTML();
    wireEvents(container);
    loadAll();
    startRefreshObserver();

    // Initial poll timer (self-stops when pane becomes inactive)
    if (_timer) clearInterval(_timer);
    _timer = setInterval(function () {
      var pane = document.getElementById('mesh-pane');
      if (pane && pane.classList.contains('active')) {
        loadAll();
      } else {
        clearInterval(_timer);
        _timer = null;
      }
    }, REFRESH_MS);
  };

  // Auto-run at DOMContentLoaded if the pane already exists in DOM
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('mesh-pane');
    if (pane) window.odRenderMesh(pane);
  });

}());
