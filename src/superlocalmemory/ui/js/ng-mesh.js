// Neural Glass — Mesh Peers Tab
// Real-time view of SLM Mesh P2P network (v3.4.3 Python broker)
// API: /mesh/peers, /mesh/status, /mesh/events, /mesh/state

(function() {
  'use strict';

  var REFRESH_INTERVAL = 5000; // 5 seconds
  var refreshTimer = null;

  // ── Install-token bootstrap (mirrors brain.js) ───────────────
  // Dashboard callers authenticate with the install token so we never
  // embed the mesh secret in JS. Token is fetched from /internal/token
  // (loopback-only endpoint) and cached in sessionStorage for the tab session.
  var TOKEN_STORAGE_KEY = 'slm_install_token';

  function readToken() {
    try { return window.sessionStorage ? window.sessionStorage.getItem(TOKEN_STORAGE_KEY) : null; }
    catch (e) { return null; }
  }

  function writeToken(value) {
    try { if (window.sessionStorage) window.sessionStorage.setItem(TOKEN_STORAGE_KEY, value); }
    catch (e) { /* storage disabled */ }
  }

  function fetchTokenFromServer() {
    return fetch('/internal/token', { credentials: 'same-origin' })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(data) {
        var tok = data && typeof data.token === 'string' ? data.token.trim() : '';
        if (tok) { writeToken(tok); return tok; }
        return null;
      })
      .catch(function() { return null; });
  }

  function ensureToken() {
    var cached = readToken();
    if (cached) return Promise.resolve(cached);
    return fetchTokenFromServer();
  }

  // Authenticated fetch: injects X-Install-Token; retries once on 401
  // (token may have been rotated since the tab was opened).
  function meshFetch(url) {
    return ensureToken().then(function(token) {
      var opts = token ? { headers: { 'X-Install-Token': token }, credentials: 'same-origin' } : {};
      return fetch(url, opts).then(function(r) {
        if (r.status === 401) {
          return fetchTokenFromServer().then(function(freshToken) {
            var retryOpts = freshToken
              ? { headers: { 'X-Install-Token': freshToken }, credentials: 'same-origin' }
              : {};
            return fetch(url, retryOpts);
          });
        }
        return r;
      });
    });
  }

  // ── Public entry point ────────────────────────────────────────
  window.loadMeshPeers = function() {
    fetchMeshStatus();
    fetchMeshPeers();
    fetchMeshEvents();
    fetchMeshState();

    // Auto-refresh while tab is active; stop when pane loses focus
    clearInterval(refreshTimer);
    refreshTimer = setInterval(function() {
      var pane = document.getElementById('mesh-pane');
      if (pane && pane.classList.contains('active')) {
        fetchMeshStatus();
        fetchMeshPeers();
        fetchMeshEvents();
        fetchMeshState();
      } else {
        // Pane is no longer active — stop the timer to avoid background polling
        clearInterval(refreshTimer);
        refreshTimer = null;
      }
    }, REFRESH_INTERVAL);
  };

  // Try BOTH brokers: daemon (port 8765 /mesh/*) AND standalone slm-mesh (port 7899 /*)
  var STANDALONE_PORT = null;

  function fetchStandaloneBroker(path) {
    var ports = [7899];
    return fetch('http://127.0.0.1:' + ports[0] + path, { signal: AbortSignal.timeout(2000) })
      .then(function(r) {
        if (r.ok) { STANDALONE_PORT = ports[0]; return r.json(); }
        STANDALONE_PORT = null;
        return null;
      })
      .catch(function() { STANDALONE_PORT = null; return null; });
  }

  function fetchMeshStatus() {
    Promise.all([
      meshFetch('/mesh/status').then(function(r) {
        if (!r.ok) return r.status === 401 ? { _auth_error: true } : null;
        return r.json();
      }).catch(function() { return null; }),
      fetchStandaloneBroker('/health')
    ]).then(function(results) {
      var daemon = results[0];
      if (daemon && daemon._auth_error) { renderMeshStatusAuthError(); return; }
      renderMeshStatus(daemon, results[1]);
    });
  }

  function fetchMeshPeers() {
    Promise.all([
      meshFetch('/mesh/peers').then(function(r) {
        if (!r.ok) return r.status === 401 ? { _auth_error: true } : { peers: [] };
        return r.json();
      }).catch(function() { return { peers: [] }; }),
      fetchStandaloneBroker('/peers')
    ]).then(function(results) {
      var daemonResult = results[0];
      if (daemonResult && daemonResult._auth_error) { renderMeshPeersAuthError(); return; }
      var daemonPeers = (daemonResult && daemonResult.peers) || [];
      var standalonePeers = (results[1] && (results[1].peers || results[1])) || [];
      if (!Array.isArray(standalonePeers)) standalonePeers = [];
      var seen = {};
      var allPeers = [];
      daemonPeers.concat(standalonePeers).forEach(function(p) {
        var id = p.peer_id || p.id || JSON.stringify(p);
        if (!seen[id]) { seen[id] = true; allPeers.push(p); }
      });
      renderMeshPeers(allPeers);
    }).catch(function() {
      var el = document.getElementById('mesh-peers-list');
      if (el) el.innerHTML =
        '<div class="text-center" style="padding:24px;color:var(--ng-text-tertiary)">' +
          '<i class="bi bi-wifi-off" style="font-size:2rem;display:block;margin-bottom:8px"></i>' +
          'Mesh broker not reachable' +
        '</div>';
    });
  }

  function fetchMeshEvents() {
    meshFetch('/mesh/events').then(function(r) {
      if (!r.ok) return { events: [] };
      return r.json();
    }).then(function(data) {
      renderMeshEvents(data.events || data || []);
    }).catch(function() {});
  }

  function fetchMeshState() {
    meshFetch('/mesh/state').then(function(r) {
      if (!r.ok) return { state: {} };
      return r.json();
    }).then(function(data) {
      renderMeshState(data.state || data || {});
    }).catch(function() {});
  }

  // ── Auth-error renderers ──────────────────────────────────────
  function renderMeshStatusAuthError() {
    var el = document.getElementById('mesh-status-cards');
    if (!el) return;
    el.innerHTML =
      '<div class="row g-3">' +
        statusCard('Status', statusDot('error') + ' Auth required', 'bi-lock') +
        statusCard('Peers', '—', 'bi-people') +
        statusCard('Uptime', '—', 'bi-clock') +
        statusCard('Brokers', '—', 'bi-hdd-stack') +
      '</div>' +
      '<div style="font-size:0.75rem;color:var(--ng-status-error);margin-top:8px;text-align:center">' +
        'Mesh endpoints returned 401. The daemon may be starting — retrying automatically.' +
      '</div>';
  }

  function renderMeshPeersAuthError() {
    var el = document.getElementById('mesh-peers-list');
    if (!el) return;
    el.innerHTML =
      '<div class="text-center" style="padding:32px;color:var(--ng-text-tertiary)">' +
        '<i class="bi bi-lock" style="font-size:2.5rem;display:block;margin-bottom:12px;opacity:0.5"></i>' +
        '<div style="font-size:0.9375rem;margin-bottom:4px">Authentication required</div>' +
        '<div style="font-size:0.8125rem">Could not authenticate with the mesh broker. ' +
        'The dashboard token may still be loading.</div>' +
      '</div>';
  }

  function renderMeshStatus(daemonData, standaloneData) {
    var el = document.getElementById('mesh-status-cards');
    if (!el) return;

    // Daemon broker (Python, integrated)
    var daemonUp = daemonData && daemonData.broker_up === true;
    var daemonUptime = daemonData ? (daemonData.uptime_s || 0) : 0;
    var daemonPeers = daemonData ? (daemonData.peer_count || 0) : 0;

    // Standalone broker (TypeScript, slm-mesh npm)
    var standaloneUp = standaloneData && standaloneData.status === 'ok';
    var standaloneUptime = standaloneData ? (standaloneData.uptime || 0) : 0;
    var standaloneVersion = standaloneData ? (standaloneData.version || '') : '';

    var anyUp = daemonUp || standaloneUp;
    var statusText = anyUp ? 'Active' : 'Offline';
    var statusKey = anyUp ? 'active' : 'dead';
    var bestUptime = Math.max(daemonUptime, standaloneUptime);

    // Combined info
    var brokerInfo = [];
    if (daemonUp) brokerInfo.push('Daemon (Python)');
    if (standaloneUp) brokerInfo.push('slm-mesh ' + standaloneVersion);

    el.innerHTML =
      '<div class="row g-3">' +
        statusCard('Broker', statusDot(statusKey) + ' ' + statusText, 'bi-wifi') +
        statusCard('Peers', daemonPeers, 'bi-people') +
        statusCard('Uptime', formatUptime(bestUptime), 'bi-clock') +
        statusCard('Brokers', brokerInfo.length, 'bi-hdd-stack') +
      '</div>' +
      '<div style="font-size:0.75rem;color:var(--ng-text-quaternary);margin-top:8px;text-align:center">' +
        (brokerInfo.length > 0 ? 'Running: ' + brokerInfo.join(' + ') : 'No brokers detected') +
        (standaloneUp ? ' (port 7899)' : '') +
        ' · Peers register via <code>mesh_summary</code> MCP tool and expire after 60s without heartbeat' +
      '</div>';
  }

  function renderMeshStatusOffline() {
    var el = document.getElementById('mesh-status-cards');
    if (!el) return;
    el.innerHTML =
      '<div class="row g-3">' +
        statusCard('Status', statusDot('dead') + ' Offline', 'bi-wifi-off') +
        statusCard('Peers', '0', 'bi-people') +
        statusCard('Messages', '0', 'bi-chat-dots') +
        statusCard('State Keys', '0', 'bi-database') +
      '</div>';
  }

  function statusCard(label, value, icon) {
    return '<div class="col-md-3 col-6">' +
      '<div class="ng-glass" style="padding:16px;text-align:center">' +
        '<i class="bi ' + icon + '" style="font-size:1.25rem;color:var(--ng-accent);display:block;margin-bottom:8px"></i>' +
        '<div class="ng-stat-value" style="font-size:1.5rem">' + value + '</div>' +
        '<div class="ng-stat-label">' + label + '</div>' +
      '</div>' +
    '</div>';
  }

  function statusDot(status) {
    var color = 'var(--ng-text-quaternary)';
    if (status === 'active' || status === 'running' || status === 'ok') color = 'var(--ng-status-success)';
    else if (status === 'stale') color = 'var(--ng-status-warning)';
    else if (status === 'dead' || status === 'error') color = 'var(--ng-status-error)';
    return '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + ';box-shadow:0 0 6px ' + color + ';margin-right:4px"></span>';
  }

  function renderMeshPeers(peers) {
    var el = document.getElementById('mesh-peers-list');
    if (!el) return;

    if (!peers || peers.length === 0) {
      el.innerHTML =
        '<div class="text-center" style="padding:32px;color:var(--ng-text-tertiary)">' +
          '<i class="bi bi-people" style="font-size:2.5rem;display:block;margin-bottom:12px;opacity:0.3"></i>' +
          '<div style="font-size:0.9375rem;margin-bottom:4px">No peers connected</div>' +
          '<div style="font-size:0.8125rem">Start another Claude Code session to see it appear here.<br>' +
          'Peers register via <code>mesh_summary</code> MCP tool.</div>' +
        '</div>';
      return;
    }

    var html = '<div class="row g-3">';
    peers.forEach(function(peer) {
      var st = peer.status || 'unknown';
      var ago = timeAgo(peer.last_heartbeat);
      html +=
        '<div class="col-md-6 col-lg-4">' +
          '<div class="ng-glass" style="padding:16px;transition:border-color 150ms">' +
            '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">' +
              '<div>' +
                '<div style="font-weight:590;color:var(--ng-text-primary);font-size:0.9375rem">' +
                  escapeHtml(peer.session_id || peer.peer_id || 'Unknown') +
                '</div>' +
                '<div style="font-size:0.75rem;color:var(--ng-text-tertiary)">' +
                  escapeHtml(peer.peer_id || '') +
                '</div>' +
              '</div>' +
              '<span class="ng-badge ng-badge-' + statusBadgeClass(st) + '">' +
                statusDot(st) + ' ' + capitalize(st) +
              '</span>' +
            '</div>' +
            '<div style="font-size:0.8125rem;color:var(--ng-text-secondary);margin-bottom:8px;min-height:36px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">' +
              escapeHtml(peer.summary || 'No summary provided') +
            '</div>' +
            '<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--ng-text-quaternary)">' +
              '<span><i class="bi bi-geo-alt"></i> ' + escapeHtml((peer.host || '127.0.0.1') + ':' + (peer.port || '?')) + '</span>' +
              '<span><i class="bi bi-clock"></i> ' + ago + '</span>' +
            '</div>' +
          '</div>' +
        '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
  }

  function renderMeshEvents(events) {
    var el = document.getElementById('mesh-events-list');
    if (!el) return;

    if (!events || events.length === 0) {
      el.innerHTML = '<div style="padding:16px;color:var(--ng-text-tertiary);text-align:center;font-size:0.8125rem">No events yet</div>';
      return;
    }

    var html = '<div style="max-height:300px;overflow-y:auto;font-size:0.8125rem">';
    events.slice(-20).reverse().forEach(function(ev) {
      html +=
        '<div style="padding:8px 12px;border-bottom:1px solid var(--ng-border-subtle);display:flex;gap:12px;align-items:flex-start">' +
          '<span style="color:var(--ng-text-quaternary);font-size:0.75rem;white-space:nowrap;min-width:70px">' +
            formatTime(ev.created_at || ev.timestamp) +
          '</span>' +
          '<span style="color:var(--ng-text-secondary)">' + escapeHtml(ev.content || ev.event || JSON.stringify(ev)) + '</span>' +
        '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
  }

  function renderMeshState(state) {
    var el = document.getElementById('mesh-state-list');
    if (!el) return;

    var keys = Object.keys(state);
    if (keys.length === 0) {
      el.innerHTML = '<div style="padding:16px;color:var(--ng-text-tertiary);text-align:center;font-size:0.8125rem">No shared state</div>';
      return;
    }

    var html = '<div style="font-size:0.8125rem">';
    keys.forEach(function(key) {
      var entry = state[key];
      var val = typeof entry === 'object' ? (entry.value || JSON.stringify(entry)) : entry;
      var setBy = typeof entry === 'object' ? (entry.set_by || '') : '';
      html +=
        '<div style="padding:8px 12px;border-bottom:1px solid var(--ng-border-subtle);display:flex;justify-content:space-between;align-items:center">' +
          '<div>' +
            '<code style="color:var(--ng-accent)">' + escapeHtml(key) + '</code>' +
            (setBy ? '<span style="color:var(--ng-text-quaternary);font-size:0.75rem;margin-left:8px">by ' + escapeHtml(setBy) + '</span>' : '') +
          '</div>' +
          '<div style="color:var(--ng-text-secondary);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
            escapeHtml(String(val).substring(0, 100)) +
          '</div>' +
        '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
  }

  // ── Helpers ────────────────────────────────────────────────

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  function capitalize(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
  }

  function formatUptime(sec) {
    if (!sec || sec <= 0) return 'N/A';
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    if (h > 24) return Math.floor(h / 24) + 'd ' + (h % 24) + 'h';
    if (h > 0) return h + 'h ' + m + 'm';
    return m + 'm';
  }

  function statusBadgeClass(status) {
    if (status === 'active' || status === 'running') return 'success';
    if (status === 'stale') return 'warning';
    if (status === 'dead' || status === 'error') return 'error';
    return 'neutral';
  }

  function timeAgo(isoStr) {
    if (!isoStr) return 'unknown';
    var diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
    if (diff < 60) return Math.floor(diff) + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
  }

  function formatTime(isoStr) {
    if (!isoStr) return '';
    try {
      var d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch(e) { return ''; }
  }

  // Cleanup on tab switch
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
      clearInterval(refreshTimer);
    }
  });

})();
