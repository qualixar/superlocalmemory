// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// Existing dashboard initiation. Local API/installation auth owns enrollment;
// this card never receives a connector key or changes transport configuration.
(function () {
  'use strict';
  var HOSTS = { muse: 'Musebot', chatgpt: 'ChatGPT Web', claude_web: 'Claude Web', claude_code_web: 'Claude Code Web', composio: 'Composio', other_mcp: 'Other MCP client' };
  function node(tag, text, cls) { var el = document.createElement(tag); if (text) el.textContent = text; if (cls) el.className = cls; return el; }
  function call(path, init) {
    var fetcher = typeof window.slmFetch === 'function' ? window.slmFetch : window.fetch;
    var options = Object.assign({ credentials: 'same-origin' }, init || {});
    return fetcher(path, options).then(function (response) { if (!response.ok) throw new Error('connection request unavailable'); return response.json(); });
  }
  function safeSignIn(value, connectionId) {
    try {
      var url = new URL(value);
      if (url.protocol !== 'https:' || url.hostname !== 'auth.superlocalmemory.com' || url.port || url.username || url.password || url.hash) return null;
      if (url.pathname !== '/owner-login' || url.href.length > 2048) return null;
      var parameters = Array.from(url.searchParams.entries());
      if (parameters.length !== 1 || parameters[0][0] !== 'connection_id' || parameters[0][1] !== connectionId) return null;
      return url.href;
    } catch (_) { return null; }
  }
  function operationKey() { var bytes = new Uint8Array(16); window.crypto.getRandomValues(bytes); return Array.from(bytes, function (x) { return x.toString(16).padStart(2, '0'); }).join(''); }
  window.odCreateAiConnectionsCard = function () {
    var card = node('section', '', 'card'); card.id = 'od-ai-connections'; card.style.marginBottom = '16px';
    var head = node('div', '', 'card-head'); head.appendChild(node('h3', 'Connect your AI')); card.appendChild(head);
    var body = node('div', '', 'card-pad'); card.appendChild(body);
    body.appendChild(node('p', 'Use this computer’s memories in your AI. You choose whether it can save new memories.'));
    body.appendChild(node('p', 'Remote access starts off. Your database stays on this computer, which must be online.', 'od-mcp-intro'));
    var status = node('p', 'Checking connection availability…'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); body.appendChild(status);
    var journey = node('ol', '', 'od-ai-journey');
    ['Choose your AI', 'Link this computer', 'Verify connection', 'Connect your AI'].forEach(function (step) { journey.appendChild(node('li', step)); }); body.appendChild(journey);
    var list = node('div'); body.appendChild(list);
    var add = node('button', 'Add AI connection', 'btn ghost sm'); add.hidden = true; add.type = 'button'; add.disabled = true; body.appendChild(add);
    var refresh = node('button', 'Refresh status', 'btn ghost sm'); refresh.type = 'button'; refresh.style.marginLeft = '8px'; body.appendChild(refresh);
    var form = node('form'); form.hidden = false; form.style.marginTop = '12px'; body.appendChild(form);
    form.appendChild(node('h4', 'Choose your AI'));
    var label = node('label', 'Choose your AI'); label.htmlFor = 'od-connection-host'; label.hidden = true; form.appendChild(label);
    var host = node('select'); host.hidden = true; host.id = 'od-connection-host'; host.style.display = 'block'; host.style.marginBottom = '12px'; host.style.cssText += ';padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--card-2);color:var(--fg-1);min-width:200px;max-width:100%'; form.appendChild(host);
    var clients = node('div', '', 'od-ai-clients'); clients.setAttribute('role', 'group'); clients.setAttribute('aria-label', 'AI services'); form.insertBefore(clients, host);
    var help = node('p', '', 'od-mcp-intro'); form.appendChild(help);
    function selectedClient() {
      Array.from(clients.children).forEach(function (button) { button.setAttribute('aria-pressed', String(button.getAttribute('data-client') === host.value)); });
      help.textContent = host.value === 'muse' ? 'Musebot uses a private adapter and its secure OAuth connector. Link this computer first; then configure the adapter in Musebot.' : host.value === 'composio' ? 'Link this computer first. Then add SuperLocalMemory as a Custom MCP in Composio using OAuth.' : host.value === 'other_mcp' ? 'Use a compatible web client that supports remote MCP and OAuth. Link this computer first, then use its MCP connection settings.' : 'Your AI account must support custom remote MCP connections with OAuth. Link this computer first, then use its connector settings.';
    }
    host.addEventListener('change', selectedClient);
    var consentLabel = node('label'); var consent = node('input'); consent.type = 'checkbox'; consent.setAttribute('data-remote-opt-in', ''); consentLabel.append(consent, document.createTextNode(' Enable remote access for this connection.')); form.appendChild(consentLabel);
    var writeLabel = node('label'); writeLabel.style.display = 'block'; writeLabel.style.margin = '8px 0'; var write = node('input'); write.type = 'checkbox'; write.setAttribute('data-permission', 'write'); writeLabel.append(write, document.createTextNode(' Allow this AI to save new memories.')); form.appendChild(writeLabel);
    var submit = node('button', 'Link this computer with GitHub', 'btn primary'); submit.type = 'submit'; form.appendChild(submit);
    var links = node('div'); body.appendChild(links);
    var metadata = null; var attempt = null; var busy = false; var storageKey = null;
    var choicesSignature = null; var listSignature = null;
    var pollTimer = null; var loading = false; var disposed = false;
    function controls() {
      submit.disabled = add.disabled || busy;
      Array.from(clients.children).forEach(function (button) { button.disabled = add.disabled || busy; });
    }
    function schedulePoll(delay) {
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(function () {
        pollTimer = null;
        if (disposed || !card.isConnected) return;
        if (busy || document.hidden) { schedulePoll(3000); return; }
        load();
      }, delay);
    }
    window.addEventListener('pagehide', function () { disposed = true; if (pollTimer !== null) window.clearTimeout(pollTimer); }, { once: true });
    function saveAttempt() {
      if (!storageKey || !attempt) throw new Error('intent persistence unavailable');
      // Only non-secret consent/request metadata. Never credentials or sign-in URLs.
      window.sessionStorage.setItem(storageKey, JSON.stringify(attempt));
    }
    function restoreAttempt() {
      var raw = window.sessionStorage.getItem(storageKey);
      if (!raw) return null;
      if (raw.length > 4096) throw new Error('invalid pending intent');
      var saved = JSON.parse(raw);
      if (!saved || !/^[a-f0-9]{32}$/.test(saved.key) || typeof saved.payload !== 'string') throw new Error('invalid pending intent');
      var payload = JSON.parse(saved.payload);
      if (!payload || payload.profile_id !== metadata.current_profile || !Object.hasOwn(HOSTS, payload.host) || payload.remote_opt_in !== true || !payload.permissions || payload.permissions.read !== true || typeof payload.permissions.write !== 'boolean' || payload.permissions.correction !== false || payload.permissions.session !== false) throw new Error('invalid pending intent');
      return { key: saved.key, payload: saved.payload, acknowledged: saved.acknowledged === true, connectionId: typeof saved.connectionId === 'string' ? saved.connectionId : null };
    }
    function load() {
      if (loading || disposed) return Promise.resolve();
      if (pollTimer !== null) { window.clearTimeout(pollTimer); pollTimer = null; }
      loading = true;
      refresh.disabled = true;
      return call('/api/v3/connections/status').then(function (data) {
        metadata = data && typeof data === 'object' ? data : null;
        var choices = metadata && Array.isArray(metadata.hosts) ? metadata.hosts.filter(function (name) { return Object.hasOwn(HOSTS, name); }) : [];
        var nextChoices = JSON.stringify(choices);
        if (nextChoices !== choicesSignature) {
          var selected = host.value; host.textContent = ''; clients.textContent = '';
          choices.forEach(function (name) {
            var option = node('option', HOSTS[name]); option.value = name; host.appendChild(option);
            var button = node('button', HOSTS[name], 'btn ghost od-ai-client'); button.type = 'button'; button.setAttribute('data-client', name);
            button.addEventListener('click', function () { host.value = name; selectedClient(); }); clients.appendChild(button);
          });
          if (choices.indexOf(selected) >= 0) host.value = selected;
          choicesSignature = nextChoices; selectedClient();
        }
        add.disabled = !(metadata && metadata.available === true && typeof metadata.current_profile === 'string' && metadata.current_profile && metadata.current_profile.length <= 256 && typeof metadata.installation_id === 'string' && metadata.installation_id && metadata.installation_id.length <= 256 && choices.length);
        if (!add.disabled) {
          var nextStorageKey = 'slm-ai-intent-v1:' + encodeURIComponent(metadata.installation_id) + ':' + encodeURIComponent(metadata.current_profile);
          if (storageKey !== nextStorageKey) {
            storageKey = nextStorageKey; links.textContent = ''; form.hidden = false; add.hidden = true; attempt = restoreAttempt(); consent.checked = false; write.checked = false; listSignature = null;
          if (attempt) {
            var restored = JSON.parse(attempt.payload);
            if (choices.indexOf(restored.host) < 0) throw new Error('pending host unavailable');
            host.value = restored.host; write.checked = restored.permissions.write; consent.checked = false;
          }
          selectedClient();
          }
        }
        status.textContent = add.disabled ? 'AI connections are currently unavailable.' : attempt ? 'A pending request was restored. Review it and confirm remote access to retry.' : 'No remote access is enabled until you approve a connection.';
        var currentConnections = metadata && Array.isArray(metadata.connections) ? metadata.connections : [];
        var step = currentConnections.some(function (c) { return c && c.verified === true && (c.state === 'connected' || c.state === 'ready_for_client' && c.mcp_url === 'https://mcp.superlocalmemory.com/mcp'); }) ? 3 : currentConnections.some(function (c) { return c && c.state === 'pending' && c.transport_state && c.transport_state !== 'authorization_required'; }) ? 2 : currentConnections.some(function (c) { return c && c.state === 'pending'; }) ? 1 : 0;
        if (!add.disabled && step === 3) status.textContent = 'This computer is verified. Follow your AI connection instructions below.';
        else if (!add.disabled && step === 2) status.textContent = 'GitHub sign-in completed. Checking the connection from this computer…';
        else if (!add.disabled && step === 1) status.textContent = 'Waiting for GitHub sign-in. Finish the opened sign-in page, or retry the same request below.';
        Array.from(journey.children).forEach(function (item, index) { if (index === step) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current'); });
        controls();
        var nextList = JSON.stringify([metadata.current_profile, metadata.connections || []]);
        if (nextList !== listSignature) {
        list.textContent = '';
        var history = node('details'); var historyRows = node('div'); var historyCount = 0;
        history.appendChild(node('summary', 'Past connections')); history.appendChild(historyRows);
        (metadata && Array.isArray(metadata.connections) ? metadata.connections : []).forEach(function (connection) {
          if (!connection || !Object.hasOwn(HOSTS, connection.host)) return;
          // Recover identity after a lost initiation response. This is a
          // non-secret idempotency key scoped by the authenticated local API.
          if (attempt && connection.intent_key === attempt.key && /^[a-f0-9]{32}$/.test(connection.connection_id)) {
            attempt.connectionId = connection.connection_id; saveAttempt();
          }
          var active = connection.state === 'connected' && connection.verified === true;
          var ready = connection.state === 'ready_for_client' && connection.verified === true && connection.mcp_url === 'https://mcp.superlocalmemory.com/mcp';
          if ((active || ready || connection.state === 'cancelled') && attempt && connection.connection_id === attempt.connectionId) { window.sessionStorage.removeItem(storageKey); attempt = null; links.textContent = ''; form.hidden = true; add.hidden = false; consent.checked = false; }
          var cancelled = connection.state === 'cancelled';
          var row = node('div', '', 'od-ai-connection');
          if (cancelled) { historyRows.appendChild(row); historyCount += 1; } else list.appendChild(row);
          var description = ready ? 'Ready for AI client — GitHub connected' : active ? 'Connected' : cancelled ? (connection.cleanup_pending ? 'Cancelled — remote cleanup pending' : 'Cancelled') : connection.state === 'pending' ? connection.transport_state === 'authorization_required' ? 'Authorization required — cancel this connection and link again' : connection.transport_state ? 'Verifying computer connection' : 'Waiting for GitHub sign-in and connection verification' : 'Not connected';
          row.appendChild(node('p', HOSTS[connection.host] + ': ' + description));
          if (ready) {
            row.appendChild(node('h4', 'Connect your AI — ' + HOSTS[connection.host]));
            row.appendChild(node('p', connection.host === 'composio' ? 'In Composio, add a Custom MCP named SuperLocalMemory, paste the MCP server URL below, and choose OAuth. In Advanced settings, use the OAuth metadata URL below. Sign in with the same GitHub account and approve access to this profile.' : connection.host === 'muse' ? 'Use the Musebot private adapter with its secure OAuth connector. Give it the MCP server URL and OAuth metadata URL below; approve access using the same GitHub account. Never paste tokens into chat.' : 'Add a remote MCP connector in your compatible AI client using the MCP server URL below and OAuth. Sign in with the same GitHub account and approve this profile. Availability depends on your AI account.'));
            var endpoint = node('input'); endpoint.type = 'text'; endpoint.readOnly = true; endpoint.value = connection.mcp_url; endpoint.setAttribute('aria-label', 'MCP server URL'); row.appendChild(endpoint);
            var oauthLabel = node('label', 'OAuth metadata URL'); var oauth = node('input'); oauth.type = 'text'; oauth.readOnly = true; oauth.value = 'https://auth.superlocalmemory.com/.well-known/oauth-authorization-server'; oauth.setAttribute('aria-label', 'OAuth metadata URL'); oauthLabel.appendChild(oauth);
            var copy = node('button', 'Copy URL', 'btn ghost sm'); copy.type = 'button'; row.appendChild(copy);
            row.appendChild(oauthLabel);
            copy.addEventListener('click', function () {
              if (window.navigator.clipboard && window.navigator.clipboard.writeText) window.navigator.clipboard.writeText(connection.mcp_url).then(function () { copy.textContent = 'Copied'; }).catch(function () { endpoint.select(); });
              else endpoint.select();
            });
          }
          if ((connection.state === 'pending' || ready || active) && /^[a-f0-9]{32}$/.test(connection.connection_id) && Number.isSafeInteger(connection.version) && connection.version >= 0) {
            var cancel = node('button', 'Cancel connection', 'btn ghost sm'); cancel.type = 'button'; row.appendChild(cancel);
            var profile = metadata.current_profile;
            cancel.addEventListener('click', function () {
              if (busy) return;
              busy = true; cancel.disabled = true; status.textContent = 'Cancelling connection…';
              call('/api/v3/connections/' + encodeURIComponent(connection.connection_id) + '/cancel', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile_id: profile, expected_version: connection.version })
              }).then(function (result) {
                if (!result || result.connection_id !== connection.connection_id || result.state !== 'cancelled' || result.verified !== false || typeof result.cleanup_pending !== 'boolean') throw new Error('unconfirmed cancellation');
                if (attempt && attempt.connectionId === connection.connection_id) { window.sessionStorage.removeItem(storageKey); attempt = null; links.textContent = ''; form.hidden = true; add.hidden = false; consent.checked = false; }
                return load();
              }).catch(function () { status.textContent = 'Cancellation could not be confirmed. Refresh status and retry.'; }).finally(function () { busy = false; cancel.disabled = false; controls(); });
            });
          }
        });
        if (historyCount) { history.firstChild.textContent = 'Past connections (' + historyCount + ')'; list.appendChild(history); }
        listSignature = nextList;
        }
        if (metadata && Array.isArray(metadata.connections) && metadata.connections.some(function (connection) { return connection && connection.state === 'pending'; })) schedulePoll(1500);
      }).catch(function () { if (attempt && attempt.acknowledged) schedulePoll(10000); metadata = null; add.disabled = true; controls(); status.textContent = 'Could not check AI connections. Refresh to retry.'; }).finally(function () { loading = false; refresh.disabled = false; });
    }
    add.addEventListener('click', function () { if (!add.disabled) { form.hidden = false; add.hidden = true; } });
    refresh.addEventListener('click', function () { if (!busy) load(); });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      if (busy || add.disabled || !metadata) return;
      if (!consent.checked) { status.textContent = 'Please enable remote access to continue.'; return; }
      var payload = JSON.stringify({ host: host.value, profile_id: metadata.current_profile, remote_opt_in: true, permissions: { read: true, write: write.checked, correction: false, session: false } });
      if (attempt && attempt.payload !== payload) {
        if (!attempt.acknowledged) { status.textContent = 'Retry the pending request before changing its settings.'; return; }
        attempt = null;
      }
      if (!attempt) attempt = { key: operationKey(), payload: payload };
      try { saveAttempt(); } catch (_) { status.textContent = 'Browser storage is unavailable. Connection was not requested.'; return; }
      var signInWindow = null;
      try { signInWindow = window.open('about:blank', '_blank'); if (signInWindow) signInWindow.opener = null; } catch (_) { signInWindow = null; }
      busy = true; controls(); links.textContent = ''; status.textContent = 'Requesting your connection…';
      call('/api/v3/connections/initiate', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': attempt.key }, body: payload }).then(function (result) {
        if (!result || result.state !== 'pending' || typeof result.connection_id !== 'string' || !result.connection_id || result.connection_id.length > 256) throw new Error('unconfirmed connection receipt');
        attempt.acknowledged = true; attempt.connectionId = result.connection_id; saveAttempt();
        // A request acknowledgement is not proof of a live authorized connection.
        status.textContent = 'Connection requested. Waiting for sign-in and verification.';
        var url = result && safeSignIn(result.authorization_url, result.connection_id);
        if (url) {
          if (signInWindow) { try { signInWindow.location.replace(url); } catch (_) { signInWindow.close(); } }
          var link = node('a', 'Continue sign-in', 'btn ghost'); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; links.appendChild(link); } else if (signInWindow) signInWindow.close();
        schedulePoll(1500);
      }).catch(function () { if (signInWindow) signInWindow.close(); status.textContent = 'Connection request could not be confirmed. Retry the same request.'; }).finally(function () { busy = false; controls(); });
    });
    load(); return card;
  };
}());
