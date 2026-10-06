// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// Existing dashboard initiation. Local API/installation auth owns enrollment;
// this card never receives a connector key or changes transport configuration.
(function () {
  'use strict';
  var HOSTS = { muse: 'Musebot', chatgpt: 'ChatGPT Web', claude_web: 'Claude Web', claude_code_web: 'Claude Code Web' };
  function node(tag, text, cls) { var el = document.createElement(tag); if (text) el.textContent = text; if (cls) el.className = cls; return el; }
  function call(path, init) {
    var fetcher = typeof window.slmFetch === 'function' ? window.slmFetch : window.fetch;
    var options = Object.assign({ credentials: 'same-origin' }, init || {});
    return fetcher(path, options).then(function (response) { if (!response.ok) throw new Error('connection request unavailable'); return response.json(); });
  }
  function safeSignIn(value) {
    try {
      var url = new URL(value);
      if (url.protocol !== 'https:' || url.hostname !== 'auth.superlocalmemory.com' || url.port || url.username || url.password || url.hash) return null;
      if (url.pathname !== '/authorize' && url.pathname !== '/owner/login') return null;
      if (url.href.length > 4096) return null;
      var forbidden = ['access_token', 'refresh_token', 'client_secret', 'token', 'secret', 'key'];
      if (forbidden.some(function (name) { return url.searchParams.has(name); })) return null;
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
    var list = node('div'); body.appendChild(list);
    var add = node('button', 'Add AI connection', 'btn primary'); add.type = 'button'; add.disabled = true; body.appendChild(add);
    var refresh = node('button', 'Refresh status', 'btn ghost sm'); refresh.type = 'button'; refresh.style.marginLeft = '8px'; body.appendChild(refresh);
    var form = node('form'); form.hidden = true; form.style.marginTop = '12px'; body.appendChild(form);
    var label = node('label', 'Choose your AI'); label.htmlFor = 'od-connection-host'; form.appendChild(label);
    var host = node('select'); host.id = 'od-connection-host'; host.style.display = 'block'; host.style.marginBottom = '12px'; host.style.cssText += ';padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--card-2);color:var(--fg-1);min-width:200px;max-width:100%'; form.appendChild(host);
    var consentLabel = node('label'); var consent = node('input'); consent.type = 'checkbox'; consent.setAttribute('data-remote-opt-in', ''); consentLabel.append(consent, document.createTextNode(' Enable remote access for this connection.')); form.appendChild(consentLabel);
    var writeLabel = node('label'); writeLabel.style.display = 'block'; writeLabel.style.margin = '8px 0'; var write = node('input'); write.type = 'checkbox'; write.setAttribute('data-permission', 'write'); writeLabel.append(write, document.createTextNode(' Allow this AI to save new memories.')); form.appendChild(writeLabel);
    var submit = node('button', 'Continue', 'btn primary'); submit.type = 'submit'; form.appendChild(submit);
    var links = node('div'); body.appendChild(links);
    var metadata = null; var attempt = null; var busy = false; var storageKey = null;
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
      refresh.disabled = true;
      return call('/api/v3/connections/status').then(function (data) {
        metadata = data && typeof data === 'object' ? data : null;
        var choices = metadata && Array.isArray(metadata.hosts) ? metadata.hosts.filter(function (name) { return Object.hasOwn(HOSTS, name); }) : [];
        host.textContent = ''; choices.forEach(function (name) { var option = node('option', HOSTS[name]); option.value = name; host.appendChild(option); });
        add.disabled = !(metadata && metadata.available === true && typeof metadata.current_profile === 'string' && metadata.current_profile && metadata.current_profile.length <= 256 && typeof metadata.installation_id === 'string' && metadata.installation_id && metadata.installation_id.length <= 256 && choices.length);
        if (!add.disabled) {
          storageKey = 'slm-ai-intent-v1:' + encodeURIComponent(metadata.installation_id) + ':' + encodeURIComponent(metadata.current_profile);
          attempt = restoreAttempt();
          if (attempt) {
            var restored = JSON.parse(attempt.payload);
            if (choices.indexOf(restored.host) < 0) throw new Error('pending host unavailable');
            host.value = restored.host; write.checked = restored.permissions.write; consent.checked = false;
          }
        }
        status.textContent = add.disabled ? 'AI connections are currently unavailable.' : attempt ? 'A pending request was restored. Review it and confirm remote access to retry.' : 'No remote access is enabled until you approve a connection.';
        list.textContent = '';
        (metadata && Array.isArray(metadata.connections) ? metadata.connections : []).forEach(function (connection) {
          if (!connection || !Object.hasOwn(HOSTS, connection.host)) return;
          var active = connection.state === 'connected' && connection.verified === true;
          if (active && attempt && connection.connection_id === attempt.connectionId) { window.sessionStorage.removeItem(storageKey); attempt = null; }
          list.appendChild(node('p', HOSTS[connection.host] + ': ' + (active ? 'Connected' : 'Not connected')));
        });
      }).catch(function () { metadata = null; add.disabled = true; status.textContent = 'Could not check AI connections. Refresh to retry.'; }).finally(function () { refresh.disabled = false; });
    }
    add.addEventListener('click', function () { if (!add.disabled) form.hidden = false; });
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
      busy = true; submit.disabled = true; links.textContent = ''; status.textContent = 'Requesting your connection…';
      call('/api/v3/connections/initiate', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': attempt.key }, body: payload }).then(function (result) {
        if (!result || result.state !== 'pending' || typeof result.connection_id !== 'string' || !result.connection_id || result.connection_id.length > 256) throw new Error('unconfirmed connection receipt');
        attempt.acknowledged = true; attempt.connectionId = result.connection_id; saveAttempt();
        // A request acknowledgement is not proof of a live authorized connection.
        status.textContent = 'Connection requested. Waiting for sign-in and verification.';
        var url = result && safeSignIn(result.authorization_url);
        if (url) { var link = node('a', 'Continue sign-in', 'btn ghost'); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; links.appendChild(link); }
      }).catch(function () { status.textContent = 'Connection request could not be confirmed. Retry the same request.'; }).finally(function () { busy = false; submit.disabled = false; });
    });
    load(); return card;
  };
}());
