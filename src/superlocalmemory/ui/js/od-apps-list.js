// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
// od-apps-list.js — "Your connected apps": one row per app authorized on this
// computer, with a confirmed "Remove access" action.
//
// Exposes: window.odCreateConnectedAppsList()
//   returns the section element, which carries:
//     .odSetConnections(connectionIds, profileId, { force })  feed it the live connections
//     .odFence()    a profile/scope refresh started: drop in-flight answers, block removal
//     .odReload()   reload from the server now
//
// API (per connection):
//   GET  /api/v3/connections/{id}/apps
//        200 {connection_id, apps:[{authorization_id, name, client_host, permissions:{read,save,session},
//                                   version, connected_at_ms, last_used_at_ms}]}
//        404 {detail:"not_found"}   503 {detail:"apps_unavailable"}
//   POST /api/v3/connections/{id}/apps/{authorization_id}/revoke   {profile_id, expected_version}
//        200 {revoked:true}   409 {detail:"version_conflict"|"profile_changed"}   503
//
// Every app name / host is third-party text: textContent only, never markup.
(function () {
  'use strict';

  var CONNECTION_ID = /^[a-f0-9]{32}$/;
  var STALE_AFTER_MS = 15000;

  function text(value, fallback, limit) {
    if (typeof value !== 'string') return fallback;
    var trimmed = value.trim();
    if (!trimmed) return fallback;
    return trimmed.length > limit ? trimmed.slice(0, limit) : trimmed;
  }

  // Defensive normalisation: the server is trusted for shape, not for content.
  function normalize(connectionId, raw) {
    if (!raw || typeof raw !== 'object') return null;
    var authorizationId = typeof raw.authorization_id === 'string' ? raw.authorization_id : '';
    if (!authorizationId || authorizationId.length > 256) return null;
    var permissions = raw.permissions && typeof raw.permissions === 'object' ? raw.permissions : {};
    return {
      connectionId: connectionId,
      authorizationId: authorizationId,
      name: text(raw.name, 'Unnamed app', 120),
      host: text(raw.client_host, '', 253),
      save: permissions.save === true,
      session: permissions.session === true,
      version: Number.isSafeInteger(raw.version) && raw.version >= 0 ? raw.version : null,
      connectedAt: typeof raw.connected_at_ms === 'number' ? raw.connected_at_ms : null,
      lastUsedAt: typeof raw.last_used_at_ms === 'number' ? raw.last_used_at_ms : null
    };
  }

  window.odCreateConnectedAppsList = function () {
    var ui = window.odAppsUi;
    var h = ui.h;

    var headingId = ui.nextId('apps-list-title');
    var heading = h('h3', { id: headingId, className: 'apps-section-title', text: 'Your connected apps', tabindex: '-1' });
    var refreshLabel = h('span', { text: 'Refresh' });
    var refresh = h('button', { type: 'button', className: 'btn ghost sm', 'aria-label': 'Refresh connected apps' }, [ui.icon('refresh', 15), refreshLabel]);
    var head = h('div', { className: 'apps-section-head' }, [
      h('div', { className: 'apps-section-headtext' }, [
        heading,
        h('p', { className: 'apps-section-sub', text: 'Apps that can use your memory right now. You can remove access at any time.' })
      ]),
      refresh
    ]);

    var notice = h('div', { className: 'apps-notice', role: 'status', 'aria-live': 'polite', hidden: true });
    var rows = h('ul', { className: 'apps-rows', 'aria-labelledby': headingId });
    var loading = h('p', { className: 'apps-loading', text: 'Loading your apps…', hidden: true });
    var addHint = h('button', { type: 'button', className: 'btn secondary sm', text: 'Add an app' });
    var empty = h('div', { className: 'apps-empty', hidden: true }, [
      h('h4', { className: 'apps-empty-title', text: 'No apps connected yet' }),
      h('p', { className: 'apps-empty-text', text: 'When you add an app, it will show up here so you can see what it can do and remove its access at any time.' }),
      addHint
    ]);
    var panel = h('div', { className: 'apps-panel' }, [notice, loading, rows, empty]);
    var root = h('section', { className: 'apps-section', 'data-apps-list': '', 'aria-labelledby': headingId }, [head, panel]);

    var state = {
      ids: [], profile: '', entries: [], generation: 0, loadedAt: 0,
      fenced: false, loading: false, error: null, blocked: false, ready: false
    };

    // ----- rendering --------------------------------------------------------
    function showNotice(message, tone, retry) {
      notice.textContent = '';
      notice.className = 'apps-notice' + (tone ? ' is-' + tone : '');
      if (!message) { notice.hidden = true; return; }
      notice.appendChild(ui.icon(tone === 'error' ? 'alert' : 'refresh', 16));
      notice.appendChild(h('span', { className: 'apps-notice-text', text: message }));
      if (retry) {
        var again = h('button', { type: 'button', className: 'btn secondary sm', text: 'Try again' });
        again.addEventListener('click', function () { load(); });
        notice.appendChild(again);
      }
      notice.hidden = false;
    }

    function chip(label, title, on) {
      return h('span', { className: 'apps-chip' + (on ? ' is-on' : ''), title: title, text: label });
    }

    function buildRow(entry) {
      var meta = [];
      var connected = ui.relativeTime(entry.connectedAt);
      if (connected) meta.push(h('span', { title: ui.fullDate(entry.connectedAt), text: 'Connected ' + connected }));
      var used = ui.relativeTime(entry.lastUsedAt);
      meta.push(h('span', used ? { title: ui.fullDate(entry.lastUsedAt), text: 'Last used ' + used } : { text: 'Not used yet' }));
      var metaRow = h('div', { className: 'apps-meta' });
      meta.forEach(function (item, index) {
        if (index) metaRow.appendChild(h('span', { className: 'apps-meta-sep', 'aria-hidden': 'true', text: '·' }));
        metaRow.appendChild(item);
      });

      var chips = h('div', { className: 'apps-chips', role: 'group', 'aria-label': 'What ' + entry.name + ' can do' }, [
        chip('Read', 'Can read your memories', false)
      ]);
      if (entry.save) chips.appendChild(chip('Save', 'Can save new memories', true));
      if (entry.session) chips.appendChild(chip('Session tools', 'Can use session tools', true));

      var titleRow = h('div', { className: 'apps-row-title' }, [h('span', { className: 'apps-name', text: entry.name })]);
      if (entry.host) titleRow.appendChild(h('span', { className: 'apps-host', text: entry.host }));

      var remove = h('button', {
        type: 'button', className: 'btn danger sm', text: 'Remove access',
        'aria-label': 'Remove access for ' + entry.name, 'data-remove-access': ''
      });
      if (entry.version === null) { remove.disabled = true; remove.title = 'This app cannot be removed from here yet.'; }
      remove.addEventListener('click', function () { askRemove(entry, remove); });
      entry.button = remove;

      var row = h('li', { className: 'apps-row', 'data-authorization-id': entry.authorizationId }, [
        h('div', { className: 'apps-avatar', 'aria-hidden': 'true', text: ui.monogram(entry.name) }),
        h('div', { className: 'apps-row-main' }, [titleRow, chips, metaRow]),
        h('div', { className: 'apps-row-actions' }, [remove])
      ]);
      entry.row = row;
      return row;
    }

    function render() {
      rows.textContent = '';
      state.entries.forEach(function (entry) { rows.appendChild(buildRow(entry)); });
      rows.hidden = state.entries.length === 0;
      var settled = state.ready && !state.loading && !state.error;
      empty.hidden = !(settled && state.entries.length === 0);
      loading.hidden = !((state.loading || !state.ready) && !state.error && state.entries.length === 0);
      applyBlock();
    }

    function applyBlock() {
      state.entries.forEach(function (entry) {
        if (entry.button) entry.button.disabled = state.blocked || entry.version === null;
      });
      refresh.disabled = state.loading;
    }

    // ----- loading ----------------------------------------------------------
    function load() {
      var generation = state.generation + 1; state.generation = generation;
      var ids = state.ids.slice(); var profile = state.profile;
      state.fenced = false; state.blocked = false;
      if (!ids.length) {
        state.entries = []; state.loading = false; state.error = null; state.ready = true; state.loadedAt = Date.now();
        showNotice(''); render();
        return Promise.resolve();
      }
      state.loading = true; state.error = null;
      if (!state.entries.length) render(); else applyBlock();
      return Promise.all(ids.map(function (id) {
        return ui.request('/api/v3/connections/' + encodeURIComponent(id) + '/apps').then(function (result) {
          if (result.ok && result.data && Array.isArray(result.data.apps)) {
            return { id: id, apps: result.data.apps.map(function (raw) { return normalize(id, raw); }).filter(Boolean) };
          }
          // 404: the connection is gone, so it has no apps. Anything else is a failure.
          if (result.status === 404) return { id: id, apps: [] };
          return { id: id, failed: true };
        }).catch(function () { return { id: id, failed: true }; });
      })).then(function (results) {
        // Stale-answer fence: a newer load, a fence, or a profile switch won.
        if (generation !== state.generation || profile !== state.profile) return;
        var entries = [];
        var failed = false;
        results.forEach(function (result) {
          if (result.failed) { failed = true; return; }
          entries = entries.concat(result.apps);
        });
        entries.sort(function (a, b) {
          return (b.lastUsedAt || 0) - (a.lastUsedAt || 0) || (b.connectedAt || 0) - (a.connectedAt || 0) || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
        });
        state.entries = entries; state.loading = false; state.ready = true; state.loadedAt = Date.now();
        state.error = failed ? 'unavailable' : null;
        if (failed) showNotice('We couldn’t load your connected apps just now. Try again in a moment.', 'error', true);
        else showNotice('');
        render();
      });
    }

    // ----- removing access --------------------------------------------------
    // Rows are rebuilt on every render, so remember the neighbouring ENTRY and
    // read its (new) button after the render.
    function neighbourOf(entry) {
      var index = state.entries.indexOf(entry);
      return state.entries[index + 1] || state.entries[index - 1] || null;
    }

    function dropEntry(entry) {
      state.entries = state.entries.filter(function (item) { return item !== entry; });
      render();
    }

    function askRemove(entry, button) {
      if (state.blocked) return;
      var dialog = ui.confirmDialog({
        title: 'Remove access for ' + entry.name + '?',
        body: entry.name + ' will no longer be able to read or save memories. You can connect it again any time.',
        confirmLabel: 'Remove access',
        cancelLabel: 'Cancel',
        returnFocus: button,
        fallbackFocus: function () { return heading; },
        onConfirm: function (controller) { return revoke(entry, controller); }
      });
      return dialog;
    }

    function revoke(entry, controller) {
      controller.setBusy(true, 'Removing…');
      var path = '/api/v3/connections/' + encodeURIComponent(entry.connectionId) + '/apps/' + encodeURIComponent(entry.authorizationId) + '/revoke';
      return ui.request(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: state.profile, expected_version: entry.version })
      }).then(function (result) {
        if (result.ok && result.data && result.data.revoked === true) {
          var neighbour = neighbourOf(entry);
          dropEntry(entry);
          controller.close();
          ui.toast('Access removed');
          var target = neighbour && neighbour.button && neighbour.button.isConnected ? neighbour.button : heading;
          if (typeof target.focus === 'function') target.focus();
          return;
        }
        if (result.status === 409 || result.status === 404) {
          controller.close();
          showNotice('This list changed. Refreshing…', 'info');
          return load();
        }
        controller.setBusy(false);
        controller.setError(result.status === 503
          ? 'Couldn’t reach the connection service. Try again.'
          : 'Couldn’t remove access. Try again.');
      }).catch(function () {
        controller.setBusy(false);
        controller.setError('Couldn’t reach the connection service. Try again.');
      });
    }

    // ----- public surface ---------------------------------------------------
    root.odSetConnections = function (connectionIds, profileId, options) {
      var ids = (Array.isArray(connectionIds) ? connectionIds : []).filter(function (id) { return typeof id === 'string' && CONNECTION_ID.test(id); });
      var profile = typeof profileId === 'string' ? profileId : '';
      var changed = ids.join(',') !== state.ids.join(',') || profile !== state.profile;
      state.ids = ids; state.profile = profile;
      var stale = Date.now() - state.loadedAt >= STALE_AFTER_MS;
      if (changed || state.fenced || stale || (options && options.force)) return load();
      state.blocked = false; applyBlock();
      return Promise.resolve();
    };
    root.odFence = function () {
      state.generation += 1; state.fenced = true; state.blocked = true; state.loading = false;
      applyBlock();
    };
    root.odReload = function () { return load(); };
    // The status call itself failed: say so instead of claiming the list is empty.
    root.odSetUnavailable = function () {
      state.generation += 1; state.loading = false; state.ready = true; state.error = 'unavailable'; state.blocked = true;
      showNotice('We can’t check your connected apps right now. Use Refresh status above to try again.', 'error', false);
      render();
    };

    refresh.addEventListener('click', function () { if (!state.loading) load(); });
    addHint.addEventListener('click', function () {
      var target = document.getElementById('apps-add-title');
      if (target) { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); target.focus(); }
    });

    render();
    return root;
  };
}());
