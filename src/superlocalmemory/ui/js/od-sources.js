// od-sources.js — "Folders" section of the Documents & Images pane: connected folders,
// add with a preview, rescan, report, release a held file, remove.
// Public: window.odRenderSources(host)
// XSS-safe: folder names, paths, relpaths and reasons are untrusted and go in with textContent
// only. Writes go through the page's fetch, which core.js wraps with the local write credential.
// Routes: GET/POST /api/v3/sources   POST /api/v3/sources/{id}/confirm|rescan|quarantine/release
//         GET /api/v3/sources/{id}/report   DELETE /api/v3/sources/{id}[?purge=1]
(function () {
  'use strict';

  var ID_RE = /^[0-9a-f]{32}$/;
  var BASE = '/api/v3/sources';
  var OFFLINE = {
    unreachable: 'The folder cannot be reached.',
    disk_changed: 'The folder is now on a different disk, so it was left alone.',
    empty_folder: 'The folder is empty now, so nothing was removed.',
    root_moved: 'The folder has moved.',
  };
  var REMOTE_SENTENCE = 'Turn remote access off first, then connect the folder. ' +
    'Remote tools could read the notes while it is on.';

  function F() { return window.odFeatures; }
  function el(tag, cls, text) { return F().el(tag, cls, text); }
  function note(host, text) { host.appendChild(el('p', 'muted', text)); }

  function stateText(s) {
    if (s.state === 'paused') return 'Paused because remote access is on.';
    if (s.state === 'offline') return 'Offline. ' + (OFFLINE[s.offline_reason] || s.offline_reason || '');
    return s.state === 'active' ? 'Connected.' : String(s.state || '');
  }

  function countsText(files) {
    var parts = Object.keys(files || {}).map(function (k) { return files[k] + ' ' + k; });
    return parts.length ? parts.join(', ') : 'no files yet';
  }

  function pairs(map) {
    return Object.keys(map || {}).map(function (k) { return k + ': ' + map[k]; }).join(', ');
  }

  // -------------------------------------------------------------------- report
  function releaseFile(id, item, row) {
    F().api('POST', BASE + '/' + id + '/quarantine/release', { relpath: item.relpath }).then(function (res) {
      if (res.ok) { row.textContent = ''; row.appendChild(el('span', 'muted', item.relpath + ' released. It is read on the next scan.')); return; }
      if (res.status === 404) return note(row, 'Not available in this build.');
      note(row, F().failText(res, 'Could not release the file.'));
    });
  }

  function fileList(title, items, onRow) {
    var box = el('div');
    box.appendChild(el('h5', null, title));
    var ul = el('ul', 'od-media-list');
    items.forEach(function (it) { ul.appendChild(onRow(it)); });
    box.appendChild(ul);
    return box;
  }

  function fileRow(it, withReason, id) {
    var li = el('li');
    li.appendChild(el('strong', null, it.relpath));
    if (withReason && it.reason) li.appendChild(el('span', 'muted', ' ' + it.reason));
    if (id) li.appendChild(F().button('Release', 'btn sm', function () { releaseFile(id, it, li); }));
    return li;
  }

  function renderReport(host, id, r) {
    host.textContent = '';
    host.appendChild(el('p', 'muted', 'Watching for changes: ' + (r.watch ? 'yes' : 'no') +
      (r.capped ? '. This folder is over the file limit; only the first files are indexed.' : '')));
    if (r.paused_reason) note(host, r.paused_reason);
    var skipped = pairs(r.skipped_by_rule);
    host.appendChild(el('p', 'muted', 'Skipped by rule: ' + (skipped || 'none')));
    var errs = r.errors || [];
    host.appendChild(el('p', 'muted', errs.length + ' error' + (errs.length === 1 ? '' : 's') + '.'));
    var held = r.quarantined || [];
    if (held.length) host.appendChild(fileList('Held back for review', held,
      function (it) { return fileRow(it, true, id); }));
    if (errs.length) host.appendChild(fileList('Errors', errs, function (it) { return fileRow(it, true); }));
    var cloud = r.cloud_only || [];
    if (cloud.length) host.appendChild(fileList('Only in the cloud, not downloaded', cloud,
      function (p) { return fileRow({ relpath: p }, false); }));
  }

  function toggleReport(panel, id) {
    if (panel.firstChild) { panel.textContent = ''; return; }
    panel.appendChild(el('p', 'muted', 'Loading…'));
    F().api('GET', BASE + '/' + id + '/report').then(function (res) {
      panel.textContent = '';
      if (res.status === 404) return F().notAvailable(panel, 'The folder report');
      if (!res.ok) return note(panel, F().failText(res, 'Could not load the report.'));
      renderReport(panel, id, res.data);
    });
  }

  // ------------------------------------------------------------------ actions
  function rescan(msg, id) {
    F().api('POST', BASE + '/' + id + '/rescan').then(function (res) {
      if (res.status === 404) msg.textContent = 'Not available in this build.';
      else msg.textContent = res.ok ? 'Scan queued.' : F().failText(res, 'Could not start the scan.');
    });
  }

  function removeFolder(ctx, s, msg, purge) {
    var erase = purge.checked;
    F().confirmThen({
      title: 'Remove folder', target: String(s.display_name || 'folder').slice(0, 80),
      consequence: erase ? 'The folder is disconnected and everything saved from it is erased. The files in the folder are not touched.'
        : 'The folder is disconnected. What was saved from it is hidden, not erased. The files in the folder are not touched.',
      confirmLabel: 'Remove',
    }, function () {
      F().api('DELETE', BASE + '/' + s.source_id + (erase ? '?purge=1' : '')).then(function (res) {
        if (res.ok) return load(ctx);
        msg.textContent = res.status === 404 ? 'Not available in this build.' : F().failText(res, 'Could not remove the folder.');
      });
    });
  }

  function actions(ctx, s, panel, msg) {
    var bar = el('div', 'od-sources-actions');
    var purge = el('input');
    purge.type = 'checkbox';
    var label = el('label', 'muted');
    label.appendChild(purge);
    label.appendChild(document.createTextNode(' Also erase what was saved from it'));
    bar.appendChild(F().button('Rescan', 'btn sm', function () { rescan(msg, s.source_id); }));
    bar.appendChild(F().button('Report', 'btn sm', function () { toggleReport(panel, s.source_id); }));
    bar.appendChild(label);
    bar.appendChild(F().button('Remove', 'btn sm', function () { removeFolder(ctx, s, msg, purge); }));
    return bar;
  }

  // --------------------------------------------------------------------- list
  function folderRow(ctx, s) {
    var li = el('li');
    li.appendChild(el('strong', null, s.display_name || 'Folder'));
    li.appendChild(el('span', 'muted', ' ' + (s.kind === 'obsidian' ? 'Obsidian vault' : 'Folder')));
    li.appendChild(el('div', 'muted', s.root_path || ''));
    li.appendChild(el('div', 'muted', stateText(s) + ' ' + countsText(s.files) +
      '. Last scan: ' + (s.last_scan_at || 'not yet') + '.'));
    if (!ID_RE.test(String(s.source_id || ''))) return li;
    var panel = el('div');
    var msg = el('p', 'muted');
    msg.setAttribute('role', 'status');
    [actions(ctx, s, panel, msg), msg, panel].forEach(function (n) { li.appendChild(n); });
    return li;
  }

  function load(ctx) {
    return F().api('GET', BASE).then(function (res) {
      ctx.list.textContent = '';
      if (res.status === 404) { ctx.form.textContent = ''; return F().notAvailable(ctx.list, 'Connected folders'); }
      if (!res.ok) return note(ctx.list, F().failText(res, 'Could not load the folders.'));
      var found = res.data.sources || [];
      if (!found.length) return note(ctx.list, 'No folders connected yet.');
      var ul = el('ul', 'od-media-list');
      found.forEach(function (s) { ul.appendChild(folderRow(ctx, s)); });
      ctx.list.appendChild(ul);
    });
  }

  // ---------------------------------------------------------------------- add
  function previewLines(p) {
    var total = Object.keys(p.files_by_type || {}).reduce(function (n, k) { return n + p.files_by_type[k]; }, 0);
    var lines = [total + ' files found (' + (pairs(p.files_by_type) || 'none') + ').',
      'Skipped by rule: ' + (pairs(p.skipped_by_rule) || 'none') + '.'];
    if (p.quarantined_count) lines.push(p.quarantined_count + ' files would be held back for review.');
    lines.push('About ' + Math.max(1, Math.round(p.est_seconds || 0)) + ' seconds. ' + (p.estimate_note || ''));
    if (p.capped) lines.push('This folder is over the file limit; only the first files will be indexed.');
    return lines.concat(p.warnings || []);
  }

  function connect(ctx, p, msg) {
    var name = String(p.root || 'folder').split(/[\\/]/).filter(Boolean).pop() || 'folder';
    F().confirmThen({
      title: 'Connect folder', target: name.slice(0, 80),
      consequence: 'Files in ' + String(p.root || '') + ' are read and saved to your memory. Nothing in the folder is changed.',
      confirmLabel: 'Connect',
    }, function () {
      F().api('POST', BASE + '/' + p.source_id + '/confirm').then(function (res) {
        if (res.ok) { ctx.preview.textContent = ''; return load(ctx); }
        var code = res.data && res.data.detail && res.data.detail.code;
        msg.textContent = code === 'remote_access_on' ? REMOTE_SENTENCE
          : res.status === 404 && !code ? 'Not available in this build.' : F().failText(res, 'Could not connect the folder.');
      });
    });
  }

  function showPreview(ctx, p) {
    ctx.preview.textContent = '';
    if (!ID_RE.test(String(p.source_id || ''))) return note(ctx.preview, 'The service sent a preview that cannot be used.');
    ctx.preview.appendChild(el('strong', null, p.root));
    previewLines(p).forEach(function (l) { note(ctx.preview, l); });
    var msg = el('p', 'muted');
    msg.setAttribute('role', 'status');
    ctx.preview.appendChild(F().button('Connect this folder', 'btn sm primary', function () { connect(ctx, p, msg); }));
    ctx.preview.appendChild(msg);
  }

  function check(ctx) {
    var path = ctx.input.value.trim();
    ctx.preview.textContent = '';
    if (!path) return note(ctx.preview, 'Type the path of a folder first.');
    F().api('POST', BASE, { path: path }).then(function (res) {
      if (res.status === 404) return note(ctx.preview, 'Not available in this build.');
      if (!res.ok) return note(ctx.preview, F().failText(res, 'That folder cannot be used.'));
      showPreview(ctx, res.data);
    });
  }

  function buildForm(ctx) {
    ctx.form.appendChild(el('p', 'muted', 'Connect a folder of notes, PDFs or pictures. Files stay where they are and are only read.'));
    ctx.input = el('input');
    ctx.input.type = 'text';
    ctx.input.setAttribute('placeholder', '/path/to/folder');
    ctx.input.setAttribute('aria-label', 'Folder path');
    ctx.form.appendChild(ctx.input);
    ctx.form.appendChild(F().button('Check folder', 'btn sm', function () { check(ctx); }));
  }

  function odRenderSources(host) {
    host.textContent = '';
    var ctx = { form: el('div'), preview: el('div'), list: el('div') };
    host.appendChild(el('h3', null, 'Folders'));
    buildForm(ctx);
    [ctx.form, ctx.preview, ctx.list].forEach(function (n) { host.appendChild(n); });
    return load(ctx);
  }

  window.odRenderSources = odRenderSources;
})();
