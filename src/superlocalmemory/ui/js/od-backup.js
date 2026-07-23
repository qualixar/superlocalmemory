/* od-backup.js — SuperLocalMemory Backup & Cloud Dashboard v1.0
 * Exposes:
 *   window.odRenderBackup(container)    — render into a supplied pane
 *   window.odOpenBackupDashboard()      — open a full-screen overlay (called from od-settings.js)
 * Wired endpoints (all confirmed against live daemon):
 *   GET  /api/backup/status
 *   POST /api/backup/create
 *   POST /api/backup/configure          body: {interval_hours, max_backups, enabled}
 *   GET  /api/backup/list
 *   GET  /api/backup/destinations
 *   POST /api/backup/connect/github     body: {pat, repo_name}
 *   DELETE /api/backup/disconnect/{id}
 *   POST /api/backup/sync
 *   POST /api/backup/export             → FileResponse (.db.gz download)
 *   GET  /api/backup/oauth/github/start → OAuth redirect or PAT form
 *   GET  /api/backup/oauth/google/start → Google OAuth redirect
 * No mock / seed data — every render goes to the live daemon.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var OVERLAY_ID = 'od-backup-overlay';
  var P = 'od-bk';
  var tokenPromise = null;

  function getToken(forceRefresh, retriedEmpty) {
    if (forceRefresh) tokenPromise = null;
    if (!tokenPromise) {
      tokenPromise = fetch('/internal/token', { credentials:'same-origin' })
        .then(function (response) {
          if (!response.ok) throw new Error('Dashboard authorization failed');
          return response.json();
        })
        .then(function (data) {
          if (!data.token && !retriedEmpty) {
            tokenPromise = null;
            return getToken(true, true);
          }
          if (!data.token) throw new Error('Dashboard authorization token is unavailable');
          return data.token;
        })
        .catch(function (error) {
          tokenPromise = null;
          throw error;
        });
    }
    return tokenPromise;
  }
  function authMutation(url, method, body) {
    function send(forceRefresh) {
      return getToken(forceRefresh).then(function (token) {
        return fetch(url, {
          method: method,
          credentials: 'same-origin',
          headers: { 'Content-Type':'application/json', 'X-Install-Token': token },
          body: body === undefined ? undefined : JSON.stringify(body)
        });
      }).then(function (response) {
        if (!forceRefresh && (response.status === 401 || response.status === 403)) {
          return send(true);
        }
        if (!response.ok) {
          return response.text().then(function (message) {
            throw new Error(message || ('Request failed (' + response.status + ')'));
          });
        }
        return response;
      });
    }
    return send(false);
  }

  /* ── Tiny utilities ──────────────────────────────────────── */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }
  function toast(msg, err) {
    if (typeof window.showToast === 'function') { window.showToast(msg); return; }
    var d = document.createElement('div');
    d.textContent = msg;
    Object.assign(d.style, { position:'fixed', bottom:'20px', right:'20px', zIndex:'99999',
      background: err ? 'var(--danger)' : 'var(--violet)', color:'#fff',
      padding:'10px 18px', borderRadius:'10px', fontSize:'13px',
      boxShadow:'0 4px 16px rgba(0,0,0,.4)', maxWidth:'340px' });
    document.body.appendChild(d);
    setTimeout(function () { d.remove(); }, 3200);
  }
  function fmt(bytes) {
    if (!bytes) return '0 MB';
    var mb = bytes / 1048576;
    return mb > 1024 ? (mb/1024).toFixed(1) + ' GB' : mb.toFixed(0) + ' MB';
  }
  function fmtDate(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  }
  function el(tag, attrs, css) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]);
    });
    if (css) Object.assign(e.style, css);
    return e;
  }
  function q(root, id) {
    return (root || document).querySelector('#' + P + '-' + id);
  }
  function set(root, id, txt) {
    var e = q(root, id); if (e) e.textContent = txt;
  }

  /* ── CSS injection (design-system classes from backup.html inline styles) ── */
  function injectStyles() {
    if (document.getElementById('od-bk-styles')) return;
    var s = document.createElement('style');
    s.id = 'od-bk-styles';
    s.textContent = [
      '.cloud-orb{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;',
      'flex-shrink:0;background:linear-gradient(150deg,var(--violet),var(--cyan));',
      'box-shadow:0 6px 20px -6px var(--violet)}',
      '.cloud-orb svg{width:26px;height:26px;color:#fff}',
      '.cloud-orb.off{background:var(--card-2);box-shadow:none}',
      '.cloud-orb.off svg{color:var(--fg-3)}',
      '.bk-conn{display:flex;align-items:center;gap:13px;padding:14px 0;',
      'border-bottom:1px solid var(--border)}',
      '.bk-conn:last-child{border-bottom:0}',
      '.bk-conn-ic{width:38px;height:38px;border-radius:10px;flex-shrink:0;',
      'display:grid;place-items:center;color:#fff}',
      '.bk-conn-ic.gh{background:#24292f}',
      '.bk-conn-ic.goog{background:#fff;border:1px solid var(--border)}',
      '.bk-conn-ic.s3{background:var(--warn)}',
      '.bk-conn.off .bk-conn-ic{filter:grayscale(1);opacity:.6}',
      '.bk-ctl{display:flex;align-items:center;justify-content:space-between;',
      'gap:16px;padding:13px 0;border-bottom:1px solid var(--border)}',
      '.bk-ctl:last-child{border-bottom:0}'
    ].join('');
    document.head.appendChild(s);
  }

  /* ── Card factory ────────────────────────────────────────── */
  // Returns {wrap, head, body} matching .card structure from design-system.css
  function card(title, sub, ic) {
    var wrap = el('div');
    wrap.className = 'card';
    Object.assign(wrap.style, { marginBottom:'16px' });
    // card-head
    var head = el('div');
    head.className = 'card-head';
    if (ic && typeof window.slmIcon === 'function') {
      var icEl = el('span', null,
        { color:'var(--violet)', width:'18px', height:'18px', display:'flex', flexShrink:'0' });
      icEl.innerHTML = window.slmIcon(ic);
      head.appendChild(icEl);
    }
    head.appendChild(el('h3', { text: title }));
    if (sub) head.appendChild(el('span', { text: sub }, { className:'sub' }));
    wrap.appendChild(head);
    var body = el('div');
    body.className = 'card-pad';
    wrap.appendChild(body);
    return { wrap: wrap, head: head, body: body };
  }

  /* ══════════════════════════════════════════════════════════
     HERO SECTION  (matches design: card.glass + cloud-orb + badge + stats)
  ══════════════════════════════════════════════════════════ */
  function buildHero(root) {
    var hero = el('section');
    hero.className = 'card glass';
    Object.assign(hero.style, { padding:'24px 26px', marginBottom:'16px' });

    var inner = el('div', null,
      { display:'flex', alignItems:'center', gap:'18px', flexWrap:'wrap' });

    // Cloud orb icon (gradient, matches design)
    var orb = el('span', { id: P + '-orb' });
    orb.className = 'cloud-orb';
    if (typeof window.slmIcon === 'function') {
      orb.innerHTML = window.slmIcon('cloud');
    } else {
      orb.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor"' +
        ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>';
    }
    inner.appendChild(orb);

    // Title + badge + sub
    var titleArea = el('div', null, { flex:'1', minWidth:'200px' });
    var titleRow = el('div', null, { display:'flex', alignItems:'center', gap:'10px' });
    var titleEl = el('h3', { id: P + '-cloud-title', text:'Checking cloud backup' },
      { fontSize:'17px', margin:'0' });
    titleRow.appendChild(titleEl);
    var syncBadge = el('span', { id: P + '-cloud-badge' });
    syncBadge.className = 'badge neutral';
    syncBadge.innerHTML = '<span class="dot"></span> Checking';
    titleRow.appendChild(syncBadge);
    titleArea.appendChild(titleRow);
    var subEl = el('p', { id: P + '-cloud-sub', text:'Connecting to your cloud…' },
      { fontSize:'13px', marginTop:'4px', color:'var(--fg-2)' });
    titleArea.appendChild(subEl);
    inner.appendChild(titleArea);

    // Stats: Last backup / Next scheduled / on-disk representation
    var statsRow = el('div', null, { display:'flex', gap:'26px', flexWrap:'wrap' });
    [
      { id:'hero-last',    lbl:'Last backup' },
      { id:'hero-next',    lbl:'Next scheduled' },
      { id:'hero-encrypt', lbl:'Backup format' }
    ].forEach(function (s) {
      var item = el('div');
      item.appendChild(el('div', { text: s.lbl },
        { fontSize:'11px', color:'var(--fg-3)' }));
      item.appendChild(el('div', { id: P + '-' + s.id, text:'…' },
        { fontSize:'16px', fontWeight:'650', marginTop:'2px' }));
      statsRow.appendChild(item);
    });
    inner.appendChild(statsRow);

    // Buttons
    var btnGroup = el('div', null,
      { display:'flex', flexDirection:'column', gap:'8px', alignItems:'flex-end' });
    var nowBtn = el('button', { type:'button', id: P + '-btn-now' });
    nowBtn.className = 'btn primary';
    if (typeof window.slmIcon === 'function') {
      nowBtn.innerHTML = window.slmIcon('cloud') + ' Back up now';
    } else {
      nowBtn.textContent = 'Back up now';
    }
    nowBtn.addEventListener('click', doBackupNow.bind(null, root));
    btnGroup.appendChild(nowBtn);

    var exportBtn = el('button', { type:'button' });
    exportBtn.className = 'btn ghost';
    exportBtn.textContent = 'Export .db.gz';
    exportBtn.addEventListener('click', function () {
      doExport(exportBtn);
    });
    btnGroup.appendChild(exportBtn);
    inner.appendChild(btnGroup);

    hero.appendChild(inner);
    return hero;
  }

  /* ══════════════════════════════════════════════════════════
     KPI STRIP  (design uses .kpi-strip + .card.kpi structure)
  ══════════════════════════════════════════════════════════ */
  function buildKPIs() {
    var kpis = [
      { id:'kpi-total', lbl:'Total backups',    icon:'cloud' },
      { id:'kpi-size',  lbl:'Local storage used', icon:'memories' },
      { id:'kpi-retain',lbl:'Retention',         icon:'clock' },
      { id:'kpi-count', lbl:'Local snapshots',   icon:'shield' }
    ];
    var strip = el('div', null, { marginBottom:'16px' });
    strip.className = 'kpi-strip';
    kpis.forEach(function (k) {
      var tile = el('div');
      tile.className = 'card kpi';
      var lbl = el('div');
      lbl.className = 'label';
      if (typeof window.slmIcon === 'function') {
        var icEl = el('span', null, { display:'contents' });
        icEl.innerHTML = window.slmIcon(k.icon);
        lbl.appendChild(icEl);
      }
      lbl.appendChild(document.createTextNode(' ' + k.lbl));
      tile.appendChild(lbl);
      var val = el('div', { id: P + '-' + k.id, text:'—' });
      val.className = 'value num';
      // Retention value ("10 snapshots") wraps at 30px — design overrides to 22px for this tile
      if (k.id === 'kpi-retain') val.style.fontSize = '22px';
      tile.appendChild(val);
      strip.appendChild(tile);
    });
    return strip;
  }

  /* ══════════════════════════════════════════════════════════
     CONNECTIONS — GitHub + Google OAuth + S3
     Design: .bk-conn rows with .bk-conn-ic.gh/.goog/.s3 brand icons
  ══════════════════════════════════════════════════════════ */
  var PROVIDERS = [
    { id:'github', label:'GitHub',
      icClass:'gh',   oauthPath:'/api/backup/oauth/github/start',
      hint:'Connect as your GitHub account to mirror backups to a private repo.' },
    { id:'google', label:'Google (Gmail)',
      icClass:'goog', oauthPath:'/api/backup/oauth/google/start',
      hint:'Back up to Google Drive via your Gmail account.' },
    { id:'s3',     label:'Custom S3 / WebDAV',
      icClass:'s3',   oauthPath:'/api/backup/oauth/s3/start',
      hint:'Bring your own S3-compatible bucket (Wasabi, MinIO, Cloudflare R2).' }
  ];

  // Inject brand SVGs after DOM settles — same pattern as design's backup.html
  function injectConnIcons(container) {
    var gh = container.querySelector('.bk-conn-ic.gh');
    if (gh && typeof window.slmIcon === 'function') gh.innerHTML = window.slmIcon('github');
    var goog = container.querySelector('.bk-conn-ic.goog');
    if (goog) goog.innerHTML =
      '<svg viewBox="0 0 24 24" width="20" height="20">' +
      '<path fill="#4285F4" d="M21.6 12.2c0-.6-.1-1.2-.2-1.8H12v3.4h5.4a4.6 4.6 0 0 1-2 3v2.5h3.2c1.9-1.7 3-4.3 3-7.1z"/>' +
      '<path fill="#34A853" d="M12 22c2.7 0 5-.9 6.6-2.4l-3.2-2.5c-.9.6-2 .9-3.4.9-2.6 0-4.8-1.7-5.6-4.1H3.1v2.6A10 10 0 0 0 12 22z"/>' +
      '<path fill="#FBBC05" d="M6.4 13.9a6 6 0 0 1 0-3.8V7.5H3.1a10 10 0 0 0 0 9z"/>' +
      '<path fill="#EA4335" d="M12 6.6c1.5 0 2.8.5 3.8 1.5l2.8-2.8A10 10 0 0 0 3.1 7.5l3.3 2.6C7.2 8.3 9.4 6.6 12 6.6z"/>' +
      '</svg>';
    var s3 = container.querySelector('.bk-conn-ic.s3');
    if (s3 && typeof window.slmIcon === 'function') s3.innerHTML = window.slmIcon('cloud');
  }

  function buildConnections(root) {
    var c = card('Connections', 'connect your own accounts', 'mesh');
    c.body.style.padding = '0 20px';
    var connList = el('div', { id: P + '-conn-list' });
    c.body.appendChild(connList);
    return c.wrap;
  }

  function renderConnections(root, destinations) {
    var connList = q(root, 'conn-list'); if (!connList) return;
    connList.innerHTML = '';
    var frag = document.createDocumentFragment();
    PROVIDERS.forEach(function (prov) {
      var dest = destinations.filter(function (d) {
        var destinationType = d.destination_type || d.type || d.provider;
        return destinationType === prov.id ||
          (prov.id === 'google' && destinationType === 'google_drive');
      })[0];
      frag.appendChild(buildConnRow(root, prov, dest || null));
    });
    connList.appendChild(frag);
    // Inject brand icons after DOM insertion
    injectConnIcons(connList);
  }

  function buildConnRow(root, prov, dest) {
    var row = el('div');
    row.className = 'bk-conn' + (dest ? '' : ' off');
    row.setAttribute('data-conn', prov.id);

    // Brand icon
    var iconWrap = el('span');
    iconWrap.className = 'bk-conn-ic ' + prov.icClass;
    row.appendChild(iconWrap);

    // Info area
    var info = el('div', null, { flex:'1' });
    info.appendChild(el('b', { text: prov.label }));
    var detailEl = el('div');
    detailEl.className = 'dim';
    Object.assign(detailEl.style, { fontSize:'12.5px' });
    if (dest) {
      var config = {};
      try { config = typeof dest.config === 'string' ? JSON.parse(dest.config) : (dest.config || {}); }
      catch (e) { config = {}; }
      var repoOrBucket = dest.repo || dest.bucket || dest.folder_id ||
        config.full_repo || config.repo || config.folder || '';
      var syncStatus = dest.last_sync_status || 'never';
      var statusText = syncStatus === 'success' && dest.last_sync_at
        ? 'Last sync succeeded ' + fmtDate(dest.last_sync_at)
        : syncStatus === 'failed'
          ? 'Last sync failed'
          : 'Connected · never synced';
      detailEl.textContent = statusText + (repoOrBucket ? ' · ' + repoOrBucket : '');
    } else {
      detailEl.textContent = prov.hint;
    }
    info.appendChild(detailEl);
    row.appendChild(info);

    // Action button
    var btnWrap = el('div', null, { display:'flex', gap:'7px', flexShrink:'0' });
    if (dest) {
      var syncBtn = el('button', { type:'button' });
      syncBtn.className = 'btn sm ghost';
      syncBtn.textContent = 'Sync all';
      syncBtn.addEventListener('click', function () {
        doSync(root);
      });
      btnWrap.appendChild(syncBtn);

      var discBtn = el('button', { type:'button' });
      discBtn.className = 'btn sm';
      discBtn.textContent = 'Disconnect';
      discBtn.addEventListener('click', function () {
        doDisconnect(root, dest.id || dest.dest_id || prov.id);
      });
      btnWrap.appendChild(discBtn);
    } else {
      var connBtn = el('button', { type:'button' });
      connBtn.className = 'btn sm primary';
      connBtn.textContent = 'Connect';
      connBtn.addEventListener('click', function () {
        openOAuth(root, prov.oauthPath, prov.label);
      });
      btnWrap.appendChild(connBtn);
    }
    row.appendChild(btnWrap);
    return row;
  }

  /* ══════════════════════════════════════════════════════════
     BACKUP SCOPE + SCHEDULE
     Design: .bk-ctl rows with .switch toggles; Schedule at bottom
  ══════════════════════════════════════════════════════════ */
  function buildScope() {
    var c = card('Backup scope', 'current managed database set', 'operations');

    // Scope items — matching design's 4 items with toggle switches
    var scopeItems = [
      { id:'scope-mem',   lbl:'Memories',
        sub:'<span class="mono">memory.db</span> · facts & conversations', on: true },
      { id:'scope-learn', lbl:'Learning data',
        sub:'<span class="mono">learning.db</span> · ranking model', on: true },
      { id:'scope-audit', lbl:'Audit & code data',
        sub:'<span class="mono">audit_chain.db · code_graph.db</span> · when present', on: true },
      { id:'scope-pending', lbl:'Pending operations',
        sub:'<span class="mono">pending.db</span> · when present', on: true }
    ];

    scopeItems.forEach(function (item) {
      var row = el('div');
      row.className = 'bk-ctl';

      var labelGroup = el('div');
      labelGroup.appendChild(el('b', { text: item.lbl }));
      var subEl = el('div');
      subEl.className = 'dim';
      subEl.style.fontSize = '12.5px';
      subEl.innerHTML = item.sub;
      labelGroup.appendChild(subEl);
      row.appendChild(labelGroup);

      var sw = el('button', { type:'button', id: P + '-' + item.id });
      sw.className = 'switch' + (item.on ? ' on' : '');
      sw.setAttribute('role', 'switch');
      sw.setAttribute('aria-checked', String(item.on));
      sw.disabled = true;
      sw.title = 'Managed database scope is fixed in this release';
      row.appendChild(sw);
      c.body.appendChild(row);
    });

    // Schedule row — matching design
    var schedRow = el('div');
    schedRow.className = 'bk-ctl';
    schedRow.appendChild(el('b', { text: 'Schedule' }));

    var segs = ['Manual', 'Daily', 'Weekly'];
    var segWrap = el('div');
    segWrap.className = 'seg';
    segs.forEach(function (s) {
      var b = el('button', { type:'button', id: P + '-seg-' + s.toLowerCase() });
      b.textContent = s;
      if (s === 'Daily') b.className = 'active';
      b.addEventListener('click', function () {
        segs.forEach(function (t) {
          var tb = document.getElementById(P + '-seg-' + t.toLowerCase());
          if (tb) tb.className = (t === s ? 'active' : '');
        });
        var schedule = s === 'Manual'
          ? { enabled: false }
          : { enabled: true, interval_hours: s === 'Daily' ? 24 : 168 };
        authMutation('/api/backup/configure', 'POST', schedule)
          .then(function (r) { return r.json(); })
          .then(function () { toast(s + ' schedule saved'); })
          .catch(function () { toast('Schedule save failed', true); });
      });
      segWrap.appendChild(b);
    });
    schedRow.appendChild(segWrap);
    c.body.appendChild(schedRow);

    return c.wrap;
  }

  /* ══════════════════════════════════════════════════════════
     BACKUP HISTORY TABLE
     Design: .tbl class, columns When/Scope/Size/Destination/Status
  ══════════════════════════════════════════════════════════ */
  // Map backup type from API to a human-readable scope string
  function fmtScope(bk) {
    var t = (bk.type || 'memory').toLowerCase();
    if (t === 'full')     return 'Memories · Learning · Config';
    if (t === 'memory')   return 'Memories';
    if (t === 'learning') return 'Learning data';
    if (t === 'config')   return 'Config & profiles';
    return esc(t);
  }

  function fmtDest(bk) {
    var d = bk.destination || bk.dest || '';
    if (!d) return 'local · ~/.slm/backups';
    return esc(d);
  }

  var _histFilter = 'all'; // all | cloud | local

  function buildHistory() {
    var c = card('Backup history', 'local snapshots and upload evidence', 'health');

    // Add filter segment to card-head (All / Cloud / Local)
    var spacer = el('div');
    spacer.className = 'spacer';
    c.head.appendChild(spacer);

    var seg = el('div');
    seg.className = 'seg';
    ['All', 'Cloud', 'Local'].forEach(function (f) {
      var b = el('button', { type:'button', id: P + '-hist-' + f.toLowerCase() });
      b.textContent = f;
      if (f === 'All') b.className = 'active';
      b.addEventListener('click', function () {
        ['all','cloud','local'].forEach(function (k) {
          var btn = document.getElementById(P + '-hist-' + k);
          if (btn) btn.className = (k === f.toLowerCase() ? 'active' : '');
        });
        _histFilter = f.toLowerCase();
      });
      seg.appendChild(b);
    });
    c.head.appendChild(seg);

    c.body.style.padding = '0';
    var wrap = el('div', { id: P + '-history-wrap' });
    c.body.appendChild(wrap);
    return c.wrap;
  }

  function renderHistory(root, backups) {
    var wrap = q(root, 'history-wrap'); if (!wrap) return;
    if (!backups || backups.length === 0) {
      wrap.innerHTML = '<div style="padding:28px;text-align:center;color:var(--fg-2);' +
        'font-size:13px">No backups yet. Click "Back up now" to create your first snapshot.</div>';
      return;
    }
    var table = el('table');
    table.className = 'tbl';
    table.id = P + '-hist-tbl';

    var thead = el('thead');
    var headTr = el('tr');
    ['When', 'Scope', 'Size', 'Destination', 'Status', ''].forEach(function (h) {
      var th = el('th', { text: h });
      headTr.appendChild(th);
    });
    thead.appendChild(headTr);
    table.appendChild(thead);

    var tbody = el('tbody');
    backups.slice(0, 20).forEach(function (bk) {
      var tr = el('tr');
      // When
      var tdWhen = el('td'); tdWhen.className = 'dim';
      tdWhen.textContent = fmtDate(bk.created);
      tr.appendChild(tdWhen);
      // Scope
      var tdScope = el('td');
      tdScope.textContent = fmtScope(bk);
      tr.appendChild(tdScope);
      // Size
      var tdSize = el('td'); tdSize.className = 'num';
      tdSize.textContent = bk.size_mb ? bk.size_mb.toFixed(0) + ' MB' : '—';
      tr.appendChild(tdSize);
      // Destination
      var tdDest = el('td'); tdDest.className = 'mono';
      tdDest.style.fontSize = '12px';
      tdDest.textContent = fmtDest(bk);
      tr.appendChild(tdDest);
      // Status badge
      var tdStatus = el('td');
      var statusBadge = el('span');
      var statusTxt = bk.status || 'Complete';
      statusBadge.className = 'badge ' +
        (statusTxt === 'Complete' || statusTxt === 'ok' ? 'ok' :
         statusTxt === 'Partial' ? 'warn' : 'danger');
      statusBadge.innerHTML = '<span class="dot"></span>' + esc(statusTxt);
      tdStatus.appendChild(statusBadge);
      tr.appendChild(tdStatus);
      // Restore is intentionally absent until the dashboard has a guarded,
      // tested restore endpoint and a daemon-safe restart workflow.
      var tdAct = el('td', null, { textAlign:'right' });
      tr.appendChild(tdAct);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.innerHTML = '';
    wrap.appendChild(table);
  }

  /* ══════════════════════════════════════════════════════════
     ACTIONS: backup now, sync, disconnect, oauth
  ══════════════════════════════════════════════════════════ */
  function doBackupNow(root) {
    var btn = q(root, 'btn-now'); if (btn) { btn.disabled = true; btn.textContent = 'Creating…'; }
    authMutation('/api/backup/create', 'POST')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        toast(d.success ? 'Backup created: ' + esc(d.filename || '') : 'Backup failed', !d.success);
        if (btn) { btn.disabled = false; btn.textContent = 'Back up now'; }
        loadAll(root);
      })
      .catch(function () {
        toast('Backup failed', true);
        if (btn) { btn.disabled = false; btn.textContent = 'Back up now'; }
      });
  }

  function doSync(root) {
    authMutation('/api/backup/sync', 'POST')
      .then(function (r) { return r.json(); })
      .then(function (d) { toast(d.success ? 'Sync started' : 'Sync failed: ' + esc(d.error || ''), !d.success); })
      .catch(function () { toast('Sync error', true); });
  }

  function doExport(button) {
    if (button) button.disabled = true;
    authMutation('/api/backup/export', 'POST')
      .then(function (response) {
        return response.blob().then(function (blob) {
          return { blob: blob, disposition: response.headers.get('Content-Disposition') || '' };
        });
      })
      .then(function (download) {
        var match = download.disposition.match(/filename="?([^";]+)"?/i);
        var link = document.createElement('a');
        var objectUrl = URL.createObjectURL(download.blob);
        link.href = objectUrl;
        link.download = match ? match[1] : 'superlocalmemory-backup.db.gz';
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(objectUrl);
        toast('Backup export downloaded');
      })
      .catch(function () { toast('Backup export failed', true); })
      .finally(function () { if (button) button.disabled = false; });
  }

  function doDisconnect(root, destId) {
    if (!window.confirm('Disconnect this cloud destination? Existing backups are not deleted.')) return;
    authMutation('/api/backup/disconnect/' + encodeURIComponent(destId), 'DELETE')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        toast(d.success ? 'Disconnected' : 'Disconnect failed: ' + esc(d.error || ''), !d.success);
        loadDestinations(root);
      })
      .catch(function () { toast('Disconnect failed', true); });
  }

  function openOAuth(root, oauthPath, providerLabel) {
    var popup = window.open(oauthPath, providerLabel + '-oauth', 'width=560,height=700');
    if (!popup) { toast('Popup blocked — allow popups for this page', true); return; }
    var timer = setInterval(function () {
      if (!popup || popup.closed) {
        clearInterval(timer);
        // A closed popup is not proof of success; refresh runtime truth only.
        loadDestinations(root, true);
      }
    }, 800);
  }

  /* ══════════════════════════════════════════════════════════
     DATA LOADING
  ══════════════════════════════════════════════════════════ */
  function loadStatus(root) {
    fetch('/api/backup/status').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;

        // Hero stats
        var lastStr = d.last_backup ? fmtDate(d.last_backup) : 'Never';
        set(root, 'hero-last',    lastStr);
        var next = !d.enabled
          ? 'Manual only'
          : (d.last_backup && d.interval_hours
            ? new Date(new Date(d.last_backup).getTime() + d.interval_hours * 3600000).toLocaleString()
            : 'Scheduled');
        set(root, 'hero-next',    next);
        set(root, 'hero-encrypt', 'Plain SQLite');

        // Hero title + badge: reflect configured and witnessed sync state.
        var hasCloud = d.cloud_destinations && d.cloud_destinations.length > 0;
        var allSuccessful = hasCloud && d.cloud_destinations.every(function (dest) {
          return dest.last_sync_status === 'success' && Boolean(dest.last_sync_at);
        });
        var failed = hasCloud && d.cloud_destinations.some(function (dest) {
          return dest.last_sync_status === 'failed';
        });
        var titleEl = q(root, 'cloud-title');
        if (titleEl) titleEl.textContent = hasCloud ? 'Cloud destination configured' : 'No cloud destination';
        var badgeEl = q(root, 'cloud-badge');
        if (badgeEl) {
          badgeEl.className = 'badge ' + (failed ? 'danger' : allSuccessful ? 'ok' : 'warn');
          badgeEl.replaceChildren();
          var dot = el('span'); dot.className = 'dot';
          badgeEl.appendChild(dot);
          badgeEl.appendChild(document.createTextNode(
            failed
              ? ' One or more syncs failed'
              : allSuccessful
                ? ' Latest reported syncs succeeded'
                : hasCloud ? ' Sync incomplete or pending' : ' Not connected'
          ));
        }
        var subEl = q(root, 'cloud-sub');
        if (subEl) {
          if (hasCloud) {
            var dest = d.cloud_destinations[0];
            var destStr = dest.display_name || dest.destination_type || 'cloud';
            subEl.textContent = 'Plain SQLite copies upload to your private ' + destStr +
              '. Access protection comes from that provider account.';
          } else {
            subEl.textContent = 'Local snapshots are plaintext SQLite. Connect a private provider only if you accept its access controls.';
          }
        }
        var orb = q(root, 'orb');
        if (orb) orb.classList.toggle('off', !hasCloud);

        // KPI values (matched to new KPI ids)
        var cnt = d.backup_count || d.learning_backup_count || 0;
        set(root, 'kpi-total',  String(cnt));
        var totalMb = d.total_size_mb || 0;
        set(root, 'kpi-size',   totalMb > 1024
          ? (totalMb / 1024).toFixed(1) + ' GB'
          : totalMb.toFixed(0) + ' MB');
        set(root, 'kpi-retain', (d.max_backups || cnt) + ' snapshots');
        set(root, 'kpi-count',  String(cnt));

        if (d.backups) renderHistory(root, d.backups);

        // Sync schedule segment buttons
        var hours = d.interval_hours || 168;
        var active = !d.enabled ? 'manual' : hours <= 24 ? 'daily' : 'weekly';
        ['daily','weekly','manual'].forEach(function (s) {
          var b = document.getElementById(P + '-seg-' + s);
          if (b) b.className = (s === active ? 'active' : '');
        });
      }).catch(function () {});
  }

  function loadDestinations(root, forceReload) {
    fetch('/api/backup/destinations').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var dests = d && d.destinations ? d.destinations : [];
        renderConnections(root, dests);
        if (forceReload) {
          toast('Destination status refreshed');
          // Re-load status without inferring whether the popup succeeded.
          loadStatus(root);
        }
      }).catch(function () {});
  }

  function loadList(root) {
    fetch('/api/backup/list').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d && d.backups) renderHistory(root, d.backups);
      }).catch(function () {});
  }

  function loadAll(root) {
    loadStatus(root);
    loadDestinations(root);
    loadList(root);
  }

  /* ══════════════════════════════════════════════════════════
     MAIN RENDER
  ══════════════════════════════════════════════════════════ */
  function odRenderBackup(container) {
    if (!container) return;
    injectStyles();
    // Clear skeleton / prior render
    Array.from(container.children).forEach(function (c) { c.style.display = 'none'; });

    var hub = el('div', null, { padding:'26px', maxWidth:'960px' });

    // Page head — matches design's <div class="page-head"> block
    var pageHead = el('div');
    pageHead.className = 'page-head';
    pageHead.appendChild(el('h2', { text:'Backup & cloud sync' }));
    var desc = el('p');
    desc.textContent = 'Your memory lives on this machine as plaintext SQLite snapshots. ' +
      'You can copy snapshots to a private GitHub repository or your Google Drive. ' +
      'Those providers control remote access; this release does not encrypt backup files.';
    pageHead.appendChild(desc);
    hub.appendChild(pageHead);

    hub.appendChild(buildHero(hub));
    hub.appendChild(buildKPIs());

    // Two-column area: Connections | Scope
    var twoCol = el('div');
    twoCol.className = 'grid';
    twoCol.style.gridTemplateColumns = '1fr 1fr';
    twoCol.style.alignItems = 'start';
    twoCol.appendChild(buildConnections(hub));
    twoCol.appendChild(buildScope());
    hub.appendChild(twoCol);

    hub.appendChild(buildHistory());
    container.insertBefore(hub, container.firstChild);
    loadAll(hub);
  }

  /* ══════════════════════════════════════════════════════════
     OVERLAY — called from od-settings.js Backup group
  ══════════════════════════════════════════════════════════ */
  function odOpenBackupDashboard() {
    var existing = document.getElementById(OVERLAY_ID);
    if (existing) { existing.style.display = 'flex'; loadAll(existing); return; }

    var overlay = el('div', { id: OVERLAY_ID }, {
      position:'fixed', inset:'0', zIndex:'9000',
      background:'var(--page)', overflowY:'auto',
      display:'flex', flexDirection:'column'
    });

    // Top bar with close button
    var topbar = el('div', null, { position:'sticky', top:'0', zIndex:'9001',
      background:'var(--card)', borderBottom:'1px solid var(--border)',
      display:'flex', alignItems:'center', gap:'12px', padding:'0 22px', height:'52px' });

    var backBtn = el('button', { type:'button' }); backBtn.className = 'btn ghost sm';
    backBtn.textContent = '← Settings';
    backBtn.addEventListener('click', function () {
      overlay.style.display = 'none';
    });
    topbar.appendChild(backBtn);
    topbar.appendChild(el('span', { text:'Backup & Cloud' },
      { fontWeight:'640', fontSize:'15px' }));

    var escHint = el('span', { text:'Esc to close' },
      { marginLeft:'auto', fontSize:'11.5px', color:'var(--fg-3)' });
    topbar.appendChild(escHint);

    overlay.appendChild(topbar);

    // Content area
    var content = el('div', null, { flex:'1', padding:'22px' });
    overlay.appendChild(content);
    document.body.appendChild(overlay);

    // ESC key close
    document.addEventListener('keydown', function onEsc(e) {
      if (e.key === 'Escape' && overlay.style.display !== 'none') {
        overlay.style.display = 'none';
      }
    });

    odRenderBackup(content);
  }

  /* ══════════════════════════════════════════════════════════
     BOOT
  ══════════════════════════════════════════════════════════ */
  window.odRenderBackup      = odRenderBackup;
  window.odOpenBackupDashboard = odOpenBackupDashboard;

  document.addEventListener('DOMContentLoaded', function () {
    // Wire into a backup-pane if the shell gains one in future
    var pane = document.getElementById('backup-pane');
    if (pane) odRenderBackup(pane);
  });

}());
