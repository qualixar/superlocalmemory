/* od-settings.js — SuperLocalMemory Settings Hub v2.1
 * ALL 13 config groups wired to live daemon endpoints.
 * Exposes: window.odRenderSettings(container)
 * Auth: X-Install-Token cached from GET /internal/token (used on all new PUTs)
 * Endpoints:
 *   GET/POST  /api/v3/mode, /api/v3/mode/set
 *   GET/PUT   /api/v3/embedding/config, POST /api/v3/embedding/test
 *   GET/PUT   /api/v3/storage/config       → restart_required
 *   GET/PUT   /api/v3/scope/config
 *   GET/PUT   /api/v3/auto-capture/config
 *   GET/PUT   /api/v3/auto-recall/config
 *   GET/PUT   /api/v3/auto-invoke/config
 *   GET/PUT   /api/v3/forgetting/config + GET /api/v3/forgetting/stats
 *   GET/POST  /api/evolution/status, /api/evolution/enable|disable|config
 *   GET/POST  /api/backup/status, /api/backup/configure, /api/backup/create
 *   GET/PUT   /api/v3/mesh/config
 *   GET/PUT   /api/v3/trust/config
 *   GET/PUT   /api/v3/daemon/config        → restart_required
 * CRIT fixes: (1) every control reads on render and persists on change;
 *             (2) all new PUTs include X-Install-Token via cached authPut helper;
 *             (3) restart_required surfaced inline after storage and daemon saves.
 * Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0
 */
(function () {
  'use strict';

  var ID = 'od-settings-hub';
  var P  = 'od-s';
  var _tokenCache    = null;

  /* ── Auth helper ──────────────────────────────────────────── */
  function tokenError(message) {
    var error = new Error(message);
    error.code = 'install_token_missing';
    return error;
  }
  function getToken(refresh) {
    if (refresh) _tokenCache = null;
    if (_tokenCache) return Promise.resolve(_tokenCache);
    return fetch('/internal/token')
      .then(function (response) {
        if (!response.ok) throw new Error('Could not refresh dashboard credential');
        return response.json();
      })
      .then(function (data) {
        var token = data.token || '';
        if (!token) throw tokenError('Dashboard credential is empty');
        _tokenCache = token;
        return token;
      });
  }
  function sendAuthenticated(url, method, body, token) {
    return fetch(url, {
      method: method,
      headers: { 'Content-Type': 'application/json', 'X-Install-Token': token },
      body: JSON.stringify(body)
    });
  }
  function requireSuccess(response) {
    if (response.ok) return Promise.resolve(response);
    var details = typeof response.json === 'function'
      ? response.json().catch(function () { return {}; })
      : Promise.resolve({});
    return details.then(function (data) {
      var d = data || {};  // 409/422 bodies put a machine code in `error`, the sentence in `detail`
      var detail = typeof d.detail === 'string' ? d.detail : (d.detail && d.detail.message);
      var error = new Error(detail || d.error || d.message || ('Request failed with status ' + response.status));
      error.status = response.status;
      error.body = d;
      throw error;
    });
  }
  /** Authenticated dashboard mutation; callers must re-read daemon truth on success. */
  function authRequest(url, method, body) {
    function attempt(refresh) {
      return getToken(refresh).then(function (token) {
        return sendAuthenticated(url, method, body, token);
      });
    }
    return attempt(false).then(function (response) {
      if (response.status === 401 || response.status === 403) {
        return attempt(true).then(requireSuccess);
      }
      return requireSuccess(response);
    }, function (error) {
      if (error && error.code === 'install_token_missing') {
        return attempt(true).then(requireSuccess);
      }
      throw error;
    });
  }
  function authPut(url, body) { return authRequest(url, 'PUT', body); }
  function authPost(url, body) { return authRequest(url, 'POST', body); }

  /* ── Helpers ──────────────────────────────────────────────── */
  /** .catch handler: the status code and the daemon's own reason, never a bare "Error". */
  function failed(id, label, after) {
    return function (e) {
      var m = (e && e.status ? 'Error ' + e.status : 'Error') + ': ' + ((e && e.message) || 'request failed');
      if (id) setSt(id, m, false);
      toast(label + ' failed — ' + m, true);
      if (after) after(e);
    };
  }
  function readOk(r) { return requireSuccess(r).then(function (ok) { return ok.json(); }); }
  function toast(msg, err) {
    if (typeof window.showToast === 'function') { window.showToast(msg); return; }
    var d = document.createElement('div');
    d.textContent = msg;
    Object.assign(d.style, { position:'fixed', bottom:'20px', right:'20px', zIndex:'9999',
      background: err ? 'var(--danger)' : 'var(--violet)', color:'#fff',
      padding:'10px 18px', borderRadius:'10px', fontSize:'13px',
      boxShadow:'0 4px 16px rgba(0,0,0,.4)', maxWidth:'340px' });
    document.body.appendChild(d);
    setTimeout(function () { d.remove(); }, 3200);
  }
  function q(s) { return document.getElementById(P + '-' + s); }
  function el(tag, a, css) {
    var e = document.createElement(tag);
    if (a) Object.keys(a).forEach(function (k) {
      if (k === 'text') e.textContent = a[k]; else e.setAttribute(k, a[k]);
    });
    if (css) Object.assign(e.style, css);
    return e;
  }
  function setSt(id, msg, ok) {
    var e = q(id + '-st'); if (!e) return;
    e.textContent = msg;
    e.style.color = ok === true ? 'var(--ok)' : ok === false ? 'var(--danger)' : 'var(--fg-2)';
    if (ok === true) setTimeout(function () { if (e) e.textContent = ''; }, 2500);
  }
  function showRestartNote(id, needed) {
    var e = q(id + '-restart'); if (!e) return;
    e.style.display = needed ? 'flex' : 'none';
  }

  /* ── Control factories ────────────────────────────────────── */
  function makeSwitch(id, on) {
    var b = el('button', { type:'button', role:'switch',
      'aria-checked': String(!!on), id: P + '-' + id });
    b.className = 'switch' + (on ? ' on' : '');
    b.addEventListener('click', function () {
      b.classList.toggle('on');
      b.setAttribute('aria-checked', String(b.classList.contains('on')));
      b.dispatchEvent(new CustomEvent('od-toggle',
        { bubbles:true, detail:{ id:id, on:b.classList.contains('on') } }));
    });
    return b;
  }
  function makeSel(id, opts) {
    var s = el('select', { id: P + '-' + id });
    Object.assign(s.style, { height:'34px', borderRadius:'var(--r-md)',
      border:'1px solid var(--border)', background:'var(--card-2)',
      color:'var(--fg)', padding:'0 10px', fontSize:'13px',
      minWidth:'140px', outline:'none' });
    opts.forEach(function (o) {
      var opt = document.createElement('option');
      opt.value = o.v; opt.textContent = o.l; s.appendChild(opt);
    });
    return s;
  }
  function makeTin(id, ph, type) {
    var i = el('input', { id: P + '-' + id, type: type || 'text', placeholder: ph || '' });
    Object.assign(i.style, { height:'34px', borderRadius:'var(--r-md)',
      border:'1px solid var(--border)', background:'var(--card-2)',
      color:'var(--fg)', padding:'0 10px', fontSize:'13px',
      minWidth:'160px', maxWidth:'260px', outline:'none' });
    return i;
  }
  /** Same styled input as makeTin, with a datalist attached via the `list`
   *  attribute so the suggestions show while typing is still free-form. */
  function makeSuggest(id, ph) {
    var i = makeTin(id, ph);
    var list = el('datalist', { id: P + '-' + id + '-list' });
    i.setAttribute('list', P + '-' + id + '-list');
    i.appendChild(list);
    return i;
  }
  function makeRange(id, min, max, step, val) {
    var w = el('div', null, { display:'flex', alignItems:'center', gap:'10px' });
    var inp = el('input', { id: P + '-' + id, type:'range',
      min:min, max:max, step:step, value:val });
    inp.style.width = '110px';
    var disp = el('span', { id: P + '-' + id + '-disp', text: val },
      { fontFamily:'var(--font-mono)', fontSize:'12px', minWidth:'36px' });
    inp.addEventListener('input', function () {
      disp.textContent = Number(this.value).toFixed(parseFloat(step) < 1 ? 2 : 0);
    });
    w.appendChild(inp); w.appendChild(disp);
    return w;
  }
  function makeBtn(id, txt, pri) {
    var b = el('button', { type:'button', id: P + '-' + id });
    b.className = 'btn sm ' + (pri ? 'primary' : 'ghost');
    b.textContent = txt;
    return b;
  }
  function makeSt(id) {
    return el('span', { id: P + '-' + id + '-st' },
      { fontSize:'12px', marginLeft:'8px', color:'var(--fg-2)' });
  }
  function bRow() {
    var d = el('div', null, { display:'flex', gap:'8px', alignItems:'center', flexWrap:'wrap' });
    Array.from(arguments).forEach(function (c) { d.appendChild(c); });
    return d;
  }
  function restartDaemonFromNote(btn) {
    // Same endpoint the Governance › System-control button uses. POST is
    // mutating, so core.js attaches the X-Install-Token automatically.
    btn.disabled = true;
    btn.textContent = 'Restarting…';
    fetch('/api/daemon/restart', { method: 'POST', credentials: 'same-origin' }).then(requireSuccess)
      .then(function () {
        btn.textContent = 'Restarting… reconnecting';
        setTimeout(function () { window.location.reload(); }, 4000);
      })
      .catch(failed(null, 'Restart request', function () {
        btn.disabled = false;
        btn.textContent = 'Restart daemon now';
      }));
  }
  function makeRestartNote(id) {
    var wrap = el('div', { id: P + '-' + id + '-restart' }, {
      display: 'none', alignItems: 'center', gap: '10px', flexWrap: 'wrap',
      fontSize: '12px', fontWeight: '600', color: '#d97706',
      padding: '5px 10px', background: 'rgba(217,119,6,.12)',
      borderRadius: 'var(--r-md,6px)'
    });
    wrap.appendChild(el('span', { text: 'Config saved — restart the daemon to apply.' }));
    var btn = el('button', { type: 'button', text: 'Restart daemon now' });
    btn.className = 'btn sm';
    btn.addEventListener('click', function () { restartDaemonFromNote(btn); });
    wrap.appendChild(btn);
    return wrap;
  }

  /* ── Layout ───────────────────────────────────────────────── */
  function drawDecay(kappa) { if (window.odDecayPreview) window.odDecayPreview.draw(kappa); }

  function makeRow(lbl, desc, key, ctrl) {
    var d = el('div', { 'data-set':'', 'data-txt':(lbl + ' ' + key).toLowerCase() });
    // Wraps: a wide control (the re-index or engine-upgrade panel) drops below
    // its label instead of squeezing the label to one letter per line.
    Object.assign(d.style, { display:'flex', alignItems:'center', flexWrap:'wrap',
      justifyContent:'space-between', gap:'16px',
      padding:'14px 20px', borderBottom:'1px solid var(--border)' });
    var lv = el('div', null, { flex:'1 1 220px', minWidth:'180px' });
    var labelId = 'od-setting-label-' + (ctrl.id || key || lbl).replace(/[^a-z0-9-]/gi, '-');
    lv.appendChild(el('b', { text: lbl, id: labelId }));
    var fields = ctrl.matches('input,select,[role=switch]') ? [ctrl] : Array.from(ctrl.querySelectorAll('input,select,[role=switch]'));
    fields.forEach(function (field) { field.setAttribute('aria-labelledby', labelId); });
    if (desc) lv.appendChild(el('span', { text: desc },
      { fontSize:'12.5px', color:'var(--fg-2)', display:'block' }));
    // The config key (e.g. "mode", "llm.provider") is informational, not
    // decorative — a user reading Settings needs to read it (it's the
    // literal key to set via config file/CLI/env var) — so --fg-2, not
    // --fg-3 (2.97:1/3.89:1, under WCAG AA 4.5:1).
    if (key) lv.appendChild(el('div', { text: key },
      { fontFamily:'var(--font-mono)', fontSize:'10.5px', color:'var(--fg-2)', marginTop:'3px' }));
    d.appendChild(lv);
    var cv = el('div', null, { flex:'0 1 auto', minWidth:'0', maxWidth:'100%' });
    cv.appendChild(ctrl); d.appendChild(cv);
    return d;
  }
  function makeGrp(title, ic, rows) {
    var g = el('div', { 'data-grp':'' });
    Object.assign(g.style, { background:'var(--card)', border:'1px solid var(--border)',
      borderRadius:'var(--r-lg)', boxShadow:'var(--sh-md)', marginBottom:'14px' });
    var head = el('div', null, { display:'flex', alignItems:'center', gap:'10px',
      padding:'16px 20px', borderBottom:'1px solid var(--border)' });
    if (ic && typeof window.slmIcon === 'function') {
      var icEl = el('span', null,
        { color:'var(--violet)', width:'18px', height:'18px', display:'flex', flexShrink:'0' });
      icEl.innerHTML = window.slmIcon(ic);
      head.appendChild(icEl);
    }
    head.appendChild(el('h3', { text: title },
      { fontSize:'14.5px', fontWeight:'640', margin:'0' }));
    g.appendChild(head);
    rows.forEach(function (r) { g.appendChild(r); });
    if (g.lastElementChild !== head) g.lastElementChild.style.borderBottom = '0';
    return g;
  }

  /* ═══════════════════════════════════════════════════════════
     Model suggestions (4.1.22) — GET /api/v3/models/catalog once, then
     populate the LLM and embedding datalists and a one-line hint per input.
     Never throws: a failed/absent catalog just leaves plain text inputs.
  ═══════════════════════════════════════════════════════════ */
  var moduleContext = { tokenError:tokenError, getToken:getToken, sendAuthenticated:sendAuthenticated, requireSuccess:requireSuccess, authRequest:authRequest, authPut:authPut, authPost:authPost, failed:failed, readOk:readOk, toast:toast, q:q, el:el, setSt:setSt, showRestartNote:showRestartNote, makeSwitch:makeSwitch, makeSel:makeSel, makeTin:makeTin, makeSuggest:makeSuggest, makeRange:makeRange, makeBtn:makeBtn, makeSt:makeSt, bRow:bRow, restartDaemonFromNote:restartDaemonFromNote, makeRestartNote:makeRestartNote, drawDecay:drawDecay, makeRow:makeRow, makeGrp:makeGrp, P:P, ID:ID };
  var SLMSettingsModels = window.SLMSettingsModels(moduleContext);
  Object.assign(moduleContext, SLMSettingsModels);
  var { clearDatalist, addOption, stripLatest, computeModelHint, computeEmbHint, updateModelHint, updateEmbHint, fillModelSuggestions, loadModelCatalog, buildMode, loadMode, saveMode, modelSaved, reindexBusy, testEndpoint, testMode, buildEmb, loadEmb, saveEmb, testEmb } = SLMSettingsModels;
  var SLMSettingsGroups = window.SLMSettingsGroups(moduleContext);
  Object.assign(moduleContext, SLMSettingsGroups);
  var { buildStorage, loadStorage, saveStorage, buildScope, loadScope, buildRuntime, loadRuntime, buildCapture, loadCapture, buildRecall, loadRecall, buildInvoke, loadInvoke, buildForgetting, loadForgetting, saveForgetting, buildEvolution, loadEvolution, buildBackup, loadBackup, saveBackupConfig, buildMesh, loadMesh, buildTrust, loadTrust, saveTrust, buildRateLimit, _rlSetLb, loadRateLimit, saveRateLimit, buildDaemon, loadDaemon, saveDaemon } = SLMSettingsGroups;
  function buildAnswerCheck() {
    var mount = el('div', { 'data-set':'', 'data-txt':'answer check laya jev sufficiency' },
      { padding:'16px 20px' });
    var g = makeGrp('Answer check', 'shield', [mount]);
    g.id = 'settings-answer-check';  // deep-link target of the Answer Check tab
    if (window.SLMAnswerCheck && typeof window.SLMAnswerCheck.renderInto === 'function') {
      window.SLMAnswerCheck.renderInto(mount);
    }
    return g;
  }

  // 4.2.30 D-28: what SLM measures, in plain words, with one switch (od-measurement.js).
  function buildMeasurement() {
    var mount = el('div', { 'data-set':'', 'data-txt':'measure measurement withholding swaps receipts help slm what works' },
      { padding:'16px 20px' });
    var g = makeGrp('Help SLM measure what works', 'shield', [mount]);
    g.id = 'settings-measurement';  // deep-link target of the dashboard banner
    if (window.SLMMeasurement && typeof window.SLMMeasurement.renderInto === 'function') {
      window.SLMMeasurement.renderInto(mount);
    }
    return g;
  }

  // 4.2.30 D-23 (§8.10.3 k): the kill switch and the bring-back-retired command (od-killswitch.js).
  function buildKillSwitch() {
    var mount = el('div', { 'data-set':'', 'data-txt':'turn off everything new 4.2.30 kill switch retired bring back memories' },
      { padding:'16px 20px' });
    var g = makeGrp('Turn off everything new in 4.2.30', 'shield', [mount]);
    g.id = 'settings-kill-switch';
    if (window.SLMKillSwitch && typeof window.SLMKillSwitch.renderInto === 'function') {
      window.SLMKillSwitch.renderInto(mount);
    }
    return g;
  }

  /* ═══════════════════════════════════════════════════════════
     Filter + Load all
  ═══════════════════════════════════════════════════════════ */
  function filterSettings(query, hub) {
    var q2 = (query || '').toLowerCase().trim();
    hub.querySelectorAll('[data-grp]').forEach(function (g) {
      var any = false;
      g.querySelectorAll('[data-set]').forEach(function (r) {
        var hit = !q2 || (r.dataset.txt || '').includes(q2);
        r.style.display = hit ? '' : 'none';
        if (hit) any = true;
      });
      g.style.display = any ? '' : 'none';
    });
  }
  function loadAll() {
    loadMode(); loadEmb(); loadStorage(); loadScope(); loadRuntime();
    loadCapture(); loadRecall(); loadInvoke(); loadForgetting(); loadEvolution();
    loadBackup(); loadMesh(); loadTrust(); loadRateLimit(); loadDaemon();
  }

  /* ═══════════════════════════════════════════════════════════
     MAIN RENDER
  ═══════════════════════════════════════════════════════════ */
  function odRenderSettings(container) {
    if (!container) return;
    // Look in THIS pane only. A background refresh renders into a fresh copy
    // while the old pane (and its hub) is still in the page; a page-wide
    // lookup found the old hub, rendered nothing, and the empty copy was then
    // swapped in: a blank Settings page.
    var existing = container.querySelector('#' + ID);
    if (existing) { loadAll(); return; }

    Array.from(container.children).forEach(function (c) { c.style.display = 'none'; });

    var hub = el('div', { id: ID });
    Object.assign(hub.style, { padding:'26px', maxWidth:'920px' });

    var headDiv = el('div', null, { marginBottom:'22px' });
    headDiv.appendChild(el('h2', { text:'Settings' },
      { fontSize:'22px', fontWeight:'650', letterSpacing:'-0.02em' }));
    headDiv.appendChild(el('p', { text:'Every SuperLocalMemory configuration, grouped and searchable — no config files to edit by hand. Settings show when a daemon restart is required.' },
      { color:'var(--fg-2)', marginTop:'5px', maxWidth:'62ch' }));
    hub.appendChild(headDiv);

    var sWrap = el('div', null, { display:'flex', alignItems:'center', gap:'9px', height:'38px',
      padding:'0 14px', background:'var(--card-2)', border:'1px solid var(--border)',
      borderRadius:'999px', color:'var(--fg-2)', marginBottom:'22px' });
    var sInp = el('input', { type:'search', placeholder:'Search every setting…', 'aria-label':'Search settings', id: P + '-search' });
    Object.assign(sInp.style, { border:'0', background:'none', outline:'none',
      color:'var(--fg)', fontSize:'13.5px', width:'100%' });
    sInp.addEventListener('input', function () { filterSettings(this.value, hub); });
    sWrap.appendChild(sInp);
    hub.appendChild(sWrap);

    var frag = document.createDocumentFragment();
    [buildMode(), buildAnswerCheck(), buildMeasurement(), buildKillSwitch(), buildEmb(), buildStorage(), buildScope(), buildRuntime(),
     buildCapture(), buildRecall(), buildInvoke(), buildForgetting(),
     buildEvolution(), buildBackup(), buildMesh(), buildTrust(),
     buildRateLimit(), buildDaemon()
    ].forEach(function (g) { frag.appendChild(g); });
    hub.appendChild(frag);

    var foot = el('div', null, { display:'flex', gap:'10px', marginTop:'8px', position:'sticky',
      bottom:'0', padding:'14px 0',
      background:'linear-gradient(transparent, var(--page) 40%)' });
    var saveAllBtn = el('button', { type:'button' }); saveAllBtn.className = 'btn primary';
    saveAllBtn.textContent = 'Save all changes';
    var discardBtn = el('button', { type:'button' }); discardBtn.className = 'btn ghost';
    discardBtn.textContent = 'Discard';
    var syncBadge = el('span', null, { alignSelf:'center', marginLeft:'auto' });
    syncBadge.className = 'badge';
    syncBadge.innerHTML = '<span class="dot"></span> Ready';
    saveAllBtn.addEventListener('click', function () {
      ['mode-save','emb-save','stor-save','forg-save','trust-save','rl-save','evo-save','bk-save','dmn-save']
        .forEach(function (i) { var b = q(i); if (b) b.click(); });
      syncBadge.className = 'badge';
      syncBadge.innerHTML = '<span class="dot"></span> Check each section';
    });
    discardBtn.addEventListener('click', function () { loadAll(); });
    foot.appendChild(saveAllBtn); foot.appendChild(discardBtn); foot.appendChild(syncBadge);
    hub.appendChild(foot);

    container.insertBefore(hub, container.firstChild);
    loadAll();
    loadModelCatalog(); // once per pane render — not repeated on loadAll() reloads
  }

  /* ═══════════════════════════════════════════════════════════
     BOOT
  ═══════════════════════════════════════════════════════════ */
  window.odRenderSettings = odRenderSettings;
  // A finished re-index changes the live model: show the daemon's new truth.
  if (window.odReindex) window.odReindex.onFinish(function () { loadEmb(); });

  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('settings-pane');
    if (pane && !document.getElementById(ID)) odRenderSettings(pane);
  });

}());
