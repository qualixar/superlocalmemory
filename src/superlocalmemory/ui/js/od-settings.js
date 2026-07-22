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
  var _captureConfig = {};  // preserves full capture config for merge-on-save
  var _tokenCache    = null;

  /* ── Auth helper ──────────────────────────────────────────── */
  function getToken() {
    if (_tokenCache) return Promise.resolve(_tokenCache);
    return fetch('/internal/token')
      .then(function (r) { return r.json(); })
      .then(function (d) { _tokenCache = d.token || ''; return _tokenCache; })
      .catch(function () { return ''; });
  }
  /** PUT with X-Install-Token. Use for all new config endpoints. */
  function authPut(url, body) {
    return getToken().then(function (tok) {
      return fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', 'X-Install-Token': tok },
        body: JSON.stringify(body)
      });
    });
  }

  /* ── Helpers ──────────────────────────────────────────────── */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }
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
    fetch('/api/daemon/restart', { method: 'POST', credentials: 'same-origin' })
      .then(function () {
        btn.textContent = 'Restarting… reconnecting';
        setTimeout(function () { window.location.reload(); }, 4000);
      })
      .catch(function () {
        btn.disabled = false;
        btn.textContent = 'Restart daemon now';
        if (typeof showToast === 'function') showToast('Restart request failed');
      });
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
  function makeRow(lbl, desc, key, ctrl) {
    var d = el('div', { 'data-set':'', 'data-txt':(lbl + ' ' + key).toLowerCase() });
    Object.assign(d.style, { display:'flex', alignItems:'center',
      justifyContent:'space-between', gap:'16px',
      padding:'14px 20px', borderBottom:'1px solid var(--border)' });
    var lv = el('div');
    lv.appendChild(el('b', { text: lbl }));
    if (desc) lv.appendChild(el('span', { text: desc },
      { fontSize:'12.5px', color:'var(--fg-2)', display:'block' }));
    if (key) lv.appendChild(el('div', { text: key },
      { fontFamily:'var(--font-mono)', fontSize:'10.5px', color:'var(--fg-3)', marginTop:'3px' }));
    d.appendChild(lv);
    var cv = el('div', null, { flexShrink:'0' }); cv.appendChild(ctrl); d.appendChild(cv);
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
     GROUP 1 — Operating Mode & LLM
  ═══════════════════════════════════════════════════════════ */
  function buildMode() {
    var modeSel = makeSel('mode', [
      { v:'a', l:'Mode A — Local Guardian (zero cloud)' },
      { v:'b', l:'Mode B — Smart Local (Ollama)' },
      { v:'c', l:'Mode C — Full Power (cloud LLM)' }
    ]);
    var provSel = makeSel('provider', [
      { v:'none', l:'None (Mode A)' }, { v:'ollama', l:'Ollama (local)' },
      { v:'openrouter', l:'OpenRouter' }, { v:'openai', l:'OpenAI' },
      { v:'anthropic', l:'Anthropic' }
    ]);
    var modelInp = makeTin('model', 'e.g. llama3.2');
    var keyInp   = makeTin('apikey', 'sk-... or your key', 'password');
    var epInp    = makeTin('endpoint', 'http://localhost:11434');
    var saveBtn  = makeBtn('mode-save', 'Save Mode', true);
    var testBtn  = makeBtn('mode-test', 'Test');
    var st = makeSt('mode');
    modeSel.addEventListener('change', function () {
      var a = this.value === 'a';
      [provSel, modelInp, keyInp, epInp].forEach(function (i) { i.disabled = a; });
    });
    saveBtn.addEventListener('click', saveMode);
    testBtn.addEventListener('click', testMode);
    return makeGrp('Operating Mode & LLM Provider', 'operations', [
      makeRow('Mode', 'Select operating mode', 'mode', modeSel),
      makeRow('Provider', 'LLM backend (Mode B/C)', 'llm.provider', provSel),
      makeRow('Model', '', 'llm.model', modelInp),
      makeRow('API Key', 'Stored locally, never sent to Qualixar', 'llm.api_key', keyInp),
      makeRow('Base URL / Endpoint', 'Custom endpoint for Ollama, LM Studio', 'llm.base_url', epInp),
      makeRow('Actions', '', '', bRow(saveBtn, testBtn, st))
    ]);
  }
  function loadMode() {
    fetch('/api/v3/mode').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var m = q('mode');
        if (m) { m.value = d.mode || 'a'; m.dispatchEvent(new Event('change')); }
        var p = q('provider'); if (p) p.value = d.provider || 'none';
        var mi = q('model'); if (mi) mi.value = d.model || '';
        var k = q('apikey');
        if (k) k.placeholder = d.has_key ? '(key saved — enter new to replace)' : 'sk-...';
        var ep = q('endpoint'); if (ep) ep.value = d.endpoint || '';
      }).catch(function () {});
  }
  function saveMode() {
    setSt('mode', 'Saving…', null);
    var body = { mode: q('mode') ? q('mode').value : 'a',
                 provider: q('provider') ? q('provider').value : 'none',
                 model: q('model') ? q('model').value : '' };
    var k = q('apikey') ? q('apikey').value : '';
    var ep = q('endpoint') ? q('endpoint').value : '';
    if (k)  { body.api_key = k; }
    if (ep) { body.base_url = ep; body.endpoint = ep; }
    fetch('/api/v3/mode/set', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
    }).then(function (r) { return r.json(); })
      .then(function () { setSt('mode', 'Saved', true); toast('Mode saved'); loadMode(); })
      .catch(function () { setSt('mode', 'Error', false); toast('Mode save failed', true); });
  }
  function testMode() {
    setSt('mode', 'Testing…', null);
    var body = { provider: q('provider') ? q('provider').value : '',
                 model:    q('model')    ? q('model').value    : '' };
    var k  = q('apikey')   ? q('apikey').value   : '';
    var ep = q('endpoint') ? q('endpoint').value : '';
    if (k)  { body.api_key = k; }
    if (ep) { body.base_url = ep; body.endpoint = ep; }
    fetch('/api/v3/provider/test', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        setSt('mode', d.success ? esc(d.message || 'OK') : 'Failed: ' + esc(d.error || ''), d.success);
      }).catch(function (e) { setSt('mode', 'Error: ' + e.message, false); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 2 — Embeddings & Retrieval
  ═══════════════════════════════════════════════════════════ */
  function buildEmb() {
    var provSel = makeSel('emb-prov', [
      { v:'default', l:'Default (local nomic-embed-text)' },
      { v:'openai',  l:'Custom OpenAI-compatible endpoint' }
    ]);
    var modelInp = makeTin('emb-model', 'e.g. Qwen3-Embedding');
    var dimInp   = makeTin('emb-dim', '768', 'number');
    dimInp.style.minWidth = '80px'; dimInp.style.maxWidth = '100px';
    var epInp  = makeTin('emb-endpoint', 'http://localhost:8045/v1/embeddings');
    var keyInp = makeTin('emb-key', 'optional key', 'password');
    var infoEl = el('span', { id: P + '-emb-info' }, { fontSize:'12px', color:'var(--fg-2)' });
    var saveBtn = makeBtn('emb-save', 'Save Embedding', true);
    var testBtn = makeBtn('emb-test', 'Test');
    var st = makeSt('emb');
    provSel.addEventListener('change', function () {
      var c = this.value === 'openai';
      [modelInp, dimInp, epInp, keyInp].forEach(function (i) { i.disabled = !c; });
    });
    saveBtn.addEventListener('click', saveEmb);
    testBtn.addEventListener('click', testEmb);
    return makeGrp('Embeddings & Retrieval', 'memories', [
      makeRow('Provider', 'Local model is private and free', 'embedding.provider', provSel),
      makeRow('Model Name', 'For custom endpoints', 'embedding.model_name', modelInp),
      makeRow('Dimensions', '768 local · 1024/3072 cloud', 'embedding.dimension', dimInp),
      makeRow('Endpoint', 'OpenAI-compatible URL', 'embedding.api_endpoint', epInp),
      makeRow('API Key', 'optional', 'embedding.api_key', keyInp),
      makeRow('Info', '', '', infoEl),
      makeRow('Actions', '', '', bRow(saveBtn, testBtn, st))
    ]);
  }
  function loadEmb() {
    fetch('/api/v3/embedding/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var p = q('emb-prov');
        if (p) { p.value = d.is_openai_compatible ? 'openai' : 'default'; p.dispatchEvent(new Event('change')); }
        var m = q('emb-model'); if (m) m.value = d.model_name || '';
        var dim = q('emb-dim'); if (dim) dim.value = d.dimension || '';
        var ep = q('emb-endpoint'); if (ep) ep.value = d.api_endpoint || '';
        var inf = q('emb-info');
        if (inf) inf.textContent = 'Current: ' + (d.model_name || 'nomic-embed-text') + ' (' + (d.dimension || 768) + 'd)';
      }).catch(function () {});
  }
  function saveEmb() {
    setSt('emb', 'Saving…', null);
    var prov = q('emb-prov') ? q('emb-prov').value : 'default';
    var body = prov === 'openai'
      ? { embedding_provider:'openai',
          embedding_endpoint:  q('emb-endpoint') ? q('emb-endpoint').value : '',
          embedding_model:     q('emb-model')    ? q('emb-model').value    : '',
          embedding_dimension: parseInt(q('emb-dim') ? q('emb-dim').value : '768') || 768,
          embedding_key:       q('emb-key')      ? q('emb-key').value      : '' }
      : { embedding_provider:'default' };
    fetch('/api/v3/mode/set', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
    }).then(function (r) { return r.json(); })
      .then(function () { setSt('emb', 'Saved', true); toast('Embedding config saved'); })
      .catch(function () { setSt('emb', 'Error', false); toast('Embedding save failed', true); });
  }
  function testEmb() {
    setSt('emb', 'Testing…', null);
    var ep = q('emb-endpoint') ? q('emb-endpoint').value : '';
    if (!ep) { setSt('emb', 'Enter endpoint first', null); return; }
    fetch('/api/v3/embedding/test', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ api_endpoint: ep,
        model_name: q('emb-model') ? q('emb-model').value : '',
        api_key:    q('emb-key')   ? q('emb-key').value   : '' })
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        setSt('emb', d.success ? esc(d.message || 'OK') : 'Failed: ' + esc(d.error || ''), d.success);
      }).catch(function (e) { setSt('emb', 'Error: ' + e.message, false); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 3 — Storage Backends
     GET/PUT /api/v3/storage/config  — PUT returns restart_required
  ═══════════════════════════════════════════════════════════ */
  function buildStorage() {
    var gBk = makeSel('stor-graph', [
      { v:'auto',   l:'auto (recommended)' },
      { v:'sqlite', l:'SQLite' },
      { v:'cozo',   l:'CozoDB' }
    ]);
    var vBk = makeSel('stor-vec', [
      { v:'auto',       l:'auto (recommended)' },
      { v:'sqlite-vec', l:'sqlite-vec' },
      { v:'lancedb',    l:'LanceDB' }
    ]);
    var dirInp = makeTin('stor-dir', '~/.superlocalmemory');
    dirInp.disabled = true;
    Object.assign(dirInp.style, { opacity:'.55', cursor:'not-allowed' });
    var saveBtn  = makeBtn('stor-save', 'Save Backends', true);
    var st       = makeSt('stor');
    var restNote = makeRestartNote('stor');
    saveBtn.addEventListener('click', saveStorage);
    return makeGrp('Storage Backends', 'operations', [
      makeRow('Graph backend', 'Engine for relationship and knowledge-graph queries', 'graph_backend', gBk),
      makeRow('Vector backend', 'Engine for semantic similarity search', 'vector_backend', vBk),
      makeRow('Base directory', 'Where SLM stores its database files (read-only)', 'base_dir', dirInp),
      makeRow('Actions', '', '', bRow(saveBtn, st, restNote))
    ]);
  }
  function loadStorage() {
    fetch('/api/v3/storage/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var g = q('stor-graph'); if (g) g.value = d.graph_backend  || 'auto';
        var v = q('stor-vec');   if (v) v.value = d.vector_backend || 'auto';
        var dir = q('stor-dir'); if (dir) dir.value = d.base_dir || '';
      }).catch(function () {});
  }
  function saveStorage() {
    setSt('stor', 'Saving…', null);
    showRestartNote('stor', false);
    authPut('/api/v3/storage/config', {
      graph_backend:  q('stor-graph') ? q('stor-graph').value : 'auto',
      vector_backend: q('stor-vec')   ? q('stor-vec').value   : 'auto'
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        setSt('stor', 'Saved', true);
        toast('Storage backends saved');
        showRestartNote('stor', !!d.restart_required);
      }).catch(function () { setSt('stor', 'Error', false); toast('Storage save failed', true); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 4 — Memory Scope
     GET/PUT /api/v3/scope/config
  ═══════════════════════════════════════════════════════════ */
  function buildScope() {
    var defSel = makeSel('scope-def', [
      { v:'personal', l:'Personal (recommended)' },
      { v:'shared',   l:'Shared' },
      { v:'global',   l:'Global' }
    ]);
    var shSw = makeSwitch('scope-shared', false);
    var glSw = makeSwitch('scope-global', false);
    var st = makeSt('scope');
    function save() {
      authPut('/api/v3/scope/config', {
        default_scope:         q('scope-def') ? q('scope-def').value : 'personal',
        recall_include_shared: shSw.classList.contains('on'),
        recall_include_global: glSw.classList.contains('on')
      }).then(function (r) { return r.json(); })
        .then(function (d) {
          setSt('scope', d.success !== false ? 'Applied' : esc(d.error || 'Error'), d.success !== false);
        })
        .catch(function () { setSt('scope', 'Error', false); toast('Scope save failed', true); });
    }
    defSel.addEventListener('change', save);
    shSw.addEventListener('od-toggle', save);
    glSw.addEventListener('od-toggle', save);
    return makeGrp('Memory Scope (auto-saves)', 'memories', [
      makeRow('Default write scope', 'Where new memories are stored — Personal is privacy-safe', 'scope.default_scope', defSel),
      makeRow('Include shared memories in recall', 'Surface memories shared across your agents', 'scope.recall_include_shared', shSw),
      makeRow('Include global memories in recall', 'Surface system-wide shared facts', 'scope.recall_include_global', glSw),
      makeRow('Status', '', '', st)
    ]);
  }
  function loadScope() {
    fetch('/api/v3/scope/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var def = q('scope-def'); if (def) def.value = d.default_scope || 'personal';
        var sh = q('scope-shared');
        if (sh) { sh.classList.toggle('on', !!d.recall_include_shared); sh.setAttribute('aria-checked', String(!!d.recall_include_shared)); }
        var gl = q('scope-global');
        if (gl) { gl.classList.toggle('on', !!d.recall_include_global); gl.setAttribute('aria-checked', String(!!d.recall_include_global)); }
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 5 — Auto-Capture
  ═══════════════════════════════════════════════════════════ */
  function buildCapture() {
    var sw    = makeSwitch('cap-en',  true);
    var swDec = makeSwitch('cap-dec', true);
    var swBug = makeSwitch('cap-bug', true);
    function save() {
      // Merge with full loaded config to preserve server-side fields the UI doesn't expose
      var body = Object.assign({}, _captureConfig, {
        enabled:           sw.classList.contains('on'),
        capture_decisions: swDec.classList.contains('on'),
        capture_bugs:      swBug.classList.contains('on')
      });
      fetch('/api/v3/auto-capture/config', {
        method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
      }).catch(function () { toast('Auto-capture save failed', true); });
    }
    sw.addEventListener('od-toggle', save);
    swDec.addEventListener('od-toggle', save);
    swBug.addEventListener('od-toggle', save);
    return makeGrp('Auto-Capture', 'brain', [
      makeRow('Enable auto-capture', 'Silently remember key facts from agent sessions', 'auto_capture.enabled', sw),
      makeRow('Capture decisions', 'Architecture and design choices', 'auto_capture.capture_decisions', swDec),
      makeRow('Capture bug fixes', 'Error patterns and their resolutions', 'auto_capture.capture_bugs', swBug)
    ]);
  }
  function loadCapture() {
    fetch('/api/v3/auto-capture/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.config) return;
        _captureConfig = Object.assign({}, d.config);
        var c = d.config;
        var map = { 'cap-en': c.enabled, 'cap-dec': c.capture_decisions, 'cap-bug': c.capture_bugs };
        Object.keys(map).forEach(function (k) {
          var e = q(k); if (!e) return;
          e.classList.toggle('on', !!map[k]);
          e.setAttribute('aria-checked', String(!!map[k]));
        });
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 6 — Auto-Recall
  ═══════════════════════════════════════════════════════════ */
  function buildRecall() {
    var sw     = makeSwitch('rec-en',   true);
    var swSess = makeSwitch('rec-sess', true);
    function save() {
      fetch('/api/v3/auto-recall/config', {
        method:'PUT', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          enabled:          sw.classList.contains('on'),
          on_session_start: swSess.classList.contains('on')
        })
      }).catch(function () { toast('Auto-recall save failed', true); });
    }
    sw.addEventListener('od-toggle', save);
    swSess.addEventListener('od-toggle', save);
    return makeGrp('Auto-Recall', 'brain', [
      makeRow('Enable auto-recall', 'Inject relevant memory into agent context', 'auto_recall.enabled', sw),
      makeRow('Recall on session start', 'Run recall automatically when a new session begins', 'auto_recall.on_session_start', swSess)
    ]);
  }
  function loadRecall() {
    fetch('/api/v3/auto-recall/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.config) return;
        var c = d.config;
        var en = q('rec-en');
        if (en) { en.classList.toggle('on', !!c.enabled); en.setAttribute('aria-checked', String(!!c.enabled)); }
        var se = q('rec-sess');
        if (se) { se.classList.toggle('on', !!c.on_session_start); se.setAttribute('aria-checked', String(!!c.on_session_start)); }
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 7 — Auto-Invoke (FOK Threshold)
  ═══════════════════════════════════════════════════════════ */
  function buildInvoke() {
    var sw    = makeSwitch('ai-en',   true);
    var range = makeRange('ai-fok', '0', '1', '0.01', '0.12');
    var arSw  = makeSwitch('ai-actr', false);
    var st    = makeSt('ai');
    function save() {
      fetch('/api/v3/auto-invoke/config', {
        method:'PUT', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          enabled:    sw.classList.contains('on'),
          min_score:  parseFloat(q('ai-fok') ? q('ai-fok').value : '0.12'),
          act_r_mode: arSw.classList.contains('on')
        })
      }).then(function (r) { return r.json(); })
        .then(function () { setSt('ai', 'Saved', true); })
        .catch(function () { toast('Auto-invoke save failed', true); });
    }
    sw.addEventListener('od-toggle', save);
    arSw.addEventListener('od-toggle', save);
    range.querySelector('input').addEventListener('change', save);
    return makeGrp('Auto-Invoke (FOK Threshold)', 'optimize', [
      makeRow('Enable auto-invoke', 'Run recall automatically when relevance score is high enough', 'auto_invoke.enabled', sw),
      makeRow('FOK threshold', '0 = always recall · 1 = never recall', 'auto_invoke.min_score', range),
      makeRow('ACT-R mode', 'Biologically-inspired memory activation formula', 'auto_invoke.act_r_mode', arSw),
      makeRow('Status', '', '', st)
    ]);
  }
  function loadInvoke() {
    fetch('/api/v3/auto-invoke/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var en = q('ai-en');
        if (en) { en.classList.toggle('on', !!d.enabled); en.setAttribute('aria-checked', String(!!d.enabled)); }
        var score = String(d.min_score != null ? d.min_score : 0.12);
        var fok = q('ai-fok'); if (fok) fok.value = score;
        var fokD = q('ai-fok-disp'); if (fokD) fokD.textContent = parseFloat(score).toFixed(2);
        var ar = q('ai-actr');
        if (ar) { ar.classList.toggle('on', !!d.act_r_mode); ar.setAttribute('aria-checked', String(!!d.act_r_mode)); }
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 8 — Forgetting Engine
     GET/PUT /api/v3/forgetting/config + GET /api/v3/forgetting/stats
  ═══════════════════════════════════════════════════════════ */
  function buildForgetting() {
    var onSw   = makeSwitch('forg-en',  false);
    var immSw  = makeSwitch('forg-imm', true);
    var arcR   = makeRange('forg-arc', '0', '1', '0.01', '0.20');
    var fgtR   = makeRange('forg-fgt', '0', '1', '0.01', '0.05');
    var intInp = makeTin('forg-int', '30', 'number');
    intInp.style.minWidth = '80px'; intInp.style.maxWidth = '100px';
    var statsEl = el('span', { id: P + '-forg-stats', text:'Loading…' },
      { fontSize:'12px', color:'var(--fg-2)', fontFamily:'var(--font-mono)' });
    var saveBtn = makeBtn('forg-save', 'Save Forgetting', true);
    var st = makeSt('forg');
    saveBtn.addEventListener('click', saveForgetting);
    onSw.addEventListener('od-toggle',  saveForgetting);
    immSw.addEventListener('od-toggle', saveForgetting);
    arcR.querySelector('input').addEventListener('change', saveForgetting);
    fgtR.querySelector('input').addEventListener('change', saveForgetting);
    intInp.addEventListener('change', saveForgetting);
    return makeGrp('Forgetting Engine', 'health', [
      makeRow('Enable forgetting', 'Gradually fade old or rarely-used memories (recommended on)', 'forgetting.enabled', onSw),
      makeRow('Protect important memories', 'Core memories are immune from being forgotten', 'forgetting.core_memory_immune', immSw),
      makeRow('Archive threshold', 'Memories below this strength move to cold archive', 'forgetting.archive_threshold', arcR),
      makeRow('Forget threshold', 'Memories below this strength are permanently deleted', 'forgetting.forget_threshold', fgtR),
      makeRow('Run every (minutes)', 'How often the forgetting engine runs in the background', 'forgetting.scheduler_interval_minutes', intInp),
      makeRow('Memory zone stats', '', 'forgetting.stats', statsEl),
      makeRow('Actions', '', '', bRow(saveBtn, st))
    ]);
  }
  function loadForgetting() {
    fetch('/api/v3/forgetting/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var en = q('forg-en');
        if (en) { en.classList.toggle('on', !!d.enabled); en.setAttribute('aria-checked', String(!!d.enabled)); }
        var imm = q('forg-imm');
        if (imm) { imm.classList.toggle('on', d.core_memory_immune !== false); imm.setAttribute('aria-checked', String(d.core_memory_immune !== false)); }
        var arc = q('forg-arc');
        if (arc) { arc.value = d.archive_threshold || 0.2; var dA = q('forg-arc-disp'); if (dA) dA.textContent = Number(d.archive_threshold || 0.2).toFixed(2); }
        var fgt = q('forg-fgt');
        if (fgt) { fgt.value = d.forget_threshold || 0.05; var dF = q('forg-fgt-disp'); if (dF) dF.textContent = Number(d.forget_threshold || 0.05).toFixed(2); }
        var it = q('forg-int'); if (it) it.value = d.scheduler_interval_minutes || 30;
      }).catch(function () {});
    fetch('/api/v3/forgetting/stats').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        var e = q('forg-stats'); if (!d || !e) return;
        var parts = [];
        if (d.total != null) parts.push('total: ' + d.total);
        if (d.zones) {
          ['active','warm','cold','archive','forgotten'].forEach(function (z) {
            if (d.zones[z] != null) parts.push(z + ': ' + d.zones[z]);
          });
        }
        e.textContent = parts.join(' · ') || 'No data';
      }).catch(function () {});
  }
  function saveForgetting() {
    setSt('forg', 'Saving…', null);
    authPut('/api/v3/forgetting/config', {
      enabled:                    q('forg-en')  ? q('forg-en').classList.contains('on')  : false,
      core_memory_immune:         q('forg-imm') ? q('forg-imm').classList.contains('on') : true,
      archive_threshold:  parseFloat(q('forg-arc') ? q('forg-arc').value : '0.20'),
      forget_threshold:   parseFloat(q('forg-fgt') ? q('forg-fgt').value : '0.05'),
      scheduler_interval_minutes: parseInt(q('forg-int') ? q('forg-int').value : '30') || 30
    }).then(function (r) { return r.json(); })
      .then(function () { setSt('forg', 'Saved', true); toast('Forgetting settings saved'); })
      .catch(function () { setSt('forg', 'Error', false); toast('Forgetting save failed', true); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 9 — Skill Evolution
     POST /api/evolution/enable|disable  (toggle);  POST /api/evolution/config (backend)
  ═══════════════════════════════════════════════════════════ */
  function buildEvolution() {
    var sw = makeSwitch('evo-en', false);
    var bkSel = makeSel('evo-bk', [
      { v:'auto',      l:'auto (detect)' }, { v:'claude',    l:'claude' },
      { v:'ollama',    l:'ollama' },         { v:'anthropic', l:'anthropic' },
      { v:'openai',    l:'openai' }
    ]);
    var saveBtn = makeBtn('evo-save', 'Save', true);
    var st = makeSt('evo');
    sw.addEventListener('od-toggle', function () {
      var on = sw.classList.contains('on');
      fetch(on ? '/api/evolution/enable' : '/api/evolution/disable', { method:'POST' })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.ok === false) {
            toast('Failed: ' + esc(d.error || ''), true);
            sw.classList.toggle('on', !on);
            sw.setAttribute('aria-checked', String(!on));
          } else {
            toast(on ? 'Evolution engine enabled' : 'Evolution engine disabled');
            loadEvolution();
          }
        }).catch(function () {
          toast('Evolution toggle failed', true);
          sw.classList.toggle('on', !on);
          sw.setAttribute('aria-checked', String(!on));
        });
    });
    saveBtn.addEventListener('click', function () {
      setSt('evo', 'Saving…', null);
      fetch('/api/evolution/config', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ backend: q('evo-bk') ? q('evo-bk').value : 'auto' })
      }).then(function (r) { return r.json(); })
        .then(function () { setSt('evo', 'Saved', true); })
        .catch(function () { setSt('evo', 'Error', false); toast('Evolution save failed', true); });
    });
    return makeGrp('Skill Evolution', 'skill', [
      makeRow('Enable evolution engine', 'Off by default — makes background LLM calls to improve skills', 'evolution.enabled', sw),
      makeRow('Backend', 'auto = cheapest capable model in your current mode', 'evolution.backend', bkSel),
      makeRow('Actions', '', '', bRow(saveBtn, st))
    ]);
  }
  function loadEvolution() {
    fetch('/api/evolution/status').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var sw = q('evo-en');
        if (sw) { sw.classList.toggle('on', !!d.enabled); sw.setAttribute('aria-checked', String(!!d.enabled)); }
        var bk = q('evo-bk');
        if (bk && d.config && d.config.backend_setting) bk.value = d.config.backend_setting;
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 10 — Backup
  ═══════════════════════════════════════════════════════════ */
  function buildBackup() {
    var swEn   = makeSwitch('bk-en', true);
    var intSel = makeSel('bk-int', [{ v:'24', l:'Daily (24 h)' }, { v:'168', l:'Weekly (168 h)' }]);
    var maxInp = makeTin('bk-max', '10', 'number');
    maxInp.style.minWidth = '80px'; maxInp.style.maxWidth = '100px';
    var saveBtn = makeBtn('bk-save', 'Save Backup Config', true);
    var nowBtn  = makeBtn('bk-now',  'Backup Now');
    var st = makeSt('bk');
    swEn.addEventListener('od-toggle', saveBackupConfig);
    saveBtn.addEventListener('click', saveBackupConfig);
    nowBtn.addEventListener('click', function () {
      setSt('bk', 'Creating…', null);
      fetch('/api/backup/create', { method:'POST' }).then(function (r) { return r.json(); })
        .then(function (d) {
          setSt('bk', d.success ? 'Done: ' + esc(d.filename || '') : 'Failed', d.success);
          toast(d.success ? 'Backup created' : 'Backup failed', !d.success);
          loadBackup();
        }).catch(function () { setSt('bk', 'Error', false); toast('Backup failed', true); });
    });
    var cloudBtn = el('button', { type:'button' });
    cloudBtn.className = 'btn sm ghost';
    cloudBtn.textContent = 'Backup & Cloud →';
    cloudBtn.addEventListener('click', function () {
      if (typeof window.odOpenBackupDashboard === 'function') {
        window.odOpenBackupDashboard();
      } else {
        toast('Backup dashboard module not loaded yet', true);
      }
    });
    return makeGrp('Backup', 'health', [
      makeRow('Auto-backup enabled', '', 'backup.enabled', swEn),
      makeRow('Backup interval', '', 'backup.interval_hours', intSel),
      makeRow('Max backups to keep', '', 'backup.max_backups', maxInp),
      makeRow('Actions', '', '', bRow(saveBtn, nowBtn, st)),
      makeRow('Cloud backup', 'Connect GitHub or Google Drive', '', cloudBtn)
    ]);
  }
  function loadBackup() {
    fetch('/api/backup/status').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var en = q('bk-en');
        if (en) { en.classList.toggle('on', d.enabled !== false); en.setAttribute('aria-checked', String(d.enabled !== false)); }
        var it = q('bk-int'); if (it) it.value = (d.interval_hours || 168) <= 24 ? '24' : '168';
        var mx = q('bk-max'); if (mx) mx.value = d.max_backups || 10;
      }).catch(function () {});
  }
  function saveBackupConfig() {
    setSt('bk', 'Saving…', null);
    fetch('/api/backup/configure', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        interval_hours: parseInt(q('bk-int') ? q('bk-int').value : '168'),
        max_backups:    parseInt(q('bk-max') ? q('bk-max').value : '10'),
        enabled:        q('bk-en') ? q('bk-en').classList.contains('on') : true
      })
    }).then(function (r) { return r.json(); })
      .then(function () { setSt('bk', 'Saved', true); toast('Backup settings saved'); })
      .catch(function () { setSt('bk', 'Error', false); toast('Backup save failed', true); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 11 — Mesh Network
     GET/PUT /api/v3/mesh/config   {enabled}
  ═══════════════════════════════════════════════════════════ */
  function buildMesh() {
    var sw = makeSwitch('mesh-en', false);
    var st = makeSt('mesh');
    sw.addEventListener('od-toggle', function () {
      authPut('/api/v3/mesh/config', { enabled: sw.classList.contains('on') })
        .then(function (r) { return r.json(); })
        .then(function () { setSt('mesh', 'Saved', true); })
        .catch(function () { setSt('mesh', 'Error', false); toast('Mesh save failed', true); });
    });
    return makeGrp('Mesh Network', 'mesh', [
      makeRow('Enable mesh sync', 'Synchronise memories across devices on the same local network', 'mesh.enabled', sw),
      makeRow('Status', '', '', st)
    ]);
  }
  function loadMesh() {
    fetch('/api/v3/mesh/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var sw = q('mesh-en');
        if (sw) { sw.classList.toggle('on', !!d.enabled); sw.setAttribute('aria-checked', String(!!d.enabled)); }
      }).catch(function () {});
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 12 — Trust & Source Quality
     GET/PUT /api/v3/trust/config  {use_trust_weighting, trust_first_party, promotion_min_trust}
  ═══════════════════════════════════════════════════════════ */
  function buildTrust() {
    var twSw = makeSwitch('trust-tw', true);
    var fpSw = makeSwitch('trust-fp', false);
    var pmR  = makeRange('trust-pm', '0', '1', '0.01', '0.5');
    var saveBtn = makeBtn('trust-save', 'Save Trust', true);
    var st = makeSt('trust');
    saveBtn.addEventListener('click', saveTrust);
    twSw.addEventListener('od-toggle', saveTrust);
    fpSw.addEventListener('od-toggle', saveTrust);
    pmR.querySelector('input').addEventListener('change', saveTrust);
    return makeGrp('Trust & Source Quality', 'lock', [
      makeRow('Weight recall by trust', 'Higher-trust sources appear first in recall results', 'trust.use_trust_weighting', twSw),
      makeRow('Prefer first-party agents', 'Your own Claude/Gemini sessions rank higher than third-party', 'trust.trust_first_party', fpSw),
      makeRow('Promotion min trust', 'Min trust score required to promote a memory to long-term storage', 'trust.promotion_min_trust', pmR),
      makeRow('Actions', '', '', bRow(saveBtn, st))
    ]);
  }
  function loadTrust() {
    fetch('/api/v3/trust/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var tw = q('trust-tw');
        if (tw) { tw.classList.toggle('on', d.use_trust_weighting !== false); tw.setAttribute('aria-checked', String(d.use_trust_weighting !== false)); }
        var fp = q('trust-fp');
        if (fp) { fp.classList.toggle('on', !!d.trust_first_party); fp.setAttribute('aria-checked', String(!!d.trust_first_party)); }
        var pm = q('trust-pm');
        if (pm) { pm.value = d.promotion_min_trust != null ? d.promotion_min_trust : 0.5; var pmD = q('trust-pm-disp'); if (pmD) pmD.textContent = Number(d.promotion_min_trust != null ? d.promotion_min_trust : 0.5).toFixed(2); }
      }).catch(function () {});
  }
  function saveTrust() {
    setSt('trust', 'Saving…', null);
    authPut('/api/v3/trust/config', {
      use_trust_weighting: q('trust-tw') ? q('trust-tw').classList.contains('on') : true,
      trust_first_party:   q('trust-fp') ? q('trust-fp').classList.contains('on') : false,
      promotion_min_trust: parseFloat(q('trust-pm') ? q('trust-pm').value : '0.5')
    }).then(function (r) { return r.json(); })
      .then(function () { setSt('trust', 'Saved', true); })
      .catch(function () { setSt('trust', 'Error', false); toast('Trust save failed', true); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 12b — Rate Limits (task #47)
     GET/PUT /api/v3/ratelimit  {write, read, window} → applied at runtime
  ═══════════════════════════════════════════════════════════ */
  function buildRateLimit() {
    var wInp = makeTin('rl-write', '100', 'number');
    wInp.style.minWidth = '90px'; wInp.style.maxWidth = '120px';
    var rInp = makeTin('rl-read', '300', 'number');
    rInp.style.minWidth = '90px'; rInp.style.maxWidth = '120px';
    var winInp = makeTin('rl-window', '60', 'number');
    winInp.style.minWidth = '90px'; winInp.style.maxWidth = '120px';
    var lbInfo = el('span', { text: '—' }, { color:'var(--fg-2)', fontSize:'12px' });
    lbInfo.id = P + '-rl-lb';
    var saveBtn = makeBtn('rl-save', 'Save Limits', true);
    var st = makeSt('rl');
    saveBtn.addEventListener('click', saveRateLimit);
    return makeGrp('Rate Limits', 'lock', [
      makeRow('Write limit (per window)', 'Max write requests per client per window — raise this for heavy multi-system load', 'ratelimit.write', wInp),
      makeRow('Read limit (per window)', 'Max read requests per client per window', 'ratelimit.read', rInp),
      makeRow('Window (seconds)', 'Length of the sliding rate-limit window', 'ratelimit.window', winInp),
      makeRow('Loopback (derived)', 'The local dashboard gets a generous multiple of these limits — updates automatically', 'ratelimit.loopback', lbInfo),
      makeRow('Actions', 'Applied immediately — no restart needed', '', bRow(saveBtn, st))
    ]);
  }
  function _rlSetLb(d) {
    var lb = q('rl-lb');
    if (lb && d) {
      lb.textContent = 'write ' + (d.loopback_write != null ? d.loopback_write : '—') +
        ' · read ' + (d.loopback_read != null ? d.loopback_read : '—');
    }
  }
  function loadRateLimit() {
    fetch('/api/v3/ratelimit').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var w = q('rl-write');   if (w) w.value = d.write != null ? d.write : 100;
        var r = q('rl-read');    if (r) r.value = d.read != null ? d.read : 300;
        var win = q('rl-window'); if (win) win.value = d.window != null ? d.window : 60;
        _rlSetLb(d);
      }).catch(function () {});
  }
  function saveRateLimit() {
    setSt('rl', 'Saving…', null);
    authPut('/api/v3/ratelimit', {
      write:  parseInt(q('rl-write')  ? q('rl-write').value  : '100') || 100,
      read:   parseInt(q('rl-read')   ? q('rl-read').value   : '300') || 300,
      window: parseInt(q('rl-window') ? q('rl-window').value : '60')  || 60
    }).then(function (r) { return r.json(); })
      .then(function (d) { setSt('rl', 'Saved', true); _rlSetLb(d); })
      .catch(function () { setSt('rl', 'Error', false); toast('Rate limit save failed', true); });
  }

  /* ═══════════════════════════════════════════════════════════
     GROUP 13 — Daemon
     GET/PUT /api/v3/daemon/config  {idle_timeout, port, legacy_port, enable_legacy_port}
     PUT returns restart_required
  ═══════════════════════════════════════════════════════════ */
  function buildDaemon() {
    var portInp = makeTin('dmn-port', '8765', 'number');
    portInp.style.minWidth = '90px'; portInp.style.maxWidth = '120px';
    var lpInp = makeTin('dmn-lport', '8767', 'number');
    lpInp.style.minWidth = '90px'; lpInp.style.maxWidth = '120px';
    var lpSw    = makeSwitch('dmn-lpen', true);
    var toInp   = makeTin('dmn-timeout', '0', 'number');
    toInp.style.minWidth = '90px'; toInp.style.maxWidth = '120px';
    var saveBtn  = makeBtn('dmn-save', 'Save Daemon', true);
    var st       = makeSt('dmn');
    var restNote = makeRestartNote('dmn');
    saveBtn.addEventListener('click', saveDaemon);
    return makeGrp('Daemon (port changes require restart)', 'settings', [
      makeRow('API port', 'Port the SLM daemon listens on — changing this requires a restart', 'daemon.port', portInp),
      makeRow('Legacy port', 'Backward-compatible API port for older integrations', 'daemon.legacy_port', lpInp),
      makeRow('Enable legacy port', 'Keep the old API port open alongside the main port', 'daemon.enable_legacy_port', lpSw),
      makeRow('Idle timeout (seconds)', '0 = daemon runs forever · >0 = auto-stop after N idle seconds', 'daemon.idle_timeout', toInp),
      makeRow('Actions', '', '', bRow(saveBtn, st, restNote))
    ]);
  }
  function loadDaemon() {
    fetch('/api/v3/daemon/config').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var p = q('dmn-port');    if (p) p.value = d.port || 8765;
        var l = q('dmn-lport');   if (l) l.value = d.legacy_port || 8767;
        var s = q('dmn-lpen');
        if (s) { s.classList.toggle('on', d.enable_legacy_port !== false); s.setAttribute('aria-checked', String(d.enable_legacy_port !== false)); }
        var t = q('dmn-timeout'); if (t) t.value = d.idle_timeout != null ? d.idle_timeout : 0;
      }).catch(function () {});
  }
  function saveDaemon() {
    setSt('dmn', 'Saving…', null);
    showRestartNote('dmn', false);
    authPut('/api/v3/daemon/config', {
      port:               parseInt(q('dmn-port')    ? q('dmn-port').value    : '8765') || 8765,
      legacy_port:        parseInt(q('dmn-lport')   ? q('dmn-lport').value   : '8767') || 8767,
      enable_legacy_port: q('dmn-lpen') ? q('dmn-lpen').classList.contains('on') : true,
      idle_timeout:       parseInt(q('dmn-timeout') ? q('dmn-timeout').value : '0') || 0
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        setSt('dmn', 'Saved', true);
        toast('Daemon settings saved');
        showRestartNote('dmn', !!d.restart_required);
      }).catch(function () { setSt('dmn', 'Error', false); toast('Daemon save failed', true); });
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
    loadMode(); loadEmb(); loadStorage(); loadScope(); loadCapture();
    loadRecall(); loadInvoke(); loadForgetting(); loadEvolution();
    loadBackup(); loadMesh(); loadTrust(); loadRateLimit(); loadDaemon();
  }

  /* ═══════════════════════════════════════════════════════════
     MAIN RENDER
  ═══════════════════════════════════════════════════════════ */
  function odRenderSettings(container) {
    if (!container) return;
    var existing = document.getElementById(ID);
    if (existing) { loadAll(); return; }

    Array.from(container.children).forEach(function (c) { c.style.display = 'none'; });

    var hub = el('div', { id: ID });
    Object.assign(hub.style, { padding:'26px', maxWidth:'920px' });

    var headDiv = el('div', null, { marginBottom:'22px' });
    headDiv.appendChild(el('h2', { text:'Settings' },
      { fontSize:'22px', fontWeight:'650', letterSpacing:'-0.02em' }));
    headDiv.appendChild(el('p', { text:'Every SuperLocalMemory configuration, grouped and searchable — no config files to edit by hand. Changes hot-reload the local daemon.' },
      { color:'var(--fg-2)', marginTop:'5px', maxWidth:'62ch' }));
    hub.appendChild(headDiv);

    var sWrap = el('div', null, { display:'flex', alignItems:'center', gap:'9px', height:'38px',
      padding:'0 14px', background:'var(--card-2)', border:'1px solid var(--border)',
      borderRadius:'999px', color:'var(--fg-2)', marginBottom:'22px' });
    var sInp = el('input', { type:'search', placeholder:'Search every setting…', id: P + '-search' });
    Object.assign(sInp.style, { border:'0', background:'none', outline:'none',
      color:'var(--fg)', fontSize:'13.5px', width:'100%' });
    sInp.addEventListener('input', function () { filterSettings(this.value, hub); });
    sWrap.appendChild(sInp);
    hub.appendChild(sWrap);

    var frag = document.createDocumentFragment();
    [buildMode(), buildEmb(), buildStorage(), buildScope(),
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
    syncBadge.className = 'badge ok';
    syncBadge.innerHTML = '<span class="dot"></span> All synced';
    saveAllBtn.addEventListener('click', function () {
      ['mode-save','emb-save','stor-save','forg-save','trust-save','rl-save','evo-save','bk-save','dmn-save']
        .forEach(function (i) { var b = q(i); if (b) b.click(); });
      saveBackupConfig();
      syncBadge.className = 'badge ok';
      syncBadge.innerHTML = '<span class="dot"></span> Saved';
      setTimeout(function () { syncBadge.innerHTML = '<span class="dot"></span> All synced'; }, 3000);
    });
    discardBtn.addEventListener('click', function () { loadAll(); });
    foot.appendChild(saveAllBtn); foot.appendChild(discardBtn); foot.appendChild(syncBadge);
    hub.appendChild(foot);

    container.insertBefore(hub, container.firstChild);
    loadAll();
  }

  /* ═══════════════════════════════════════════════════════════
     BOOT
  ═══════════════════════════════════════════════════════════ */
  window.odRenderSettings = odRenderSettings;

  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('settings-pane');
    if (pane && !document.getElementById(ID)) odRenderSettings(pane);
  });

}());
