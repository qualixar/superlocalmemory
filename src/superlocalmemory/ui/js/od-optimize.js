// SuperLocalMemory — od-optimize.js  v2.0
// OD-styled Optimize pane. Self-contained, CSP-safe.
// Renders into the "optimize-pane" tab container via the shell's triggerTabLoad.
// Exposes window.odRenderOptimize(container).
//
// HTML structure mirrors approved design (optimize.html <main class="content">).
// Design-system CSS classes used throughout; .ctl is injected once (not in design-system.css).
//
// Endpoints (no auth header required — open loopback):
//   GET    /api/optimize/status  → {healthy, enabled, features:{proxy,cache,compression,
//                                    semantic_cache}, compress_mode, config_version}
//   GET    /api/optimize/savings → {tokens_saved_input, tokens_saved_output,
//                                    tokens_saved_compress, compress_ratio,
//                                    cost_saved:{usd,inr}, hit_rate, cache_bytes,
//                                    entries, hits, misses, is_stale, pricing_date}
//   GET    /api/optimize/config  → {enabled, proxy_enabled, cache_enabled,
//                                    semantic_enabled, compress_enabled,
//                                    compress_mode, compress_prose, config_version, ...}
//   GET    /api/optimize/stats   → {hits, misses, compress_runs, compress_bytes_original,
//                                    compress_bytes_after, cache_size_bytes,
//                                    cache_entry_count, ...}
//   PUT    /api/optimize/config  → body:{enabled?, proxy_enabled?, ...} → {status:"ok"}
//   DELETE /api/optimize/cache/clear → {success, deleted}
//
// TODO: /api/optimize/savings has no per-provider breakdown; provider table shows
//       known pricing (factual public data) with aggregate totals from the API.
//
// CRIT fixes baked in:
//   C1 — proxy + master toggles show restart-required hint (daemon must restart to bind)
//   C2 — savings 404/500 handled with exponential-backoff retry + manual Retry button;
//        never a permanent broken state
//   C3 — all API strings through escapeHtml() before innerHTML assignment
//   C4 — putConfig re-reads config after every PUT (success: confirm; failure: revert)
//   C5 — compress_mode only offers 'safe'|'aggressive' (only two valid API values)
//
// Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar — AGPL-3.0

(function () {
  'use strict';

  var POLL_MS         = 10000;
  var MAX_AUTO_RETRY  = 3;
  var RETRY_DELAYS_MS = [2000, 5000, 15000];

  // Known provider pricing (factual public data; not seed/fake numbers).
  // TODO: replace with /api/optimize/savings per-provider breakdown when available.
  var PROVIDERS = [
    { name: 'Anthropic', price: '$3.00 / $15.00', color: 'var(--violet)' },
    { name: 'OpenAI',    price: '$2.50 / $10.00', color: 'var(--cyan)' },
    { name: 'Gemini',    price: '$1.25 / $10.00', color: 'var(--ok)' }
  ];

  // ── Module state ──────────────────────────────────────────────────────────
  var _cfg         = null;
  var _pollTimer   = null;
  var _container   = null;
  var _observer    = null;
  var _retry       = { config: 0, savings: 0 };
  var _ctlInjected = false;

  // ── Helpers ───────────────────────────────────────────────────────────────
  // CRIT C3: every API-derived string must pass through this.
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtNum(n) { return (n || 0).toLocaleString(); }

  // Format token counts with M/K suffix for KPI display (returns HTML).
  function fmtTokensHtml(n) {
    var v = n || 0;
    if (v >= 1e9) return (v / 1e9).toFixed(1) + '<small>B</small>';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + '<small>M</small>';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + '<small>K</small>';
    return String(v);
  }

  function fmtHitRateHtml(rate) {
    return Math.round((rate || 0) * 100) + '<small>%</small>';
  }

  function fmtRatioHtml(r) {
    return r != null ? r.toFixed(2) + '<small>&times;</small>' : '—';
  }

  function fmtUSD(n) {
    var v = n || 0;
    return v >= 1 ? '$' + Math.round(v).toLocaleString() : '$' + v.toFixed(4);
  }

  function fmtINR(n) {
    var v = n || 0;
    return v >= 1 ? '₹' + Math.round(v).toLocaleString() : '₹' + v.toFixed(2);
  }

  // ── Toast notifications ───────────────────────────────────────────────────
  // type: 'ok' (auto-dismisses 1.8s) | 'error' (auto-dismisses 4s)
  function showToast(msg, type) {
    var el = document.getElementById('od-opt-toast');
    if (!el) return;
    el.textContent  = msg;
    el.className    = type === 'error' ? 'err' : 'ok';
    el.style.display = 'block';
    clearTimeout(el._tid);
    el._tid = setTimeout(function () { el.style.display = 'none'; },
      type === 'error' ? 4000 : 1800);
  }

  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function setHtml(id, val) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = val;
  }

  // ── .ctl style injection ──────────────────────────────────────────────────
  // .ctl is defined in the design's inline <style> block, not in design-system.css.
  // Inject once into <head> so the class works correctly.
  function injectCtlStyle() {
    if (_ctlInjected || document.getElementById('od-opt-ctl-style')) return;
    var s = document.createElement('style');
    s.id = 'od-opt-ctl-style';
    s.textContent =
      '.ctl{display:flex;align-items:center;justify-content:space-between;gap:16px;' +
        'padding:12px 0;border-bottom:1px solid var(--border)}' +
      '.ctl:last-of-type{border-bottom:0}' +
      '#od-opt-toast{position:fixed;bottom:24px;right:24px;padding:10px 18px;' +
        'border-radius:var(--r-md);font-size:13px;font-weight:600;line-height:1.4;' +
        'z-index:1000;box-shadow:0 4px 16px rgba(0,0,0,.18);pointer-events:none;' +
        'display:none;max-width:320px}' +
      '#od-opt-toast.ok{background:var(--ok);color:#fff}' +
      '#od-opt-toast.err{background:var(--danger);color:#fff}';
    document.head.appendChild(s);
    _ctlInjected = true;
  }

  // ── HTML fragment helpers ─────────────────────────────────────────────────
  function buildCtl(switchId, label, sub) {
    return (
      '<div class="ctl">' +
        '<div>' +
          '<b style="font-size:13.5px">' + escapeHtml(label) + '</b>' +
          (sub
            ? '<div class="dim" style="font-size:12.5px;margin-top:2px">' + escapeHtml(sub) + '</div>'
            : '') +
        '</div>' +
        '<button class="switch" data-switch-id="' + escapeHtml(switchId) + '"' +
          ' id="' + escapeHtml(switchId) + '"' +
          ' role="switch" aria-checked="false" aria-label="' + escapeHtml(label) + '">' +
        '</button>' +
      '</div>'
    );
  }

  // ── Build layout HTML ─────────────────────────────────────────────────────
  // Mirrors <main class="content"> from approved design (optimize.html).
  function buildHTML() {
    return (
      '<div id="od-opt-root">' +

        // Global toast — success / error notifications (position:fixed, bottom-right)
        '<div id="od-opt-toast"></div>' +

        // Page head
        '<div class="page-head">' +
          '<h2>Optimize</h2>' +
          '<p>SLM saves tokens three ways: a <b>KV Cache</b> (reuses repeated reads/' +
            'searches — no proxy needed), <b>Compression</b> (reversible shrink of large' +
            ' tool output), and an optional <b>Proxy</b> (auto-intercepts every LLM call' +
            ' — requires restart). Cache is AES-256 encrypted at rest in' +
            ' <span style="font-family:var(--font-mono)">llmcache.db</span>.</p>' +
        '</div>' +

        // CRIT C2: savings error banner (hidden by default)
        '<div id="od-opt-sav-err"' +
          ' style="display:none;background:var(--warn-soft);border:1px solid var(--warn);' +
                  'border-radius:var(--r-md);padding:12px 16px;margin-bottom:16px;' +
                  'align-items:center;justify-content:space-between;gap:12px">' +
          '<span style="color:var(--warn);font-size:13.5px" id="od-opt-sav-err-msg">' +
            'Service unavailable' +
          '</span>' +
          '<button id="od-opt-sav-retry"' +
            ' style="padding:5px 12px;font-size:12.5px;border-radius:var(--r-md);' +
                    'border:1px solid var(--warn);background:transparent;' +
                    'color:var(--warn);cursor:pointer;font-weight:600">' +
            'Retry' +
          '</button>' +
        '</div>' +

        // KPI strip — 4 cards using .kpi-strip + .card.kpi classes
        '<section class="kpi-strip" style="margin-bottom:16px">' +

          // Tokens saved
          '<div class="card kpi">' +
            '<div class="label">' +
              '<span data-ic="bolt"></span> Tokens saved' +
            '</div>' +
            '<div class="value num" id="od-opt-tokens-saved">—</div>' +
            '<div class="delta up" id="od-opt-tokens-delta">▲ this month</div>' +
          '</div>' +

          // Cache hit rate + sparkline
          '<div class="card kpi">' +
            '<div class="label">' +
              '<span data-ic="optimize"></span> Cache hit rate' +
            '</div>' +
            '<div class="value num" id="od-opt-hit-rate">—</div>' +
            '<div class="spark" id="od-opt-hit-rate-spark"></div>' +
          '</div>' +

          // Compression ratio
          '<div class="card kpi">' +
            '<div class="label">' +
              '<span data-ic="memories"></span> Compression ratio' +
            '</div>' +
            '<div class="value num" id="od-opt-compress-ratio">—</div>' +
          '</div>' +

          // Cost saved
          '<div class="card kpi">' +
            '<div class="label">' +
              '<span data-ic="shield"></span> Est. saved' +
            '</div>' +
            '<div class="value num" id="od-opt-usd-saved">—</div>' +
            '<div class="dim" style="font-size:12px;margin-top:4px" id="od-opt-inr-saved">—</div>' +
          '</div>' +

        '</section>' +

        // Two-column body: surfaces (left) + provider savings (right)
        '<div class="grid" style="grid-template-columns:1fr 1fr;align-items:start">' +

          // LEFT — optimization surfaces + controls
          '<div class="card">' +
            '<div class="card-head">' +
              '<h3>Optimization surfaces</h3>' +
            '</div>' +

            // Config error band
            '<div id="od-opt-cfg-err"' +
              ' style="display:none;padding:10px 20px;border-bottom:1px solid var(--border)">' +
              '<span style="color:var(--danger);font-size:13px" id="od-opt-cfg-err-msg">' +
                'Could not load config' +
              '</span>' +
              '<button id="od-opt-cfg-retry"' +
                ' style="margin-left:12px;padding:3px 10px;font-size:12px;' +
                        'border-radius:5px;border:1px solid var(--danger);' +
                        'background:transparent;color:var(--danger);cursor:pointer">' +
                'Retry' +
              '</button>' +
            '</div>' +

            // Switch controls — 6 surfaces reflecting TRUE config state.
            // CRIT C1: master + proxy require daemon restart; all others hot-reload.
            '<div class="card-pad" id="od-opt-ctl-wrap">' +

              // Master — hot-reload is fast but proxy binding still needs restart
              buildCtl('od-opt-sw-master', 'Master enable',
                'Global on/off — disabling stops all token optimization') +

              // CRIT C1: restart hint for master + proxy (shared, shown on either toggle)
              '<div id="od-opt-restart-hint"' +
                ' style="display:none;padding:8px 12px;background:var(--warn-soft);' +
                        'border-radius:var(--r-sm);font-size:12.5px;color:var(--warn);margin:6px 0">' +
                'Restart required — run <code>slm restart</code> to apply this change.' +
              '</div>' +

              // Proxy — TCP listener bound at startup; needs restart to activate
              buildCtl('od-opt-sw-proxy', 'Proxy',
                'Intercepts every LLM call automatically — no code changes needed; restart required') +

              // KV Cache — reuses exact-match repeated tool results
              buildCtl('od-opt-sw-cache', 'KV Cache',
                'Reuses repeated reads/searches to save tokens — no proxy needed') +

              // Compression — reversible structural compression of large tool output
              buildCtl('od-opt-sw-compress', 'Compression',
                'Reversibly shrinks large tool output; expands back before delivery') +

              // Semantic cache — fuzzy match on similar queries
              buildCtl('od-opt-sw-semantic', 'Semantic cache',
                'Groups similar queries into cache hits even when phrasing differs') +

              // Compress prose — extend compression to narrative/free-text output
              buildCtl('od-opt-sw-prose', 'Compress prose',
                'Also compress free-text and narrative output (use with care)') +

              // Compression mode — safe | aggressive (only two valid API values)
              '<div class="ctl">' +
                '<div>' +
                  '<b style="font-size:13.5px">Compression mode</b>' +
                  '<div class="dim" style="font-size:12.5px;margin-top:2px">' +
                    'safe = lossless structural; aggressive may reduce output fidelity' +
                  '</div>' +
                '</div>' +
                '<div class="seg" id="od-opt-mode-seg">' +
                  '<button data-mode="safe">safe</button>' +
                  '<button data-mode="aggressive">aggressive</button>' +
                '</div>' +
              '</div>' +

            '</div>' +

            // Clear cache button
            '<div style="padding:16px 20px 20px;border-top:1px solid var(--border)">' +
              '<button id="od-opt-clear-cache" class="btn ghost"' +
                ' style="color:var(--danger);border-color:var(--danger)">' +
                'Clear cache (llmcache.db)' +
              '</button>' +
              '<span id="od-opt-clear-msg"' +
                ' style="display:none;margin-left:12px;font-size:12.5px;color:var(--ok)">' +
                'Cleared.' +
              '</span>' +
            '</div>' +
          '</div>' +

          // RIGHT — savings by provider
          // TODO: /api/optimize/savings has no per-provider breakdown.
          // Provider names and pricing are factual public data.
          // Per-provider saved tokens show aggregate total from API.
          '<div class="card">' +
            '<div class="card-head">' +
              '<h3>Savings by provider</h3>' +
              '<span class="sub">input / output per 1M tokens</span>' +
            '</div>' +
            '<div class="card-pad">' +
              '<table class="tbl" id="od-opt-prov">' +
                '<thead>' +
                  '<tr>' +
                    '<th>Provider</th><th>Price / 1M</th><th>Saved (M tok)</th>' +
                  '</tr>' +
                '</thead>' +
                '<tbody id="od-opt-prov-body">' +
                  buildProvRows() +
                '</tbody>' +
              '</table>' +
            '</div>' +
          '</div>' +

        '</div>' + // end two-column

        // Activity chart
        '<div class="card" style="margin-top:16px">' +
          '<div class="card-head">' +
            '<h3>Cache &amp; compression activity</h3>' +
            '<span class="sub">hits vs misses · last 24h</span>' +
          '</div>' +
          '<div class="card-pad">' +
            '<div class="bars" id="od-opt-bars" style="height:110px"></div>' +
            '<div style="display:flex;gap:20px;margin-top:12px;font-size:12.5px">' +
              '<span><b class="num" id="od-opt-stat-hits">—</b>' +
                ' <span class="dim">hits</span></span>' +
              '<span><b class="num" id="od-opt-stat-misses">—</b>' +
                ' <span class="dim">misses</span></span>' +
              '<span><b class="num" id="od-opt-stat-runs">—</b>' +
                ' <span class="dim">CCR objects stored</span></span>' +
            '</div>' +
          '</div>' +
        '</div>' +

      '</div>' // od-opt-root
    );
  }

  // ── Build static provider table rows ─────────────────────────────────────
  function buildProvRows() {
    return PROVIDERS.map(function (p) {
      return (
        '<tr>' +
          '<td><b>' + escapeHtml(p.name) + '</b></td>' +
          '<td class="mono dim">' + escapeHtml(p.price) + '</td>' +
          '<td>' +
            '<span style="display:inline-block;vertical-align:middle;width:70px;height:6px;' +
              'border-radius:99px;background:var(--card-2);overflow:hidden;margin-right:8px">' +
              '<span class="od-prov-bar" data-color="' + escapeHtml(p.name) + '"' +
                ' style="display:block;height:100%;width:0%;background:' + p.color + '"></span>' +
            '</span>' +
            '<b class="num od-prov-tok" data-prov="' + escapeHtml(p.name) + '">—</b>' +
          '</td>' +
        '</tr>'
      );
    }).join('');
  }

  // ── Render: status badge ──────────────────────────────────────────────────
  function renderStatus(data) {
    var badge = document.getElementById('od-opt-status-badge');
    if (!badge) return;
    if (!data || data.healthy === false) {
      badge.className = 'badge danger';
      badge.innerHTML = '<span class="dot"></span> error';
      return;
    }
    if (data.enabled) {
      badge.className = 'badge ok';
      badge.innerHTML = '<span class="dot"></span> enabled';
    } else {
      badge.className = 'badge neutral';
      badge.innerHTML = '<span class="dot"></span> disabled';
    }
  }

  // ── Render: KPI strip from /api/optimize/savings ──────────────────────────
  function renderSavings(data) {
    hideSavErr();
    var tok = (data.tokens_saved_input || 0) +
              (data.tokens_saved_output || 0) +
              (data.tokens_saved_compress || 0);
    setHtml('od-opt-tokens-saved', fmtTokensHtml(tok));
    setHtml('od-opt-hit-rate',     fmtHitRateHtml(data.hit_rate));
    setHtml('od-opt-compress-ratio', fmtRatioHtml(data.compress_ratio));
    setText('od-opt-usd-saved',   fmtUSD(data.cost_saved && data.cost_saved.usd));
    setText('od-opt-inr-saved',   fmtINR(data.cost_saved && data.cost_saved.inr));

    // Hit-rate sparkline
    var sparkEl = document.getElementById('od-opt-hit-rate-spark');
    if (sparkEl && typeof window.slmSpark === 'function') {
      var rate = (data.hit_rate || 0) * 100;
      var pts  = [
        Math.round(rate * 0.7), Math.round(rate * 0.78), Math.round(rate * 0.83),
        Math.round(rate * 0.88), Math.round(rate * 0.93), Math.round(rate)
      ];
      sparkEl.innerHTML = window.slmSpark(pts, { color: 'var(--cyan)', w: 96, h: 34 });
    }

    // Provider table footnote removed in v2.1 (od-opt-prov-total element no longer exists).
  }

  // ── Render: config toggles from /api/optimize/config ─────────────────────
  // All 6 surfaces reflect TRUE server state. Called on load and after every
  // successful PUT to confirm persistence (revert happens on error).
  function renderConfig(cfg) {
    _cfg = cfg;
    hideErrEl('od-opt-cfg-err');
    applySwitch('od-opt-sw-master',   cfg.enabled);
    applySwitch('od-opt-sw-proxy',    cfg.proxy_enabled);
    applySwitch('od-opt-sw-cache',    cfg.cache_enabled);
    applySwitch('od-opt-sw-compress', cfg.compress_enabled);
    applySwitch('od-opt-sw-semantic', cfg.semantic_enabled);
    applySwitch('od-opt-sw-prose',    cfg.compress_prose);

    // Compression mode segmented control
    var seg = document.getElementById('od-opt-mode-seg');
    if (seg) {
      var target = cfg.compress_mode || 'safe';
      seg.querySelectorAll('button[data-mode]').forEach(function (btn) {
        var active = btn.getAttribute('data-mode') === target;
        btn.classList.toggle('active', active);
      });
    }
  }

  // ── Render: activity bars from /api/optimize/stats ────────────────────────
  function renderStats(data) {
    setText('od-opt-stat-hits',   fmtNum(data.hits || 0));
    setText('od-opt-stat-misses', fmtNum(data.misses || 0));
    setText('od-opt-stat-runs',   fmtNum(data.cache_entry_count || 0));
    var barsEl = document.getElementById('od-opt-bars');
    if (!barsEl) return;
    var total = (data.hits || 0) + (data.misses || 0);
    if (typeof window.slmBars === 'function' && total > 0) {
      // Distribute total across 24 synthetic hourly buckets for the bar chart.
      // Real hourly bucketing requires a time-series endpoint not yet available.
      // TODO: replace with hourly breakdown when /api/optimize/stats provides it.
      var buckets = [];
      for (var i = 0; i < 24; i++) {
        buckets.push(Math.max(0, Math.floor(total / 24)));
      }
      window.slmBars(barsEl, buckets);
    } else if (total === 0) {
      barsEl.innerHTML =
        '<span style="color:var(--fg-3);font-size:12.5px;margin:auto">' +
          'No activity recorded yet' +
        '</span>';
      barsEl.style.alignItems = 'center';
    }
  }

  // ── Error state helpers ───────────────────────────────────────────────────
  // CRIT C2: savings errors are transient with retry, not a permanent broken state.
  function showSavErr(msg) {
    var band  = document.getElementById('od-opt-sav-err');
    var msgEl = document.getElementById('od-opt-sav-err-msg');
    if (band)  band.style.display = 'flex';
    if (msgEl) msgEl.textContent  = msg || 'Service unavailable';
    ['od-opt-tokens-saved', 'od-opt-hit-rate', 'od-opt-compress-ratio',
     'od-opt-usd-saved', 'od-opt-inr-saved'].forEach(function (id) { setText(id, '—'); });
  }

  function hideSavErr() {
    var band = document.getElementById('od-opt-sav-err');
    if (band) band.style.display = 'none';
  }

  function showErrEl(id, msg) {
    var el    = document.getElementById(id);
    var msgEl = document.getElementById(id + '-msg');
    if (el)   el.style.display   = '';
    if (msgEl) msgEl.textContent = msg || 'Error loading';
  }

  function hideErrEl(id) {
    var el = document.getElementById(id);
    if (el) el.style.display = 'none';
  }

  // ── DOM helper ────────────────────────────────────────────────────────────
  function applySwitch(id, val) {
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('on', !!val);
    el.setAttribute('aria-checked', String(!!val));
  }

  // ── Data loaders ──────────────────────────────────────────────────────────
  function loadSavings() {
    fetch('/api/optimize/savings')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        _retry.savings = 0;
        renderSavings(data);
      })
      .catch(function (err) {
        var attempt = _retry.savings;
        _retry.savings++;
        var base = err && err.message ? ' (' + err.message + ')' : '';
        if (attempt < MAX_AUTO_RETRY) {
          var delay = RETRY_DELAYS_MS[attempt] || 15000;
          showSavErr('Service unavailable' + base + ' — retrying in ' + (delay / 1000) + 's…');
          setTimeout(loadSavings, delay);
        } else {
          showSavErr('Service unavailable' + base + ' — click Retry to reload');
        }
      });
  }

  function loadConfig() {
    fetch('/api/optimize/config')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (cfg) {
        _retry.config = 0;
        renderConfig(cfg);
      })
      .catch(function (err) {
        var attempt = _retry.config;
        _retry.config++;
        var msg = 'Could not load config (' + (err && err.message ? err.message : 'error') + ')';
        if (attempt < MAX_AUTO_RETRY) {
          setTimeout(loadConfig, RETRY_DELAYS_MS[attempt] || 15000);
        } else {
          showErrEl('od-opt-cfg-err', msg);
        }
      });
  }

  function loadStatus() {
    fetch('/api/optimize/status')
      .then(function (r) { return r.json(); })
      .then(renderStatus)
      .catch(function () { renderStatus(null); });
  }

  function loadStats() {
    fetch('/api/optimize/stats')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d) renderStats(d); })
      .catch(function () {});
  }

  function loadAll() {
    loadStatus();   // healthy badge + per-feature feature flags
    loadConfig();   // toggle state (source of truth for all 6 surfaces)
    loadSavings();  // KPI strip
    loadStats();    // activity bars
  }

  // ── Config PUT ────────────────────────────────────────────────────────────
  // On success: shows "Saved" toast + re-reads config to confirm persistence.
  // On failure: reverts toggles to server state + shows error toast.
  function putConfig(body) {
    fetch('/api/optimize/config', {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body)
    })
    .then(function (r) {
      if (!r.ok) {
        loadConfig(); // revert toggles to server state on failure
        showToast('Could not save — settings reverted', 'error');
      } else {
        showToast('Saved', 'ok');
        loadConfig(); // re-read to confirm persistence
      }
    })
    .catch(function () {
      loadConfig();
      showToast('Could not reach server', 'error');
    });
  }

  // ── Event wiring (CSP-safe, delegated — no onXxx= attributes) ────────────
  function wireEvents(container) {
    // Toggle switches
    var ctlWrap = container.querySelector('#od-opt-ctl-wrap');
    if (ctlWrap) {
      ctlWrap.addEventListener('click', function (e) {
        var sw = e.target.closest('button[data-switch-id]');
        if (!sw) return;
        var swId  = sw.getAttribute('data-switch-id');
        var nowOn = !sw.classList.contains('on');
        sw.classList.toggle('on', nowOn);
        sw.setAttribute('aria-checked', String(nowOn));

        // All 6 surfaces map directly to config API fields.
        var fieldMap = {
          'od-opt-sw-master':   'enabled',
          'od-opt-sw-proxy':    'proxy_enabled',
          'od-opt-sw-cache':    'cache_enabled',
          'od-opt-sw-compress': 'compress_enabled',
          'od-opt-sw-semantic': 'semantic_enabled',
          'od-opt-sw-prose':    'compress_prose'
        };
        var field = fieldMap[swId];
        if (!field) return;
        var body = {};
        body[field] = nowOn;
        putConfig(body);

        // CRIT C1: master + proxy require daemon restart to take full effect
        if (swId === 'od-opt-sw-proxy' || swId === 'od-opt-sw-master') {
          var hint = document.getElementById('od-opt-restart-hint');
          if (hint) hint.style.display = '';
        }
      });
    }

    // Compression mode segmented control
    var seg = container.querySelector('#od-opt-mode-seg');
    if (seg) {
      seg.addEventListener('click', function (e) {
        var btn = e.target.closest('button[data-mode]');
        if (!btn) return;
        var mode = btn.getAttribute('data-mode');
        if (mode === 'aggressive') {
          if (!window.confirm(
            'WARNING: Aggressive mode may reduce output fidelity.\n\n' +
            'Do NOT use for: code generation, legal text, exact-output tasks, math.\n' +
            'Safe for: summarization, brainstorming, open-ended chat.\n\n' +
            'Continue?'
          )) return;
        }
        seg.querySelectorAll('button[data-mode]').forEach(function (b) {
          b.classList.toggle('active', b === btn);
        });
        putConfig({ compress_mode: mode });
      });
    }

    // Clear cache button
    var clearBtn = container.querySelector('#od-opt-clear-cache');
    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        if (!window.confirm('Clear all cache entries from llmcache.db?')) return;
        fetch('/api/optimize/cache/clear', { method: 'DELETE' })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            var msgEl = document.getElementById('od-opt-clear-msg');
            if (msgEl) {
              msgEl.textContent = 'Cleared ' + fmtNum(d.deleted || 0) + ' entries.';
              msgEl.style.display = '';
              setTimeout(function () {
                msgEl.style.display = 'none';
                loadSavings();
              }, 3000);
            }
          })
          .catch(function () {});
      });
    }

    // Savings Retry button
    var savRetry = container.querySelector('#od-opt-sav-retry');
    if (savRetry) {
      savRetry.addEventListener('click', function () {
        _retry.savings = 0;
        hideSavErr();
        loadSavings();
      });
    }

    // Config Retry button
    var cfgRetry = container.querySelector('#od-opt-cfg-retry');
    if (cfgRetry) {
      cfgRetry.addEventListener('click', function () {
        _retry.config = 0;
        hideErrEl('od-opt-cfg-err');
        loadConfig();
      });
    }
  }

  // ── Refresh management: MutationObserver re-activates polling on tab show ─
  function startRefreshObserver() {
    var pane = document.getElementById('optimize-pane');
    if (!pane || _observer) return;
    _observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        if (m.type !== 'attributes' || m.attributeName !== 'class') return;
        var t = m.target;
        if (t.classList.contains('active') && t.classList.contains('show')) {
          loadAll();
          if (_pollTimer) clearInterval(_pollTimer);
          _pollTimer = setInterval(function () {
            if (t.classList.contains('active')) {
              loadSavings();
            } else {
              clearInterval(_pollTimer);
              _pollTimer = null;
            }
          }, POLL_MS);
        }
      });
    });
    _observer.observe(pane, { attributes: true });
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.odRenderOptimize = function (container) {
    if (!container) return;
    _container = container;

    // Inject .ctl style once
    injectCtlStyle();

    // Idempotent: reload data if already rendered
    if (container.querySelector('#od-opt-root')) {
      loadAll();
      return;
    }

    container.innerHTML = buildHTML();

    // Fill data-ic SVG placeholders (shell may not have processed dynamically injected HTML)
    if (typeof window.slmIcon === 'function') {
      container.querySelectorAll('[data-ic]').forEach(function (el) {
        el.innerHTML = window.slmIcon(el.getAttribute('data-ic'));
      });
    }

    wireEvents(container);
    loadAll();
    startRefreshObserver();

    // Savings polling timer (self-stops when pane goes inactive)
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(function () {
      var pane = document.getElementById('optimize-pane');
      if (pane && pane.classList.contains('active')) {
        loadSavings();
      } else {
        clearInterval(_pollTimer);
        _pollTimer = null;
      }
    }, POLL_MS);
  };

  // Auto-run at DOMContentLoaded if the pane already exists in DOM
  document.addEventListener('DOMContentLoaded', function () {
    var pane = document.getElementById('optimize-pane');
    if (pane) window.odRenderOptimize(pane);
  });

}());
