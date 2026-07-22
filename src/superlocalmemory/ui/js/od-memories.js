// od-memories.js — Memories · Timeline · Knowledge Clusters (OD design port)
// Render: #memories-pane  |  Public: window.odRenderMemories(container)
// CSP-safe: data-od-act event delegation. XSS-safe: _esc() on all API values.
// nosemgrep: innerHTML — all dynamic values escaped via _esc()
//
// Endpoints (live daemon 127.0.0.1:8765):
//   GET /api/memories?limit=N&offset=N[&category=X]
//       → { memories[{id,memory_id,content,category,importance(0-1),
//                      access_count,created_at,project_name}], total }
//   GET /api/memories/{id}/facts → { facts[{fact_type,content}] }
//   GET /api/v3/timeline/?range=Xd&group_by=category&limit=N
//       → { events[{id,timestamp,category}] }
//   GET /api/clusters → { clusters[{cluster_id,member_count,categories,summary}] }
//   GET /api/clusters/{id}?limit=N → { members[], summary }
// TODO: /api/memories/{id}/score-breakdown — Semantic/BM25/Entity/Temporal not yet exposed.
(function () {
  'use strict';

  // ── Utilities ───────────────────────────────────────────────────────────────

  function _esc(s) {
    if (typeof escapeHtml === 'function') return escapeHtml(s);
    var d = document.createElement('div');
    d.textContent = String(s == null ? '' : s);
    return d.innerHTML;
  }

  function _fmt(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso) : d.toLocaleDateString();
  }

  function _ago(iso) {
    if (!iso) return '';
    var s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 0) s = 0;
    if (s < 60)    return Math.floor(s) + 's ago';
    if (s < 3600)  return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
  }

  // importance arrives as 0–1 float; display as X/10 integer
  function _imp(raw) { return Math.round((parseFloat(raw) || 0) * 10); }

  // importance (0–1) → badge class for Score column
  function _scoreCls(raw) {
    var v = parseFloat(raw) || 0;
    return v >= 0.7 ? 'ok' : v >= 0.4 ? 'warn' : 'danger';
  }

  // API category → badge class
  function _catCls(cat) {
    var MAP = { semantic: 'violet', episodic: 'cyan', opinion: 'warn',
                temporal: 'ok', consolidation: 'danger' };
    return MAP[String(cat).toLowerCase()] || 'neutral';
  }

  // Normalize /api/search result → memory shape (fact_id→id, score→importance)
  function _normSearch(r) { return {id:r.fact_id||r.memory_id||'',content:r.content||'',category:r.category||'semantic',project_name:r.project_name||'',importance:r.score||r.confidence||0,created_at:r.created_at||'',access_count:r.access_count||0}; }
  // shared_with is stored as a JSON array string; show it comma-separated.
  function _sharedToStr(v) {
    if (!v) return '';
    if (Array.isArray(v)) return v.join(', ');
    try { var a = JSON.parse(v); return Array.isArray(a) ? a.join(', ') : ''; }
    catch (e) { return ''; }
  }

  // Write-auth token: reuses dashboard.js closure when available, else fails gracefully
  function _getMutToken() { return typeof window.dashboardInstallToken==='function'?window.dashboardInstallToken():Promise.resolve(''); }

  // ── CSS (injected once) ─────────────────────────────────────────────────────

  var _cssInjected = false;
  function _injectCSS() {
    if (_cssInjected) return;
    _cssInjected = true;
    var s = document.createElement('style');
    s.dataset.odModule = 'memories';
    s.textContent =
      '.od-drawer{position:fixed;top:0;right:0;height:100vh;width:420px;max-width:92vw;' +
      'background:var(--card);border-left:1px solid var(--border);box-shadow:var(--sh-lg);' +
      'transform:translateX(100%);transition:transform .28s cubic-bezier(.22,1,.36,1);' +
      'z-index:60;overflow-y:auto;padding:24px;}' +
      '.od-drawer.open{transform:none;}' +
      '.od-drawer-scrim{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:59;' +
      'opacity:0;pointer-events:none;transition:opacity .25s;}' +
      '.od-drawer-scrim.on{opacity:1;pointer-events:auto;}' +
      '.od-prov{padding:14px;border-radius:var(--r-md);background:var(--card-2);' +
      'border:1px solid var(--border);margin-top:12px;}';
    document.head.appendChild(s);
  }

  // ── Module state ────────────────────────────────────────────────────────────

  var _st = {
    rootId:    null,
    activeTab: 'all',
    category:  null,   // null = all; string = filtered category
    sort:      'created',
    page:      0,
    pageSize:  50,
    total:     0,
    memories:  [],
    catCounts: {},     // pre-fetched per-category totals
    tlRange:   '30d',
    cluLoaded: false,  // reset in render() to avoid stale state on re-entry
  };

  // Known categories from the daemon (pre-fetched at render time)
  var KNOWN_CATS = ['semantic', 'episodic', 'opinion', 'temporal', 'consolidation'];

  // ── Main entry ──────────────────────────────────────────────────────────────

  function render(container) {
    if (!container) return;
    _injectCSS();
    var id = 'od-mem-' + Math.random().toString(36).slice(2, 8);
    _st = Object.assign({}, _st, {
      rootId: id, page: 0, category: null, sort: 'created',
      memories: [], catCounts: {}, cluLoaded: false,
    });
    // Clear per-id timeline cache for this render instance
    _tlLoaded = {};
    container.innerHTML = _scaffold(id);
    _wire(container, id);
    // Fire initial loads in parallel
    _loadMem(id);
    _loadCatCounts(id);
  }

  // ── Scaffold HTML ────────────────────────────────────────────────────────────

  function _scaffold(id) {
    return (
      '<div id="' + id + '">' +
        '<div class="page-head">' +
          '<h2>Everything you\'ve remembered</h2>' +
          '<p id="' + id + '-sub" style="color:var(--fg-2);margin-top:5px">Loading…</p>' +
        '</div>' +
        '<div class="tabs" id="' + id + '-tabs">' +
          '<button class="tab active" data-od-act="tab" data-tab="all">' +
            'All memories <span class="cnt" id="' + id + '-cnt-all">…</span></button>' +
          '<button class="tab" data-od-act="tab" data-tab="timeline">Creation timeline</button>' +
          '<button class="tab" data-od-act="tab" data-tab="clusters">' +
            'Knowledge clusters <span class="cnt" id="' + id + '-cnt-clusters">…</span></button>' +
        '</div>' +
        '<div class="tabpane active" id="' + id + '-pane-all">' + _allScaffold(id) + '</div>' +
        '<div class="tabpane" id="' + id + '-pane-timeline">' + _tlScaffold(id) + '</div>' +
        '<div class="tabpane" id="' + id + '-pane-clusters">' +
          '<div class="launch-grid" id="' + id + '-clu-grid">' +
            _loading('Loading clusters…') +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  function _allScaffold(id) {
    return (
      '<div style="margin-bottom:16px">' +
        '<input data-od-act="search" placeholder="Search all memories…" autocomplete="off" ' +
          'style="width:100%;padding:8px 12px;border:1px solid var(--border);' +
          'border-radius:var(--r-md);background:var(--card-2);color:var(--fg);' +
          'font-size:13.5px;outline:none">' +
      '</div>' +
      // Filter bar: category chips (populated after cat-count fetch) + sort seg
      '<div id="' + id + '-cats" ' +
        'style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;align-items:center">' +
        '<div class="chip on" data-od-act="cat" data-cat="">All categories</div>' +
        '<div style="flex:1"></div>' +
        // Scope view: Mine (this profile) / Shared with me / Global / Everything
        '<div class="seg" title="Which memories to show">' +
          '<button class="active" data-od-act="scope-view" data-scope="mine">Mine</button>' +
          '<button data-od-act="scope-view" data-scope="shared">Shared</button>' +
          '<button data-od-act="scope-view" data-scope="global">Global</button>' +
          '<button data-od-act="scope-view" data-scope="all">All</button>' +
        '</div>' +
        '<div class="seg">' +
          '<button class="active" data-od-act="sort" data-sort="created">Newest</button>' +
          '<button data-od-act="sort" data-sort="score">Score</button>' +
          '<button data-od-act="sort" data-sort="importance">Importance</button>' +
        '</div>' +
      '</div>' +
      '<div class="card" id="' + id + '-tbl-wrap">' + _loading('Loading memories…') + '</div>' +
      '<div id="' + id + '-pg" style="margin-top:10px"></div>'
    );
  }

  function _tlScaffold(id) {
    return (
      '<div class="card">' +
        '<div class="card-head">' +
          '<h3>Memory creation timeline</h3>' +
          '<span class="sub" id="' + id + '-tl-sub">facts stored per day</span>' +
          '<div style="flex:1"></div>' +
          '<div class="seg">' +
            '<button data-od-act="tl-range" data-range="7d">7d</button>' +
            '<button class="active" data-od-act="tl-range" data-range="30d">30d</button>' +
          '</div>' +
        '</div>' +
        '<div class="card-pad">' +
          '<div class="bars" id="' + id + '-bars" style="height:120px">' +
            _loading('Loading timeline…') +
          '</div>' +
          '<div style="display:flex;justify-content:space-between;margin-top:10px;' +
            'font-size:11.5px;color:var(--fg-3)">' +
            '<span id="' + id + '-tl-start"></span><span>today</span>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="card" style="margin-top:16px">' +
        '<div class="card-head"><h3>By fact type</h3></div>' +
        '<div class="card-pad" id="' + id + '-ftypes">' + _loading('Loading breakdown…') + '</div>' +
      '</div>'
    );
  }

  function _loading(msg) {
    return '<div style="padding:40px;text-align:center;color:var(--fg-2);font-size:13px">' +
      _esc(msg) + '</div>';
  }

  // ── Tab switching ────────────────────────────────────────────────────────────

  function _switchTab(id, tab) {
    _st = Object.assign({}, _st, { activeTab: tab });
    var root = document.getElementById(id);
    if (!root) return;
    root.querySelectorAll('#' + id + '-tabs .tab').forEach(function (b) {
      b.classList.toggle('active', b.dataset.tab === tab);
    });
    root.querySelectorAll('.tabpane').forEach(function (p) { p.classList.remove('active'); });
    var pane = document.getElementById(id + '-pane-' + tab);
    if (pane) pane.classList.add('active');
    if (tab === 'timeline') _loadTimeline(id);
    if (tab === 'clusters')  _loadClusters(id);
  }

  // ── Category counts (pre-fetch real totals) ──────────────────────────────────

  function _loadCatCounts(id) {
    Promise.all(KNOWN_CATS.map(function (cat) {
      return fetch('/api/memories?limit=1&category=' + encodeURIComponent(cat))
        .then(function (r) { return r.json(); })
        .then(function (d) { return { cat: cat, total: d.total || 0 }; })
        .catch(function () { return { cat: cat, total: 0 }; });
    })).then(function (results) {
      var counts = {};
      results.forEach(function (r) { if (r.total > 0) counts[r.cat] = r.total; });
      _st = Object.assign({}, _st, { catCounts: counts });
      _rebuildCatBar(id);
    });
  }

  // Rebuild the full filter bar (chips + sort) from current _st
  function _rebuildCatBar(id) {
    var bar = document.getElementById(id + '-cats');
    if (!bar) return;
    var active = _st.category || '';
    var html = '<div class="chip' + (!active ? ' on' : '') +
      '" data-od-act="cat" data-cat="">All categories</div>';
    Object.keys(_st.catCounts).forEach(function (cat) {
      html += '<div class="chip' + (_st.category === cat ? ' on' : '') +
        '" data-od-act="cat" data-cat="' + _esc(cat) + '">' +
        _esc(cat) + ' <span class="cnt">' +
        _st.catCounts[cat].toLocaleString() + '</span></div>';
    });
    var sv = _st.scopeView || 'mine';
    html += '<div style="flex:1"></div>' +
      '<div class="seg" title="Which memories to show">' +
        ['mine', 'shared', 'global', 'all'].map(function (s) {
          var label = { mine: 'Mine', shared: 'Shared', global: 'Global', all: 'All' }[s];
          return '<button class="' + (sv === s ? 'active' : '') +
            '" data-od-act="scope-view" data-scope="' + s + '">' + label + '</button>';
        }).join('') +
      '</div>' +
      '<div class="seg">' +
        '<button class="' + (_st.sort === 'created' ? 'active' : '') +
          '" data-od-act="sort" data-sort="created">Newest</button>' +
        '<button class="' + (_st.sort === 'score' ? 'active' : '') +
          '" data-od-act="sort" data-sort="score">Score</button>' +
        '<button class="' + (_st.sort === 'importance' ? 'active' : '') +
          '" data-od-act="sort" data-sort="importance">Importance</button>' +
      '</div>';
    bar.innerHTML = html;
  }

  // ── Memories fetch & render ──────────────────────────────────────────────────

  function _loadMem(id) {
    var url = '/api/memories?limit=' + _st.pageSize +
      '&offset=' + (_st.page * _st.pageSize);
    if (_st.category) url += '&category=' + encodeURIComponent(_st.category);
    if (_st.scopeView && _st.scopeView !== 'mine') {
      url += '&scope=' + encodeURIComponent(_st.scopeView);
    }
    var wrap = document.getElementById(id + '-tbl-wrap');
    if (wrap) wrap.innerHTML = _loading('Loading memories…');

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        _st = Object.assign({}, _st, { memories: d.memories || [], total: d.total || 0 });
        _renderTable(id, d);
      })
      .catch(function (err) {
        var w = document.getElementById(id + '-tbl-wrap');
        if (w) w.innerHTML = '<div class="card-pad" style="color:var(--danger);text-align:center">' +
          'Failed to load memories: ' + _esc(err.message) + '</div>';
      });
  }

  function _renderTable(id, d) {
    var mems   = d.memories || [];
    var total  = d.total || 0;
    var wrap   = document.getElementById(id + '-tbl-wrap');
    var cntEl  = document.getElementById(id + '-cnt-all');
    var subEl  = document.getElementById(id + '-sub');

    if (cntEl) cntEl.textContent = total.toLocaleString();
    if (subEl) {
      var _projs = mems.reduce(function (s, m) {
        if (m.project_name) s.add(m.project_name);
        return s;
      }, new Set()).size;
      var _pStr = _projs > 0
        ? ' across ' + _projs + ' project' + (_projs === 1 ? '' : 's')
        : '';
      subEl.textContent = total.toLocaleString() + ' memories' + _pStr +
        '. Every memory shows its provenance — why it was remembered and how strongly it scored.';
    }
    if (!wrap) return;

    if (mems.length === 0) {
      wrap.innerHTML = '<div class="card-pad" style="text-align:center;padding:40px;' +
        'color:var(--fg-2)">No memories found. Try a different filter.</div>';
      _renderPag(id, d);
      return;
    }

    var rows = mems.map(function (m, idx) {
      var imp      = _imp(m.importance);
      var impCls   = imp >= 8 ? 'ok' : imp >= 5 ? 'warn' : 'neutral';
      var scorePct = Math.round((parseFloat(m.importance) || 0) * 100);
      var scoreCls = _scoreCls(m.importance);
      var cat      = m.category || 'semantic';
      var preview  = (m.content || '').substring(0, 120);
      if ((m.content || '').length > 120) preview += '…';

      return '<tr class="row" data-od-act="drawer" data-idx="' + idx + '" style="cursor:pointer">' +
        '<td style="max-width:420px">' +
          '<b class="mono dim" style="font-size:11px">#' +
            _esc((m.id || '').substring(0, 8)) +
          '</b>' +
          '<div style="margin-top:2px">' + _esc(preview) + '</div>' +
        '</td>' +
        '<td><span class="badge ' + _esc(_catCls(cat)) + '">' + _esc(cat) + '</span></td>' +
        '<td class="mono dim" style="font-size:12px">' + _esc(m.project_name || '—') + '</td>' +
        '<td><b class="num">' + imp + '</b>/10</td>' +
        '<td><span class="badge ' + _esc(scoreCls) + '">' + scorePct + '%</span></td>' +
        '<td class="mono dim" style="font-size:12px">' + _esc(_fmt(m.created_at)) + '</td>' +
      '</tr>';
    }).join('');

    wrap.innerHTML =
      '<table class="tbl">' +
        '<thead><tr>' +
          '<th>Memory</th><th>Category</th><th>Project</th>' +
          '<th>Importance</th><th>Score</th><th>Created</th>' +
        '</tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>';

    _renderPag(id, d);
  }

  function _renderPag(id, d) {
    var el   = document.getElementById(id + '-pg');
    if (!el) return;
    var tot  = d.total || 0;
    var lim  = d.limit || _st.pageSize;
    var off  = d.offset || (_st.page * _st.pageSize);
    var pg   = Math.floor(off / lim);
    var last = Math.max(0, Math.ceil(tot / lim) - 1);
    var show = Math.min(off + lim, tot);
    var from = tot === 0 ? 0 : off + 1;

    el.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:center;' +
        'font-size:12.5px;color:var(--fg-2)">' +
        '<span>Showing ' + from + '–' + show + ' of ' + tot.toLocaleString() + '</span>' +
        '<div class="seg">' +
          '<button ' + (pg <= 0 ? 'disabled style="opacity:.4"' :
            'data-od-act="pg" data-page="' + (pg - 1) + '"') + '>← Prev</button>' +
          '<button disabled style="opacity:.7;cursor:default">Page ' +
            (pg + 1) + ' / ' + (last + 1) + '</button>' +
          '<button ' + (pg >= last ? 'disabled style="opacity:.4"' :
            'data-od-act="pg" data-page="' + (pg + 1) + '"') + '>Next →</button>' +
        '</div>' +
      '</div>';
  }

  // ── Sort (client-side on current page) ──────────────────────────────────────

  function _doSort(id, by) {
    _st = Object.assign({}, _st, { sort: by });
    var mems = _st.memories.slice();
    // 'score' and 'importance' both rank by the same underlying importance float
    if (by === 'importance' || by === 'score') {
      mems.sort(function (a, b) {
        return (parseFloat(b.importance) || 0) - (parseFloat(a.importance) || 0);
      });
    } else {
      mems.sort(function (a, b) {
        return String(b.created_at || '').localeCompare(String(a.created_at || ''));
      });
    }
    _st = Object.assign({}, _st, { memories: mems });
    _renderTable(id, {
      memories: mems, total: _st.total,
      limit: _st.pageSize, offset: _st.page * _st.pageSize,
    });
  }

  // ── Search — POST /api/search (full corpus, 0.5–1.4 s warm) ─────────────────

  var _searchTimer = null;
  function _doSearch(id, q) {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(function () {
      _st = Object.assign({}, _st, { searchQ: q || null });
      if (!q) { _loadMem(id); return; }
      var wrap = document.getElementById(id + '-tbl-wrap');
      if (wrap) wrap.innerHTML = _loading('Searching…');
      fetch('/api/search', { method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: 50 })
      }).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (d) {
        var mems = (d.results || []).map(_normSearch);
        _st = Object.assign({}, _st, { memories: mems, total: mems.length });
        _renderTable(id, { memories: mems, total: mems.length, limit: mems.length, offset: 0 });
      }).catch(function (err) {
        var w = document.getElementById(id + '-tbl-wrap');
        if (w) w.innerHTML = '<div class="card-pad" style="color:var(--danger);text-align:center;' +
          'padding:24px">Search failed: ' + _esc(err.message) + '</div>';
      });
    }, 350);
  }

  // ── Timeline fetch & render ──────────────────────────────────────────────────

  var _tlLoaded = {};
  function _loadTimeline(id) {
    var range = _st.tlRange;
    if (_tlLoaded[id + '-' + range]) return;
    var barsEl  = document.getElementById(id + '-bars');
    var ftEl    = document.getElementById(id + '-ftypes');
    var startEl = document.getElementById(id + '-tl-start');

    fetch('/api/v3/timeline/?range=' + range + '&group_by=category&limit=1000')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var events = d.events || [];
        _tlLoaded[id + '-' + range] = true;
        if (events.length === 0) {
          if (barsEl) barsEl.innerHTML =
            '<div style="display:grid;place-items:center;width:100%;height:100%;' +
            'color:var(--fg-3);font-size:13px">No memory events in this range.</div>';
          if (ftEl) ftEl.innerHTML =
            '<div style="color:var(--fg-3);font-size:13px">No category data available.</div>';
          return;
        }
        _renderBars(id, events, range, barsEl, startEl);
        _renderFtypes(id, events, ftEl);
      })
      .catch(function (err) {
        if (barsEl) barsEl.innerHTML =
          '<div style="color:var(--fg-3);font-size:13px;text-align:center;padding:16px">' +
          'Timeline unavailable: ' + _esc(err.message) + '</div>';
        // TODO: endpoint returns 404 when timeline feature is disabled
      });
  }

  function _renderBars(id, events, range, barsEl, startEl) {
    if (!barsEl) return;
    var days   = range === '30d' ? 30 : 7;
    var today  = new Date();
    var counts = new Array(days).fill(0);
    events.forEach(function (ev) {
      var diff = Math.floor((today - new Date(ev.timestamp)) / 86400000);
      if (diff >= 0 && diff < days) counts[days - 1 - diff]++;
    });
    if (typeof slmBars === 'function') {
      slmBars(barsEl, counts);
    } else {
      var maxVal = Math.max.apply(null, counts) || 1;
      barsEl.innerHTML = counts.map(function (v) {
        return '<i style="height:' + Math.max(4, (v / maxVal) * 100) + '%' +
          '" title="' + v + ' events"></i>';
      }).join('');
    }
    if (startEl) startEl.textContent = days + 'd ago';
    var subEl = document.getElementById(id + '-tl-sub');
    if (subEl) subEl.textContent = events.length.toLocaleString() + ' events · last ' + days + ' days';
  }

  function _renderFtypes(id, events, ftEl) {
    if (!ftEl) return;
    var catMap = {};
    events.forEach(function (ev) {
      catMap[ev.category || 'unknown'] = (catMap[ev.category || 'unknown'] || 0) + 1;
    });
    var total  = events.length;
    var colors = { temporal:'var(--ok)', semantic:'var(--violet)', episodic:'var(--cyan)', opinion:'var(--warn)' };
    var sorted = Object.keys(catMap).sort(function (a, b) { return catMap[b] - catMap[a]; });
    ftEl.innerHTML = sorted.map(function (cat) {
      var pct   = Math.round(catMap[cat] / total * 100);
      var color = colors[cat] || 'var(--fg-3)';
      return '<div style="margin-bottom:13px">' +
        '<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">' +
          '<span class="mono">' + _esc(cat) + '</span><b>' + pct + '%</b>' +
        '</div>' +
        '<div class="meter"><i style="width:' + pct + '%;background:' + color + '"></i></div>' +
      '</div>';
    }).join('');
  }

  // ── Clusters fetch & render ──────────────────────────────────────────────────

  function _loadClusters(id) {
    if (_st.cluLoaded) return;
    var grid = document.getElementById(id + '-clu-grid');
    if (!grid) return;
    fetch('/api/clusters')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        _st = Object.assign({}, _st, { cluLoaded: true });
        var clus  = d.clusters || [];
        var cntEl = document.getElementById(id + '-cnt-clusters');
        if (cntEl) cntEl.textContent = clus.length.toLocaleString();
        if (clus.length === 0) {
          grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:40px;' +
            'color:var(--fg-2)">No clusters yet. Clusters form as memories accumulate.</div>';
          return;
        }
        _renderClusters(id, clus, grid);
      })
      .catch(function (err) {
        if (grid) grid.innerHTML =
          '<div style="grid-column:1/-1;text-align:center;padding:24px;' +
          'color:var(--danger);font-size:13px">Failed to load clusters: ' +
          _esc(err.message) + '</div>';
      });
  }

  var _PAL = ['var(--violet)', 'var(--cyan)', 'var(--warn)', 'var(--ok)', 'var(--danger)'];

  function _renderClusters(id, clus, grid) {
    grid.innerHTML = clus.map(function (c, i) {
      var color   = _PAL[i % _PAL.length];
      var summary = c.summary || c.categories || '';
      var preview = summary.length > 80 ? summary.substring(0, 80) + '…' : summary;
      var cid     = String(c.cluster_id || '');
      return '<div class="launch-card card" style="cursor:pointer"' +
        ' data-od-act="expand-cluster" data-cid="' + _esc(cid) + '">' +
        '<div class="ic" style="background:color-mix(in srgb,' + color + ' 15%,transparent);color:' + color + '">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="21" height="21">' +
            '<path d="M12 3l1.9 4.6L18 9l-3.5 3 1 5-3.5-2.5L8.5 17l1-5L6 9l4.1-1.4z"/>' +
          '</svg>' +
        '</div>' +
        '<h3>' + _esc(preview) + '</h3>' +
        '<p><b class="num">' + _esc(String(c.member_count)) + '</b> memories · expand →</p>' +
        '<div style="display:none;margin-top:14px;border-top:1px solid var(--border);' +
          'padding-top:12px;font-size:13px" id="' + id + '-clu-' + _esc(cid) + '"></div>' +
      '</div>';
    }).join('');
  }

  function _expandCluster(id, cid) {
    var detEl = document.getElementById(id + '-clu-' + cid);
    if (!detEl) return;
    if (detEl.style.display !== 'none') { detEl.style.display = 'none'; return; }
    detEl.style.display = 'block';
    if (detEl.dataset.loaded) return;
    detEl.dataset.loaded = '1';
    detEl.textContent = 'Loading members…';
    fetch('/api/clusters/' + encodeURIComponent(cid) + '?limit=5')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var members = d.members || [];
        if (members.length === 0) { detEl.textContent = 'No members found.'; return; }
        detEl.innerHTML = members.map(function (m, i) {
          var txt = m.content || m.summary || '';
          return '<div style="margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid var(--border)">' +
            '<div style="font-size:12.5px">' + (i + 1) + '. ' +
              _esc(txt.substring(0, 120)) + '</div></div>';
        }).join('');
      })
      .catch(function () { detEl.textContent = 'Failed to load members.'; });
  }

  // ── Provenance drawer ────────────────────────────────────────────────────────

  function _ensureDrawer() {
    if (document.getElementById('od-mem-drawer')) return;
    var scrim = document.createElement('div');
    scrim.id = 'od-mem-drawer-scrim';
    scrim.className = 'od-drawer-scrim';
    scrim.dataset.odAct = 'close-drawer';
    document.body.appendChild(scrim);
    var drawer = document.createElement('aside');
    drawer.id = 'od-mem-drawer';
    drawer.className = 'od-drawer';
    document.body.appendChild(drawer);
  }

  function _openDrawer(idx) {
    var mem = _st.memories[idx];
    if (!mem) return;
    _ensureDrawer();
    var drawer = document.getElementById('od-mem-drawer');
    var scrim  = document.getElementById('od-mem-drawer-scrim');
    if (!drawer || !scrim) return;

    var imp      = _imp(mem.importance);
    var cat      = mem.category || 'semantic';
    var scorePct = Math.round((parseFloat(mem.importance) || 0) * 100);

    drawer.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:flex-start">' +
        '<div>' +
          '<div style="font-size:12px;color:var(--fg-2)">Memory</div>' +
          '<h2 style="font-size:19px;margin-top:2px">' +
            _esc('#' + (mem.id || '').substring(0, 12)) + '</h2>' +
        '</div>' +
        '<button class="btn icon ghost" data-od-act="close-drawer">✕</button>' +
      '</div>' +
      '<div class="od-prov" style="font-size:14px;line-height:1.7">' +
        _esc(mem.content || '') +
      '</div>' +
      '<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">' +
        '<span class="badge ' + _esc(_catCls(cat)) + '">' + _esc(cat) + '</span>' +
        '<span class="badge neutral">' + _esc(mem.project_name || 'no project') + '</span>' +
        '<span class="badge ' + (mem.scope === 'global' ? 'success' : mem.scope === 'shared' ? 'warn' : 'neutral') +
          '">' + _esc(mem.scope || 'personal') + '</span>' +
      '</div>' +
      // Scope & sharing control (C2) — set personal/shared/global from the UI
      '<h3 style="font-size:13px;margin:20px 0 8px">Scope &amp; sharing</h3>' +
      '<div class="od-prov" style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">' +
        '<select id="od-scope-sel" style="padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
          ['personal', 'shared', 'global'].map(function (s) {
            return '<option value="' + s + '"' + ((mem.scope || 'personal') === s ? ' selected' : '') + '>' + s + '</option>';
          }).join('') +
        '</select>' +
        '<input id="od-scope-shared" placeholder="share with profiles (comma-sep)" ' +
          'value="' + _esc(_sharedToStr(mem.shared_with)) + '" ' +
          'style="flex:1;min-width:140px;padding:6px 8px;border-radius:6px;background:var(--card-2);color:var(--fg);border:1px solid var(--border);font-size:13px">' +
        '<button data-od-act="set-scope" data-mid="' + _esc(mem.id) + '" ' +
          'style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Apply</button>' +
      '</div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Why this was remembered</h3>' +
      '<div class="od-prov">' +
        '<div style="font-size:13px;color:var(--fg-2);line-height:1.6">' +
          'Importance: <b>' + imp + '/10</b>. ' +
          'Score: <b>' + scorePct + '%</b>. ' +
          'Recalled <b>' + _esc(String(mem.access_count || 0)) + '</b> time' +
          (mem.access_count === 1 ? '' : 's') + '. ' +
          'Stored ' + _esc(_ago(mem.created_at)) + '.' +
        '</div>' +
        // TODO: /api/memories/{id}/score-breakdown — Semantic/BM25/EntityGraph/Temporal
        //       decomposition is not yet exposed by the daemon; show when available.
      '</div>' +
      '<h3 style="font-size:13px;margin:20px 0 8px">Atomic facts</h3>' +
      '<div id="od-drawer-facts" style="color:var(--fg-3);font-size:13px">Loading facts…</div>' +
      '<div style="display:flex;gap:8px;margin-top:20px;padding-top:12px;border-top:1px solid var(--border)">' +
        '<button data-od-act="edit-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);cursor:pointer;font-size:13px">Edit</button>' +
        '<button data-od-act="del-mem" data-mid="' + _esc(mem.id) + '" style="padding:6px 14px;border:1px solid var(--danger);border-radius:6px;background:transparent;color:var(--danger);cursor:pointer;font-size:13px">Delete</button>' +
      '</div><div id="od-mem-act" style="margin-top:10px"></div>';

    drawer.classList.add('open');
    scrim.classList.add('on');
    if (mem.id) _loadFacts(mem.id);
  }

  function _closeDrawer() {
    var d = document.getElementById('od-mem-drawer');
    var s = document.getElementById('od-mem-drawer-scrim');
    if (d) d.classList.remove('open');
    if (s) s.classList.remove('on');
  }

  function _loadFacts(memId) {
    var el = document.getElementById('od-drawer-facts');
    if (!el) return;
    fetch('/api/memories/' + encodeURIComponent(memId) + '/facts')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var facts = d.facts || [];
        if (facts.length === 0) {
          el.textContent = 'No atomic facts recorded for this memory.'; return;
        }
        el.innerHTML = facts.map(function (f) {
          return '<div class="od-prov" style="margin-bottom:8px">' +
            '<span class="badge neutral" style="font-size:10px">' +
              _esc(f.fact_type || 'fact') +
            '</span>' +
            '<div style="font-size:13px;line-height:1.6;margin-top:4px">' +
              _esc((f.content || '').substring(0, 200)) +
            '</div></div>';
        }).join('');
      })
      .catch(function () { if (el) el.textContent = 'Could not load facts.'; });
  }

  function _startEdit(mid) { // show inline edit form in drawer
    var m = _st.memories.filter(function (x) { return x.id === mid; })[0];
    var el = document.getElementById('od-mem-act'); if (!el || !m) return;
    el.innerHTML = '<textarea id="od-ea" rows="4" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--card-2);color:var(--fg);font-size:13px">' + _esc(m.content) + '</textarea><button data-od-act="save-edit" data-mid="' + _esc(mid) + '" style="display:block;margin-top:6px;padding:6px 14px;border:none;border-radius:6px;background:var(--violet);color:#fff;cursor:pointer">Save</button>';
  }
  function _startDel(mid) { // show delete confirmation in drawer
    var el = document.getElementById('od-mem-act'); if (!el) return;
    el.innerHTML = '<p style="font-size:13px;color:var(--fg-2);margin-bottom:10px">Permanently delete this memory? This cannot be undone.</p><button data-od-act="confirm-del" data-mid="' + _esc(mid) + '" style="padding:6px 14px;border-radius:6px;background:var(--danger);color:#fff;border:none;cursor:pointer;margin-right:8px">Confirm delete</button><button data-od-act="cancel-act" style="padding:6px 14px;border-radius:6px;background:var(--card-2);border:1px solid var(--border);color:var(--fg);cursor:pointer">Cancel</button>';
  }

  // ── Event delegation ─────────────────────────────────────────────────────────
  function _wire(container, id) {
    container.addEventListener('click', function (e) {
      var el  = e.target.closest('[data-od-act]');
      if (!el) return;
      var act = el.dataset.odAct;

      if (act === 'tab') {
        _switchTab(id, el.dataset.tab);
        return;
      }
      if (act === 'cat') {
        // Toggle on/off chip states without a full re-render
        var bar = document.getElementById(id + '-cats');
        if (bar) bar.querySelectorAll('[data-od-act="cat"]').forEach(function (c) {
          c.classList.toggle('on', c.dataset.cat === el.dataset.cat);
        });
        _st = Object.assign({}, _st, { category: el.dataset.cat || null, page: 0 });
        _loadMem(id);
        return;
      }
      if (act === 'sort') {
        // Update active state on sort buttons in-place
        var root = document.getElementById(id);
        if (root) root.querySelectorAll('[data-od-act="sort"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.sort === el.dataset.sort);
        });
        _doSort(id, el.dataset.sort);
        return;
      }
      if (act === 'scope-view') {
        var sroot = document.getElementById(id);
        if (sroot) sroot.querySelectorAll('[data-od-act="scope-view"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.scope === el.dataset.scope);
        });
        _st = Object.assign({}, _st, { scopeView: el.dataset.scope, page: 0 });
        _loadMem(id);
        return;
      }
      if (act === 'drawer') {
        var row = el.closest('tr[data-od-act]');
        var idx = parseInt((row || el).dataset.idx, 10);
        if (!isNaN(idx)) _openDrawer(idx);
        return;
      }
      if (act === 'close-drawer') { _closeDrawer(); return; }
      if (act === 'pg') {
        var pg = parseInt(el.dataset.page, 10);
        if (!isNaN(pg)) { _st = Object.assign({}, _st, { page: pg }); _loadMem(id); }
        return;
      }
      if (act === 'expand-cluster') { _expandCluster(id, el.dataset.cid); return; }
      if (act === 'tl-range') {
        var root2 = document.getElementById(id);
        if (root2) root2.querySelectorAll('[data-od-act="tl-range"]').forEach(function (b) {
          b.classList.toggle('active', b.dataset.range === el.dataset.range);
        });
        _tlLoaded[id + '-' + _st.tlRange] = false; // invalidate cache
        _st = Object.assign({}, _st, { tlRange: el.dataset.range });
        _loadTimeline(id);
        return;
      }
    });

    // Search — input event
    container.addEventListener('input', function (e) {
      if (e.target.dataset.odAct === 'search') _doSearch(id, e.target.value.trim());
    });

    // Drawer actions — document-level (drawer is outside container), guarded once
    if (!window._slmMemDrawerWired) {
      window._slmMemDrawerWired = true;
      document.addEventListener('click', function (e) {
        var el = e.target.closest('[data-od-act]');
        if (!el) return;
        var act = el.dataset.odAct;
        if (act === 'close-drawer') { _closeDrawer(); return; }
        if (act === 'edit-mem') { _startEdit(el.dataset.mid); return; }
        if (act === 'del-mem')  { _startDel(el.dataset.mid); return; }
        if (act === 'cancel-act') { var ac = document.getElementById('od-mem-act'); if (ac) ac.innerHTML = ''; return; }
        if (act === 'set-scope') {
          var smid = el.dataset.mid;
          var scopeVal = (document.getElementById('od-scope-sel') || {}).value || 'personal';
          var sharedVal = ((document.getElementById('od-scope-shared') || {}).value || '').trim();
          if (scopeVal === 'shared' && !sharedVal) {
            showToast('Enter at least one profile to share with'); return;
          }
          _getMutToken().then(function (tok) {
            if (!tok) { showToast('Auth unavailable — reload the page'); return; }
            fetch('/api/memories/' + encodeURIComponent(smid) + '/scope', {
              method: 'PATCH', credentials: 'same-origin',
              headers: { 'Content-Type': 'application/json', 'X-Install-Token': tok },
              body: JSON.stringify({ scope: scopeVal, shared_with: sharedVal }),
            })
              .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
              .then(function () { showToast('Scope updated → ' + scopeVal); _closeDrawer(); _loadMem(_st.rootId); })
              .catch(function (e) { showToast('Scope update failed: ' + e.message); });
          });
          return;
        }
        if (act === 'save-edit' || act === 'confirm-del') {
          var rid = _st.rootId, isDel = act === 'confirm-del', mid = el.dataset.mid;
          var content = isDel ? '' : ((document.getElementById('od-ea') || {}).value || '').trim();
          if (!isDel && !content) { showToast('Content cannot be empty'); return; }
          _getMutToken().then(function (tok) {
            if (!tok) { showToast('Auth unavailable — reload the page'); return; }
            var opts = isDel
              ? {method:'DELETE',credentials:'same-origin',headers:{'X-Install-Token':tok}}
              : {method:'PATCH',credentials:'same-origin',headers:{'Content-Type':'application/json','X-Install-Token':tok},body:JSON.stringify({content:content})};
            fetch('/api/memories/' + encodeURIComponent(mid), opts)
              .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
              .then(function () { showToast(isDel ? 'Memory deleted' : 'Memory updated'); _closeDrawer(); _loadMem(rid); })
              .catch(function (e) { showToast((isDel ? 'Delete' : 'Edit') + ' failed: ' + e.message); });
          });
          return;
        }
      });
    }
  }

  // ── Public API + auto-init ───────────────────────────────────────────────────

  window.odRenderMemories = render;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      var p = document.getElementById('memories-pane');
      if (p) render(p);
    });
  } else {
    var _p = document.getElementById('memories-pane');
    if (_p) render(_p);
  }

}());
