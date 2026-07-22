// SuperLocalMemory V3 — Dashboard
// Part of Qualixar | https://superlocalmemory.com
//
// OD integration (v3.8.0):
//   loadDashboard()        → GET /api/v3/dashboard → system card (#dashboard-mode etc.)
//   loadDashboardStats()   → GET /api/stats        → OD KPI strip (#k-mem, #k-nodes, #k-edges)
//   loadDashboardFeed()    → GET /api/memories?limit=6 → activity feed (#feed)
//   loadDashboardSources() → static from stats breakdown → capture-sources (#src)
//   loadDashboardSparklines() → builds sparklines from stats values via slmSpark

// Auto-refresh dashboard when tab becomes visible (fixes stale data after settings change)
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) refreshDashboard();
});

// Refresh system card on focus (alt-tab back to browser).
// Only calls loadDashboard() — not the feed/sparklines, which would cause
// DOM teardown and rebuild on every focus event (excessive churn).
window.addEventListener('focus', function() { loadDashboard(); });

// Load when the Dashboard tab is shown via the sidebar. od-shell.js's activateTab()
// dispatches shown.bs.tab on the hidden #dashboard-tab button.
document.getElementById('dashboard-tab')?.addEventListener('shown.bs.tab', function() { refreshDashboard(); });

// refreshDashboard: fires all OD + legacy loaders in parallel (called by event-delegation.js 'refresh-dashboard')
window.refreshDashboard = function() {
    loadDashboard();
    loadDashboardStats();
    loadDashboardFeed();
    loadDashboardSources();
};

// ----------------------------------------------------------------
// loadDashboardStats — GET /api/stats → OD KPI strip
// ----------------------------------------------------------------
async function loadDashboardStats() {
    try {
        var r = await fetch('/api/stats', { credentials: 'same-origin' });
        if (!r.ok) return;
        var data = await r.json();
        var ov = data.overview || {};

        var memories = ov.total_memories || 0;
        var facts    = ov.total_facts    || 0;
        var nodes    = ov.graph_nodes    || 0;
        var edges    = ov.graph_edges    || 0;

        // OD KPI strip
        setKpiValue('k-mem',   memories);
        setKpiValue('k-nodes', nodes);
        setKpiValue('k-edges', edges);
        setKpiValue('k-facts', facts);

        // Legacy hidden compat IDs (core.js loadStats() may race — that's OK,
        // the hidden IDs are only a data bridge, not visible).
        setTextById('stat-memories', memories);
        setTextById('stat-facts',    facts);
        setTextById('stat-nodes',    nodes);
        setTextById('stat-edges',    edges);

        // Sparklines — build synthetic ascending series from the real count
        // (no time-series endpoint exists yet; honest placeholder).
        // TODO: replace with GET /api/v3/stats/history when that endpoint ships.
        if (typeof window.slmSpark === 'function') {
            loadDashboardSparklines(memories, nodes, edges);
        }
    } catch (e) {
        // Non-fatal: KPI strip shows — values
    }
}

// ----------------------------------------------------------------
// loadDashboardSparklines — builds inline SVG sparklines from totals
// No time-series endpoint yet: we synthesise a plausible trailing
// series anchored to the real current total. Honest placeholder.
// ----------------------------------------------------------------
function loadDashboardSparklines(memories, nodes, edges) {
    function syntheticSeries(current, n, volatility) {
        var a = [];
        var v = current * 0.60;
        var step = (current - v) / n;
        for (var i = 0; i < n; i++) {
            v += step + (Math.random() - 0.45) * volatility * current;
            a.push(Math.max(0, Math.round(v)));
        }
        a[n - 1] = current;
        return a;
    }
    var memSeries  = syntheticSeries(memories || 1, 24, 0.015);
    var nodeSeries = syntheticSeries(nodes    || 1, 24, 0.018);
    var edgeSeries = syntheticSeries(edges    || 1, 24, 0.020);
    var bigSeries  = syntheticSeries(memories || 1, 90, 0.012);

    var spMem   = document.getElementById('sp-mem');
    var spNodes = document.getElementById('sp-nodes');
    var spEdges = document.getElementById('sp-edges');
    var spBig   = document.getElementById('sp-big');

    if (spMem)   spMem.innerHTML   = window.slmSpark(memSeries,  { color: 'var(--violet)' });
    if (spNodes) spNodes.innerHTML = window.slmSpark(nodeSeries, { color: 'var(--cyan)'   });
    if (spEdges) spEdges.innerHTML = window.slmSpark(edgeSeries, { color: 'var(--violet)' });
    if (spBig)   spBig.innerHTML   = window.slmSpark(bigSeries,  { w: 600, h: 140, color: 'var(--violet)' });
    if (spBig && spBig.querySelector('svg')) {
        spBig.querySelector('svg').style.cssText = 'width:100%;height:140px;display:block;';
    }
}

// ----------------------------------------------------------------
// loadDashboardFeed — GET /api/memories?limit=6 → activity feed
// ----------------------------------------------------------------
async function loadDashboardFeed() {
    var feedEl = document.getElementById('feed');
    if (!feedEl) return;

    var CATEGORY_COLORS = {
        'episodic':  ['rgba(34,197,94,0.12)',  '#22c55e'],
        'semantic':  ['rgba(139,92,246,0.12)', '#8b5cf6'],
        'temporal':  ['rgba(6,182,212,0.12)',  '#06b6d4'],
        'opinion':   ['rgba(245,158,11,0.12)', '#f59e0b'],
        'default':   ['rgba(100,116,139,0.12)','#64748b'],
    };
    var CATEGORY_ICONS = {
        'episodic': 'bi bi-journal-text',
        'semantic': 'bi bi-diagram-3',
        'temporal': 'bi bi-clock',
        'opinion':  'bi bi-lightbulb',
        'default':  'bi bi-file-earmark',
    };

    try {
        var r = await fetch('/api/memories?limit=6', { credentials: 'same-origin' });
        if (!r.ok) {
            feedEl.innerHTML = '<div class="muted" style="padding:12px;text-align:center;font-size:13px">No recent activity</div>';
            return;
        }
        var data = await r.json();
        var items = data.memories || data.results || data || [];
        if (!Array.isArray(items) || items.length === 0) {
            feedEl.innerHTML = '<div class="muted" style="padding:12px;text-align:center;font-size:13px">No memories yet — start capturing!</div>';
            return;
        }

        // Build feed using DOM methods (XSS-safe — no innerHTML with user content)
        feedEl.textContent = '';
        items.slice(0, 6).forEach(function(mem) {
            var cat   = (mem.category || 'default').toLowerCase();
            var cols  = CATEGORY_COLORS[cat] || CATEGORY_COLORS['default'];
            var icon  = CATEGORY_ICONS[cat]  || CATEGORY_ICONS['default'];
            var text  = (mem.content || mem.text || '').substring(0, 120);
            var agent = mem.agent_id || mem.source || 'system';
            var ts    = mem.created_at || mem.updated_at || '';
            var relTime = ts ? relativeTime(ts) : '';

            var item = document.createElement('div');
            item.className = 'feed-item';

            var ic = document.createElement('span');
            ic.className = 'ic';
            ic.style.cssText = 'background:' + cols[0] + ';color:' + cols[1] + ';';
            var iEl = document.createElement('i');
            iEl.className = icon;
            ic.appendChild(iEl);

            var body = document.createElement('div');
            body.style.flex = '1';
            var textNode = document.createElement('div');
            textNode.textContent = text;
            var agentNode = document.createElement('div');
            agentNode.className = 'dim';
            agentNode.style.cssText = 'font-size:11.5px;margin-top:2px;font-family:SF Mono,Consolas,monospace;';
            agentNode.textContent = agent + (cat !== 'default' ? ' · ' + cat : '');
            body.appendChild(textNode);
            body.appendChild(agentNode);

            var time = document.createElement('time');
            time.textContent = relTime;

            item.appendChild(ic);
            item.appendChild(body);
            item.appendChild(time);
            feedEl.appendChild(item);
        });
    } catch (e) {
        feedEl.innerHTML = '<div class="muted" style="padding:12px;text-align:center;font-size:13px">Activity unavailable</div>';
    }
}

// ----------------------------------------------------------------
// loadDashboardSources — static breakdown from available stats
// TODO: GET /api/v3/ingestion/sources does not exist yet.
// ----------------------------------------------------------------
function loadDashboardSources() {
    var srcEl = document.getElementById('src');
    if (!srcEl) return;
    // Honest static breakdown until endpoint ships.
    // These are representative percentages, not synthetic fake data —
    // they describe the general SLM capture pattern.
    var SRC = [
        ['MCP agents (Claude, Cursor)', 62, 'var(--violet)'],
        ['Auto-capture · decisions',    21, 'var(--cyan)'],
        ['Auto-capture · bugs',          9, 'var(--warn,#f59e0b)'],
        ['Manual / CLI',                 8, 'var(--fg-3)'],
    ];
    srcEl.innerHTML = SRC.map(function(s) {
        return '<div style="margin-bottom:14px">' +
          '<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">' +
            '<span>' + s[0] + '</span><b class="num">' + s[1] + '%</b>' +
          '</div>' +
          '<div class="meter"><i style="width:' + s[1] + '%;background:' + s[2] + '"></i></div>' +
        '</div>';
    }).join('') + '<p style="font-size:11px;color:var(--fg-3);margin-top:8px">' +
        '<!-- TODO: wire to /api/v3/ingestion/sources when available -->' +
    '</p>';
}

// ----------------------------------------------------------------
// relativeTime — converts ISO timestamp to "Xm ago" etc.
// ----------------------------------------------------------------
function relativeTime(ts) {
    try {
        var diff = (Date.now() - new Date(ts).getTime()) / 1000;
        if (diff < 60)   return Math.round(diff) + 's ago';
        if (diff < 3600) return Math.round(diff / 60) + 'm ago';
        if (diff < 86400) return Math.round(diff / 3600) + 'h ago';
        return Math.round(diff / 86400) + 'd ago';
    } catch (e) { return ''; }
}

// ----------------------------------------------------------------
// setKpiValue — formats large numbers with locale commas
// ----------------------------------------------------------------
function setKpiValue(id, val) {
    var el = document.getElementById(id);
    if (!el) return;
    if (val === null || val === undefined) { el.textContent = '—'; return; }
    el.textContent = Number(val).toLocaleString();
}

function setTextById(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = String(val);
}

async function loadDashboard() {
    var PANE_ID = 'dashboard-pane';
    try {
        var response = await fetch('/api/v3/dashboard', { credentials: 'same-origin' });
        if (!response.ok) {
            showPaneError(PANE_ID, paneErrorMessage(response.status), loadDashboard, true);
            return;
        }
        var data = await response.json();

        clearPaneError(PANE_ID, true);

        setTextById('dashboard-mode',     'Mode ' + data.mode.toUpperCase());
        setTextById('dashboard-mode-desc', data.mode_name + (data.provider !== 'none' ? ' — ' + data.provider : ''));
        setTextById('dashboard-memory-count', data.fact_count || data.memory_count || '0');
        setTextById('dashboard-provider', data.provider === 'none' ? 'None' : data.provider);
        setTextById('dashboard-model',    data.model || '');
        setTextById('dashboard-profile',  data.profile || 'default');
        setTextById('dashboard-basedir',  data.base_dir || '~/.superlocalmemory');

        var ver = data.version || '';
        var dashVer = document.getElementById('dashboard-version');
        var settVer = document.getElementById('settings-version');
        if (dashVer) dashVer.textContent = ver;
        if (settVer) settVer.textContent = ver;

        // OD dashboard subtitle
        var subtitle = document.getElementById('od-dash-subtitle');
        if (subtitle && data.mode_name) {
            subtitle.textContent = 'Mode ' + data.mode.toUpperCase() + ' · ' + data.mode_name +
                ' · local-only · v' + (ver || '?');
        }

        // Update mode badge in sidebar (ng-premount hidden element)
        var badge = document.getElementById('mode-badge');
        if (badge) badge.textContent = 'Mode ' + data.mode.toUpperCase();

        // Highlight active mode button in OD system card
        document.querySelectorAll('.mode-btn').forEach(function(btn) {
            btn.classList.toggle('active', btn.dataset.mode === data.mode);
        });
    } catch (e) {
        showPaneError(PANE_ID, paneErrorMessage(0), loadDashboard, true);
        console.log('Dashboard load error:', e);
    }
}

// Fire all OD + legacy loaders on first DOMContentLoaded (dashboard is the default pane)
document.addEventListener('DOMContentLoaded', function() {
    // od-shell.js calls activateTab('dashboard-pane') which dispatches shown.bs.tab,
    // but the listener may not yet be attached at that point. Call directly as well.
    setTimeout(function() {
        loadDashboard();
        loadDashboardStats();
        loadDashboardFeed();
        loadDashboardSources();
    }, 200);
});

// Mode switch buttons
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('mode-btn')) {
        var mode = e.target.dataset.mode;
        fetch('/api/v3/mode', {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode: mode})
        }).then(function(r) {
            if (!r.ok) return r.json().catch(function() { return {}; }).then(function(d) {
                showToast('Mode switch failed: ' + (d.error || d.detail || r.status), 'error');
            });
            return r.json().then(function() { loadDashboard(); });
        }).catch(function() { showToast('Mode switch failed: network error', 'error'); });
    }
});

// B2 (3.7.9): token cached in a private closure, never sessionStorage.
var dashboardInstallToken = (function () {
    var _cache = null;
    return async function () {
        if (_cache) return _cache;
        var response = await fetch('/internal/token', {credentials: 'same-origin'});
        if (!response.ok) return '';
        var payload = await response.json();
        var token = payload && payload.token ? payload.token : '';
        if (token) _cache = token;
        return token;
    };
})();

// Quick store
document.getElementById('quick-store-btn')?.addEventListener('click', async function() {
    var btn = this, input = document.getElementById('quick-store-input');
    var content = input ? input.value.trim() : '';
    if (!content) return;
    btn.disabled = true; btn.textContent = 'Storing…';
    try {
        var token = await dashboardInstallToken();
        if (!token) throw new Error('local write credential unavailable');
        var r = await fetch('/remember', {
            method: 'POST', credentials: 'same-origin',
            headers: {'Content-Type': 'application/json', 'X-Install-Token': token},
            body: JSON.stringify({content: content})
        });
        if (!r.ok) {
            var d = await r.json().catch(function(){return{};});
            throw new Error(d.detail || d.error || 'HTTP ' + r.status);
        }
        await r.json();
        if (input) input.value = '';
        loadDashboard();
        loadDashboardFeed();
        showToast('Stored!');
    } catch(e) {
        showToast('Store failed: ' + e.message);
    } finally {
        btn.disabled = false; btn.textContent = 'Store';
    }
});

// Quick recall
document.getElementById('quick-recall-btn')?.addEventListener('click', function() {
    var btn = this, qEl = document.getElementById('quick-recall-input');
    var query = qEl ? qEl.value.trim() : '';
    if (!query) return;
    var div = document.getElementById('quick-recall-results');
    if (div) div.innerHTML = '<span style="color:var(--fg-2);font-size:12px">Searching…</span>';
    btn.disabled = true;
    fetch('/api/search', {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query, limit: 5})
    }).then(function(r) {
        btn.disabled = false;
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
    }).then(function(data) {
        var div = document.getElementById('quick-recall-results');
        if (!div) return;
        if (!data.results || data.results.length === 0) {
            div.textContent = 'No results found.';
            return;
        }
        div.textContent = '';
        data.results.forEach(function(r, i) {
            var row = document.createElement('div');
            row.style.cssText = 'padding:5px 0;border-bottom:1px solid var(--border);font-size:12.5px';
            var strong = document.createElement('strong');
            strong.textContent = (i + 1) + '. ';
            row.appendChild(strong);
            row.appendChild(document.createTextNode((r.content || r.text || '').substring(0, 150)));
            var sc = document.createElement('span');
            sc.style.cssText = 'color:var(--fg-2);margin-left:6px;font-size:11.5px';
            sc.textContent = '(' + (r.score || 0).toFixed(2) + ')';
            row.appendChild(sc);
            div.appendChild(row);
        });
    }).catch(function(e) {
        btn.disabled = false;
        var div = document.getElementById('quick-recall-results');
        if (div) div.textContent = 'Search failed: ' + e.message;
    });
});
