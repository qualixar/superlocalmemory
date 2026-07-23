// SuperLocalMemory V3 — Dashboard
// Part of Qualixar | https://superlocalmemory.com
//
// OD integration (v3.8.0):
//   loadDashboard()        → GET /api/v3/dashboard → system card (#dashboard-mode etc.)
//   loadDashboardStats()   → GET /api/stats        → OD KPI strip (#k-mem, #k-nodes, #k-edges)
//   loadDashboardFeed()    → GET /api/memories?limit=6 → activity feed (#feed)
//   loadDashboardSources() → real provenance from /api/stats → capture-sources (#src)
//   loadDashboardTimeline() → real daily fact counts from /api/timeline → #sp-big

var DASHBOARD_CACHE_MS = 30000;
var dashboardLastRefresh = 0;
var dashboardRefreshInFlight = null;
var dashboardCacheGeneration = 0;
var dashboardRefreshQueued = false;

window.slmInvalidateDashboardCache = function() {
    dashboardCacheGeneration += 1;
    dashboardLastRefresh = 0;
    if (dashboardRefreshInFlight) dashboardRefreshQueued = true;
};

// Auto-refresh dashboard when tab becomes visible (fixes stale data after settings change)
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) refreshDashboard();
});

// Focus and tab navigation use a short cache window. This keeps normal
// browser switching from repeatedly querying a large local database.
window.addEventListener('focus', function() { refreshDashboard(); });

// Load when the Dashboard tab is shown via the sidebar. od-shell.js's activateTab()
// dispatches shown.bs.tab on the hidden #dashboard-tab button.
document.getElementById('dashboard-tab')?.addEventListener('shown.bs.tab', function() { refreshDashboard(); });

// refreshDashboard coalesces concurrent startup/navigation events and keeps a
// 30-second snapshot. Mutations pass {force:true} to re-read daemon truth.
window.refreshDashboard = function(options) {
    var force = options && options.force === true;
    var now = Date.now();
    if (dashboardRefreshInFlight) {
        if (force) dashboardRefreshQueued = true;
        return dashboardRefreshInFlight;
    }
    if (!force && dashboardLastRefresh && now - dashboardLastRefresh < DASHBOARD_CACHE_MS) {
        return Promise.resolve();
    }
    var refreshGeneration = dashboardCacheGeneration;
    dashboardRefreshInFlight = Promise.all([
        loadDashboard(),
        loadDashboardStats(),
        loadDashboardTimeline(),
        loadDashboardFeed()
    ]).then(function(results) {
        dashboardLastRefresh = refreshGeneration === dashboardCacheGeneration && results.every(Boolean)
            ? Date.now()
            : 0;
    }).finally(function() {
        dashboardRefreshInFlight = null;
    }).then(function() {
        if (!dashboardRefreshQueued) return;
        dashboardRefreshQueued = false;
        return window.refreshDashboard({ force: true });
    });
    return dashboardRefreshInFlight;
};

// ----------------------------------------------------------------
// loadDashboardStats — GET /api/stats → OD KPI strip
// ----------------------------------------------------------------
async function loadDashboardStats() {
    try {
        var r = await fetch('/api/stats', { credentials: 'same-origin' });
        if (!r.ok) return false;
        var data = await r.json();
        var ov = data.overview || {};

        var memories = ov.total_memories || 0;
        var facts    = ov.total_facts    || 0;
        var nodes    = ov.graph_nodes    || 0;
        var edges    = ov.graph_edges    || 0;
        var clusters = ov.total_clusters || 0;

        // OD KPI strip
        setKpiValue('k-mem',   memories);
        setKpiValue('k-nodes', nodes);
        setKpiValue('k-edges', edges);
        setKpiValue('k-facts', facts);
        setKpiValue('k-clu',   clusters);

        // Legacy hidden compat IDs (core.js loadStats() may race — that's OK,
        // the hidden IDs are only a data bridge, not visible).
        setTextById('stat-memories', memories);
        setTextById('stat-facts',    facts);
        setTextById('stat-nodes',    nodes);
        setTextById('stat-edges',    edges);
        loadDashboardSources(data);
        return true;
    } catch (e) {
        // Non-fatal: KPI strip shows — values
        return false;
    }
}

// ----------------------------------------------------------------
// loadDashboardTimeline — renders real daily fact counts from /api/timeline
// ----------------------------------------------------------------
async function loadDashboardTimeline() {
    var spBig = document.getElementById('sp-big');
    if (!spBig) return true;
    try {
        var response = await fetch('/api/timeline?days=365&group_by=day&include_categories=false', {
            credentials: 'same-origin'
        });
        if (!response.ok) throw new Error('timeline unavailable');
        var stats = await response.json();
        var points = Array.isArray(stats.timeline) ? stats.timeline : [];
        var byDate = {};
        points.forEach(function(point) {
            if (point && point.period) byDate[String(point.period)] = Number(point.count) || 0;
        });
        var counts = [];
        var today = new Date();
        today.setHours(0, 0, 0, 0);
        for (var dayOffset = 364; dayOffset >= 0; dayOffset--) {
            var day = new Date(today);
            day.setDate(today.getDate() - dayOffset);
            counts.push(byDate[day.toISOString().slice(0, 10)] || 0);
        }
        var total = counts.reduce(function(sum, value) { return sum + value; }, 0);
        var avgEl = document.getElementById('k-avg-day');
        if (avgEl) avgEl.textContent = (total / 365).toFixed(1);

        if (!total || typeof window.slmSpark !== 'function') {
            spBig.textContent = 'No dated memory history recorded for this profile.';
            spBig.className = 'muted';
            return true;
        }
        spBig.className = '';
        spBig.innerHTML = window.slmSpark(counts, {
            w: 600,
            h: 140,
            color: 'var(--violet)'
        });
        if (spBig.querySelector('svg')) {
            spBig.querySelector('svg').style.cssText = 'width:100%;height:140px;display:block;';
        }
        return true;
    } catch (e) {
        spBig.textContent = 'Memory history unavailable.';
        spBig.className = 'muted';
        return false;
    }
}

// ----------------------------------------------------------------
// loadDashboardFeed — GET /api/memories?limit=6 → activity feed
// ----------------------------------------------------------------
async function loadDashboardFeed() {
    var feedEl = document.getElementById('feed');
    if (!feedEl) return true;

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
            return false;
        }
        var data = await r.json();
        var items = data.memories || data.results || data || [];
        if (!Array.isArray(items) || items.length === 0) {
            feedEl.innerHTML = '<div class="muted" style="padding:12px;text-align:center;font-size:13px">No memories yet — start capturing!</div>';
            return true;
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
        return true;
    } catch (e) {
        feedEl.innerHTML = '<div class="muted" style="padding:12px;text-align:center;font-size:13px">Activity unavailable</div>';
        return false;
    }
}

// ----------------------------------------------------------------
// loadDashboardSources — profile-scoped provenance returned by /api/stats
// ----------------------------------------------------------------
function loadDashboardSources(stats) {
    var srcEl = document.getElementById('src');
    if (!srcEl) return;
    var sourceStatus = stats && stats.ingestion_sources_status;
    if (sourceStatus && sourceStatus.available === false) {
        srcEl.textContent = 'Provenance metrics are temporarily unavailable.';
        srcEl.className = 'muted';
        return;
    }
    var sources = stats && Array.isArray(stats.ingestion_sources)
        ? stats.ingestion_sources
        : [];
    if (!sources.length) {
        srcEl.textContent = 'No provenance recorded for this profile.';
        srcEl.className = 'muted';
        return;
    }

    var LABELS = {
        'store': 'Historical / engine store',
        'http': 'Dashboard, hooks, and API',
        'http-observe': 'Auto-capture observations',
        'mcp-offline-worker': 'Queued MCP ingestion',
        'cli-sync': 'CLI sync',
        'unknown': 'Unclassified'
    };
    var COLORS = ['var(--violet)', 'var(--cyan)', 'var(--warn,#f59e0b)', 'var(--fg-3)'];
    var total = sources.reduce(function(sum, source) {
        return sum + (Number(source.count) || 0);
    }, 0);

    srcEl.className = '';
    srcEl.textContent = '';
    sources.forEach(function(source, index) {
        var count = Number(source.count) || 0;
        var percent = total ? Math.round((count / total) * 100) : 0;
        var row = document.createElement('div');
        row.style.marginBottom = '14px';

        var heading = document.createElement('div');
        heading.style.cssText = 'display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px';
        var label = document.createElement('span');
        label.textContent = LABELS[source.source_type] || source.source_type || 'Unclassified';
        var value = document.createElement('b');
        value.className = 'num';
        value.textContent = count.toLocaleString() + ' · ' + percent + '%';
        heading.appendChild(label);
        heading.appendChild(value);

        var meter = document.createElement('div');
        meter.className = 'meter';
        var fill = document.createElement('i');
        fill.style.width = percent + '%';
        fill.style.background = COLORS[index % COLORS.length];
        meter.appendChild(fill);

        row.appendChild(heading);
        row.appendChild(meter);
        srcEl.appendChild(row);
    });
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
            return false;
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
        return true;
    } catch (e) {
        showPaneError(PANE_ID, paneErrorMessage(0), loadDashboard, true);
        console.log('Dashboard load error:', e);
        return false;
    }
}

// Fire all OD + legacy loaders on first DOMContentLoaded (dashboard is the default pane)
document.addEventListener('DOMContentLoaded', function() {
    // od-shell.js calls activateTab('dashboard-pane') which dispatches shown.bs.tab,
    // but the listener may not yet be attached at that point. Call directly as well.
    setTimeout(function() {
        refreshDashboard({ force: true });
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
        refreshDashboard({ force: true });
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
        slmInvalidatesCache: false,
        slmRequiresWriteAuth: false,
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
