// SuperLocalMemory V3 — Dashboard
// Part of Qualixar | https://superlocalmemory.com

// Auto-refresh dashboard when tab becomes visible (fixes stale data after settings change)
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) loadDashboard();
});

// Also refresh when navigating back to dashboard tab in SPA
window.addEventListener('hashchange', function() {
    if (window.location.hash === '' || window.location.hash === '#dashboard') {
        loadDashboard();
    }
});

// Refresh on focus (covers alt-tab back to browser)
window.addEventListener('focus', function() { loadDashboard(); });

async function loadDashboard() {
    var PANE_ID = 'dashboard-pane';
    try {
        var response = await fetch('/api/v3/dashboard');
        if (!response.ok) {
            showPaneError(PANE_ID, paneErrorMessage(response.status), loadDashboard, true);
            return;
        }
        var data = await response.json();

        clearPaneError(PANE_ID, true);

        document.getElementById('dashboard-mode').textContent = 'Mode ' + data.mode.toUpperCase();
        document.getElementById('dashboard-mode-desc').textContent = data.mode_name + (data.provider !== 'none' ? ' — ' + data.provider : '');
        document.getElementById('dashboard-memory-count').textContent = data.fact_count || data.memory_count || '0';
        document.getElementById('dashboard-provider').textContent = data.provider === 'none' ? 'None' : data.provider;
        document.getElementById('dashboard-model').textContent = data.model || '';
        document.getElementById('dashboard-profile').textContent = data.profile || 'default';
        document.getElementById('dashboard-basedir').textContent = data.base_dir || '~/.superlocalmemory';
        var ver = data.version || '';
        var dashVer = document.getElementById('dashboard-version');
        var settVer = document.getElementById('settings-version');
        if (dashVer) dashVer.textContent = ver;
        if (settVer) settVer.textContent = ver;

        // Update mode badge in navbar
        var badge = document.getElementById('mode-badge');
        if (badge) badge.textContent = 'Mode ' + data.mode.toUpperCase();

        // Highlight active mode button
        document.querySelectorAll('.mode-btn').forEach(function(btn) {
            btn.classList.toggle('active', btn.dataset.mode === data.mode);
        });
    } catch (e) {
        showPaneError(PANE_ID, paneErrorMessage(0), loadDashboard, true);
        console.log('Dashboard load error:', e);
    }
}

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

// Quick store
document.getElementById('quick-store-btn')?.addEventListener('click', function() {
    var input = document.getElementById('quick-store-input');
    var content = input.value.trim();
    if (!content) return;
    fetch('/remember', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({content: content})
    }).then(function(r) {
        if (!r.ok) return r.json().catch(function() { return {}; }).then(function(d) {
            showToast('Store failed: ' + (d.detail || d.error || r.status), 'error');
        });
        return r.json();
    }).then(function(data) {
        if (!data) return;
        input.value = '';
        loadDashboard();
        showToast('Stored!');
    }).catch(function() { showToast('Store failed: network error', 'error'); });
});

// Quick recall
document.getElementById('quick-recall-btn')?.addEventListener('click', function() {
    var query = document.getElementById('quick-recall-input').value.trim();
    if (!query) return;
    fetch('/api/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query, limit: 5})
    }).then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
    }).then(function(data) {
        var div = document.getElementById('quick-recall-results');
        if (!data.results || data.results.length === 0) {
            div.textContent = 'No results found.';
            return;
        }
        // Build results using DOM methods for safety
        div.textContent = '';
        data.results.forEach(function(r, i) {
            var row = document.createElement('div');
            row.className = 'border-bottom py-1';
            var strong = document.createElement('strong');
            strong.textContent = (i + 1) + '. ';
            row.appendChild(strong);
            var text = document.createTextNode((r.content || r.text || '').substring(0, 150) + ' ');
            row.appendChild(text);
            var scoreSpan = document.createElement('span');
            scoreSpan.className = 'text-muted';
            scoreSpan.textContent = '(' + (r.score || 0).toFixed(2) + ')';
            row.appendChild(scoreSpan);
            div.appendChild(row);
        });
    }).catch(function(e) {
        var div = document.getElementById('quick-recall-results');
        if (div) div.textContent = 'Search failed. Is the daemon running?';
    });
});
