// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
// Lifecycle tab — state distribution, compaction, transitions (v2.8)
// NOTE: All dynamic values pass through escapeHtml() or textContent for DOM insertion.

var _lifecycleData = null;

async function loadLifecycle() {
    try {
        // V3.4.11: Load tier stats from the authoritative source (atomic_facts.lifecycle)
        // and merge into the existing lifecycle view for a unified display.
        var tierResponse = await fetch('/api/tiers/stats');
        var tierData = await tierResponse.json();

        var response = await fetch('/api/lifecycle/status');
        var data = await response.json();
        _lifecycleData = data;

        // Override state counts with tier stats (atomic_facts is the truth)
        if (tierData && tierData.total > 0) {
            data.states = data.states || {};
            data.states.active = tierData.active;
            data.states.warm = tierData.warm;
            data.states.cold = tierData.cold;
            data.states.archived = tierData.archived;
            data.total_memories = tierData.total;
            data._pinned = tierData.pinned;
            data.available = true;
        }

        if (!data.available) {
            showEmpty('lifecycle-states-row', 'hourglass-split', 'Lifecycle engine not available. Upgrade to v2.8.');
            return;
        }

        renderLifecycleStates(data);
        renderLifecycleProgress(data);
        renderLifecycleAgeStats(data);
        renderLifecycleTransitions(data);

        // V3.4.11: Show pinned count below the state cards
        var pinnedInfo = document.getElementById('lifecycle-pinned-info');
        if (!pinnedInfo) {
            pinnedInfo = document.createElement('div');
            pinnedInfo.id = 'lifecycle-pinned-info';
            pinnedInfo.className = 'small text-muted mb-3';
            var statesRow = document.getElementById('lifecycle-states-row');
            if (statesRow) statesRow.parentNode.insertBefore(pinnedInfo, statesRow.nextSibling);
        }
        var pinCount = data._pinned || 0;
        pinnedInfo.innerHTML = '<i class="bi bi-pin-angle"></i> ' + pinCount + ' pinned facts (protected from demotion). ' +
            '<span class="text-muted" style="font-size:11px;">Active = full weight | Warm = 0.7x | Cold = 0.3x | Archived = deep recall only</span>';

        var badge = document.getElementById('lifecycle-profile-badge');
        if (badge) badge.textContent = data.active_profile || 'default';
    } catch (error) {
        console.error('Error loading lifecycle:', error);
    }
}

function renderLifecycleStates(data) {
    var states = data.states || {};
    var mapping = {
        active: 'lc-active-count',
        warm: 'lc-warm-count',
        cold: 'lc-cold-count',
        archived: 'lc-archived-count',
        tombstoned: 'lc-tombstoned-count'
    };
    for (var state in mapping) {
        animateCounter(mapping[state], states[state] || 0);
    }
    animateCounter('lc-total-count', data.total_memories || 0);
}

function renderLifecycleProgress(data) {
    var bar = document.getElementById('lifecycle-progress-bar');
    if (!bar) return;
    var states = data.states || {};
    var total = data.total_memories || 1;
    var colors = {
        active: '#198754',
        warm: '#ffc107',
        cold: '#0dcaf0',
        archived: '#6c757d',
        tombstoned: '#dc3545'
    };

    bar.textContent = '';
    var hasSegments = false;

    var stateKeys = ['active', 'warm', 'cold', 'archived', 'tombstoned'];
    for (var i = 0; i < stateKeys.length; i++) {
        var state = stateKeys[i];
        var count = states[state] || 0;
        if (count > 0) {
            hasSegments = true;
            var pct = ((count / total) * 100).toFixed(1);
            var segment = document.createElement('div');
            segment.className = 'progress-bar';
            segment.setAttribute('role', 'progressbar');
            segment.style.width = pct + '%';
            segment.style.backgroundColor = colors[state];
            segment.title = state + ': ' + count + ' (' + pct + '%)';
            segment.textContent = state;
            bar.appendChild(segment);
        }
    }

    if (!hasSegments) {
        var fallback = document.createElement('div');
        fallback.className = 'progress-bar bg-success';
        fallback.style.width = '100%';
        fallback.textContent = 'All Active';
        bar.appendChild(fallback);
    }
}

function renderLifecycleAgeStats(data) {
    var container = document.getElementById('lifecycle-age-content');
    if (!container) return;
    var stats = data.age_stats || {};
    if (Object.keys(stats).length === 0) {
        container.textContent = '';
        var empty = document.createElement('span');
        empty.className = 'text-muted';
        empty.textContent = 'No age data available yet.';
        container.appendChild(empty);
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['State', 'Avg Age', 'Newest', 'Oldest'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var stateOrder = ['active', 'warm', 'cold', 'archived'];
    var badgeColors = { active: 'success', warm: 'warning', cold: 'info', archived: 'secondary' };

    for (var i = 0; i < stateOrder.length; i++) {
        var s = stateOrder[i];
        if (stats[s]) {
            var row = document.createElement('tr');

            var stateCell = document.createElement('td');
            var stateBadge = document.createElement('span');
            stateBadge.className = 'badge bg-' + (badgeColors[s] || 'secondary');
            stateBadge.textContent = s;
            stateCell.appendChild(stateBadge);
            row.appendChild(stateCell);

            var avgCell = document.createElement('td');
            avgCell.textContent = (stats[s].avg_days || 0) + 'd';
            row.appendChild(avgCell);

            var minCell = document.createElement('td');
            minCell.textContent = (stats[s].min_days || 0) + 'd';
            row.appendChild(minCell);

            var maxCell = document.createElement('td');
            maxCell.textContent = (stats[s].max_days || 0) + 'd';
            row.appendChild(maxCell);

            tbody.appendChild(row);
        }
    }
    table.appendChild(tbody);

    container.textContent = '';
    container.appendChild(table);
}

function renderLifecycleTransitions(data) {
    var container = document.getElementById('lifecycle-transitions-content');
    if (!container) return;
    var transitions = data.recent_transitions || [];
    if (transitions.length === 0) {
        container.textContent = '';
        var empty = document.createElement('span');
        empty.className = 'text-muted';
        empty.textContent = 'No transitions yet. Memories start as Active and transition based on usage.';
        container.appendChild(empty);
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Memory', 'State', 'Last Transition'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    for (var i = 0; i < transitions.length; i++) {
        var t = transitions[i];
        var row = document.createElement('tr');

        var memCell = document.createElement('td');
        var preview = (t.content_preview || '').substring(0, 40);
        memCell.textContent = '#' + t.memory_id + ' ' + preview + (preview.length >= 40 ? '...' : '');
        memCell.title = t.content_preview || '';
        row.appendChild(memCell);

        var stateCell = document.createElement('td');
        var badge = document.createElement('span');
        badge.className = 'badge bg-secondary';
        badge.textContent = t.current_state || '';
        stateCell.appendChild(badge);
        row.appendChild(stateCell);

        var transCell = document.createElement('td');
        transCell.className = 'small text-muted';
        transCell.textContent = JSON.stringify(t.last_transition || {});
        row.appendChild(transCell);

        tbody.appendChild(row);
    }
    table.appendChild(tbody);

    container.textContent = '';
    container.appendChild(table);
}

async function compactDryRun() {
    try {
        var response = await fetch('/api/lifecycle/compact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dry_run: true })
        });
        var data = await response.json();
        var resultsDiv = document.getElementById('compaction-results');
        var titleEl = document.getElementById('compaction-results-title');
        var contentEl = document.getElementById('compaction-results-content');
        resultsDiv.classList.remove('d-none');
        titleEl.textContent = 'Compaction Preview (Dry Run)';
        contentEl.textContent = '';

        if (data.recommendations === 0) {
            var ok = document.createElement('span');
            ok.className = 'text-success';
            var icon = document.createElement('i');
            icon.className = 'bi bi-check-circle';
            ok.appendChild(icon);
            ok.appendChild(document.createTextNode(' No compaction needed. All memories are in optimal states.'));
            contentEl.appendChild(ok);
        } else {
            var p = document.createElement('p');
            p.className = 'mb-2';
            p.textContent = data.recommendations + ' memories would be transitioned:';
            contentEl.appendChild(p);

            var table = document.createElement('table');
            table.className = 'table table-sm mb-0';
            var thead = document.createElement('thead');
            var headRow = document.createElement('tr');
            ['Memory ID', 'From', 'To'].forEach(function(h) {
                var th = document.createElement('th');
                th.textContent = h;
                headRow.appendChild(th);
            });
            thead.appendChild(headRow);
            table.appendChild(thead);

            var tbody = document.createElement('tbody');
            var details = data.details || [];
            for (var i = 0; i < details.length; i++) {
                var row = document.createElement('tr');
                var idCell = document.createElement('td');
                idCell.textContent = '#' + details[i].memory_id;
                row.appendChild(idCell);
                var fromCell = document.createElement('td');
                fromCell.textContent = details[i].from || '';
                row.appendChild(fromCell);
                var toCell = document.createElement('td');
                toCell.textContent = details[i].to || '';
                row.appendChild(toCell);
                tbody.appendChild(row);
            }
            table.appendChild(tbody);
            contentEl.appendChild(table);
        }
    } catch (e) {
        console.error('Compaction preview error:', e);
    }
}

async function compactExecute() {
    if (!confirm('This will transition memories to lower lifecycle states. Continue?')) return;
    try {
        var response = await fetch('/api/lifecycle/compact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dry_run: false })
        });
        var data = await response.json();
        var resultsDiv = document.getElementById('compaction-results');
        var contentEl = document.getElementById('compaction-results-content');
        resultsDiv.classList.remove('d-none');
        document.getElementById('compaction-results-title').textContent = 'Compaction Results';

        contentEl.textContent = '';
        var ok = document.createElement('span');
        ok.className = 'text-success';
        var icon = document.createElement('i');
        icon.className = 'bi bi-check-circle';
        ok.appendChild(icon);
        ok.appendChild(document.createTextNode(' ' + (data.transitioned || 0) + ' memories transitioned successfully.'));
        contentEl.appendChild(ok);

        loadLifecycle(); // Refresh
    } catch (e) {
        console.error('Compaction error:', e);
    }
}


// ---- V3.4.11: Tier Management (pinning + evaluation) ----

async function evaluateTiersNow() {
    showToast('Evaluating tiers...');
    try {
        var response = await fetch('/api/tiers/evaluate', { method: 'POST' });
        var data = await response.json();
        if (data.success) {
            var s = data.stats;
            var demoted = (s.demoted_to_warm || 0) + (s.demoted_to_cold || 0) + (s.demoted_to_archive || 0);
            showToast('Tier evaluation: ' + demoted + ' facts demoted, ' + s.pinned_protected + ' pinned protected');
            loadTierStats();
            loadLifecycle();
        }
    } catch (error) {
        showToast('Tier evaluation failed');
    }
}

async function pinFact(factId, reason) {
    try {
        var response = await fetch('/api/tiers/pin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fact_id: factId, reason: reason || '' })
        });
        var data = await response.json();
        if (data.success) {
            showToast('Fact pinned');
            loadTierStats();
        }
    } catch (error) {
        showToast('Failed to pin fact');
    }
}

async function unpinFact(factId) {
    try {
        var response = await fetch('/api/tiers/unpin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fact_id: factId })
        });
        if (response.ok) {
            showToast('Fact unpinned');
            loadTierStats();
        }
    } catch (error) {
        showToast('Failed to unpin fact');
    }
}
