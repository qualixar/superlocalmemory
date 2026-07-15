// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
// Behavioral Learning tab — outcomes, patterns, cross-project transfers (v2.8)
// NOTE: All dynamic values use textContent or escapeHtml() from core.js before DOM insertion.

var _behavioralData = null;

async function loadBehavioral() {
    try {
        var response = await fetch('/api/behavioral/status');
        var data = await response.json();
        _behavioralData = data;

        if (!data.available) {
            showEmpty('behavioral-patterns-content', 'lightbulb', 'Behavioral learning not available. Upgrade to v2.8.');
            return;
        }

        renderBehavioralStats(data);
        renderBehavioralPatterns(data);
        renderBehavioralTransfers(data);
        renderBehavioralOutcomes(data);

        var badge = document.getElementById('behavioral-profile-badge');
        if (badge) badge.textContent = data.active_profile || 'default';
    } catch (error) {
        console.error('Error loading behavioral:', error);
    }
}

function renderBehavioralStats(data) {
    var stats = data.stats || {};
    animateCounter('bh-success-count', stats.success_count || 0);
    animateCounter('bh-failure-count', stats.failure_count || 0);
    animateCounter('bh-partial-count', stats.partial_count || 0);
    animateCounter('bh-patterns-count', stats.patterns_count || 0);
}

function renderBehavioralPatterns(data) {
    var container = document.getElementById('behavioral-patterns-content');
    if (!container) return;
    var patterns = data.patterns || [];
    container.textContent = '';

    if (patterns.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No patterns learned yet. Report outcomes to start learning.';
        container.appendChild(empty);
        return;
    }

    for (var i = 0; i < patterns.length; i++) {
        var p = patterns[i];
        var successRate = Math.round((p.success_rate || 0) * 100);
        var confPct = Math.round((p.confidence || 0) * 100);
        var barColor = successRate >= 70 ? 'bg-success' : (successRate >= 40 ? 'bg-warning' : 'bg-danger');

        var row = document.createElement('div');
        row.className = 'd-flex align-items-center mb-2';

        // Pattern key label
        var label = document.createElement('div');
        label.style.minWidth = '140px';
        var labelCode = document.createElement('code');
        labelCode.className = 'small';
        labelCode.textContent = p.pattern_key || '';
        label.appendChild(labelCode);

        // Success rate progress bar
        var barWrap = document.createElement('div');
        barWrap.className = 'flex-grow-1 mx-2';
        var progress = document.createElement('div');
        progress.className = 'progress';
        progress.style.height = '20px';
        progress.style.borderRadius = '10px';
        var barEl = document.createElement('div');
        barEl.className = 'progress-bar ' + barColor;
        barEl.style.width = successRate + '%';
        barEl.style.borderRadius = '10px';
        barEl.style.fontSize = '0.7rem';
        barEl.textContent = successRate + '% success';
        progress.appendChild(barEl);
        barWrap.appendChild(progress);

        // Evidence count
        var evidence = document.createElement('small');
        evidence.className = 'text-muted';
        evidence.style.minWidth = '50px';
        evidence.style.textAlign = 'right';
        evidence.textContent = (p.evidence_count || 0) + ' ev.';

        // Confidence badge
        var confBadge = document.createElement('span');
        confBadge.className = 'badge ms-2 ' + (confPct >= 70 ? 'bg-success' : (confPct >= 40 ? 'bg-warning' : 'bg-secondary'));
        confBadge.style.minWidth = '50px';
        confBadge.textContent = confPct + '%';

        row.appendChild(label);
        row.appendChild(barWrap);
        row.appendChild(evidence);
        row.appendChild(confBadge);
        container.appendChild(row);
    }
}

function renderBehavioralTransfers(data) {
    var container = document.getElementById('behavioral-transfers-content');
    if (!container) return;
    var transfers = data.transfers || [];
    container.textContent = '';

    if (transfers.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No cross-project transfers yet. Patterns transfer automatically when confidence is high.';
        container.appendChild(empty);
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Pattern', 'From Project', 'To Project', 'Confidence', 'Date'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    for (var i = 0; i < transfers.length; i++) {
        var t = transfers[i];
        var row = document.createElement('tr');

        var patternCell = document.createElement('td');
        var patternCode = document.createElement('code');
        patternCode.className = 'small';
        patternCode.textContent = t.pattern_key || '';
        patternCell.appendChild(patternCode);
        row.appendChild(patternCell);

        var fromCell = document.createElement('td');
        var fromBadge = document.createElement('span');
        fromBadge.className = 'badge bg-secondary';
        fromBadge.textContent = t.from_project || '';
        fromCell.appendChild(fromBadge);
        row.appendChild(fromCell);

        var toCell = document.createElement('td');
        var toBadge = document.createElement('span');
        toBadge.className = 'badge bg-primary';
        toBadge.textContent = t.to_project || '';
        toCell.appendChild(toBadge);
        row.appendChild(toCell);

        var confCell = document.createElement('td');
        confCell.textContent = Math.round((t.confidence || 0) * 100) + '%';
        row.appendChild(confCell);

        var dateCell = document.createElement('td');
        dateCell.className = 'small text-muted';
        dateCell.textContent = formatDate(t.transferred_at || '');
        row.appendChild(dateCell);

        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

function renderBehavioralOutcomes(data) {
    var container = document.getElementById('behavioral-outcomes-content');
    if (!container) return;
    var outcomes = data.recent_outcomes || [];
    container.textContent = '';

    if (outcomes.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No outcomes recorded yet. Use the form above or the report_outcome MCP tool.';
        container.appendChild(empty);
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Memory IDs', 'Outcome', 'Action Type', 'Date'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var outcomeBadgeColors = {
        success: 'bg-success',
        failure: 'bg-danger',
        partial: 'bg-warning'
    };

    var tbody = document.createElement('tbody');
    for (var i = 0; i < outcomes.length; i++) {
        var o = outcomes[i];
        var row = document.createElement('tr');

        var idsCell = document.createElement('td');
        var memIds = o.memory_ids || [];
        idsCell.textContent = memIds.join(', ');
        row.appendChild(idsCell);

        var outcomeCell = document.createElement('td');
        var outBadge = document.createElement('span');
        outBadge.className = 'badge ' + (outcomeBadgeColors[o.outcome] || 'bg-secondary');
        outBadge.textContent = o.outcome || '';
        outcomeCell.appendChild(outBadge);
        row.appendChild(outcomeCell);

        var actionCell = document.createElement('td');
        actionCell.textContent = o.action_type || '';
        row.appendChild(actionCell);

        var dateCell = document.createElement('td');
        dateCell.className = 'small text-muted';
        dateCell.textContent = formatDate(o.created_at || '');
        row.appendChild(dateCell);

        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

async function reportOutcome() {
    var memIdsInput = document.getElementById('bh-memory-ids');
    var outcomeSelect = document.getElementById('bh-outcome');
    var actionSelect = document.getElementById('bh-action-type');
    var contextInput = document.getElementById('bh-context');

    var rawIds = (memIdsInput.value || '').trim();
    if (!rawIds) {
        showToast('Enter at least one memory ID.');
        return;
    }

    var memoryIds = rawIds.split(',').map(function(id) { return id.trim(); }).filter(function(id) { return id.length > 0; });

    try {
        var response = await fetch('/api/behavioral/report-outcome', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                memory_ids: memoryIds,
                outcome: outcomeSelect.value,
                action_type: actionSelect.value,
                context: contextInput.value.trim() || undefined
            })
        });
        var data = await response.json();
        if (response.ok) {
            showToast('Outcome reported successfully.');
            memIdsInput.value = '';
            contextInput.value = '';
            loadBehavioral(); // Refresh
        } else {
            showToast(data.detail || 'Failed to report outcome.');
        }
    } catch (error) {
        console.error('Error reporting outcome:', error);
        showToast('Error reporting outcome.');
    }
}
