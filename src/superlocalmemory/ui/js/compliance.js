// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
// Compliance tab — audit trail, retention policies, ABAC (v2.8)
// NOTE: All dynamic values use textContent or escapeHtml() from core.js before DOM insertion.

var _complianceData = null;

async function loadCompliance() {
    var filterEl = document.getElementById('cp-audit-filter');
    var filterValue = filterEl ? filterEl.value : '';

    try {
        var statusResp = await fetch('/api/compliance/status');
        if (!statusResp.ok) throw new Error('HTTP ' + statusResp.status);
        var data = await statusResp.json();
        _complianceData = data;

        if (!data.available) {
            showEmpty('compliance-audit-content', 'shield-lock', 'Compliance engine not available. Upgrade to v2.8.');
            return;
        }

        renderComplianceStats(data);
        renderCompliancePolicies(data);

        // Task C: fetch real audit trail from dedicated endpoint with filters
        var auditUrl = '/api/compliance/audit?limit=100';
        if (filterValue) auditUrl += '&event_type=' + encodeURIComponent(filterValue);
        var auditResp = await fetch(auditUrl);
        var auditData = auditResp.ok ? await auditResp.json() : null;
        renderComplianceAudit(auditData || data);

        var badge = document.getElementById('compliance-profile-badge');
        if (badge) badge.textContent = data.active_profile || 'default';

        // Task C: show chain-integrity indicator
        var chainEl = document.getElementById('cp-chain-integrity');
        if (chainEl && auditData) {
            var ok = auditData.chain_verified !== false;
            chainEl.innerHTML = ok
                ? '<span class="badge bg-success"><i class="bi bi-shield-check"></i> Chain verified</span>'
                : '<span class="badge bg-danger"><i class="bi bi-shield-x"></i> Chain integrity issue</span>';
        }
    } catch (error) {
        console.error('Error loading compliance:', error);
    }
}

// Task D: Fix KPI field reads — /api/compliance/status returns top-level fields,
// NOT a nested .stats object. audit_events_count, retention_policies (array),
// abac_policies_count are all top-level.
function renderComplianceStats(data) {
    animateCounter('cp-audit-count', data.audit_events_count || 0);
    animateCounter('cp-retention-count', (data.retention_policies || []).length);
    animateCounter('cp-abac-count', data.abac_policies_count || 0);
}

function renderCompliancePolicies(data) {
    var container = document.getElementById('compliance-policies-content');
    if (!container) return;
    var policies = data.retention_policies || [];
    container.textContent = '';

    if (policies.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No retention policies configured. Create one above or use the set_retention_policy MCP tool.';
        container.appendChild(empty);
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Policy Name', 'Retention (days)', 'Category', 'Action', 'Created', ''].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    for (var i = 0; i < policies.length; i++) {
        var pol = policies[i];
        var row = document.createElement('tr');

        var nameCell = document.createElement('td');
        var nameIcon = document.createElement('i');
        nameIcon.className = 'bi bi-shield-check text-success me-1';
        nameCell.appendChild(nameIcon);
        nameCell.appendChild(document.createTextNode(pol.name || ''));
        row.appendChild(nameCell);

        var daysCell = document.createElement('td');
        daysCell.textContent = (pol.retention_days || 0) + ' days';
        row.appendChild(daysCell);

        var catCell = document.createElement('td');
        if (pol.category) {
            var catBadge = document.createElement('span');
            catBadge.className = 'badge bg-info';
            catBadge.textContent = pol.category;
            catCell.appendChild(catBadge);
        } else {
            catCell.textContent = 'All';
        }
        row.appendChild(catCell);

        var actionCell = document.createElement('td');
        var actionColors = { archive: 'bg-secondary', tombstone: 'bg-danger', notify: 'bg-warning' };
        var actionBadge = document.createElement('span');
        actionBadge.className = 'badge ' + (actionColors[pol.action] || 'bg-secondary');
        actionBadge.textContent = pol.action || '';
        actionCell.appendChild(actionBadge);
        row.appendChild(actionCell);

        var dateCell = document.createElement('td');
        dateCell.className = 'small text-muted';
        dateCell.textContent = formatDate(pol.created_at || '');
        row.appendChild(dateCell);

        // Task E: Delete button per row — calls DELETE /api/compliance/retention-policy?name=
        var actCell = document.createElement('td');
        var delBtn = document.createElement('button');
        delBtn.className = 'btn btn-sm btn-outline-danger';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', (function(policyName) {
            return async function() {
                var confirmed = await confirmDestructive({
                    title: 'Delete retention policy',
                    target: policyName,
                    consequence: 'This policy will stop applying to all memories.',
                    confirmLabel: 'Delete',
                });
                if (!confirmed) return;
                delBtn.disabled = true;
                try {
                    var r = await fetch(
                        '/api/compliance/retention-policy?name=' + encodeURIComponent(policyName),
                        { method: 'DELETE' }
                    );
                    var d = await r.json().catch(function() { return {}; });
                    if (d.success !== false) {
                        showToast('Policy deleted.');
                        loadCompliance();
                    } else {
                        showToast((d.error || 'Delete failed.'));
                        delBtn.disabled = false;
                    }
                } catch (e) {
                    showToast('Network error deleting policy.');
                    delBtn.disabled = false;
                }
            };
        }(pol.name || '')));
        actCell.appendChild(delBtn);
        row.appendChild(actCell);

        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

// Task C: renderComplianceAudit accepts data from either the status endpoint
// (recent_audit_events) or the dedicated audit endpoint (events). Tries both.
function renderComplianceAudit(data) {
    var container = document.getElementById('compliance-audit-content');
    if (!container) return;
    var events = data.events || data.recent_audit_events || data.audit_events || [];
    container.textContent = '';

    if (events.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'text-center text-muted py-3';
        empty.textContent = 'No audit events recorded yet.';
        container.appendChild(empty);
        return;
    }

    var wrapper = document.createElement('div');
    wrapper.style.maxHeight = '400px';
    wrapper.style.overflowY = 'auto';

    var table = document.createElement('table');
    table.className = 'table table-sm table-hover mb-0';
    var thead = document.createElement('thead');
    thead.style.position = 'sticky';
    thead.style.top = '0';
    thead.style.backgroundColor = 'var(--bs-body-bg)';
    var headRow = document.createElement('tr');
    ['Timestamp', 'Event', 'Actor', 'Action', 'Target', 'Result'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var eventBadgeColors = {
        recall: 'bg-primary',
        remember: 'bg-success',
        delete: 'bg-danger',
        lifecycle_transition: 'bg-warning',
        access_denied: 'bg-danger',
        retention_enforced: 'bg-warning'
    };

    var tbody = document.createElement('tbody');
    for (var i = 0; i < events.length; i++) {
        var ev = events[i];
        var row = document.createElement('tr');

        var tsCell = document.createElement('td');
        tsCell.className = 'small text-muted';
        tsCell.textContent = formatDateFull(ev.timestamp || '');
        row.appendChild(tsCell);

        var typeCell = document.createElement('td');
        var typeBadge = document.createElement('span');
        typeBadge.className = 'badge ' + (eventBadgeColors[ev.event_type] || 'bg-secondary');
        typeBadge.textContent = ev.event_type || '';
        typeCell.appendChild(typeBadge);
        row.appendChild(typeCell);

        var actorCell = document.createElement('td');
        actorCell.textContent = ev.actor || '';
        row.appendChild(actorCell);

        var actionCell = document.createElement('td');
        actionCell.textContent = ev.action || '';
        row.appendChild(actionCell);

        var targetCell = document.createElement('td');
        targetCell.className = 'small';
        targetCell.textContent = ev.target || '';
        row.appendChild(targetCell);

        var resultCell = document.createElement('td');
        if (ev.result === 'success' || ev.result === 'allowed') {
            var okBadge = document.createElement('span');
            okBadge.className = 'badge bg-success';
            okBadge.textContent = ev.result;
            resultCell.appendChild(okBadge);
        } else if (ev.result === 'denied' || ev.result === 'error') {
            var failBadge = document.createElement('span');
            failBadge.className = 'badge bg-danger';
            failBadge.textContent = ev.result;
            resultCell.appendChild(failBadge);
        } else {
            resultCell.textContent = ev.result || '';
        }
        row.appendChild(resultCell);

        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
}

async function createRetentionPolicy() {
    var nameInput = document.getElementById('cp-policy-name');
    var daysInput = document.getElementById('cp-retention-days');
    var categoryInput = document.getElementById('cp-category');
    var actionSelect = document.getElementById('cp-action');

    var policyName = (nameInput.value || '').trim();
    if (!policyName) {
        showToast('Enter a policy name.');
        return;
    }

    var days = parseInt(daysInput.value, 10);
    if (isNaN(days) || days <= 0) {
        showToast('Enter a valid number of days.');
        return;
    }

    try {
        var response = await fetch('/api/compliance/retention-policy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: policyName,
                retention_days: days,
                category: categoryInput.value.trim() || undefined,
                action: actionSelect.value
            })
        });
        var data = await response.json();
        if (response.ok) {
            showToast('Retention policy created.');
            nameInput.value = '';
            daysInput.value = '90';
            categoryInput.value = '';
            loadCompliance(); // Refresh
        } else {
            showToast(data.detail || 'Failed to create policy.');
        }
    } catch (error) {
        console.error('Error creating retention policy:', error);
        showToast('Error creating retention policy.');
    }
}
