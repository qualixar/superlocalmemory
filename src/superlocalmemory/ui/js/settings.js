// SuperLocalMemory V3.4.10 "Fortress" - Settings, Backup & Cloud Sync
// Depends on: core.js, profiles.js (loadProfilesTable)

async function loadSettings() {
    loadProfilesTable();
    loadBackupStatus();
    loadBackupList();
    loadLearningDataStats();
    loadCloudDestinations();
}

async function loadLearningDataStats() {
    try {
        var response = await fetch('/api/feedback/stats');
        var data = await response.json();
        var container = document.getElementById('learning-data-stats');
        if (!container) return;

        container.textContent = '';
        var row = document.createElement('div');
        row.className = 'row g-2';

        var stats = [
            { value: String(data.total_signals || 0), label: 'Feedback Signals' },
            { value: data.ranking_phase || 'baseline', label: 'Ranking Phase' },
            { value: Math.round(data.progress || 0) + '%', label: 'ML Progress' },
        ];

        stats.forEach(function(s) {
            var col = document.createElement('div');
            col.className = 'col-4';
            var stat = document.createElement('div');
            stat.className = 'backup-stat';
            var val = document.createElement('div');
            val.className = 'value';
            val.textContent = s.value;
            var lbl = document.createElement('div');
            lbl.className = 'label';
            lbl.textContent = s.label;
            stat.appendChild(val);
            stat.appendChild(lbl);
            col.appendChild(stat);
            row.appendChild(col);
        });
        container.appendChild(row);
    } catch (error) {
        // Silent — learning stats are optional
    }
}

async function backupLearningDb() {
    try {
        var response = await fetch('/api/learning/backup', { method: 'POST' });
        var data = await response.json();
        if (data.success) {
            showToast('Learning DB backed up: ' + (data.filename || 'learning.db.bak'));
        } else {
            showToast('Backup created at ~/.superlocalmemory/learning.db.bak');
        }
    } catch (error) {
        // Fallback: just tell user the manual path
        showToast('Manual backup: cp ~/.superlocalmemory/learning.db ~/.superlocalmemory/learning.db.bak');
    }
}

async function loadBackupStatus() {
    try {
        var response = await fetch('/api/backup/status');
        var data = await response.json();
        renderBackupStatus(data);
        document.getElementById('backup-interval').value = data.interval_hours <= 24 ? '24' : '168';
        document.getElementById('backup-max').value = data.max_backups || 10;
        document.getElementById('backup-enabled').checked = data.enabled !== false;
    } catch (error) {
        var container = document.getElementById('backup-status');
        var alert = document.createElement('div');
        alert.className = 'alert alert-warning mb-0';
        alert.textContent = 'Auto-backup not available. Update to v2.4.0+.';
        container.textContent = '';
        container.appendChild(alert);
    }
}

function renderBackupStatus(data) {
    var container = document.getElementById('backup-status');
    container.textContent = '';

    var lastBackup = data.last_backup ? formatDateFull(data.last_backup) : 'Never';
    var nextBackup = data.next_backup || 'N/A';
    if (nextBackup === 'overdue') nextBackup = 'Overdue';
    else if (nextBackup !== 'N/A' && nextBackup !== 'unknown') nextBackup = formatDateFull(nextBackup);

    var statusColor = data.enabled ? 'text-success' : 'text-secondary';
    var statusText = data.enabled ? 'Active' : 'Disabled';

    var row = document.createElement('div');
    row.className = 'row g-2 mb-2';

    var stats = [
        { value: statusText, label: 'Status', cls: statusColor },
        { value: String(data.backup_count || 0), label: 'Backups', cls: '' },
        { value: (data.total_size_mb || 0) + ' MB', label: 'Storage', cls: '' }
    ];

    stats.forEach(function(s) {
        var col = document.createElement('div');
        col.className = 'col-4';
        var stat = document.createElement('div');
        stat.className = 'backup-stat';
        var val = document.createElement('div');
        val.className = 'value ' + s.cls;
        val.textContent = s.value;
        var lbl = document.createElement('div');
        lbl.className = 'label';
        lbl.textContent = s.label;
        stat.appendChild(val);
        stat.appendChild(lbl);
        col.appendChild(stat);
        row.appendChild(col);
    });
    container.appendChild(row);

    var details = [
        { label: 'Last backup:', value: lastBackup },
        { label: 'Next backup:', value: nextBackup },
        { label: 'Interval:', value: data.interval_display || '-' }
    ];
    details.forEach(function(d) {
        var div = document.createElement('div');
        div.className = 'small text-muted';
        var strong = document.createElement('strong');
        strong.textContent = d.label + ' ';
        div.appendChild(strong);
        div.appendChild(document.createTextNode(d.value));
        container.appendChild(div);
    });
}

async function saveBackupConfig() {
    try {
        var response = await fetch('/api/backup/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                interval_hours: parseInt(document.getElementById('backup-interval').value),
                max_backups: parseInt(document.getElementById('backup-max').value),
                enabled: document.getElementById('backup-enabled').checked
            })
        });
        var data = await response.json();
        // API wraps status inside data.status on configure response
        var status = data.status || data;
        renderBackupStatus(status);
        showToast('Backup settings saved');
    } catch (error) {
        console.error('Error saving backup config:', error);
        showToast('Failed to save backup settings');
    }
}

async function createBackupNow() {
    showToast('Creating backup...');
    try {
        var response = await fetch('/api/backup/create', { method: 'POST' });
        var data = await response.json();
        if (data.success) {
            showToast('Backup created: ' + data.filename);
            loadBackupStatus();
            loadBackupList();
        } else {
            showToast('Backup failed');
        }
    } catch (error) {
        console.error('Error creating backup:', error);
        showToast('Backup failed');
    }
}

async function loadBackupList() {
    try {
        var response = await fetch('/api/backup/list');
        var data = await response.json();
        renderBackupList(data.backups || []);
    } catch (error) {
        var container = document.getElementById('backup-list');
        container.textContent = 'Backup list unavailable';
    }
}

function renderBackupList(backups) {
    var container = document.getElementById('backup-list');
    if (!backups || backups.length === 0) {
        showEmpty('backup-list', 'archive', 'No backups yet. Create your first backup above.');
        return;
    }

    var table = document.createElement('table');
    table.className = 'table table-sm';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['Filename', 'Size', 'Age', 'Created'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    backups.forEach(function(b) {
        var row = document.createElement('tr');
        var age = b.age_hours < 48 ? Math.round(b.age_hours) + 'h ago' : Math.round(b.age_hours / 24) + 'd ago';
        var cells = [b.filename, b.size_mb + ' MB', age, formatDateFull(b.created)];
        cells.forEach(function(text) {
            var td = document.createElement('td');
            td.textContent = text;
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    container.textContent = '';
    container.appendChild(table);
}


// ---- Cloud Backup (v3.4.10) ----

async function loadCloudDestinations() {
    var container = document.getElementById('cloud-destinations');
    if (!container) return;

    try {
        var response = await fetch('/api/backup/destinations');
        var data = await response.json();
        var destinations = data.destinations || [];
        renderCloudDestinations(destinations, container);
        updateAccountWidget(destinations);
    } catch (error) {
        container.innerHTML = '<div class="text-muted small">Cloud backup not available in this version.</div>';
    }
}

function updateAccountWidget(destinations) {
    // Update BOTH widgets: old navbar (if present) and new Neural Glass sidebar
    _updateNavbarWidget(destinations);
    _updateSidebarWidget(destinations);
}

function _updateSidebarWidget(destinations) {
    var avatar = document.getElementById('ng-account-avatar');
    var name = document.getElementById('ng-account-name');
    var status = document.getElementById('ng-account-status');
    var dot = document.getElementById('ng-account-dot');
    var actions = document.getElementById('ng-account-actions');

    if (!avatar || !name) return;

    if (!destinations || destinations.length === 0) {
        avatar.innerHTML = '<i class="bi bi-cloud-slash" style="font-size:13px;opacity:0.4;"></i>';
        name.textContent = 'Not connected';
        status.textContent = 'No cloud backup';
        dot.style.background = '#444';
        if (actions) actions.style.display = 'block';
        return;
    }

    var primary = destinations[0];
    var config = {};
    try { config = JSON.parse(primary.config || '{}'); } catch(e) {}

    if (primary.destination_type === 'google_drive') {
        var email = config.email || 'Google Drive';
        // Keep the dashboard fully local. CSP deliberately forbids remote
        // avatar requests, and a provider icon conveys the same state.
        avatar.innerHTML = '<i class="bi bi-google" style="font-size:14px;color:#4285f4;"></i>';
        name.textContent = email.split('@')[0];
        name.title = email;
    } else if (primary.destination_type === 'github') {
        var username = config.username || 'GitHub';
        avatar.innerHTML = '<i class="bi bi-github" style="font-size:14px;"></i>';
        name.textContent = username;
    }

    // Sync status
    var hasSuccess = destinations.some(function(d) { return d.last_sync_status === 'success'; });
    var hasFailed = destinations.some(function(d) { return d.last_sync_status === 'failed'; });
    var allNever = destinations.every(function(d) { return d.last_sync_status === 'never'; });

    if (hasFailed) {
        dot.style.background = '#ff4757';
        status.textContent = 'Sync failed';
        status.style.color = '#ff4757';
    } else if (hasSuccess) {
        dot.style.background = '#00D4AA';
        status.textContent = destinations.length + ' destination' + (destinations.length > 1 ? 's' : '') + ' synced';
        status.style.color = '#00D4AA';
    } else if (allNever) {
        dot.style.background = '#f39c12';
        status.textContent = 'Connected \u2014 not yet synced';
        status.style.color = '#f39c12';
    }

    // Show actions row with disconnect options if connected
    if (actions) actions.style.display = 'block';
}

function _updateNavbarWidget(destinations) {
    var avatar = document.getElementById('account-avatar');
    var label = document.getElementById('account-label');
    var syncDot = document.getElementById('account-sync-dot');
    var accountList = document.getElementById('account-list');
    var syncLabel = document.getElementById('account-sync-label');

    if (!avatar || !label) return;

    if (!destinations || destinations.length === 0) {
        avatar.innerHTML = '<i class="bi bi-cloud-slash" style="font-size:12px;opacity:0.6;"></i>';
        label.textContent = 'Not connected';
        syncDot.style.background = '#666';
        syncDot.title = 'No cloud backup';
        if (accountList) accountList.innerHTML = '<span style="color:#666;">No accounts connected</span>';
        if (syncLabel) syncLabel.textContent = 'Last sync: Never';
        return;
    }

    // Find primary destination (first one)
    var primary = destinations[0];
    var config = {};
    try { config = JSON.parse(primary.config || '{}'); } catch(e) {}

    // Set avatar
    var displayName = '';
    if (primary.destination_type === 'google_drive') {
        displayName = config.email || 'Google Drive';
        avatar.innerHTML = '<i class="bi bi-google" style="font-size:12px;color:#4285f4;"></i>';
    } else if (primary.destination_type === 'github') {
        displayName = config.username || 'GitHub';
        avatar.innerHTML = '<i class="bi bi-github" style="font-size:12px;"></i>';
    }

    // Set label (show name or email, truncated)
    var shortName = displayName.split('@')[0];
    if (shortName.length > 15) shortName = shortName.substring(0, 13) + '..';
    label.textContent = shortName;

    // Sync status dot
    var hasSuccess = destinations.some(function(d) { return d.last_sync_status === 'success'; });
    var hasFailed = destinations.some(function(d) { return d.last_sync_status === 'failed'; });
    var allNever = destinations.every(function(d) { return d.last_sync_status === 'never'; });

    if (hasFailed) {
        syncDot.style.background = '#ff4757';
        syncDot.title = 'Sync failed — click to fix';
    } else if (hasSuccess) {
        syncDot.style.background = '#00D4AA';
        syncDot.title = 'Cloud backup active';
    } else if (allNever) {
        syncDot.style.background = '#f39c12';
        syncDot.title = 'Connected but not yet synced';
    }

    // Build the account list in dropdown
    if (accountList) {
        var html = '';
        destinations.forEach(function(dest) {
            var cfg = {};
            try { cfg = JSON.parse(dest.config || '{}'); } catch(e) {}
            var icon = dest.destination_type === 'google_drive'
                ? '<i class="bi bi-google" style="color:#4285f4;font-size:14px;"></i>'
                : '<i class="bi bi-github" style="font-size:14px;"></i>';
            var name = dest.display_name || dest.destination_type;
            var statusCls = dest.last_sync_status === 'success' ? 'synced'
                : dest.last_sync_status === 'failed' ? 'failed' : 'never';
            var statusText = dest.last_sync_status === 'success' ? 'Synced'
                : dest.last_sync_status === 'failed' ? 'Failed' : 'Pending';

            html += '<div class="account-dest-item">' +
                icon +
                '<span style="flex:1;color:#e0e0e0;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + name + '</span>' +
                '<span class="account-dest-badge ' + statusCls + '">' + statusText + '</span>' +
                '<button class="btn btn-sm" data-act-click="disconnect-destination" data-dest-id="' + dest.id + '" style="padding:0;border:0;color:#555;font-size:12px;" title="Disconnect"><i class="bi bi-x"></i></button>' +
                '</div>';
        });
        accountList.innerHTML = html;
    }

    // Last sync time
    if (syncLabel) {
        var lastSynced = destinations.filter(function(d) { return d.last_sync_at; })
            .sort(function(a, b) { return (b.last_sync_at || '').localeCompare(a.last_sync_at || ''); });
        if (lastSynced.length > 0 && lastSynced[0].last_sync_at) {
            syncLabel.textContent = 'Last sync: ' + formatDateFull(lastSynced[0].last_sync_at);
        } else {
            syncLabel.textContent = 'Last sync: Never';
        }
    }
}

function renderCloudDestinations(destinations, container) {
    container.textContent = '';

    if (destinations.length === 0) {
        container.innerHTML = '<div class="text-muted small mb-2">No cloud destinations configured. Connect Google Drive or GitHub below.</div>';
    } else {
        destinations.forEach(function(dest) {
            var card = document.createElement('div');
            card.className = 'card mb-2';
            card.style.background = 'rgba(255,255,255,0.03)';
            card.style.border = '1px solid rgba(255,255,255,0.08)';

            var icon = dest.destination_type === 'google_drive' ? 'cloud' : 'github';
            var statusBadge = dest.last_sync_status === 'success'
                ? '<span class="badge bg-success">Synced</span>'
                : dest.last_sync_status === 'failed'
                ? '<span class="badge bg-danger">Failed</span>'
                : '<span class="badge bg-secondary">Never synced</span>';

            var lastSync = dest.last_sync_at ? formatDateFull(dest.last_sync_at) : 'Never';

            card.innerHTML = '<div class="card-body p-2">' +
                '<div class="d-flex justify-content-between align-items-center">' +
                '<div><i class="bi bi-' + icon + '"></i> <strong>' + dest.display_name + '</strong> ' + statusBadge + '</div>' +
                '<button class="btn btn-outline-danger btn-sm" data-act-click="disconnect-destination" data-dest-id="' + dest.id + '"><i class="bi bi-x-circle"></i></button>' +
                '</div>' +
                '<div class="small text-muted mt-1">Last sync: ' + lastSync + '</div>' +
                '</div>';
            container.appendChild(card);
        });
    }
}

async function connectGitHub() {
    showToast('Opening GitHub login...');
    // Open GitHub OAuth in a popup window
    var w = 600, h = 700;
    var left = (screen.width - w) / 2, top = (screen.height - h) / 2;
    var popup = window.open(
        '/api/backup/oauth/github/start',
        'github_oauth',
        'width=' + w + ',height=' + h + ',left=' + left + ',top=' + top
    );

    // Poll for completion
    var pollTimer = setInterval(function() {
        if (popup && popup.closed) {
            clearInterval(pollTimer);
            loadCloudDestinations();
            loadBackupStatus();
        }
    }, 1000);
}

async function connectGoogleDrive() {
    showToast('Opening Google login...');
    // Open Google OAuth in a popup window
    var w = 600, h = 700;
    var left = (screen.width - w) / 2, top = (screen.height - h) / 2;
    var popup = window.open(
        '/api/backup/oauth/google/start',
        'google_oauth',
        'width=' + w + ',height=' + h + ',left=' + left + ',top=' + top
    );

    // Poll for completion
    var pollTimer = setInterval(function() {
        if (popup && popup.closed) {
            clearInterval(pollTimer);
            loadCloudDestinations();
            loadBackupStatus();
        }
    }, 1000);
}

async function disconnectDestination(destId) {
    if (!confirm('Disconnect this backup destination? Your backups on the cloud will remain.')) return;

    try {
        var response = await fetch('/api/backup/disconnect/' + destId, { method: 'DELETE' });
        if (response.ok) {
            showToast('Destination disconnected');
            loadCloudDestinations();
            loadBackupStatus();
        } else {
            showToast('Failed to disconnect');
        }
    } catch (error) {
        showToast('Failed to disconnect');
    }
}

async function syncCloudNow() {
    showToast('Starting cloud sync...');
    try {
        var response = await fetch('/api/backup/sync', { method: 'POST' });
        var data = await response.json();
        if (data.success) {
            showToast('Backup created. Upload running in background.');
            // Poll for completion every 10 seconds for 5 minutes
            var polls = 0;
            var pollInterval = setInterval(async function() {
                polls++;
                if (polls > 30) { clearInterval(pollInterval); return; }
                try {
                    var statusResp = await fetch('/api/backup/destinations');
                    var statusData = await statusResp.json();
                    var dests = statusData.destinations || [];
                    var allDone = dests.length > 0 && dests.every(function(d) {
                        return d.last_sync_status === 'success' || d.last_sync_status === 'failed';
                    });
                    if (allDone) {
                        clearInterval(pollInterval);
                        var succeeded = dests.filter(function(d) { return d.last_sync_status === 'success'; }).length;
                        showToast('Cloud sync done: ' + succeeded + '/' + dests.length + ' destinations');
                        loadCloudDestinations();
                        loadBackupStatus();
                    }
                } catch(e) { /* ignore polling errors */ }
            }, 10000);
        } else {
            showToast('Cloud sync failed');
        }
    } catch (error) {
        showToast('Cloud sync failed');
    }
}

async function exportBackup() {
    showToast('Preparing backup export...');
    try {
        window.location.href = '/api/backup/export';
    } catch (error) {
        showToast('Export failed');
    }
}
