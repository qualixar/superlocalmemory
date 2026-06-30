// SuperLocalMemory V2 - Profile Management
// Depends on: core.js

async function loadProfiles() {
    try {
        var response = await fetch('/api/profiles');
        var data = await response.json();
        var select = document.getElementById('profile-select');
        select.textContent = '';
        var profiles = data.profiles || [];
        var active = data.active_profile || 'default';

        if (profiles.length === 0) {
            // Fresh install fallback — show default profile
            var opt = document.createElement('option');
            opt.value = 'default';
            opt.textContent = 'default (0)';
            opt.selected = true;
            select.appendChild(opt);
        } else {
            profiles.forEach(function(p) {
                var opt = document.createElement('option');
                opt.value = p.name;
                opt.textContent = p.name + ' (' + (p.memory_count || 0) + ')';
                if (p.name === active) opt.selected = true;
                select.appendChild(opt);
            });
        }
    } catch (error) {
        console.error('Error loading profiles:', error);
        // On error, ensure dropdown shows at least 'default'
        var select = document.getElementById('profile-select');
        if (select && select.options.length === 0) {
            select.textContent = '';
            var opt = document.createElement('option');
            opt.value = 'default';
            opt.textContent = 'default';
            opt.selected = true;
            select.appendChild(opt);
        }
    }
}

async function createProfile(nameOverride) {
    var name = nameOverride || document.getElementById('new-profile-name').value.trim();
    if (!name) {
        name = prompt('Enter new profile name:');
        if (!name || !name.trim()) return;
        name = name.trim();
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
        showToast('Invalid name. Use letters, numbers, dashes, underscores.');
        return;
    }

    try {
        var response = await fetch('/api/profiles/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_name: name })
        });
        var data = await response.json();
        if (response.status === 409) {
            showToast('Profile "' + name + '" already exists');
            return;
        }
        if (!response.ok) {
            showToast(data.detail || 'Failed to create profile');
            return;
        }
        showToast('Profile "' + name + '" created');
        var input = document.getElementById('new-profile-name');
        if (input) input.value = '';
        // Force reload with small delay to ensure backend has persisted
        setTimeout(function() {
            loadProfiles();
            if (typeof loadProfilesTable === 'function') loadProfilesTable();
        }, 300);
    } catch (error) {
        console.error('Error creating profile:', error);
        showToast('Error creating profile');
    }
}

async function deleteProfile(name) {
    if (name === 'default') {
        showToast('Cannot delete the default profile');
        return;
    }
    if (!confirm('Delete profile "' + name + '"?\nIts memories will be moved to the default profile.')) {
        return;
    }
    try {
        var response = await fetch('/api/profiles/' + encodeURIComponent(name), {
            method: 'DELETE'
        });
        var data = await response.json();
        if (!response.ok) {
            showToast(data.detail || 'Failed to delete profile');
            return;
        }
        showToast(data.message || 'Profile deleted');
        loadProfiles();
        loadProfilesTable();
        loadStats();
    } catch (error) {
        console.error('Error deleting profile:', error);
        showToast('Error deleting profile');
    }
}

async function loadProfilesTable() {
    var container = document.getElementById('profiles-table');
    if (!container) return;
    try {
        var response = await fetch('/api/profiles');
        var data = await response.json();
        var profiles = data.profiles || [];
        var active = data.active_profile || 'default';

        if (profiles.length === 0) {
            showEmpty('profiles-table', 'people', 'No profiles found.');
            return;
        }

        var table = document.createElement('table');
        table.className = 'table table-sm mb-0';
        var thead = document.createElement('thead');
        var headRow = document.createElement('tr');
        ['Name', 'Memories', 'Status', 'Actions'].forEach(function(h) {
            var th = document.createElement('th');
            th.textContent = h;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        var tbody = document.createElement('tbody');
        profiles.forEach(function(p) {
            var row = document.createElement('tr');

            var nameCell = document.createElement('td');
            var nameIcon = document.createElement('i');
            nameIcon.className = 'bi bi-person me-1';
            nameCell.appendChild(nameIcon);
            nameCell.appendChild(document.createTextNode(p.name));
            row.appendChild(nameCell);

            var countCell = document.createElement('td');
            countCell.textContent = (p.memory_count || 0) + ' memories';
            row.appendChild(countCell);

            var statusCell = document.createElement('td');
            if (p.name === active) {
                var badge = document.createElement('span');
                badge.className = 'badge bg-success';
                badge.textContent = 'Active';
                statusCell.appendChild(badge);
            } else {
                var switchBtn = document.createElement('button');
                switchBtn.className = 'btn btn-sm btn-outline-primary';
                switchBtn.textContent = 'Switch';
                switchBtn.addEventListener('click', (function(n) {
                    return function() { switchProfile(n); };
                })(p.name));
                statusCell.appendChild(switchBtn);
            }
            row.appendChild(statusCell);

            var actionsCell = document.createElement('td');
            if (p.name !== 'default') {
                var delBtn = document.createElement('button');
                delBtn.className = 'btn btn-sm btn-outline-danger btn-delete-profile';
                delBtn.title = 'Delete profile';
                var delIcon = document.createElement('i');
                delIcon.className = 'bi bi-trash';
                delBtn.appendChild(delIcon);
                delBtn.addEventListener('click', (function(n) {
                    return function() { deleteProfile(n); };
                })(p.name));
                actionsCell.appendChild(delBtn);
            } else {
                var protectedBadge = document.createElement('span');
                protectedBadge.className = 'badge bg-secondary';
                protectedBadge.textContent = 'Protected';
                actionsCell.appendChild(protectedBadge);
            }
            row.appendChild(actionsCell);

            tbody.appendChild(row);
        });
        table.appendChild(tbody);

        container.textContent = '';
        container.appendChild(table);
    } catch (error) {
        console.error('Error loading profiles table:', error);
        showEmpty('profiles-table', 'exclamation-triangle', 'Failed to load profiles');
    }
}

async function switchProfile(profileName) {
    try {
        var response = await fetch('/api/profiles/' + encodeURIComponent(profileName) + '/switch', {
            method: 'POST'
        });
        var data = await response.json();
        if (data.success || data.active_profile) {
            showToast('Switched to profile: ' + profileName);
            loadProfiles();
            loadStats();
            if (typeof loadGraph === 'function') loadGraph();
            loadProfilesTable();
            // v2.7.4: Reload ALL tabs for new profile
            if (typeof loadLearning === 'function') loadLearning();
            if (typeof refreshFeedbackStats === 'function') refreshFeedbackStats();
            if (typeof loadLearningDataStats === 'function') loadLearningDataStats();
            if (typeof loadAgents === 'function') loadAgents();
            if (typeof loadMemories === 'function') loadMemories();
            if (typeof loadTimeline === 'function') loadTimeline();
            if (typeof loadEvents === 'function') loadEvents();
            // v2.8 tabs
            if (typeof loadLifecycle === 'function') loadLifecycle();
            if (typeof loadBehavioral === 'function') loadBehavioral();
            if (typeof loadCompliance === 'function') loadCompliance();
            var activeTab = document.querySelector('#mainTabs .nav-link.active');
            if (activeTab) activeTab.click();
        } else {
            showToast('Failed to switch profile');
        }
    } catch (error) {
        console.error('Error switching profile:', error);
        showToast('Error switching profile');
    }
}
