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

var PROFILE_NAME_RE = /^[a-zA-Z0-9_-]+$/;

// POST /api/profiles/create → { ok, status, detail }. Pure API step, no UI.
async function _postCreateProfile(name) {
    try {
        var response = await fetch('/api/profiles/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_name: name })
        });
        var data = {};
        try { data = await response.json(); } catch (e) { /* empty body */ }
        return { ok: response.ok, status: response.status, detail: data.detail || '' };
    } catch (error) {
        console.error('Error creating profile:', error);
        return { ok: false, status: 0, detail: 'Network error creating profile' };
    }
}

// Refresh every profile-aware surface after a successful create.
function _afterProfileMutation() {
    // Small delay so the backend has persisted to both stores before re-read.
    setTimeout(function () {
        loadProfiles();
        if (typeof loadProfilesTable === 'function') loadProfilesTable();
    }, 300);
}

// Entry point used by the sidebar "+", the legacy Profiles-tab form, and the
// data-act-click="create-profile" button. Resolves a name from (a) an explicit
// override, (b) the legacy #new-profile-name input, or (c) an OD-styled modal —
// never the native prompt() (crude for the non-technical dashboard users).
async function createProfile(nameOverride) {
    var name = (typeof nameOverride === 'string' && nameOverride) ? nameOverride : '';
    if (!name) {
        var input = document.getElementById('new-profile-name');
        if (input && input.value.trim()) name = input.value.trim();
    }
    if (!name) {
        openCreateProfileModal();
        return;
    }
    if (!PROFILE_NAME_RE.test(name)) {
        showToast('Invalid name. Use letters, numbers, dashes, underscores.');
        return;
    }
    var res = await _postCreateProfile(name);
    if (res.status === 409) { showToast('Profile "' + name + '" already exists'); return; }
    if (!res.ok) { showToast(res.detail || 'Failed to create profile'); return; }
    showToast('Profile "' + name + '" created');
    var legacyInput = document.getElementById('new-profile-name');
    if (legacyInput) legacyInput.value = '';
    _afterProfileMutation();
}

// OD-styled create-profile dialog. Lazily built, reused across opens. Inline
// validation + 409 handling; ESC / backdrop / Cancel dismiss; Enter submits.
function openCreateProfileModal() {
    var existing = document.getElementById('od-create-profile-overlay');
    if (existing) { existing.remove(); }

    var overlay = document.createElement('div');
    overlay.id = 'od-create-profile-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Create a new profile');
    overlay.style.cssText =
        'position:fixed;inset:0;z-index:11000;display:flex;align-items:center;' +
        'justify-content:center;background:rgba(0,0,0,0.45);backdrop-filter:blur(2px);';

    var card = document.createElement('div');
    card.style.cssText =
        'width:min(420px,92vw);background:var(--card,#1a1f2e);color:var(--fg,#e8ecf3);' +
        'border:1px solid var(--border,rgba(255,255,255,0.1));border-radius:14px;' +
        'box-shadow:0 20px 60px rgba(0,0,0,0.4);padding:22px 22px 18px;';

    var h = document.createElement('h3');
    h.textContent = 'Create a new profile';
    h.style.cssText = 'margin:0 0 4px;font-size:1.05rem;font-weight:600;';
    var sub = document.createElement('p');
    sub.textContent = 'Each profile is a fully isolated memory space.';
    sub.style.cssText = 'margin:0 0 16px;font-size:0.8125rem;color:var(--fg-3,#8b93a7);';

    var input = document.createElement('input');
    input.type = 'text';
    input.maxLength = 32;
    input.placeholder = 'e.g. work, personal, project-x';
    input.setAttribute('aria-label', 'Profile name');
    input.style.cssText =
        'width:100%;box-sizing:border-box;padding:10px 12px;font-size:0.9rem;' +
        'background:var(--page,rgba(255,255,255,0.04));color:var(--fg,#e8ecf3);' +
        'border:1px solid var(--border,rgba(255,255,255,0.14));border-radius:8px;outline:none;';

    var err = document.createElement('div');
    err.style.cssText = 'min-height:18px;margin:6px 2px 0;font-size:0.75rem;color:#ff6b6b;';

    var actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:8px;justify-content:flex-end;margin-top:16px;';

    var cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = 'Cancel';
    cancel.style.cssText =
        'padding:8px 14px;font-size:0.85rem;border-radius:8px;cursor:pointer;' +
        'background:transparent;color:var(--fg-3,#8b93a7);' +
        'border:1px solid var(--border,rgba(255,255,255,0.14));';

    var create = document.createElement('button');
    create.type = 'button';
    create.textContent = 'Create profile';
    create.style.cssText =
        'padding:8px 16px;font-size:0.85rem;border-radius:8px;cursor:pointer;' +
        'background:var(--violet,#7c5cff);color:#fff;border:1px solid transparent;font-weight:600;';

    actions.appendChild(cancel);
    actions.appendChild(create);
    card.appendChild(h);
    card.appendChild(sub);
    card.appendChild(input);
    card.appendChild(err);
    card.appendChild(actions);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    setTimeout(function () { input.focus(); }, 30);

    function close() {
        document.removeEventListener('keydown', onKey);
        overlay.remove();
    }
    function onKey(e) {
        if (e.key === 'Escape') { close(); }
        else if (e.key === 'Enter') { submit(); }
    }
    async function submit() {
        var name = input.value.trim();
        if (!name) { err.textContent = 'Please enter a profile name.'; return; }
        if (!PROFILE_NAME_RE.test(name)) {
            err.textContent = 'Use only letters, numbers, dashes and underscores.';
            return;
        }
        create.disabled = true;
        create.textContent = 'Creating…';
        err.textContent = '';
        var res = await _postCreateProfile(name);
        if (res.status === 409) {
            err.textContent = 'A profile named "' + name + '" already exists.';
            create.disabled = false; create.textContent = 'Create profile';
            return;
        }
        if (!res.ok) {
            err.textContent = res.detail || 'Failed to create profile.';
            create.disabled = false; create.textContent = 'Create profile';
            return;
        }
        close();
        showToast('Profile "' + name + '" created');
        _afterProfileMutation();
    }

    cancel.addEventListener('click', close);
    create.addEventListener('click', submit);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
    document.addEventListener('keydown', onKey);
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
        var acknowledged = response.ok && data.success === true &&
            data.active_profile === profileName && Number.isInteger(data.generation);
        if (acknowledged) {
            // A profile switch is a full context change: reload the whole
            // dashboard so EVERY pane, KPI, graph, and table reflects the new
            // profile. Piecemeal per-pane refresh (the legacy loadX pile) left
            // stale cross-profile data in any pane that wasn't re-fetched, and
            // the OD dashboard's panes aren't driven by those legacy loaders.
            showToast('Switched to profile: ' + profileName + ' — refreshing…');
            setTimeout(function () { window.location.reload(); }, 350);
            return true;
        } else {
            showToast(data.detail || 'Daemon did not acknowledge the requested profile');
            // Restore the selector from daemon runtime truth after any failure
            // or mismatched acknowledgement.
            loadProfiles();
            return false;
        }
    } catch (error) {
        console.error('Error switching profile:', error);
        showToast('Error switching profile');
        loadProfiles();
        return false;
    }
}
