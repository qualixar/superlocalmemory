// SuperLocalMemory V3 — IDE Connections
// Displays detected IDEs and allows connecting them to SLM.

async function loadIDEStatus() {
    var CID = 'ide-list-body';
    try {
        var response = await fetch('/api/v3/ide/status');
        if (!response.ok) {
            showPaneError(CID, paneErrorMessage(response.status), loadIDEStatus, false);
            return;
        }
        var data = await response.json();

        var ides = data.ides || [];

        // AC5: empty array → empty state
        if (ides.length === 0) {
            showEmpty(CID, 'laptop', 'No IDEs detected');
            return;
        }

        var tbody = document.getElementById(CID);
        tbody.textContent = '';
        ides.forEach(function(ide) {
            var tr = document.createElement('tr');

            // IDE name cell
            var tdName = document.createElement('td');
            var strong = document.createElement('strong');
            strong.textContent = ide.name;
            tdName.appendChild(strong);
            tr.appendChild(tdName);

            // Installed status cell
            var tdInstalled = document.createElement('td');
            var badge = document.createElement('span');
            if (ide.installed) {
                badge.className = 'badge bg-success';
                badge.textContent = 'Installed';
            } else {
                badge.className = 'badge bg-secondary';
                badge.textContent = 'Not Found';
            }
            tdInstalled.appendChild(badge);
            tr.appendChild(tdInstalled);

            // Config path cell
            var tdPath = document.createElement('td');
            tdPath.className = 'text-muted small';
            tdPath.textContent = ide.config_path || '';
            tr.appendChild(tdPath);

            // Action cell
            var tdAction = document.createElement('td');
            if (ide.installed) {
                var btn = document.createElement('button');
                btn.className = 'btn btn-sm btn-outline-primary ide-connect-btn';
                btn.dataset.ide = ide.id;
                btn.textContent = 'Connect';
                tdAction.appendChild(btn);
            }
            tr.appendChild(tdAction);

            tbody.appendChild(tr);
        });
    } catch (e) {
        showPaneError('ide-list-body', paneErrorMessage(0), loadIDEStatus, false);
        console.log('IDE status error:', e);
    }
}

// Delegate click handler for individual IDE connect buttons
document.addEventListener('click', function(e) {
    var btn = e.target.closest('.ide-connect-btn');
    if (!btn) return;

    var ideId = btn.dataset.ide;
    fetch('/api/v3/ide/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ide: ideId })
    }).then(function(r) {
        return r.json();
    }).then(function(data) {
        alert(data.success ? 'Connected: ' + ideId : 'Failed to connect');
        loadIDEStatus();
    }).catch(function(e) {
        console.log('IDE connect error:', e);
    });
});

// Connect all detected IDEs
var connectAllBtn = document.getElementById('ide-connect-all-btn');
if (connectAllBtn) {
    connectAllBtn.addEventListener('click', function() {
        fetch('/api/v3/ide/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        }).then(function(r) {
            return r.json();
        }).then(function(data) {
            var results = data.results || {};
            var count = Object.values(results).filter(function(s) {
                return s === 'connected';
            }).length;
            alert('Connected ' + count + ' IDEs');
            loadIDEStatus();
        }).catch(function(e) {
            console.log('IDE connect-all error:', e);
        });
    });
}

document.getElementById('ide-tab')?.addEventListener('shown.bs.tab', loadIDEStatus);
