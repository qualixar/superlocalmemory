// SuperLocalMemory V3 — Math Health
// Displays status of the mathematical scoring, consistency, and lifecycle layers.

async function loadMathHealth() {
    var CID = 'math-health-cards';
    try {
        var response = await fetch('/api/v3/math/health');
        if (!response.ok) {
            showPaneError(CID, paneErrorMessage(response.status), loadMathHealth, false);
            return;
        }
        var data = await response.json();

        var layers = data.health || {};

        // AC5: empty object → empty state
        if (!layers || Object.keys(layers).length === 0) {
            showEmpty(CID, 'calculator', 'No math health data available');
            return;
        }

        var container = document.getElementById(CID);
        container.textContent = '';

        var colors = { fisher: 'primary', sheaf: 'success', langevin: 'info' };
        var icons = { fisher: 'bi-graph-up', sheaf: 'bi-diagram-3', langevin: 'bi-activity' };

        Object.keys(layers).forEach(function(key) {
            var layer = layers[key];
            var col = document.createElement('div');
            col.className = 'col-md-4';

            var card = document.createElement('div');
            card.className = 'card h-100';

            // Card header
            var header = document.createElement('div');
            header.className = 'card-header bg-' + (colors[key] || 'secondary') + ' text-white';
            var h6 = document.createElement('h6');
            h6.className = 'mb-0';
            var icon = document.createElement('i');
            icon.className = 'bi ' + (icons[key] || 'bi-gear');
            h6.appendChild(icon);
            h6.appendChild(document.createTextNode(' ' + key.charAt(0).toUpperCase() + key.slice(1)));
            header.appendChild(h6);
            card.appendChild(header);

            // Card body
            var body = document.createElement('div');
            body.className = 'card-body';

            var desc = document.createElement('p');
            desc.className = 'text-muted';
            desc.textContent = layer.description || '';
            body.appendChild(desc);

            var ul = document.createElement('ul');
            ul.className = 'list-unstyled mb-0';

            // Status item
            var liStatus = document.createElement('li');
            liStatus.appendChild(document.createTextNode('Status: '));
            var statusBadge = document.createElement('span');
            var _s = (layer.status || 'active').toLowerCase();
            var _badgeClass = _s === 'error' || _s === 'critical' ? 'bg-danger' :
                              _s === 'warning' || _s === 'degraded' ? 'bg-warning text-dark' :
                              'bg-success';
            statusBadge.className = 'badge ' + _badgeClass;
            statusBadge.textContent = layer.status || 'active';
            liStatus.appendChild(statusBadge);
            ul.appendChild(liStatus);

            // Mode item (if present)
            if (layer.mode) {
                var liMode = document.createElement('li');
                liMode.appendChild(document.createTextNode('Mode: '));
                var modeStrong = document.createElement('strong');
                modeStrong.textContent = layer.mode;
                liMode.appendChild(modeStrong);
                ul.appendChild(liMode);
            }

            // Threshold item (if present)
            if (layer.threshold) {
                var liThresh = document.createElement('li');
                liThresh.appendChild(document.createTextNode('Threshold: '));
                var threshStrong = document.createElement('strong');
                threshStrong.textContent = layer.threshold;
                liThresh.appendChild(threshStrong);
                ul.appendChild(liThresh);
            }

            // Temperature item (if present)
            if (layer.temperature) {
                var liTemp = document.createElement('li');
                liTemp.appendChild(document.createTextNode('Temperature: '));
                var tempStrong = document.createElement('strong');
                tempStrong.textContent = layer.temperature;
                liTemp.appendChild(tempStrong);
                ul.appendChild(liTemp);
            }

            body.appendChild(ul);
            card.appendChild(body);
            col.appendChild(card);
            container.appendChild(col);
        });
    } catch (e) {
        showPaneError('math-health-cards', paneErrorMessage(0), loadMathHealth, false);
        console.log('Math health error:', e);
    }
}

document.getElementById('math-health-tab')?.addEventListener('shown.bs.tab', loadMathHealth);
