// SuperLocalMemory V3 - Clusters View
// Part of Qualixar | https://superlocalmemory.com

async function loadClusters() {
    showLoading('clusters-list', 'Loading clusters...');
    try {
        var response = await fetch('/api/clusters');
        var data = await response.json();
        renderClusters(data.clusters);
    } catch (error) {
        console.error('Error loading clusters:', error);
        showEmpty('clusters-list', 'collection', 'Failed to load clusters');
    }
}

function renderClusters(clusters) {
    var container = document.getElementById('clusters-list');
    if (!clusters || clusters.length === 0) {
        showEmpty('clusters-list', 'collection', 'No clusters found yet. Clusters form automatically as you store related memories.');
        return;
    }

    var colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#30cfd0', '#764ba2', '#f5576c'];
    container.textContent = '';

    clusters.forEach(function(cluster, idx) {
        var color = colors[idx % colors.length];

        var card = document.createElement('div');
        card.className = 'card mb-2';
        card.style.borderLeft = '4px solid ' + color;

        var body = document.createElement('div');
        body.className = 'card-body py-2 px-3';
        body.style.cursor = 'pointer';

        // Header row
        var headerRow = document.createElement('div');
        headerRow.className = 'd-flex justify-content-between align-items-center';

        var title = document.createElement('h6');
        title.className = 'mb-0';
        title.textContent = 'Cluster ' + cluster.cluster_id;

        var badges = document.createElement('div');
        var countBadge = document.createElement('span');
        countBadge.className = 'badge bg-secondary me-1';
        countBadge.textContent = cluster.member_count + ' memories';
        badges.appendChild(countBadge);

        if (cluster.avg_importance) {
            var impBadge = document.createElement('span');
            impBadge.className = 'badge bg-outline-primary';
            impBadge.style.cssText = 'border:1px solid #667eea; color:#667eea;';
            impBadge.textContent = 'imp: ' + parseFloat(cluster.avg_importance).toFixed(1);
            badges.appendChild(impBadge);
        }

        var expandIcon = document.createElement('i');
        expandIcon.className = 'bi bi-chevron-down ms-2';
        expandIcon.style.transition = 'transform 0.2s';
        badges.appendChild(expandIcon);

        headerRow.appendChild(title);
        headerRow.appendChild(badges);
        body.appendChild(headerRow);

        // Summary line (categories if available)
        if (cluster.categories) {
            var catLine = document.createElement('small');
            catLine.className = 'text-muted';
            catLine.textContent = cluster.categories;
            body.appendChild(catLine);
        }

        // Expandable member area (hidden by default)
        var memberArea = document.createElement('div');
        memberArea.className = 'mt-2';
        memberArea.style.display = 'none';
        memberArea.id = 'cluster-members-' + cluster.cluster_id;

        var loadingText = document.createElement('div');
        loadingText.className = 'text-center text-muted small py-2';
        loadingText.textContent = 'Loading members...';
        memberArea.appendChild(loadingText);

        body.appendChild(memberArea);
        card.appendChild(body);
        container.appendChild(card);

        // Click to expand/collapse
        var expanded = false;
        body.addEventListener('click', function(e) {
            expanded = !expanded;
            memberArea.style.display = expanded ? 'block' : 'none';
            expandIcon.style.transform = expanded ? 'rotate(180deg)' : 'rotate(0)';

            if (expanded && memberArea.children.length === 1 && memberArea.children[0] === loadingText) {
                loadClusterMembers(cluster.cluster_id, memberArea);
            }
        });
    });
}

async function loadClusterMembers(clusterId, container) {
    try {
        var response = await fetch('/api/clusters/' + clusterId + '?limit=10');
        var data = await response.json();
        container.textContent = '';

        // Show cluster summary if available
        if (data.summary) {
            var summaryDiv = document.createElement('div');
            summaryDiv.className = 'alert alert-light border-start border-3 border-primary py-2 px-3 mb-2 small';
            var summaryLabel = document.createElement('strong');
            summaryLabel.textContent = 'Summary: ';
            summaryDiv.appendChild(summaryLabel);
            summaryDiv.appendChild(document.createTextNode(data.summary));
            container.appendChild(summaryDiv);
        }

        if (!data.members || data.members.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'text-muted small';
            empty.textContent = 'No members found.';
            container.appendChild(empty);
            return;
        }

        data.members.forEach(function(m, i) {
            var row = document.createElement('div');
            row.className = 'border-bottom py-1';
            if (i === data.members.length - 1) row.className = 'py-1';

            var content = document.createElement('div');
            content.className = 'small';
            var text = m.content || m.summary || '';
            content.textContent = (i + 1) + '. ' + (text.length > 150 ? text.substring(0, 150) + '...' : text);
            row.appendChild(content);

            var meta = document.createElement('div');
            meta.className = 'text-muted';
            meta.style.fontSize = '0.7rem';
            var parts = [];
            if (m.category) parts.push(m.category);
            if (m.importance) parts.push('imp: ' + m.importance);
            if (m.created_at) parts.push(m.created_at.substring(0, 10));
            meta.textContent = parts.join(' | ');
            row.appendChild(meta);

            container.appendChild(row);
        });

        // View in graph button
        var graphBtn = document.createElement('button');
        graphBtn.className = 'btn btn-sm btn-outline-primary mt-2';
        graphBtn.textContent = 'View in Knowledge Graph';
        graphBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            filterGraphToCluster(clusterId);
        });
        container.appendChild(graphBtn);

    } catch (error) {
        container.textContent = '';
        var errDiv = document.createElement('div');
        errDiv.className = 'text-danger small';
        errDiv.textContent = 'Failed to load: ' + error.message;
        container.appendChild(errDiv);
    }
}

function filterGraphToCluster(clusterId) {
    var graphTab = document.querySelector('a[href="#graph"]');
    if (graphTab) graphTab.click();

    setTimeout(function() {
        if (typeof filterState !== 'undefined' && typeof filterByCluster === 'function' && typeof renderGraph === 'function') {
            filterState.cluster_id = clusterId;
            var filtered = filterByCluster(originalGraphData, clusterId);
            renderGraph(filtered);
            var url = new URL(window.location);
            url.searchParams.set('cluster_id', clusterId);
            window.history.replaceState({}, '', url);
        }
    }, 300);
}

function filterGraphByEntity(entity) {
    var graphTab = document.querySelector('a[href="#graph"]');
    if (graphTab) graphTab.click();

    setTimeout(function() {
        if (typeof filterState !== 'undefined' && typeof filterByEntity === 'function' && typeof renderGraph === 'function') {
            filterState.entity = entity;
            var filtered = filterByEntity(originalGraphData, entity);
            renderGraph(filtered);
        }
    }, 300);
}

function showClusterMemories(clusterId) {
    var memoriesTab = document.querySelector('a[href="#memories"]');
    if (memoriesTab) memoriesTab.click();
    if (typeof showToast === 'function') showToast('Filtering memories for cluster ' + clusterId);
}
