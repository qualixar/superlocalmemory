// SuperLocalMemory V2.6.5 - Interactive Knowledge Graph - UI Elements Module
// Copyright (c) 2026 Varun Pratap Bhardwaj — GNU Affero General Public License v3.0 or later
// Part of modular graph visualization system (split from monolithic graph-cytoscape.js)

// ============================================================================
// FILTER BADGE UI
// ============================================================================

function updateFilterBadge() {
    const statusFull = document.getElementById('graph-status-full');
    const statusFiltered = document.getElementById('graph-status-filtered');
    const filterDescription = document.getElementById('graph-filter-description');
    const filterCount = document.getElementById('graph-filter-count');
    const statusFullText = document.getElementById('graph-status-full-text');

    const hasFilter = filterState.cluster_id || filterState.entity;

    if (hasFilter) {
        // FILTERED STATE - Show prominent alert with "Show All Memories" button
        if (statusFull) statusFull.style.display = 'none';
        if (statusFiltered) statusFiltered.style.display = 'block';

        // Update filter description
        if (filterDescription) {
            const text = filterState.cluster_id
                ? `Viewing Cluster ${filterState.cluster_id}`
                : `Viewing: ${filterState.entity}`;
            filterDescription.textContent = text;
        }

        // Update count (will be set after graph renders)
        if (filterCount && graphData && graphData.nodes) {
            filterCount.textContent = `(${graphData.nodes.length} ${graphData.nodes.length === 1 ? 'memory' : 'memories'})`;
        }

        console.log('[updateFilterBadge] Showing FILTERED state');
    } else {
        // FULL GRAPH STATE - Show normal status with refresh button
        if (statusFull) statusFull.style.display = 'block';
        if (statusFiltered) statusFiltered.style.display = 'none';

        // Update count
        if (statusFullText && graphData && graphData.nodes) {
            const maxNodes = document.getElementById('graph-max-nodes')?.value || 50;
            statusFullText.textContent = `Showing ${graphData.nodes.length} of ${maxNodes} memories`;
        }

        console.log('[updateFilterBadge] Showing FULL state');
    }
}

// ============================================================================
// GRAPH STATS
// ============================================================================

function updateGraphStats(data) {
    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
        // Clear and rebuild safely
        statsEl.textContent = '';

        const nodeBadge = document.createElement('span');
        nodeBadge.className = 'badge bg-primary';
        nodeBadge.textContent = `${data.nodes.length} nodes`;
        statsEl.appendChild(nodeBadge);

        const edgeBadge = document.createElement('span');
        edgeBadge.className = 'badge bg-secondary';
        edgeBadge.textContent = `${data.links.length} edges`;
        statsEl.appendChild(document.createTextNode(' '));
        statsEl.appendChild(edgeBadge);

        const clusterBadge = document.createElement('span');
        clusterBadge.className = 'badge bg-info';
        clusterBadge.textContent = `${data.clusters?.length || 0} clusters`;
        statsEl.appendChild(document.createTextNode(' '));
        statsEl.appendChild(clusterBadge);
    }
}

// ============================================================================
// LOADING SPINNER
// ============================================================================

function showLoadingSpinner() {
    const container = document.getElementById('graph-container');
    if (container) {
        container.textContent = ''; // Clear safely

        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'text-align:center; padding:100px;';

        const spinner = document.createElement('div');
        spinner.className = 'spinner-border text-primary';
        spinner.setAttribute('role', 'status');
        wrapper.appendChild(spinner);

        const text = document.createElement('p');
        text.style.marginTop = '20px';
        text.textContent = 'Loading graph...';
        wrapper.appendChild(text);

        container.appendChild(wrapper);
    }
}

function hideLoadingSpinner() {
    // Do nothing - renderGraph() already cleared the spinner
    // If we clear here, we destroy the Cytoscape canvas!
    console.log('[hideLoadingSpinner] Graph already rendered, spinner cleared by renderGraph()');
}

function showError(message) {
    const container = document.getElementById('graph-container');
    if (container) {
        container.textContent = ''; // Clear safely

        const alert = document.createElement('div');
        alert.className = 'alert alert-danger';
        alert.setAttribute('role', 'alert');
        alert.style.margin = '50px';
        alert.textContent = message;
        container.appendChild(alert);
    }
}

// ============================================================================
// LAYOUT MANAGEMENT
// ============================================================================

function saveLayoutPositions() {
    if (!cy) return;

    const positions = {};
    cy.nodes().forEach(node => {
        positions[node.id()] = node.position();
    });

    try {
        localStorage.setItem('slm_graph_layout', JSON.stringify(positions));
    } catch (e) {
        console.warn('Failed to save graph layout:', e);
    }
}

function restoreSavedLayout() {
    if (!cy) return;

    try {
        const saved = localStorage.getItem('slm_graph_layout');
        if (saved) {
            const positions = JSON.parse(saved);
            cy.nodes().forEach(node => {
                const pos = positions[node.id()];
                if (pos) {
                    node.position(pos);
                }
            });
        }
    } catch (e) {
        console.warn('Failed to restore graph layout:', e);
    }
}

function changeGraphLayout(layoutName) {
    if (!cy) return;

    currentLayout = layoutName;
    const layout = cy.layout(getLayoutConfig(layoutName));
    layout.run();

    // Save preference
    localStorage.setItem('slm_graph_layout_preference', layoutName);
}

// ============================================================================
// EXPAND NEIGHBORS
// ============================================================================

function expandNeighbors(memoryId) {
    if (!cy) return;

    const node = cy.getElementById(String(memoryId));
    if (!node || node.length === 0) return;

    // Hide all nodes and edges
    cy.elements().addClass('dimmed');

    // Show target node + neighbors + connecting edges
    node.removeClass('dimmed');
    node.neighborhood().removeClass('dimmed');
    node.connectedEdges().removeClass('dimmed');

    // Fit view to visible elements
    cy.fit(node.neighborhood().union(node), 50);
}

// ============================================================================
// SCREEN READER STATUS
// ============================================================================

function updateScreenReaderStatus(message) {
    let statusRegion = document.getElementById('graph-sr-status');
    if (!statusRegion) {
        statusRegion = document.createElement('div');
        statusRegion.id = 'graph-sr-status';
        statusRegion.setAttribute('role', 'status');
        statusRegion.setAttribute('aria-live', 'polite');
        statusRegion.setAttribute('aria-atomic', 'true');
        statusRegion.style.cssText = 'position:absolute; left:-10000px; width:1px; height:1px; overflow:hidden;';
        document.body.appendChild(statusRegion);
    }
    statusRegion.textContent = message;
}
